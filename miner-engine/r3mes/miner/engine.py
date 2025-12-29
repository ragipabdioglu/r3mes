#!/usr/bin/env python3
"""
R3MES Miner Engine

Production-ready mining engine that:
1. Trains BitNet model with LoRA adapters
2. Uploads gradients to IPFS (active role)
3. Submits IPFS hash to blockchain via gRPC
4. Tracks mining statistics and rewards
"""

import sys
import os
from pathlib import Path

# Add miner-engine directory to path for backward compatibility
miner_engine_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(miner_engine_dir))

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Iterator
import time
import hashlib
import base64

from core.bitlinear import BitLinear
from core.trainer import LoRATrainer
from core.serialization import LoRASerializer
from core.binary_serialization import BinaryGradientSerializer
from core.deterministic import configure_deterministic_execution
from core.coordinator import OffChainDistributedCoordinator
from core.gradient_accumulator import GradientAccumulator
from core.gradient_compression import compress_gradients, CompressedGradient
from utils.ipfs_client import IPFSClient
from utils.gpu_detection import GPUArchitectureDetector
from utils.shard_assignment import calculate_shard_id
from utils.environment_validator import EnvironmentValidator
from utils.log_streamer import setup_miner_logging
from bridge.blockchain_client import BlockchainClient
from bridge.arrow_flight_client import ArrowFlightClient
from r3mes.miner.vram_profiler import detect_vram_profile, apply_profile_to_model, create_optimizer_from_profile
from r3mes.miner.llama_loader import load_llama3_8b_model, get_model_info
from r3mes.miner.model_loader import load_model_with_enforced_lora, validate_lora_only_training
from r3mes.miner.lora_manager import LoRAManager
from r3mes.miner.task_pool_client import TaskPoolClient
from r3mes.miner.chunk_processor import ChunkProcessor
import logging

# Initialize logger early for use throughout the module
logger = logging.getLogger(__name__)


class SimpleBitNetModel(nn.Module):
    """Simple BitNet model for testing."""
    
    def __init__(self, hidden_size: int = 768, num_layers: int = 2, lora_rank: int = 8):
        super().__init__()
        self.layers = nn.ModuleList([
            BitLinear(hidden_size, hidden_size, lora_rank=lora_rank, deterministic=True)
            for _ in range(num_layers)
        ])
        self.output = BitLinear(hidden_size, hidden_size, lora_rank=lora_rank, deterministic=True)
    
    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        x = self.output(x)
        return x


class MinerEngine:
    """Main miner engine class."""
    
    def __init__(
        self,
        private_key: str,
        blockchain_url: Optional[str] = None,
        chain_id: str = "remes-test",
        model_hidden_size: int = 768,
        lora_rank: int = 8,
        learning_rate: float = 1e-4,
        deterministic: bool = True,
        gradient_accumulation_steps: int = 4,
        top_k_compression: float = 0.1,
        use_tls: bool = False,
        tls_cert_file: Optional[str] = None,
        tls_key_file: Optional[str] = None,
        tls_ca_file: Optional[str] = None,
        tls_server_name: Optional[str] = None,
    ):
        """
        Initialize miner engine.
        
        Args:
            private_key: Private key for blockchain transactions
            blockchain_url: gRPC endpoint URL
            chain_id: Chain ID
            model_hidden_size: Model hidden size
            lora_rank: LoRA rank
            learning_rate: Learning rate for training
            deterministic: Enable deterministic execution (default: True)
            gradient_accumulation_steps: Number of gradient accumulation steps
            top_k_compression: Top-k compression ratio (0.0-1.0)
            use_tls: Enable TLS/mTLS for gRPC connection to Go node
            tls_cert_file: Path to client TLS certificate (PEM)
            tls_key_file: Path to client TLS private key (PEM)
            tls_ca_file: Path to CA certificate for server verification (PEM)
            tls_server_name: Expected server name for TLS verification
        """
        # Get blockchain URL from parameter, environment variable, or default
        # In production, R3MES_NODE_GRPC_URL must be set (no localhost fallback)
        is_production = os.getenv("R3MES_ENV", "development").lower() == "production"
        
        if blockchain_url:
            self.blockchain_url = blockchain_url
        else:
            blockchain_url_env = os.getenv("R3MES_NODE_GRPC_URL")
            if not blockchain_url_env:
                if is_production:
                    raise ValueError(
                        "R3MES_NODE_GRPC_URL environment variable must be set in production. "
                        "Do not use localhost in production."
                    )
                # Development fallback
                self.blockchain_url = "localhost:9090"
                logger.warning("R3MES_NODE_GRPC_URL not set, using localhost fallback (development only)")
            else:
                self.blockchain_url = blockchain_url_env
                # Validate that production doesn't use localhost
                if is_production and ("localhost" in self.blockchain_url or "127.0.0.1" in self.blockchain_url):
                    raise ValueError(
                        f"R3MES_NODE_GRPC_URL cannot use localhost in production: {self.blockchain_url}"
                    )
        
        self.private_key = private_key
        self.chain_id = chain_id
        
        # Serving node configuration (from environment variables)
        self.serving_node_enabled = os.getenv("R3MES_ENABLE_SERVING_NODE", "false").lower() == "true"
        self._serving_node_initialized = False
        self.available_lora_list = []
        
        # Blockchain client (needed for global seed retrieval)
        self.blockchain_client = BlockchainClient(
            node_url=self.blockchain_url,
            private_key=private_key,
            chain_id=chain_id,
            use_tls=use_tls,
            tls_cert_file=tls_cert_file,
            tls_key_file=tls_key_file,
            tls_ca_file=tls_ca_file,
            tls_server_name=tls_server_name,
        )
        
        # Get global seed from blockchain for deterministic execution
        global_seed = None
        if deterministic:
            # Try to get global seed from blockchain (training round 0 for initial setup)
            # In production, this would be called with the actual training round ID
            global_seed = self.blockchain_client.get_global_seed(training_round_id=0)
            if global_seed is None:
                logger.warning("Could not retrieve global seed from blockchain, using fixed seed for testing")
        
        # Configure deterministic execution with global seed
        if deterministic:
            configure_deterministic_execution(global_seed=global_seed)
        
        # Environment validation (strict version matching for deterministic results)
        self.env_validator = EnvironmentValidator()
        is_valid, errors = self.env_validator.validate_all()
        if not is_valid:
            logger.warning("WARNING: Environment validation failed!")
            for error in errors:
                logger.warning(f"   - {error}")
            logger.warning("Continuing anyway, but results may not be deterministic.")
        else:
            logger.info("Environment validation passed")
        
        # Print environment info
        env_info = self.env_validator.get_environment_info()
        logger.info(f"Environment: Python {env_info.get('python_version')}, PyTorch {env_info.get('pytorch_version')}")
        if env_info.get('cuda_available'):
            logger.info(f"CUDA: {env_info.get('cuda_version')}, cuDNN: {env_info.get('cudnn_version')}")
        
        # GPU detection
        self.gpu_detector = GPUArchitectureDetector()
        self.gpu_architecture = self.gpu_detector.get_architecture()
        logger.info(f"GPU Architecture: {self.gpu_architecture}")
        
        # Configure deterministic CUDA based on GPU architecture
        if deterministic:
            self.gpu_detector.configure_deterministic_cuda()
        
        # Adaptive VRAM Scaling: Detect VRAM and apply profile automatically
        if torch.cuda.is_available():
            try:
                self.vram_profile = detect_vram_profile()
                # Override gradient_accumulation_steps with profile value
                gradient_accumulation_steps = self.vram_profile["gradient_accumulation"]
                logger.info(f"Overriding gradient_accumulation_steps to {gradient_accumulation_steps} (from VRAM profile)")
            except Exception as e:
                logger.warning(f"Failed to detect VRAM profile: {e}")
                logger.warning("Using default configuration")
                self.vram_profile = None
        else:
            self.vram_profile = None
        
        # Load model - Try GGUF first (llama-cpp-python), then PyTorch, then fallback
        use_gguf = os.getenv("R3MES_USE_GGUF", "true").lower() == "true"
        use_real_model = os.getenv("R3MES_USE_LLAMA3", "true").lower() == "true"
        
        # Try GGUF model first (preferred for lower memory usage)
        if use_gguf and use_real_model:
            try:
                from r3mes.miner.gguf_loader import find_gguf_model, load_gguf_model
                
                # Find GGUF model file
                model_path = os.getenv("R3MES_GGUF_MODEL_PATH")
                if model_path:
                    # Normalize path (resolve relative paths)
                    if not Path(model_path).is_absolute():
                        model_path = str(Path.cwd() / model_path)
                    else:
                        model_path = str(Path(model_path).resolve())
                else:
                    model_path = find_gguf_model()
                    # find_gguf_model may return None, and it already handles path resolution
                
                if model_path and os.path.exists(model_path):
                    logger.info(f"Loading GGUF model: {model_path}")
                    n_ctx = int(os.getenv("R3MES_GGUF_N_CTX", "2048"))
                    self.gguf_loader = load_gguf_model(
                        model_path=model_path,
                        n_gpu_layers=-1,  # Use all GPU layers
                        n_ctx=n_ctx,
                    )
                    self.model = self.gguf_loader.llm
                    self.tokenizer = None  # GGUF models don't use separate tokenizer
                    self.gguf_loader_instance = self.gguf_loader  # Store reference for inference
                    logger.info("✅ GGUF model loaded successfully (llama-cpp-python)")
                    use_real_model = False  # Skip PyTorch loading
                else:
                    logger.warning("GGUF model file not found, trying PyTorch loader...")
            except ImportError as e:
                logger.warning(f"llama-cpp-python not available: {e}")
                logger.warning("Falling back to PyTorch loader...")
            except Exception as e:
                logger.warning(f"Failed to load GGUF model: {e}")
                logger.warning("Falling back to PyTorch loader...")
        
        # Try PyTorch model loader (fallback)
        if use_real_model:
            try:
                logger.info("Loading Llama 3 8B model from HuggingFace (PyTorch)...")
                self.model, self.tokenizer = load_llama3_8b_model(
                    model_name=os.getenv("R3MES_MODEL_NAME", "meta-llama/Meta-Llama-3-8B"),
                    lora_rank=lora_rank,
                    lora_alpha=16,
                    lora_dropout=0.1,
                    device_map="auto",
                    use_quantization=True,
                    quantization_bits=4,  # 4-bit quantization for VRAM efficiency
                )
                
                # Print model info
                model_info = get_model_info(self.model)
                logger.info(f"Model loaded: {model_info['trainable_parameters']:,} trainable parameters")
                
                # LoRA is already applied by load_llama3_8b_model
                # Validate LoRA-only training (skip for now, will validate after model loading)
                pass
                
            except Exception as e:
                logger.warning(f"Failed to load Llama 3 8B model: {e}")
                logger.warning("Falling back to SimpleBitNetModel for testing")
                use_real_model = False
        
        if not use_real_model:
            # Fallback to SimpleBitNetModel for testing/development
            logger.info("Using SimpleBitNetModel (test model)")
            base_model = SimpleBitNetModel(
                hidden_size=model_hidden_size,
                num_layers=2,
                lora_rank=lora_rank,
            )
            
            # LoRA-Enforced Architecture: Load model with mandatory LoRA
            self.model = load_model_with_enforced_lora(
                base_model,
                lora_rank=lora_rank,
                lora_alpha=16,
            )
            self.tokenizer = None  # SimpleBitNetModel doesn't use tokenizer
        
        # Validate LoRA-only training
        validate_lora_only_training(self.model)
        
        # Initialize serving node LoRAs if enabled (after model is loaded)
        if self.serving_node_enabled and not self._serving_node_initialized:
            if self.initialize_serving_node_loras():
                # Register as serving node
                self.register_as_serving_node()
                # Start inference server
                self._start_inference_server()
                # Start heartbeat thread
                self._start_heartbeat_thread()
                self._serving_node_initialized = True
        
        # Apply VRAM profile optimizations to model
        if self.vram_profile:
            self.model = apply_profile_to_model(self.model, self.vram_profile)
        
        # Create trainer with VRAM-aware optimizer
        if self.vram_profile:
            # Get LoRA parameters for optimizer
            lora_params = []
            for module in self.model.modules():
                if isinstance(module, BitLinear):
                    lora_params.append(module.lora_A)
                    lora_params.append(module.lora_B)
            
            # Create optimizer from profile
            profile_optimizer = create_optimizer_from_profile(
                lora_params,
                self.vram_profile,
                learning_rate=learning_rate
            )
            
            # Create trainer with custom optimizer
            self.trainer = LoRATrainer(
                self.model,
                learning_rate=learning_rate,
                deterministic=True,
                custom_optimizer=profile_optimizer,
            )
        else:
            # Use default trainer
            self.trainer = LoRATrainer(
                self.model,
                learning_rate=learning_rate,
                deterministic=True,
            )
        
        # Serializer
        self.serializer = LoRASerializer()
        # Binary serializer (Protocol Buffers) - optional, falls back to pickle if unavailable
        self.binary_serializer = BinaryGradientSerializer()
        
        # Gradient accumulator (bandwidth optimization)
        # Use profile value if available, otherwise use provided value
        final_gradient_accumulation = (
            self.vram_profile["gradient_accumulation"] 
            if self.vram_profile 
            else gradient_accumulation_steps
        )
        self.gradient_accumulator = GradientAccumulator(
            accumulation_steps=final_gradient_accumulation
        )
        self.top_k_compression = top_k_compression
        logger.info(f"Gradient accumulation: {final_gradient_accumulation} steps")
        logger.info(f"Top-k compression: {top_k_compression*100:.1f}% (keeping top {top_k_compression*100:.1f}% of values)")
        
        # IPFS client (active role)
        # Try to ensure IPFS is available (embedded daemon)
        try:
            from r3mes.utils.ipfs_manager import ensure_ipfs_available, is_ipfs_running
            if not is_ipfs_running():
                logger.info("IPFS daemon not running. Setting up embedded IPFS...")
                success, message = ensure_ipfs_available()
                if success:
                    logger.info(f"{message}")
                else:
                    logger.warning(f"{message}")
                    logger.warning("Will try to use system IPFS or fallback to simulated mode")
        except ImportError:
            # IPFS manager not available, continue with standard client
            pass
        except Exception as e:
            logger.warning(f"Error setting up embedded IPFS: {e}")
            logger.warning("Will try to use system IPFS or fallback to simulated mode")
        
        self.ipfs_client = IPFSClient()
        if not self.ipfs_client.is_connected():
            logger.warning("Warning: IPFS not connected, using simulated mode")
        
        # Task Pool Client for claiming and processing chunks
        max_prefetch = int(os.getenv("R3MES_MAX_PREFETCH", "5"))
        self.task_pool_client = TaskPoolClient(
            grpc_client=self.blockchain_client,
            max_prefetch=max_prefetch
        )
        
        # Chunk Processor for processing fixed-size chunks
        local_batch_size = int(os.getenv("R3MES_LOCAL_BATCH_SIZE", "1"))
        self.chunk_processor = ChunkProcessor(local_batch_size=local_batch_size)
        
        # Blockchain client already initialized above for global seed retrieval
        
        # Arrow Flight client (optional - for zero-copy tensor transfer)
        # Falls back to gRPC if Arrow Flight unavailable
        # Extract host from blockchain_url (remove port if present)
        arrow_flight_host = self.blockchain_url.split(":")[0] if ":" in self.blockchain_url else self.blockchain_url
        # In production, validate that host is not localhost
        if is_production and (arrow_flight_host == "localhost" or arrow_flight_host == "127.0.0.1"):
            raise ValueError(
                f"Arrow Flight host cannot be localhost in production: {arrow_flight_host}"
            )
        arrow_flight_port = int(os.getenv("R3MES_ARROW_FLIGHT_PORT", "8815"))
        self.arrow_flight_client = ArrowFlightClient(
            host=arrow_flight_host,
            port=arrow_flight_port,
        )
        if self.arrow_flight_client.is_connected():
            logger.info("Arrow Flight connected (zero-copy tensor transfer enabled)")
        else:
            logger.info("Arrow Flight not available, using gRPC for tensor transfer")
        
        # Distributed training coordinator
        self.coordinator = OffChainDistributedCoordinator(
            ipfs_client=self.ipfs_client,
            blockchain_client=self.blockchain_client,
            model_version=str(model_hidden_size),  # Use hidden_size as version for now
        )
        
        # Mining state
        self.training_round_id = 0
        self.total_submissions = 0
        self.successful_submissions = 0
        
        # Initialize stats collector for WebSocket integration
        try:
            from r3mes.miner.stats_server import initialize_stats_collector
            initialize_stats_collector(self)
        except ImportError:
            # Stats collector not available, continue without it
            pass
        
        # Start HTTP stats server for Desktop Launcher integration
        # This runs in a background thread and exposes /stats endpoint on port 8080
        try:
            from r3mes.miner.stats_http_server import start_stats_server, is_server_running
            if not is_server_running():
                stats_port = int(os.getenv("R3MES_STATS_PORT", "8080"))
                stats_host = os.getenv("R3MES_STATS_HOST", "0.0.0.0")
                start_stats_server(port=stats_port, host=stats_host, blocking=False)
                logger.info(f"Stats HTTP server started on {stats_host}:{stats_port}")
        except Exception as e:
            logger.warning(f"Failed to start stats HTTP server: {e}")
        
        # Setup logging with WebSocket streaming
        # Get miner address from private key
        try:
            from bridge.crypto import private_key_to_address
            miner_address = private_key_to_address(private_key)
        except ImportError:
            logger.debug("bridge.crypto not available, miner_address will be None")
            miner_address = None
        except Exception as e:
            logger.warning(f"Failed to derive miner address from private key: {e}")
            miner_address = None
        
        # Setup logger with WebSocket streaming
        # Get WebSocket port from environment or use REST port (1317) as default
        ws_port = os.getenv("R3MES_WS_PORT")
        if not ws_port:
            # Try to extract port from REST URL if available
            rest_url = os.getenv("R3MES_BLOCKCHAIN_REST_URL")
            if rest_url and ":" in rest_url.split("://")[-1]:
                ws_port = rest_url.split(":")[-1].split("/")[0]
            else:
                ws_port = "1317"  # Default Cosmos SDK REST port
        
        ws_path = os.getenv("R3MES_WS_PATH", "/ws?topic=miner_logs")
        ws_host = self.blockchain_url.split(":")[0] if ":" in self.blockchain_url else self.blockchain_url
        ws_url = f"ws://{ws_host}:{ws_port}{ws_path}"
        self.logger = setup_miner_logging(
            miner_address=miner_address,
            ws_url=ws_url,
            log_level=logging.INFO,
        )
    
    def _get_global_seed_for_round(self, training_round_id: int) -> Optional[int]:
        """
        Get global seed for a specific training round from blockchain.
        
        PRODUCTION MODE: Fail-fast if seed cannot be retrieved
        TEST MODE: Allow fallback (controlled by R3MES_TEST_MODE environment variable)
        """
        import os
        
        is_test_mode = os.getenv("R3MES_TEST_MODE", "false").lower() == "true"
        
        try:
            seed = self.blockchain_client.get_global_seed(training_round_id=training_round_id)
            if seed is None:
                if is_test_mode:
                    logger.warning(
                        f"Could not retrieve global seed for round {training_round_id} (TEST MODE - using fallback)"
                    )
                    return None
                else:
                    # PRODUCTION MODE: Fail-fast
                    raise RuntimeError(
                        f"CRITICAL: Could not retrieve global seed for training round {training_round_id}. "
                        f"Deterministic execution requires valid seed from blockchain. "
                        f"Miner must stop execution."
                    )
            return seed
        except Exception as e:
            if is_test_mode:
                logger.warning(
                    f"Could not retrieve global seed for round {training_round_id}: {e} (TEST MODE - using fallback)"
                )
                return None
            else:
                # PRODUCTION MODE: Fail-fast
                raise RuntimeError(
                    f"CRITICAL: Failed to retrieve global seed for training round {training_round_id}: {e}. "
                    f"Deterministic execution requires valid seed from blockchain. "
                    f"Miner must stop execution."
                ) from e
    
    def train_and_submit(self, num_iterations: int = 1):
        """
        Train model and submit gradients to blockchain.
        
        Args:
            num_iterations: Number of training iterations
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting mining loop ({num_iterations} iterations)")
        logger.info(f"{'='*60}\n")
        
        # Get global seed for current training round
        # In production, this MUST succeed (fail-fast)
        global_seed = self._get_global_seed_for_round(self.training_round_id)
        if global_seed is not None:
            logger.info(f"🔒 Using global seed from blockchain: {global_seed} (training round {self.training_round_id})")
            # Reconfigure deterministic execution with new seed
            configure_deterministic_execution(global_seed=global_seed)
        else:
            # This should only happen in TEST MODE
            import os
            is_test_mode = os.getenv("R3MES_TEST_MODE", "false").lower() == "true"
            if is_test_mode:
                logger.warning("⚠️  Using previously configured seed (could not retrieve from blockchain - TEST MODE)")
            else:
                # PRODUCTION MODE: This should never happen (fail-fast already raised)
                raise RuntimeError(
                    "CRITICAL: Global seed is None in production mode. "
                    "This should have been caught by fail-fast mechanism."
                )
        
        for iteration in range(num_iterations):
            logger.info(f"Iteration {iteration + 1}/{num_iterations}")
            logger.debug("-" * 60)
            
            # Initialize task and chunk_data for this iteration
            task = None
            chunk_data = None
            
            try:
                # 1. Get task from task pool (claim chunk)
                active_pool_id = self.task_pool_client.get_active_pool()
                if not active_pool_id:
                    logger.warning("No active task pool found, using random data for testing")
                    # Fallback to random data if no task pool available
                    inputs = torch.randn(32, 768)
                    targets = torch.randn(32, 768)
                else:
                    # Claim task from pool
                    task = self.task_pool_client.get_next_task()
                    if not task:
                        logger.warning("No available tasks in pool, using random data for testing")
                        inputs = torch.randn(32, 768)
                        targets = torch.randn(32, 768)
                    else:
                        logger.info(f"Claimed task: chunk_id={task['chunk_id']}, data_hash={task['data_hash'][:16]}...")
                        
                        # 2. Download chunk data from IPFS
                        chunk_data = self.task_pool_client.download_chunk_data(
                            data_hash=task['data_hash'],
                            ipfs_client=self.ipfs_client
                        )
                        
                        if chunk_data is None:
                            logger.error(f"Failed to download chunk data for chunk_id={task['chunk_id']}")
                            logger.warning("Falling back to random data")
                            inputs = torch.randn(32, 768)
                            targets = torch.randn(32, 768)
                            task = None  # Don't complete task if data download failed
                        else:
                            # Extract input_ids and labels from chunk data
                            inputs = chunk_data.get("input_ids")
                            targets = chunk_data.get("labels", inputs)
                            
                            # Ensure tensors are properly shaped
                            if isinstance(inputs, torch.Tensor):
                                # Add batch dimension if missing
                                if inputs.dim() == 1:
                                    inputs = inputs.unsqueeze(0)
                            if isinstance(targets, torch.Tensor):
                                if targets.dim() == 1:
                                    targets = targets.unsqueeze(0)
                            
                            logger.info(f"Using real chunk data: shape={inputs.shape if isinstance(inputs, torch.Tensor) else 'unknown'}")
                
                # 3. Training step with real or fallback data
                loss, gradients_dict = self.trainer.train_step(inputs, targets)
                logger.info(f"Training loss: {loss:.6f}")
                
                # Calculate gradient norm from gradients dict
                def calculate_gradient_norm(gradients_dict: Dict[str, torch.Tensor]) -> float:
                    """Calculate L2 norm of all gradients."""
                    total_norm = 0.0
                    for grad in gradients_dict.values():
                        if grad is not None and isinstance(grad, torch.Tensor):
                            param_norm = grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    return total_norm ** (1. / 2)
                
                gradient_norm = calculate_gradient_norm(gradients_dict)
                
                # Get accuracy from trainer if available
                accuracy = 0.0
                if hasattr(self.trainer, 'get_validation_accuracy'):
                    try:
                        accuracy = self.trainer.get_validation_accuracy()
                    except (AttributeError, RuntimeError) as e:
                        logger.debug(f"Could not get validation accuracy: {e}")
                elif hasattr(self.trainer, 'last_accuracy'):
                    accuracy = self.trainer.last_accuracy
                elif hasattr(self.trainer, 'metrics') and 'accuracy' in self.trainer.metrics:
                    accuracy = self.trainer.metrics['accuracy']
                
                # Update training metrics for stats collector
                try:
                    from r3mes.miner.stats_server import get_stats_collector
                    stats_collector = get_stats_collector()
                    if stats_collector:
                        epoch = self.training_round_id
                        stats_collector.update_training_metrics(epoch, float(loss), accuracy, gradient_norm)
                except (ImportError, AttributeError):
                    # Stats collector not available
                    pass
                
                # Store gradient names for later reconstruction
                gradient_names = list(gradients_dict.keys())
                
                # Convert gradients dict to list (gradient_accumulator expects List[Tensor])
                gradients_list = list(gradients_dict.values())
                
                # 1.5. Accumulate gradients (bandwidth optimization)
                accumulated_gradients_list = self.gradient_accumulator.accumulate(gradients_list)
                
                # If accumulation not complete, skip submission
                if accumulated_gradients_list is None:
                    logger.debug(f"Accumulating... ({self.gradient_accumulator.step_count}/{self.gradient_accumulator.accumulation_steps})")
                    continue
                
                logger.info(f"Accumulation complete ({self.gradient_accumulator.accumulation_steps} steps)")
                
                # Reconstruct gradients dictionary from accumulated list
                accumulated_gradients_dict = dict(zip(gradient_names, accumulated_gradients_list))
                
                # 1.6. Compress gradients (top-k compression for bandwidth optimization)
                if self.top_k_compression > 0 and self.top_k_compression < 1.0:
                    compressed = compress_gradients(accumulated_gradients_list, k=self.top_k_compression)
                    compression_ratio = compressed._calculate_compression_ratio()
                    logger.info(f"Compression complete: {compression_ratio*100:.1f}% size reduction")
                    # Use compressed gradients for hash computation
                    # Note: In production, you might want to hash the compressed format
                    gradients_for_hash = accumulated_gradients_dict
                else:
                    gradients_for_hash = accumulated_gradients_dict
                
                # 2. Compute gradient hash from accumulated gradients
                gradient_hash = self.trainer.compute_gradient_hash(gradients_for_hash)
                logger.debug(f"Gradient hash: {gradient_hash[:16]}...")
                
                # 2.5. Get or start training round
                if self.training_round_id == 0:
                    # Start new training round
                    training_round = self.coordinator.start_training_round()
                    self.training_round_id = training_round.round_id
                else:
                    # Get existing training round
                    training_round = self.coordinator.get_training_round(self.training_round_id)
                    if training_round is None:
                        # Round not found, start new one
                        training_round = self.coordinator.start_training_round()
                        self.training_round_id = training_round.round_id
                
                # 2.6. Get global seed for this training round (seed locking)
                global_seed = self._get_global_seed_for_round(self.training_round_id)
                if global_seed is not None:
                    # Reconfigure deterministic execution with locked seed
                    configure_deterministic_execution(global_seed=global_seed)
                    logger.debug(f"Seed locked: {global_seed} (training round {self.training_round_id})")
                
                # 3. Get miner address for metadata (needed for binary serialization)
                miner_address = self.blockchain_client.get_miner_address()
                
                # 4. Serialize LoRA state (using accumulated gradients)
                # Update trainer with accumulated gradients for serialization
                lora_state = self.trainer.get_lora_state_dict()
                # Note: In production, accumulated gradients would be used to update LoRA state
                
                # Try binary serialization first (Protocol Buffers - ~30% bandwidth reduction)
                try:
                    serialized = self.binary_serializer.serialize_gradients(
                        accumulated_gradients_dict,  # Dictionary format
                        metadata=self.trainer.get_training_metadata(),
                        training_round_id=self.training_round_id,
                        miner_address=miner_address,
                        gradient_hash=gradient_hash,
                    )
                    logger.info("Using Protocol Buffers serialization (binary format)")
                except Exception as e:
                    # Fallback to pickle-based serialization
                    logger.warning(f"Binary serialization failed, using pickle: {e}")
                    serialized = self.serializer.serialize_lora_state(
                        lora_state,
                        metadata=self.trainer.get_training_metadata(),
                    )
                
                serialized_size_mb = len(serialized) / (1024 * 1024)
                logger.info(f"Serialized LoRA size: {serialized_size_mb:.4f} MB")
                
                # 4.5. Upload to IPFS (active role)
                # Optionally use Arrow Flight for zero-copy transfer if available
                if self.arrow_flight_client.is_connected():
                    # Try Arrow Flight first (zero-copy) - expects List[Tensor]
                    flight_path = self.arrow_flight_client.upload_gradients(
                        accumulated_gradients_list,  # List format
                        metadata={
                            "miner": miner_address,
                            "training_round_id": self.training_round_id,
                            "gradient_hash": gradient_hash,
                        }
                    )
                    if flight_path:
                        logger.info(f"Uploaded via Arrow Flight (zero-copy): {flight_path}")
                        # Still upload to IPFS for compatibility
                        logger.info("Uploading to IPFS (backup)...")
                        ipfs_hash = self.ipfs_client.upload_lora_state(serialized)
                        logger.info(f"IPFS hash: {ipfs_hash}")
                    else:
                        # Fallback to IPFS
                        logger.info("Uploading to IPFS...")
                        ipfs_hash = self.ipfs_client.upload_lora_state(serialized)
                        logger.info(f"IPFS hash: {ipfs_hash}")
                else:
                    # Standard IPFS upload
                    logger.info("Uploading to IPFS...")
                    ipfs_hash = self.ipfs_client.upload_lora_state(serialized)
                    logger.info(f"IPFS hash: {ipfs_hash}")
                
                # 4.6. Complete task if we claimed one and successfully processed it
                if task and chunk_data is not None:
                    success = self.task_pool_client.complete_task(
                        chunk_id=task['chunk_id'],
                        gradient_hash=ipfs_hash
                    )
                    if success:
                        logger.info(f"Task {task['chunk_id']} completed successfully")
                    else:
                        logger.warning(f"Failed to complete task {task['chunk_id']}")
                
                # 4.7. (Opsiyonel) Proof of Replication (PoRep) üret ve IPFS'e yükle
                porep_proof_ipfs_hash = None
                try:
                    # PoRep, Go tarafındaki VerifyPoRep ile uyumlu olacak şekilde üretilir.
                    # data_hash: SHA256(serialized gradient bytes) -> hex string
                    data_hash_bytes = hashlib.sha256(serialized).digest()
                    data_hash_hex = data_hash_bytes.hex()

                    # replica_hash: SHA256(data || miner_address) -> hex string
                    replica = serialized + miner_address.encode("utf-8")
                    replica_hash_bytes = hashlib.sha256(replica).digest()
                    replica_hash_hex = replica_hash_bytes.hex()

                    # Basit Merkle ağacı (Go'daki createMerkleTree ile aynı mantık)
                    porep_chunk_size = int(os.getenv("R3MES_POREP_CHUNK_SIZE", "1024"))
                    chunk_size = porep_chunk_size
                    if len(serialized) == 0:
                        chunks = [b""]
                    else:
                        chunks = [
                            serialized[i : i + chunk_size]
                            for i in range(0, len(serialized), chunk_size)
                        ]
                    level = [hashlib.sha256(chunk).digest() for chunk in chunks]
                    while len(level) > 1:
                        next_level = []
                        for i in range(0, len(level), 2):
                            if i + 1 < len(level):
                                combined = level[i] + level[i + 1]
                            else:
                                combined = level[i] + level[i]
                            next_level.append(hashlib.sha256(combined).digest())
                        level = next_level
                    merkle_root = level[0]

                    # replication_id: hash(miner_address + []byte(data_hash_hex) + ipfs_hash)
                    replication_input = (
                        miner_address.encode("utf-8")
                        + data_hash_hex.encode("utf-8")
                        + ipfs_hash.encode("utf-8")
                    )
                    replication_id = hashlib.sha256(replication_input).hexdigest()

                    # storage_proof: hash(replica_hash_bytes + miner_address + ipfs_hash)
                    storage_input = (
                        replica_hash_bytes
                        + miner_address.encode("utf-8")
                        + ipfs_hash.encode("utf-8")
                    )
                    storage_proof_bytes = hashlib.sha256(storage_input).digest()

                    porep_obj = {
                        "data_hash": data_hash_hex,
                        "replica_hash": replica_hash_hex,
                        "merkle_proof": base64.b64encode(merkle_root).decode("ascii"),
                        "storage_proof": base64.b64encode(storage_proof_bytes).decode(
                            "ascii"
                        ),
                        "replication_id": replication_id,
                        "miner_address": miner_address,
                        "timestamp": int(time.time()),
                    }
                    porep_proof_ipfs_hash = self.ipfs_client.upload_json(porep_obj)
                    logger.info(f"PoRep proof IPFS hash: {porep_proof_ipfs_hash}")
                except Exception as e:
                    self.logger.warning(f"⚠️  Failed to generate/upload PoRep proof: {e}")
                    porep_proof_ipfs_hash = None
                
                # 4.7. Calculate deterministic shard assignment
                block_hash = self.blockchain_client.get_block_hash()
                total_shards = int(os.getenv("R3MES_TOTAL_SHARDS", "100"))
                shard_id = calculate_shard_id(
                    miner_address=miner_address,
                    block_hash=block_hash,
                    training_round_id=self.training_round_id,
                    total_shards=total_shards,
                )
                
                # 4.8. Calculate claimed loss (BitNet integer format)
                # Get average loss from training history for this round
                # Loss-Based Spot Checking: This will be verified by validators through forward pass
                claimed_loss = None
                if hasattr(self.trainer, 'loss_history') and len(self.trainer.loss_history) > 0:
                    # Use the most recent loss value
                    recent_loss = self.trainer.loss_history[-1]
                    # Convert to BitNet integer format (multiply by scale factor for precision, then round)
                    # BitNet uses integer representation, so we scale and round
                    loss_scale_factor = int(os.getenv("R3MES_LOSS_SCALE_FACTOR", "1000"))
                    claimed_loss_int = int(recent_loss * loss_scale_factor)  # Scale to integer
                    claimed_loss = str(claimed_loss_int)
                    logger.info(f"Claimed loss (BitNet integer): {claimed_loss}")
                else:
                    logger.warning("No loss history available, claimed_loss will be empty")
                
                # Register gradient with coordinator
                gradient_metadata = self.coordinator.register_gradient(
                    round_id=self.training_round_id,
                    miner_address=miner_address,
                    ipfs_hash=ipfs_hash,
                    gradient_hash=gradient_hash,
                    shard_id=shard_id,
                    gpu_architecture=self.gpu_architecture,
                    porep_proof_ipfs_hash=porep_proof_ipfs_hash,
                )
                logger.info(f"Registered gradient {gradient_metadata.gradient_id} in round {self.training_round_id} (shard {shard_id})")
                
                # 5. Submit to blockchain (only hash + metadata)
                logger.info("Submitting to blockchain...")
                
                # Get model version from environment or use default
                model_version = os.getenv("R3MES_MODEL_VERSION", "v1.0.0")
                response = self.blockchain_client.submit_gradient(
                    miner_address=miner_address,
                    ipfs_hash=ipfs_hash,
                    model_version=model_version,
                    training_round_id=self.training_round_id,
                    shard_id=shard_id,  # Deterministic shard assignment
                    gradient_hash=gradient_hash,
                    gpu_architecture=self.gpu_architecture,
                    claimed_loss=claimed_loss,  # Loss-Based Spot Checking: miner's claimed loss
                    porep_proof_ipfs_hash=porep_proof_ipfs_hash,
                )
                
                if response.get("success"):
                    self.successful_submissions += 1
                    logger.info(f"Gradient submitted successfully! ID: {response.get('stored_gradient_id')}, TX: {response.get('tx_hash')}")
                else:
                    logger.error(f"Submission failed: {response.get('error')}")
                
                self.total_submissions += 1
                self.training_round_id += 1
                
                logger.debug("")
                
            except Exception as e:
                logger.error(f"Error in iteration {iteration + 1}: {e}", exc_info=True)
                continue
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("Mining Summary")
        logger.info(f"{'='*60}")
        logger.info(f"Total submissions: {self.total_submissions}")
        logger.info(f"Successful: {self.successful_submissions}")
        logger.info(f"Failed: {self.total_submissions - self.successful_submissions}")
        logger.info("")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get mining statistics.
        
        Returns:
            Dictionary with mining statistics including GPU stats and training metrics
        """
        # Import stats collector
        try:
            from r3mes.miner.stats_server import get_stats_collector
            stats_collector = get_stats_collector()
            
            if stats_collector:
                miner_stats = stats_collector.get_miner_stats()
                training_metrics = stats_collector.get_training_metrics()
                
                return {
                    "miner_stats": {
                        "gpu_temp": miner_stats.gpu_temp,
                        "fan_speed": miner_stats.fan_speed,
                        "vram_usage": miner_stats.vram_usage,
                        "power_draw": miner_stats.power_draw,
                        "hash_rate": miner_stats.hash_rate,
                        "uptime": miner_stats.uptime,
                        "timestamp": miner_stats.timestamp,
                    },
                    "training_metrics": {
                        "epoch": training_metrics.epoch if training_metrics else 0,
                        "loss": training_metrics.loss if training_metrics else 0.0,
                        "accuracy": training_metrics.accuracy if training_metrics else 0.0,
                        "gradient_norm": training_metrics.gradient_norm if training_metrics else 0.0,
                        "timestamp": training_metrics.timestamp if training_metrics else int(time.time()),
                    },
                    "mining_stats": {
                        "total_submissions": self.total_submissions,
                        "successful_submissions": self.successful_submissions,
                        "training_round_id": self.training_round_id,
                    },
                }
        except ImportError:
            # Stats collector not available
            pass
        
        # Fallback: return basic stats
        return {
            "mining_stats": {
                "total_submissions": self.total_submissions,
                "successful_submissions": self.successful_submissions,
                "training_round_id": self.training_round_id,
                "gpu_architecture": self.gpu_architecture,
                "lora_size_mb": self.trainer.estimate_lora_size_mb(),
            },
        }
    
    # =========================================================================
    # TRAP JOB VERIFICATION (FAZ 5 Integration)
    # =========================================================================
    
    def _verify_trap_job(
        self,
        task: Dict[str, Any],
        gradient_hash: str,
        gradients_dict: Dict[str, torch.Tensor],
    ) -> bool:
        """
        Verify if task is a trap job and validate result.
        
        This is called internally after gradient computation to check
        if the task was a trap job from the Genesis Vault.
        
        Args:
            task: Task info from task pool
            gradient_hash: Computed gradient hash
            gradients_dict: Computed gradients
        
        Returns:
            True if not a trap or trap verification passed
        """
        try:
            from core.trap_jobs import TrapJobVerifier, GenesisVaultManager
            from core.similarity import SimilarityVerifier
            
            # Check if this is a trap job (negative chunk_id)
            chunk_id = task.get('chunk_id', 0)
            if chunk_id >= 0:
                # Not a trap job
                return True
            
            # Load Genesis Vault
            vault_path = os.getenv("R3MES_GENESIS_VAULT_PATH", "genesis_vault_entries.json")
            if not os.path.exists(vault_path):
                logger.warning(f"Genesis Vault not found at {vault_path}, skipping trap verification")
                return True
            
            vault_manager = GenesisVaultManager(vault_path)
            trap_verifier = TrapJobVerifier(vault_manager)
            
            # Get entry ID from task metadata
            entry_id = task.get('trap_entry_id', f"trap_{abs(chunk_id)}")
            miner_address = self.blockchain_client.get_miner_address()
            
            # Verify trap result
            result = trap_verifier.verify_trap_result(
                entry_id=entry_id,
                miner_gradient_hash=gradient_hash,
                miner_address=miner_address,
            )
            
            if result.is_valid:
                logger.info(f"✅ Trap verification PASSED (chunk_id={chunk_id})")
                return True
            else:
                logger.warning(
                    f"⚠️ Trap verification FAILED (chunk_id={chunk_id}, "
                    f"similarity={result.similarity_score:.4f})"
                )
                # In production, this would trigger slashing
                return False
                
        except ImportError:
            logger.debug("Trap job modules not available, skipping verification")
            return True
        except Exception as e:
            logger.error(f"Error during trap verification: {e}")
            return True  # Don't fail on verification errors
    
    # =========================================================================
    # SERVING NODE METHODS
    # =========================================================================
    
    def initialize_serving_node_loras(self) -> bool:
        """
        Initialize LoRA adapters for serving node.
        
        Downloads and caches LoRA adapters from blockchain/IPFS.
        
        Returns:
            True if initialization successful
        """
        try:
            from r3mes.miner.lora_manager import LoRAManager
            
            logger.info("Initializing serving node LoRAs...")
            
            # Get available LoRAs from blockchain
            lora_list = self.blockchain_client.get_available_loras()
            if not lora_list:
                logger.warning("No LoRAs available on blockchain")
                return False
            
            self.available_lora_list = lora_list
            logger.info(f"Found {len(lora_list)} available LoRAs")
            
            # Initialize LoRA manager
            cache_dir = os.getenv("R3MES_LORA_CACHE_DIR", ".r3mes/lora_cache")
            self.lora_manager = LoRAManager(
                model=self.model,
                cache_dir=cache_dir,
                ipfs_client=self.ipfs_client,
            )
            
            # Pre-download top LoRAs
            max_preload = int(os.getenv("R3MES_LORA_PRELOAD_COUNT", "3"))
            for lora_info in lora_list[:max_preload]:
                try:
                    self.lora_manager.load_lora(lora_info['ipfs_hash'])
                    logger.info(f"Pre-loaded LoRA: {lora_info['version']}")
                except Exception as e:
                    logger.warning(f"Failed to pre-load LoRA {lora_info['version']}: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize serving node LoRAs: {e}")
            return False
    
    def register_as_serving_node(self) -> bool:
        """
        Register this node as a serving node on blockchain.
        
        Returns:
            True if registration successful
        """
        try:
            miner_address = self.blockchain_client.get_miner_address()
            
            # Get node capabilities
            capabilities = {
                "gpu_architecture": self.gpu_architecture,
                "vram_gb": self.vram_profile.get("vram_gb", 0) if self.vram_profile else 0,
                "max_batch_size": self.vram_profile.get("batch_size", 1) if self.vram_profile else 1,
                "supported_models": ["llama3-8b", "bitnet"],
            }
            
            result = self.blockchain_client.register_node(
                node_address=miner_address,
                node_type="serving",
                capabilities=capabilities,
            )
            
            if result.get("success"):
                logger.info(f"✅ Registered as serving node: {miner_address}")
                return True
            else:
                logger.error(f"Failed to register as serving node: {result.get('error')}")
                return False
                
        except Exception as e:
            logger.error(f"Error registering as serving node: {e}")
            return False
    
    def _start_inference_server(self) -> None:
        """Start the inference server for serving requests."""
        try:
            from r3mes.miner.inference_server import start_inference_server
            
            port = int(os.getenv("R3MES_INFERENCE_PORT", "8080"))
            
            # Start server in background thread
            import threading
            server_thread = threading.Thread(
                target=start_inference_server,
                args=(self.model, self.tokenizer, port),
                daemon=True,
            )
            server_thread.start()
            
            logger.info(f"🚀 Inference server started on port {port}")
            
        except Exception as e:
            logger.error(f"Failed to start inference server: {e}")
    
    def _start_heartbeat_thread(self) -> None:
        """Start heartbeat thread for serving node health reporting."""
        try:
            import threading
            
            def heartbeat_loop():
                interval = int(os.getenv("R3MES_HEARTBEAT_INTERVAL", "30"))
                while True:
                    try:
                        miner_address = self.blockchain_client.get_miner_address()
                        self.blockchain_client.send_heartbeat(
                            node_address=miner_address,
                            node_type="serving",
                        )
                        logger.debug("💓 Heartbeat sent")
                    except Exception as e:
                        logger.warning(f"Heartbeat failed: {e}")
                    
                    time.sleep(interval)
            
            heartbeat_thread = threading.Thread(
                target=heartbeat_loop,
                daemon=True,
            )
            heartbeat_thread.start()
            
            logger.info("💓 Heartbeat thread started")
            
        except Exception as e:
            logger.error(f"Failed to start heartbeat thread: {e}")


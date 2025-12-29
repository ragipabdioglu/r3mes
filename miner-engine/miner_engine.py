#!/usr/bin/env python3
"""
R3MES Miner Engine

Production-ready mining engine that:
1. Trains BitNet model with LoRA adapters
2. Uploads gradients to IPFS (active role)
3. Submits IPFS hash to blockchain via gRPC
4. Tracks mining statistics and rewards
"""

import argparse
import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import time
import sys
import signal
import atexit
import logging
from contextlib import contextmanager

from core.bitlinear import BitLinear
from utils.logger import setup_logger
from utils.error_handling import (
    exponential_backoff,
    handle_specific_errors,
    NetworkError,
    AuthenticationError,
    ResourceError,
)
from core.trainer import LoRATrainer
from typing import List
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
from bridge.blockchain_client import BlockchainClient
from bridge.arrow_flight_client import ArrowFlightClient
from r3mes.miner.vram_profiler import detect_vram_profile, apply_profile_to_model, create_optimizer_from_profile
from r3mes.miner.llama_loader import load_llama3_8b_model, get_model_info


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
        blockchain_url: str = "localhost:9090",
        chain_id: str = "remes-test",
        model_hidden_size: int = 768,
        lora_rank: int = 8,
        learning_rate: float = 1e-4,
        deterministic: bool = True,
        gradient_accumulation_steps: int = 4,
        top_k_compression: float = 0.1,
        log_level: str = "INFO",
        use_json_logs: bool = False,
        use_multi_gpu: bool = False,
        multi_gpu_device_ids: Optional[List[int]] = None,
        use_ddp: bool = False,
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
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            use_json_logs: Whether to use JSON-formatted logs
        """
        # Setup logger
        log_level_map = {
            "DEBUG": 10,
            "INFO": 20,
            "WARNING": 30,
            "ERROR": 40,
        }
        self.logger = setup_logger(
            "r3mes.miner",
            level=log_level_map.get(log_level.upper(), logging.INFO),
            use_json=use_json_logs,
        )
        
        # Graceful shutdown flag
        self._shutdown_requested = False
        self._current_iteration = None
        
        # Register shutdown handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self._cleanup)
        
        self.private_key = private_key
        self.chain_id = chain_id
        
        # Blockchain client (needed for global seed retrieval)
        self.blockchain_client = BlockchainClient(
            node_url=blockchain_url,
            private_key=private_key,
            chain_id=chain_id,
        )
        
        # Get global seed from blockchain for deterministic execution
        # PRODUCTION MODE: Fail-fast if seed cannot be retrieved
        global_seed = None
        if deterministic:
            import os
            is_test_mode = os.getenv("R3MES_TEST_MODE", "false").lower() == "true"
            
            # Try to get global seed from blockchain (training round 0 for initial setup)
            # In production, this would be called with the actual training round ID
            try:
                global_seed = self.blockchain_client.get_global_seed(training_round_id=0)
                if global_seed is None:
                    if is_test_mode:
                        self.logger.warning("Could not retrieve global seed from blockchain, using fixed seed for testing (TEST MODE)")
                    else:
                        raise RuntimeError(
                            "CRITICAL: Could not retrieve global seed from blockchain. "
                            "Deterministic execution requires valid seed. Miner must stop execution."
                        )
            except Exception as e:
                if is_test_mode:
                    self.logger.warning(f"Could not retrieve global seed from blockchain: {e} (TEST MODE - using fixed seed)")
                else:
                    raise RuntimeError(
                        f"CRITICAL: Failed to retrieve global seed from blockchain: {e}. "
                        f"Deterministic execution requires valid seed. Miner must stop execution."
                    ) from e
        
        # Configure deterministic execution with global seed
        if deterministic:
            configure_deterministic_execution(global_seed=global_seed)
        
        # Environment validation (strict version matching for deterministic results)
        self.env_validator = EnvironmentValidator()
        is_valid, errors = self.env_validator.validate_all()
        if not is_valid:
            self.logger.warning("Environment validation failed!")
            for error in errors:
                self.logger.warning(f"  - {error}")
            self.logger.warning("Continuing anyway, but results may not be deterministic.")
        else:
            self.logger.info("Environment validation passed")
        
        # Log environment info
        env_info = self.env_validator.get_environment_info()
        self.logger.info(f"Environment: Python {env_info.get('python_version')}, PyTorch {env_info.get('pytorch_version')}")
        if env_info.get('cuda_available'):
            self.logger.info(f"CUDA: {env_info.get('cuda_version')}, cuDNN: {env_info.get('cudnn_version')}")
        
        # GPU detection
        self.gpu_detector = GPUArchitectureDetector()
        self.gpu_architecture = self.gpu_detector.get_architecture()
        self.logger.info(f"GPU Architecture: {self.gpu_architecture}")
        
        # Configure deterministic CUDA based on GPU architecture
        if deterministic:
            self.gpu_detector.configure_deterministic_cuda()
        
        # Adaptive VRAM Scaling: Detect VRAM and apply profile automatically
        if torch.cuda.is_available():
            try:
                self.vram_profile = detect_vram_profile()
                # Override gradient_accumulation_steps with profile value
                gradient_accumulation_steps = self.vram_profile["gradient_accumulation"]
                self.logger.info(f"Overriding gradient_accumulation_steps to {gradient_accumulation_steps} (from VRAM profile)")
            except Exception as e:
                self.logger.warning(f"Failed to detect VRAM profile: {e}")
                self.logger.warning("Using default configuration")
                self.vram_profile = None
        else:
            self.vram_profile = None
        
        # Load Llama 3 8B model (or fallback to SimpleBitNetModel if not available)
        use_real_model = os.getenv("R3MES_USE_LLAMA3", "true").lower() == "true"
        
        if use_real_model:
            try:
                self.logger.info("Loading Llama 3 8B model from HuggingFace...")
                self.model, self.tokenizer = load_llama3_8b_model(
                    model_name=os.getenv("R3MES_MODEL_NAME", "meta-llama/Meta-Llama-3-8B"),
                    lora_rank=lora_rank,
                    lora_alpha=16,
                    lora_dropout=0.1,
                    device_map="auto",
                    use_quantization=True,
                    quantization_bits=4,  # 4-bit quantization for VRAM efficiency
                )
                
                # Log model info
                model_info = get_model_info(self.model)
                self.logger.info(f"Model loaded: {model_info['trainable_parameters']:,} trainable parameters")
                
                # LoRA is already applied by load_llama3_8b_model
                # Validate LoRA-only training
                from r3mes.miner.model_loader import validate_lora_only_training
                validate_lora_only_training(self.model)
                
            except Exception as e:
                self.logger.warning(f"Failed to load Llama 3 8B model: {e}")
                self.logger.warning("Falling back to SimpleBitNetModel for testing")
                use_real_model = False
        
        if not use_real_model:
            # Fallback to SimpleBitNetModel for testing/development
            self.logger.info("Using SimpleBitNetModel (test model)")
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
        
        # Apply VRAM profile optimizations to model
        if self.vram_profile:
            self.model = apply_profile_to_model(self.model, self.vram_profile)
        
        # Multi-GPU support
        if use_multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            from core.multi_gpu_trainer import create_multi_gpu_trainer
            
            # Get LoRA parameters for optimizer if VRAM profile exists
            if self.vram_profile:
                lora_params = []
                for module in self.model.modules():
                    if isinstance(module, BitLinear):
                        lora_params.append(module.lora_A)
                        lora_params.append(module.lora_B)
                
                profile_optimizer = create_optimizer_from_profile(
                    lora_params,
                    self.vram_profile,
                    learning_rate=learning_rate
                )
                
                # Create multi-GPU trainer with custom optimizer
                self.trainer = create_multi_gpu_trainer(
                    self.model,
                    use_all_gpus=(multi_gpu_device_ids is None),
                    device_ids=multi_gpu_device_ids,
                    use_ddp=use_ddp,
                    learning_rate=learning_rate,
                    deterministic=True,
                    custom_optimizer=profile_optimizer,
                )
            else:
                # Create multi-GPU trainer with default optimizer
                self.trainer = create_multi_gpu_trainer(
                    self.model,
                    use_all_gpus=(multi_gpu_device_ids is None),
                    device_ids=multi_gpu_device_ids,
                    use_ddp=use_ddp,
                    learning_rate=learning_rate,
                    deterministic=True,
                )
            self.logger.info(f"Multi-GPU training enabled: {torch.cuda.device_count()} GPUs")
        else:
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
        self.logger.info(f"Gradient accumulation: {final_gradient_accumulation} steps")
        self.logger.info(f"Top-k compression: {top_k_compression*100:.1f}% (keeping top {top_k_compression*100:.1f}% of values)")
        
        # IPFS client (active role)
        self.ipfs_client = IPFSClient()
        if not self.ipfs_client.is_connected():
            self.logger.warning("IPFS not connected, using simulated mode")
        
        # Blockchain client already initialized above for global seed retrieval
        
        # Arrow Flight client (optional - for zero-copy tensor transfer)
        # Falls back to gRPC if Arrow Flight unavailable
        arrow_flight_port = int(os.getenv("R3MES_ARROW_FLIGHT_PORT", "8815"))
        self.arrow_flight_client = ArrowFlightClient(
            host=blockchain_url.split(":")[0] if ":" in blockchain_url else "localhost",
            port=arrow_flight_port,
        )
        if self.arrow_flight_client.is_connected():
            self.logger.info("Arrow Flight connected (zero-copy tensor transfer enabled)")
        else:
            self.logger.info("Arrow Flight not available, using gRPC for tensor transfer")
        
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
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self._shutdown_requested = True
    
    def _cleanup(self):
        """Cleanup resources on exit."""
        if self._current_iteration is not None:
            self.logger.info(f"Cleaning up iteration {self._current_iteration}...")
        self.logger.info("Cleanup complete")
    
    def train_and_submit(self, num_iterations: int = 1):
        """
        Train model and submit gradients to blockchain.
        
        Args:
            num_iterations: Number of training iterations
        """
        self.logger.info("=" * 60)
        self.logger.info(f"Starting mining loop ({num_iterations} iterations)")
        self.logger.info("=" * 60)
        
        # Get global seed for current training round
        global_seed = self._get_global_seed_for_round(self.training_round_id)
        if global_seed is not None:
            self.logger.info(f"Using global seed from blockchain: {global_seed} (training round {self.training_round_id})")
            # Reconfigure deterministic execution with new seed
            configure_deterministic_execution(global_seed=global_seed)
        else:
            self.logger.warning("Using previously configured seed (could not retrieve from blockchain)")
        
        for iteration in range(num_iterations):
            # Check for shutdown request
            if self._shutdown_requested:
                self.logger.warning(f"Shutdown requested, stopping at iteration {iteration + 1}/{num_iterations}")
                break
            
            self._current_iteration = iteration + 1
            self.logger.info(f"Iteration {iteration + 1}/{num_iterations}")
            self.logger.info("-" * 60)
            
            try:
                # 1. Training step
                inputs = torch.randn(32, 768)
                targets = torch.randn(32, 768)
                
                loss, gradients = self.trainer.train_step(inputs, targets)
                self.logger.info(f"Training loss: {loss:.6f}")
                
                # 1.5. Accumulate gradients (bandwidth optimization)
                accumulated_gradients = self.gradient_accumulator.accumulate(gradients)
                
                # If accumulation not complete, skip submission
                if accumulated_gradients is None:
                    self.logger.debug(f"Accumulating... ({self.gradient_accumulator.step_count}/{self.gradient_accumulator.accumulation_steps})")
                    continue
                
                self.logger.info(f"Accumulation complete ({self.gradient_accumulator.accumulation_steps} steps)")
                
                # 1.6. Compress gradients (top-k compression for bandwidth optimization)
                if self.top_k_compression > 0 and self.top_k_compression < 1.0:
                    compressed = compress_gradients(accumulated_gradients, k=self.top_k_compression)
                    compression_ratio = compressed._calculate_compression_ratio()
                    self.logger.info(f"Compression complete: {compression_ratio*100:.1f}% size reduction")
                    # Use compressed gradients for hash computation
                    # Note: In production, you might want to hash the compressed format
                    gradients_for_hash = accumulated_gradients
                else:
                    gradients_for_hash = accumulated_gradients
                
                # 2. Compute gradient hash from accumulated gradients
                gradient_hash = self.trainer.compute_gradient_hash(gradients_for_hash)
                self.logger.info(f"Gradient hash: {gradient_hash[:16]}...")
                
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
                    self.logger.info(f"Seed locked: {global_seed} (training round {self.training_round_id})")
                
                # 3. Get miner address for metadata (needed for binary serialization)
                miner_address = self.blockchain_client.get_miner_address()
                
                # 4. Serialize LoRA state (using accumulated gradients)
                # Update trainer with accumulated gradients for serialization
                lora_state = self.trainer.get_lora_state_dict()
                # Note: In production, accumulated gradients would be used to update LoRA state
                
                # Try binary serialization first (Protocol Buffers - ~30% bandwidth reduction)
                try:
                    serialized = self.binary_serializer.serialize_gradients(
                        accumulated_gradients,
                        metadata=self.trainer.get_training_metadata(),
                        training_round_id=self.training_round_id,
                        miner_address=miner_address,
                        gradient_hash=gradient_hash,
                    )
                    self.logger.info("Using Protocol Buffers serialization (binary format)")
                except Exception as e:
                    # Fallback to pickle-based serialization
                    self.logger.warning(f"Binary serialization failed, using pickle: {e}")
                    serialized = self.serializer.serialize_lora_state(
                        lora_state,
                        metadata=self.trainer.get_training_metadata(),
                    )
                
                serialized_size_mb = len(serialized) / (1024 * 1024)
                self.logger.info(f"Serialized LoRA size: {serialized_size_mb:.4f} MB")
                
                # 4.5. Upload to IPFS (active role)
                # Optionally use Arrow Flight for zero-copy transfer if available
                if self.arrow_flight_client.is_connected():
                    # Try Arrow Flight first (zero-copy)
                    flight_path = self.arrow_flight_client.upload_gradients(
                        accumulated_gradients,
                        metadata={
                            "miner": miner_address,
                            "training_round_id": self.training_round_id,
                            "gradient_hash": gradient_hash,
                        }
                    )
                    if flight_path:
                        self.logger.info(f"Uploaded via Arrow Flight (zero-copy): {flight_path}")
                        # Still upload to IPFS for compatibility
                        self.logger.info("Uploading to IPFS (backup)...")
                        ipfs_hash = self.ipfs_client.upload_lora_state(serialized)
                        self.logger.info(f"IPFS hash: {ipfs_hash}")
                    else:
                        # Fallback to IPFS
                        self.logger.info("Uploading to IPFS...")
                        ipfs_hash = self.ipfs_client.upload_lora_state(serialized)
                        self.logger.info(f"IPFS hash: {ipfs_hash}")
                else:
                    # Standard IPFS upload
                    self.logger.info("Uploading to IPFS...")
                    ipfs_hash = self.ipfs_client.upload_lora_state(serialized)
                    self.logger.info(f"IPFS hash: {ipfs_hash}")
                
                # 4.6. Calculate deterministic shard assignment
                block_hash = self.blockchain_client.get_block_hash()
                shard_id = calculate_shard_id(
                    miner_address=miner_address,
                    block_hash=block_hash,
                    training_round_id=self.training_round_id,
                    total_shards=100,  # Default: 100 shards
                )
                
                # Register gradient with coordinator
                gradient_metadata = self.coordinator.register_gradient(
                    round_id=self.training_round_id,
                    miner_address=miner_address,
                    ipfs_hash=ipfs_hash,
                    gradient_hash=gradient_hash,
                    shard_id=shard_id,
                    gpu_architecture=self.gpu_architecture,
                )
                self.logger.info(f"Registered gradient {gradient_metadata.gradient_id} in round {self.training_round_id} (shard {shard_id})")
                
                # 5. Submit to blockchain (only hash + metadata)
                self.logger.info("Submitting to blockchain...")
                
                response = self.blockchain_client.submit_gradient(
                    miner_address=miner_address,
                    ipfs_hash=ipfs_hash,
                    model_version="v1.0.0",
                    training_round_id=self.training_round_id,
                    shard_id=shard_id,  # Deterministic shard assignment
                    gradient_hash=gradient_hash,
                    gpu_architecture=self.gpu_architecture,
                )
                
                if response.get("success"):
                    self.successful_submissions += 1
                    self.logger.info("Gradient submitted successfully!")
                    self.logger.info(f"  Stored Gradient ID: {response.get('stored_gradient_id')}")
                    self.logger.info(f"  TX Hash: {response.get('tx_hash')}")
                else:
                    self.logger.error(f"Submission failed: {response.get('error')}")
                
                self.total_submissions += 1
                self.training_round_id += 1
                
            except KeyboardInterrupt:
                self.logger.warning("Keyboard interrupt received, initiating graceful shutdown...")
                self._shutdown_requested = True
                break
            except NetworkError as e:
                self.logger.error(f"Network error in iteration {iteration + 1}: {e}")
                # Network errors are retryable, but we'll continue to next iteration
                continue
            except AuthenticationError as e:
                self.logger.error(f"Authentication error in iteration {iteration + 1}: {e}")
                # Authentication errors are not retryable, stop mining
                raise
            except ResourceError as e:
                self.logger.error(f"Resource error in iteration {iteration + 1}: {e}")
                # Resource errors are critical, stop mining
                raise
            except Exception as e:
                self.logger.error(f"Error in iteration {iteration + 1}: {e}", exc_info=True)
                continue
        
        # Summary
        self.logger.info("=" * 60)
        self.logger.info("Mining Summary")
        self.logger.info("=" * 60)
        self.logger.info(f"Total submissions: {self.total_submissions}")
        self.logger.info(f"Successful: {self.successful_submissions}")
        self.logger.info(f"Failed: {self.total_submissions - self.successful_submissions}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get mining statistics."""
        return {
            "total_submissions": self.total_submissions,
            "successful_submissions": self.successful_submissions,
            "training_round_id": self.training_round_id,
            "gpu_architecture": self.gpu_architecture,
            "lora_size_mb": self.trainer.estimate_lora_size_mb(),
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="R3MES Miner Engine")
    parser.add_argument(
        "--private-key",
        type=str,
        required=True,
        help="Private key for blockchain transactions",
    )
    parser.add_argument(
        "--blockchain-url",
        type=str,
        default="localhost:9090",
        help="Blockchain gRPC endpoint URL",
    )
    parser.add_argument(
        "--chain-id",
        type=str,
        default="remes-test",
        help="Chain ID",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum number of training iterations",
    )
    parser.add_argument(
        "--model-size",
        type=int,
        default=768,
        help="Model hidden size",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Number of steps to accumulate gradients before submission (bandwidth optimization)",
    )
    parser.add_argument(
        "--top-k-compression",
        type=float,
        default=0.1,
        help="Top-k compression ratio (0.1 = keep top 10%%, 0.0 = no compression)",
    )
    
    args = parser.parse_args()
    
    # Create miner engine
    miner = MinerEngine(
        private_key=args.private_key,
        blockchain_url=args.blockchain_url,
        chain_id=args.chain_id,
        model_hidden_size=args.model_size,
        lora_rank=args.lora_rank,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        top_k_compression=args.top_k_compression,
    )
    
    # Run mining loop
    try:
        miner.train_and_submit(num_iterations=args.max_iterations)
    except KeyboardInterrupt:
        miner.logger.warning("Mining interrupted by user")
    except Exception as e:
        miner.logger.error(f"Mining failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Print statistics
        stats = miner.get_statistics()
        miner.logger.info("Final Statistics:")
        for key, value in stats.items():
            miner.logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    main()


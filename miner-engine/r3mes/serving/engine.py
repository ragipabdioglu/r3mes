#!/usr/bin/env python3
"""
R3MES Serving Engine

Production-ready serving engine that:
1. Loads AI model from IPFS hash
2. Listens for inference requests from blockchain
3. Executes inference using BitNet + DoRA + RAG pipeline
4. Uploads results to IPFS
5. Submits results to blockchain via gRPC

Architecture:
    ServingEngine
        │
        ├── InferencePipeline (BitNet + DoRA + RAG)
        │   ├── RAGRetriever (context augmentation)
        │   ├── HybridRouter (expert selection)
        │   ├── TieredCache (adapter caching)
        │   └── InferenceBackend (model execution)
        │
        ├── BlockchainClient (gRPC)
        └── IPFSClient (model/data storage)
"""

import argparse
import asyncio
import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
import time
import sys
import signal
import atexit
import logging
import json
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from utils.logger import setup_logger
from utils.error_handling import (
    exponential_backoff,
    handle_specific_errors,
    NetworkError,
    AuthenticationError,
    ResourceError,
)
from utils.ipfs_client import IPFSClient
from bridge.blockchain_client import BlockchainClient
from bridge.crypto import derive_address_from_public_key
from bridge.transaction_builder import private_key_to_public_key
from r3mes.miner.model_loader import load_model_with_enforced_lora

# Import InferencePipeline components
from .inference_pipeline import (
    InferencePipeline,
    PipelineConfig,
    PipelineResult,
    PipelineMetrics,
    create_pipeline,
)


class EngineState(Enum):
    """Serving engine state."""
    INITIALIZING = "initializing"
    LOADING_MODEL = "loading_model"
    READY = "ready"
    PROCESSING = "processing"
    SHUTTING_DOWN = "shutting_down"
    ERROR = "error"


@dataclass
class EngineHealth:
    """Health status for serving engine."""
    state: EngineState = EngineState.INITIALIZING
    is_healthy: bool = False
    is_ready: bool = False
    pipeline_initialized: bool = False
    model_loaded: bool = False
    last_inference_time: Optional[float] = None
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0.0
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "is_healthy": self.is_healthy,
            "is_ready": self.is_ready,
            "pipeline_initialized": self.pipeline_initialized,
            "model_loaded": self.model_loaded,
            "last_inference_time": self.last_inference_time,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "avg_latency_ms": self.avg_latency_ms,
            "error_rate": self.failed_requests / max(1, self.total_requests),
            "error_message": self.error_message,
        }


class ServingEngine:
    """
    Main serving engine class with InferencePipeline integration.
    
    Integrates BitNet + DoRA + RAG pipeline for production inference.
    """
    
    def __init__(
        self,
        private_key: str,
        blockchain_url: str = "localhost:9090",
        chain_id: str = "remes-test",
        model_ipfs_hash: Optional[str] = None,
        model_version: str = "v1.0.0",
        log_level: str = "INFO",
        use_json_logs: bool = False,
        # Pipeline configuration
        enable_rag: bool = True,
        enable_caching: bool = True,
        vram_capacity_mb: int = 2048,
        ram_capacity_mb: int = 8192,
        adapter_dir: Optional[str] = None,
    ):
        """
        Initialize serving engine with InferencePipeline.
        
        Args:
            private_key: Private key for blockchain transactions
            blockchain_url: gRPC endpoint URL
            chain_id: Chain ID
            model_ipfs_hash: IPFS hash of model to load
            model_version: Model version to serve
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            use_json_logs: Whether to use JSON-formatted logs
            enable_rag: Enable RAG context retrieval
            enable_caching: Enable tiered adapter caching
            vram_capacity_mb: VRAM cache capacity in MB
            ram_capacity_mb: RAM cache capacity in MB
            adapter_dir: Directory for DoRA adapters
        """
        # Production localhost validation
        is_production = os.getenv("R3MES_ENV", "development").lower() == "production"
        if is_production:
            blockchain_host = blockchain_url.split(":")[0] if ":" in blockchain_url else blockchain_url
            if blockchain_host.lower() in ("localhost", "127.0.0.1", "::1") or blockchain_host.startswith("127."):
                raise ValueError(
                    f"blockchain_url cannot use localhost in production: {blockchain_url}. "
                    "Please set blockchain_url to a production gRPC endpoint."
                )
        
        # Setup logger
        log_level_map = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
        self.logger = setup_logger(
            "r3mes.serving",
            level=log_level_map.get(log_level.upper(), logging.INFO),
            use_json=use_json_logs,
        )
        
        # Engine state
        self._health = EngineHealth()
        self._shutdown_requested = False
        self._shutdown_event = asyncio.Event()
        self._latency_samples: List[float] = []
        
        # Register shutdown handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self._cleanup_sync)
        
        self.private_key = private_key
        self.chain_id = chain_id
        self.model_version = model_version
        self.model_ipfs_hash = model_ipfs_hash
        
        # Blockchain client
        self.blockchain_client = BlockchainClient(
            node_url=blockchain_url,
            chain_id=chain_id,
            private_key=private_key,
        )
        
        # Derive serving node address from private key
        try:
            private_key_bytes = bytes.fromhex(private_key)
            public_key_bytes = private_key_to_public_key(private_key_bytes)
            self.serving_node_address = derive_address_from_public_key(public_key_bytes)
        except Exception as e:
            self.logger.warning(f"Could not derive address from private key: {e}")
            self.serving_node_address = None
        
        # IPFS client
        self.ipfs_client = IPFSClient()
        
        # Device detection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Pipeline configuration
        self._pipeline_config = PipelineConfig(
            enable_rag=enable_rag,
            vram_capacity_mb=vram_capacity_mb if enable_caching else 0,
            ram_capacity_mb=ram_capacity_mb if enable_caching else 0,
            adapter_dir=adapter_dir or os.getenv("R3MES_ADAPTER_DIR", ".r3mes/adapters"),
            disk_cache_dir=os.getenv("R3MES_CACHE_DIR", ".r3mes/dora_cache"),
        )
        
        # InferencePipeline (lazy initialization)
        self._pipeline: Optional[InferencePipeline] = None
        
        # Legacy compatibility
        self.model = None
        self.is_available = False
        
        self._health.is_healthy = True
        self.logger.info(f"ServingEngine initialized (device: {self.device})")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._shutdown_requested = True
        self._health.state = EngineState.SHUTTING_DOWN
        self._shutdown_event.set()
    
    def _cleanup_sync(self):
        """Synchronous cleanup for atexit."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._cleanup_async())
            else:
                loop.run_until_complete(self._cleanup_async())
        except RuntimeError:
            # No event loop, do basic cleanup
            if self.model is not None:
                del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    async def _cleanup_async(self):
        """Async cleanup for graceful shutdown."""
        self.logger.info("Cleaning up serving engine resources...")
        
        # Shutdown pipeline
        if self._pipeline:
            try:
                await self._pipeline.shutdown()
                self.logger.info("Pipeline shutdown complete")
            except Exception as e:
                self.logger.error(f"Error shutting down pipeline: {e}")
        
        # Update blockchain status
        try:
            self.update_status(is_available=False)
        except Exception as e:
            self.logger.warning(f"Could not update blockchain status: {e}")
        
        # Cleanup model
        if self.model is not None:
            del self.model
            self.model = None
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._health.is_healthy = False
        self._health.is_ready = False
        self.logger.info("Serving engine cleanup complete")
    
    # =========================================================================
    # Health & Metrics API
    # =========================================================================
    
    def get_health(self) -> Dict[str, Any]:
        """Get engine health status (for /health endpoint)."""
        return self._health.to_dict()
    
    def is_healthy(self) -> bool:
        """Check if engine is healthy (for liveness probe)."""
        return self._health.is_healthy and not self._shutdown_requested
    
    def is_ready(self) -> bool:
        """Check if engine is ready to serve (for readiness probe)."""
        return (
            self._health.is_ready 
            and self._health.pipeline_initialized 
            and not self._shutdown_requested
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get Prometheus-compatible metrics."""
        metrics = {
            "serving_engine_requests_total": self._health.total_requests,
            "serving_engine_requests_success": self._health.successful_requests,
            "serving_engine_requests_failed": self._health.failed_requests,
            "serving_engine_latency_avg_ms": self._health.avg_latency_ms,
            "serving_engine_ready": 1 if self.is_ready() else 0,
            "serving_engine_healthy": 1 if self.is_healthy() else 0,
        }
        
        # Add pipeline metrics if available
        if self._pipeline:
            pipeline_stats = self._pipeline.get_stats()
            metrics.update({
                "pipeline_total_requests": pipeline_stats.get("total_requests", 0),
                "pipeline_error_rate": pipeline_stats.get("error_rate", 0),
            })
            
            # Cache metrics
            if "cache" in pipeline_stats:
                cache_stats = pipeline_stats["cache"]
                metrics.update({
                    "cache_vram_used_mb": cache_stats.get("vram", {}).get("used_mb", 0),
                    "cache_ram_used_mb": cache_stats.get("ram", {}).get("used_mb", 0),
                    "cache_hits": cache_stats.get("stats", {}).get("hits", 0),
                    "cache_misses": cache_stats.get("stats", {}).get("misses", 0),
                })
        
        return metrics
    
    def _update_latency_stats(self, latency_ms: float):
        """Update latency statistics."""
        self._latency_samples.append(latency_ms)
        # Keep last 100 samples
        if len(self._latency_samples) > 100:
            self._latency_samples = self._latency_samples[-100:]
        self._health.avg_latency_ms = sum(self._latency_samples) / len(self._latency_samples)
    
    # =========================================================================
    # Pipeline Management
    # =========================================================================
    
    async def initialize_pipeline(self) -> bool:
        """
        Initialize the InferencePipeline.
        
        Returns:
            True if initialization successful
        """
        if self._pipeline and self._health.pipeline_initialized:
            return True
        
        try:
            self._health.state = EngineState.INITIALIZING
            self.logger.info("Initializing InferencePipeline...")
            
            # Create pipeline with configuration
            self._pipeline = create_pipeline(config=self._pipeline_config)
            
            # Initialize pipeline components
            success = await self._pipeline.initialize()
            
            if success:
                self._health.pipeline_initialized = True
                self.logger.info("InferencePipeline initialized successfully")
                return True
            else:
                self._health.error_message = "Pipeline initialization failed"
                self._health.state = EngineState.ERROR
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
            self._health.error_message = str(e)
            self._health.state = EngineState.ERROR
            return False
    
    async def load_model(self, model_ipfs_hash: Optional[str] = None) -> bool:
        """
        Load model from IPFS hash and initialize pipeline.
        
        Args:
            model_ipfs_hash: IPFS hash of model (uses self.model_ipfs_hash if not provided)
            
        Returns:
            True if model loaded successfully
        """
        try:
            self._health.state = EngineState.LOADING_MODEL
            hash_to_load = model_ipfs_hash or self.model_ipfs_hash
            
            if not hash_to_load:
                self.logger.error("No model IPFS hash provided")
                return False
            
            self.logger.info(f"Loading model from IPFS: {hash_to_load}")
            
            # Download model from IPFS
            model_path = self.ipfs_client.get(hash_to_load, output_dir="models")
            if not model_path:
                self.logger.error(f"Failed to download model from IPFS: {hash_to_load}")
                return False
            
            self.logger.info(f"Model downloaded to: {model_path}")
            
            # Initialize pipeline if not already done
            if not self._health.pipeline_initialized:
                if not await self.initialize_pipeline():
                    return False
            
            # Load model into pipeline backend
            success = await self._pipeline.load_model(model_path)
            
            if success:
                self.model_ipfs_hash = hash_to_load
                self._health.model_loaded = True
                self._health.state = EngineState.READY
                self._health.is_ready = True
                self.logger.info(f"Model loaded successfully: {hash_to_load}")
                return True
            else:
                self._health.error_message = "Failed to load model into pipeline"
                self._health.state = EngineState.ERROR
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading model: {e}", exc_info=True)
            self._health.error_message = str(e)
            self._health.state = EngineState.ERROR
            return False
    
    def update_status(self, is_available: bool, model_version: Optional[str] = None):
        """
        Update serving node status on blockchain.
        
        Args:
            is_available: Whether node is available for requests
            model_version: Model version (optional)
        """
        try:
            self.logger.info(f"Updating serving node status: available={is_available}, version={model_version or self.model_version}")
            self.is_available = is_available
            
            if self.blockchain_client and self.serving_node_address:
                result = self.blockchain_client.update_serving_node_status(
                    serving_node=self.serving_node_address,
                    is_available=is_available,
                    model_version=model_version or self.model_version,
                    model_ipfs_hash=self.model_ipfs_hash,
                )
                if not result.get("success", False):
                    self.logger.warning(f"Failed to update status on blockchain: {result.get('error', 'Unknown error')}")
            else:
                self.logger.warning("Blockchain client or serving node address not available")
            
        except Exception as e:
            self.logger.error(f"Error updating status: {e}", exc_info=True)
    
    async def process_inference_request(self, request_id: str, input_data_ipfs_hash: str) -> Optional[str]:
        """
        Process an inference request using InferencePipeline.
        
        Args:
            request_id: Inference request ID
            input_data_ipfs_hash: IPFS hash of input data
            
        Returns:
            IPFS hash of result, or None if failed
        """
        start_time = time.perf_counter()
        self._health.total_requests += 1
        self._health.state = EngineState.PROCESSING
        
        try:
            self.logger.info(f"Processing inference request: {request_id}")
            
            # Ensure pipeline is ready
            if not self._pipeline or not self._health.pipeline_initialized:
                self.logger.error("Pipeline not initialized")
                self._health.failed_requests += 1
                return None
            
            # Download input data from IPFS
            input_path = self.ipfs_client.get(input_data_ipfs_hash, output_dir="inference_inputs")
            if not input_path:
                self.logger.error(f"Failed to download input data: {input_data_ipfs_hash}")
                self._health.failed_requests += 1
                return None
            
            # Parse input data
            with open(input_path, 'rb') as f:
                input_data = f.read()
            
            try:
                input_json = json.loads(input_data.decode('utf-8'))
                prompt = input_json.get('prompt', '')
                # Extract optional parameters
                skip_rag = input_json.get('skip_rag', False)
                force_experts = input_json.get('force_experts', None)
                temperature = input_json.get('temperature', 0.7)
            except (json.JSONDecodeError, UnicodeDecodeError):
                prompt = input_data.decode('utf-8', errors='ignore')
                skip_rag = False
                force_experts = None
                temperature = 0.7
            
            # Run inference through pipeline
            result: PipelineResult = await self._pipeline.run(
                query=prompt,
                skip_rag=skip_rag,
                force_experts=force_experts,
                temperature=temperature,
            )
            
            if not result.success:
                self.logger.error(f"Pipeline inference failed: {result.error}")
                self._health.failed_requests += 1
                return None
            
            # Prepare result
            result_json = {
                "request_id": request_id,
                "result": result.text if result.text else str(result.output.tolist()),
                "model_version": self.model_version,
                "metrics": result.metrics.to_dict(),
                "experts_used": [{"id": e[0], "weight": e[1]} for e in result.experts_used],
                "rag_context_used": result.rag_context is not None,
            }
            result_data = json.dumps(result_json).encode('utf-8')
            
            # Upload result to IPFS
            result_hash = self.ipfs_client.add_bytes(result_data)
            if not result_hash:
                self.logger.error("Failed to upload result to IPFS")
                self._health.failed_requests += 1
                return None
            
            # Update stats
            latency_ms = (time.perf_counter() - start_time) * 1000
            self._update_latency_stats(latency_ms)
            self._health.successful_requests += 1
            self._health.last_inference_time = time.time()
            self._health.state = EngineState.READY
            
            self.logger.info(f"Inference completed: request_id={request_id}, latency={latency_ms:.1f}ms, result_hash={result_hash}")
            return result_hash
            
        except Exception as e:
            self.logger.error(f"Error processing inference request: {e}", exc_info=True)
            self._health.failed_requests += 1
            self._health.state = EngineState.READY
            return None
    
    def submit_inference_result(self, request_id: str, result_ipfs_hash: str, latency_ms: int):
        """
        Submit inference result to blockchain.
        
        Args:
            request_id: Inference request ID
            result_ipfs_hash: IPFS hash of result
            latency_ms: Inference latency in milliseconds
        """
        try:
            self.logger.info(f"Submitting inference result: request_id={request_id}, latency={latency_ms}ms")
            
            if self.blockchain_client and self.serving_node_address:
                result = self.blockchain_client.submit_inference_result(
                    serving_node=self.serving_node_address,
                    request_id=request_id,
                    result_ipfs_hash=result_ipfs_hash,
                    latency_ms=latency_ms,
                )
                if not result.get("success", False):
                    self.logger.error(f"Failed to submit result to blockchain: {result.get('error', 'Unknown error')}")
                    return False
                self.logger.info(f"Result submitted to blockchain: TX {result.get('tx_hash', 'pending')}")
                return True
            else:
                self.logger.warning("Blockchain client or serving node address not available")
                return False
            
        except Exception as e:
            self.logger.error(f"Error submitting inference result: {e}", exc_info=True)
            return False
    
    async def start_async(self):
        """
        Start serving engine (async, non-blocking).
        
        Initializes pipeline, loads model, and starts polling for requests.
        """
        self.logger.info("Starting serving engine...")
        
        try:
            # Initialize pipeline
            if not await self.initialize_pipeline():
                self.logger.error("Failed to initialize pipeline, cannot start serving")
                return False
            
            # Load model if hash is provided
            if self.model_ipfs_hash:
                if not await self.load_model():
                    self.logger.error("Failed to load model, cannot start serving")
                    return False
            else:
                # No model hash - initialize pipeline without model for testing
                self._health.state = EngineState.READY
                self._health.is_ready = True
                self.logger.warning("No model IPFS hash provided, running in test mode")
            
            # Warmup pipeline
            try:
                await self._pipeline.warmup()
                self.logger.info("Pipeline warmup complete")
            except Exception as e:
                self.logger.warning(f"Pipeline warmup failed: {e}")
            
            # Update status to available
            self.update_status(is_available=True)
            
            self.logger.info("Serving engine started, polling for inference requests")
            
            # Main loop with graceful shutdown support
            poll_interval = float(os.getenv("R3MES_POLL_INTERVAL", "5.0"))
            
            while not self._shutdown_requested:
                try:
                    await self._poll_and_process_requests()
                except Exception as e:
                    self.logger.error(f"Error in polling loop: {e}", exc_info=True)
                
                # Wait with shutdown event support
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=poll_interval
                    )
                    # Shutdown event was set
                    break
                except asyncio.TimeoutError:
                    # Normal timeout, continue polling
                    pass
            
            # Graceful shutdown
            await self._cleanup_async()
            self.logger.info("Serving engine stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting serving engine: {e}", exc_info=True)
            self._health.error_message = str(e)
            self._health.state = EngineState.ERROR
            return False
    
    def start(self):
        """
        Start serving engine (sync wrapper for async start).
        
        This method runs the async start_async in an event loop.
        For async contexts, use start_async() directly.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return asyncio.create_task(self.start_async())
            else:
                return asyncio.run(self.start_async())
        except RuntimeError:
            return asyncio.run(self.start_async())
    
    async def _poll_and_process_requests(self):
        """
        Poll blockchain for new inference requests and process them.
        """
        try:
            if not self.blockchain_client or not self.serving_node_address:
                self.logger.warning("Blockchain client or serving node address not available")
                return
            
            # Query blockchain for pending inference requests
            pending_requests = self.blockchain_client.get_pending_inference_requests(
                serving_node=self.serving_node_address
            )
            
            if not pending_requests:
                return
            
            self.logger.info(f"Found {len(pending_requests)} pending inference requests")
            
            # Process each request
            for request in pending_requests:
                if self._shutdown_requested:
                    break
                
                request_id = request.get("request_id", "")
                input_data_ipfs_hash = request.get("input_data_ipfs_hash", "")
                
                if not request_id or not input_data_ipfs_hash:
                    self.logger.warning(f"Invalid request: {request}")
                    continue
                
                # Record start time for latency measurement
                start_time = time.time()
                
                # Process inference request (now async)
                result_ipfs_hash = await self.process_inference_request(request_id, input_data_ipfs_hash)
                
                # Calculate latency
                latency_ms = int((time.time() - start_time) * 1000)
                
                if result_ipfs_hash:
                    self.submit_inference_result(request_id, result_ipfs_hash, latency_ms)
                else:
                    self.logger.error(f"Failed to process inference request: {request_id}")
                    self.submit_inference_error(request_id, "Processing failed", latency_ms)
                
        except Exception as e:
            self.logger.error(f"Error polling and processing requests: {e}", exc_info=True)
    
    def submit_inference_error(self, request_id: str, error_message: str, latency_ms: int):
        """
        Submit inference error to blockchain.
        
        Args:
            request_id: Inference request ID
            error_message: Error message
            latency_ms: Processing time before error
        """
        try:
            self.logger.info(f"Submitting inference error: request_id={request_id}, error={error_message}")
            
            if self.blockchain_client and self.serving_node_address:
                result = self.blockchain_client.submit_inference_error(
                    serving_node=self.serving_node_address,
                    request_id=request_id,
                    error_message=error_message,
                    latency_ms=latency_ms,
                )
                if not result.get("success", False):
                    self.logger.error(f"Failed to submit error to blockchain: {result.get('error', 'Unknown error')}")
                    return False
                self.logger.info(f"Error submitted to blockchain: TX {result.get('tx_hash', 'pending')}")
                return True
            else:
                self.logger.warning("Blockchain client or serving node address not available")
                return False
            
        except Exception as e:
            self.logger.error(f"Error submitting inference error: {e}", exc_info=True)
            return False
    
    # =========================================================================
    # Direct Inference API (for HTTP/gRPC endpoints)
    # =========================================================================
    
    async def infer(
        self,
        query: str,
        skip_rag: bool = False,
        force_experts: Optional[List[str]] = None,
        **kwargs
    ) -> PipelineResult:
        """
        Run inference directly (for HTTP/gRPC endpoints).
        
        Args:
            query: User query text
            skip_rag: Skip RAG context retrieval
            force_experts: Force specific DoRA experts
            **kwargs: Additional inference options
            
        Returns:
            PipelineResult with output and metrics
        """
        if not self._pipeline or not self._health.pipeline_initialized:
            raise RuntimeError("Pipeline not initialized")
        
        self._health.total_requests += 1
        start_time = time.perf_counter()
        
        try:
            result = await self._pipeline.run(
                query=query,
                skip_rag=skip_rag,
                force_experts=force_experts,
                **kwargs
            )
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            self._update_latency_stats(latency_ms)
            
            if result.success:
                self._health.successful_requests += 1
                self._health.last_inference_time = time.time()
            else:
                self._health.failed_requests += 1
            
            return result
            
        except Exception as e:
            self._health.failed_requests += 1
            raise
    
    async def infer_streaming(self, query: str, **kwargs):
        """
        Run streaming inference (for real-time output).
        
        Args:
            query: User query text
            **kwargs: Additional inference options
            
        Yields:
            Generated tokens/text chunks
        """
        if not self._pipeline:
            raise RuntimeError("Pipeline not initialized")
        
        # Use streaming pipeline if available
        from r3mes.serving.inference_pipeline import StreamingInferencePipeline
        
        if isinstance(self._pipeline, StreamingInferencePipeline):
            async for token in self._pipeline.run_streaming(query, **kwargs):
                yield token
        else:
            # Fallback to non-streaming
            result = await self.infer(query, **kwargs)
            yield result.output
    
    # =========================================================================
    # Adapter Management
    # =========================================================================
    
    async def preload_adapters(self, adapter_ids: List[str]):
        """Preload DoRA adapters into cache."""
        if self._pipeline:
            await self._pipeline.preload_adapters(adapter_ids)
    
    def add_rag_document(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add document to RAG index."""
        if self._pipeline:
            self._pipeline.add_rag_document(doc_id, content, metadata)
    
    @property
    def pipeline(self) -> Optional[InferencePipeline]:
        """Get the underlying InferencePipeline."""
        return self._pipeline


def main():
    """Main entry point for serving engine."""
    parser = argparse.ArgumentParser(description="R3MES Serving Engine")
    parser.add_argument("--private-key", required=True, help="Private key for blockchain transactions")
    parser.add_argument("--blockchain-url", default="localhost:9090", help="Blockchain gRPC URL")
    parser.add_argument("--chain-id", default="remes-test", help="Chain ID")
    parser.add_argument("--model-ipfs-hash", help="IPFS hash of model to load")
    parser.add_argument("--model-version", default="v1.0.0", help="Model version")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    parser.add_argument("--json-logs", action="store_true", help="Use JSON-formatted logs")
    # Pipeline options
    parser.add_argument("--enable-rag", action="store_true", default=True, help="Enable RAG context retrieval")
    parser.add_argument("--disable-rag", action="store_true", help="Disable RAG context retrieval")
    parser.add_argument("--vram-capacity", type=int, default=2048, help="VRAM cache capacity in MB")
    parser.add_argument("--ram-capacity", type=int, default=8192, help="RAM cache capacity in MB")
    parser.add_argument("--adapter-dir", help="Directory for DoRA adapters")
    
    args = parser.parse_args()
    
    try:
        # Create and start serving engine
        engine = ServingEngine(
            private_key=args.private_key,
            blockchain_url=args.blockchain_url,
            chain_id=args.chain_id,
            model_ipfs_hash=args.model_ipfs_hash,
            model_version=args.model_version,
            log_level=args.log_level,
            use_json_logs=args.json_logs,
            enable_rag=args.enable_rag and not args.disable_rag,
            vram_capacity_mb=args.vram_capacity,
            ram_capacity_mb=args.ram_capacity,
            adapter_dir=args.adapter_dir,
        )
        
        # Start serving (blocking)
        engine.start()
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


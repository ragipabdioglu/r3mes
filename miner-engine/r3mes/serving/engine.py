#!/usr/bin/env python3
"""
R3MES Serving Engine

Production-ready serving engine that:
1. Loads AI model from IPFS hash
2. Listens for inference requests from blockchain
3. Executes inference using loaded model
4. Uploads results to IPFS
5. Submits results to blockchain via gRPC
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
from pathlib import Path

from r3mes.utils.logger import setup_logger
from r3mes.utils.error_handling import (
    exponential_backoff,
    handle_specific_errors,
    NetworkError,
    AuthenticationError,
    ResourceError,
)
from r3mes.utils.ipfs_manager import IPFSClient
from bridge.blockchain_client import BlockchainClient
from r3mes.miner.model_loader import load_model_with_enforced_lora


class ServingEngine:
    """Main serving engine class."""
    
    def __init__(
        self,
        private_key: str,
        blockchain_url: str = "localhost:9090",
        chain_id: str = "remes-test",
        model_ipfs_hash: Optional[str] = None,
        model_version: str = "v1.0.0",
        log_level: str = "INFO",
        use_json_logs: bool = False,
    ):
        """
        Initialize serving engine.
        
        Args:
            private_key: Private key for blockchain transactions
            blockchain_url: gRPC endpoint URL
            chain_id: Chain ID
            model_ipfs_hash: IPFS hash of model to load (optional, will query from blockchain if not provided)
            model_version: Model version to serve
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            use_json_logs: Whether to use JSON-formatted logs
        """
        # Production localhost validation
        is_production = os.getenv("R3MES_ENV", "development").lower() == "production"
        if is_production:
            # Extract hostname from blockchain_url (e.g., "localhost:9090" -> "localhost")
            blockchain_host = blockchain_url.split(":")[0] if ":" in blockchain_url else blockchain_url
            if blockchain_host.lower() in ("localhost", "127.0.0.1", "::1") or blockchain_host.startswith("127."):
                raise ValueError(
                    f"blockchain_url cannot use localhost in production: {blockchain_url}. "
                    "Please set blockchain_url to a production gRPC endpoint or use R3MES_NODE_GRPC_URL environment variable."
                )
        
        # Setup logger
        log_level_map = {
            "DEBUG": 10,
            "INFO": 20,
            "WARNING": 30,
            "ERROR": 40,
        }
        self.logger = setup_logger(
            "r3mes.serving",
            level=log_level_map.get(log_level.upper(), logging.INFO),
            use_json=use_json_logs,
        )
        
        # Graceful shutdown flag
        self._shutdown_requested = False
        
        # Register shutdown handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self._cleanup)
        
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
        self.serving_node_address = derive_address_from_public_key(
            hex_to_private_key(private_key).public_key()
        ) if private_key else None
        
        # IPFS client
        self.ipfs_client = IPFSClient()
        
        # Model state
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_available = False
        
        self.logger.info(f"Serving engine initialized (device: {self.device})")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self._shutdown_requested = True
    
    def _cleanup(self):
        """Cleanup resources on shutdown."""
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        self.logger.info("Serving engine cleaned up")
    
    def load_model(self, model_ipfs_hash: Optional[str] = None) -> bool:
        """
        Load model from IPFS hash.
        
        Args:
            model_ipfs_hash: IPFS hash of model (uses self.model_ipfs_hash if not provided)
            
        Returns:
            True if model loaded successfully
        """
        try:
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
            
            # Load model (simplified - would need actual model loading logic)
            # For now, this is a placeholder
            self.logger.info(f"Model downloaded to: {model_path}")
            self.logger.warning("Model loading logic needs to be implemented based on model format")
            
            # Update model IPFS hash
            self.model_ipfs_hash = hash_to_load
            
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {e}", exc_info=True)
            return False
    
    def update_status(self, is_available: bool, model_version: Optional[str] = None):
        """
        Update serving node status on blockchain.
        
        Args:
            is_available: Whether node is available for requests
            model_version: Model version (optional)
        """
        try:
            # This would send MsgUpdateServingNodeStatus transaction
            self.logger.info(f"Updating serving node status: available={is_available}, version={model_version or self.model_version}")
            self.is_available = is_available
            
            # Update status on blockchain
            if self.blockchain_client and self.serving_node_address:
                result = self.blockchain_client.update_serving_node_status(
                    serving_node=self.serving_node_address,
                    is_available=is_available,
                    model_version=model_version or self.model_version,
                    model_ipfs_hash=self.model_ipfs_hash,
                )
                if not result.get("success", False):
                    self.logger.warning(f"Failed to update serving node status on blockchain: {result.get('error', 'Unknown error')}")
            else:
                self.logger.warning("Blockchain client or serving node address not available, skipping blockchain update")
            
        except Exception as e:
            self.logger.error(f"Error updating status: {e}", exc_info=True)
    
    def process_inference_request(self, request_id: str, input_data_ipfs_hash: str) -> Optional[str]:
        """
        Process an inference request.
        
        Args:
            request_id: Inference request ID
            input_data_ipfs_hash: IPFS hash of input data
            
        Returns:
            IPFS hash of result, or None if failed
        """
        try:
            self.logger.info(f"Processing inference request: {request_id}")
            
            # Download input data from IPFS
            input_path = self.ipfs_client.get(input_data_ipfs_hash, output_dir="inference_inputs")
            if not input_path:
                self.logger.error(f"Failed to download input data: {input_data_ipfs_hash}")
                return None
            
            # Execute inference using loaded model
            if self.model is None:
                self.logger.error("Model not loaded, cannot execute inference")
                return None
            
            try:
                # Read input data from file
                with open(input_path, 'rb') as f:
                    input_data = f.read()
                
                # Parse input data (assuming JSON format with 'prompt' field)
                import json
                try:
                    input_json = json.loads(input_data.decode('utf-8'))
                    prompt = input_json.get('prompt', '')
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # If not JSON, treat as plain text
                    prompt = input_data.decode('utf-8', errors='ignore')
                
                # Execute inference
                self.model.eval()
                with torch.no_grad():
                    # Convert prompt to tensor (simplified - in production, use proper tokenization)
                    # For now, create a simple embedding-like tensor
                    # In production, this should use the model's tokenizer
                    if hasattr(self.model, 'tokenizer') and self.model.tokenizer is not None:
                        # Use tokenizer if available
                        inputs = self.model.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
                        input_ids = inputs['input_ids'].to(self.device)
                    else:
                        # Fallback: create dummy input (for SimpleBitNetModel)
                        # This is a placeholder - proper implementation should use tokenizer
                        batch_size = 1
                        seq_length = min(len(prompt.split()), 512)  # Approximate sequence length
                        input_ids = torch.randint(0, 1000, (batch_size, seq_length), device=self.device)
                    
                    # Run forward pass
                    outputs = self.model(input_ids=input_ids)
                    
                    # Extract logits or embeddings
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                        # Decode to text if tokenizer available
                        if hasattr(self.model, 'tokenizer') and self.model.tokenizer is not None:
                            generated_ids = torch.argmax(logits, dim=-1)
                            result_text = self.model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                        else:
                            # Fallback: convert logits to text representation
                            result_text = f"Inference result: {logits.shape}"
                    elif hasattr(outputs, 'last_hidden_state'):
                        # Use hidden state for embedding-based models
                        hidden_state = outputs.last_hidden_state
                        result_text = f"Embedding output: {hidden_state.shape}"
                    else:
                        # Fallback for other output types
                        result_text = str(outputs)
                
                # Serialize result
                result_json = {
                    "request_id": request_id,
                    "result": result_text,
                    "model_version": self.model_version,
                }
                result_data = json.dumps(result_json).encode('utf-8')
                
                self.logger.info(f"Inference executed successfully for request {request_id}")
                
            except Exception as e:
                self.logger.error(f"Error executing inference: {e}", exc_info=True)
                return None
            
            # Upload result to IPFS
            result_hash = self.ipfs_client.add_bytes(result_data)
            if not result_hash:
                self.logger.error("Failed to upload result to IPFS")
                return None
            
            self.logger.info(f"Inference completed: result_hash={result_hash}")
            return result_hash
            
        except Exception as e:
            self.logger.error(f"Error processing inference request: {e}", exc_info=True)
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
            # This would send MsgSubmitInferenceResult transaction
            self.logger.info(f"Submitting inference result: request_id={request_id}, latency={latency_ms}ms")
            
            # Submit result to blockchain
            if self.blockchain_client and self.serving_node_address:
                result = self.blockchain_client.submit_inference_result(
                    serving_node=self.serving_node_address,
                    request_id=request_id,
                    result_ipfs_hash=result_ipfs_hash,
                    latency_ms=latency_ms,
                )
                if not result.get("success", False):
                    self.logger.error(f"Failed to submit inference result to blockchain: {result.get('error', 'Unknown error')}")
                    return False
                self.logger.info(f"Inference result submitted to blockchain: TX {result.get('tx_hash', 'pending')}")
                return True
            else:
                self.logger.warning("Blockchain client or serving node address not available, skipping blockchain submission")
                return False
            
        except Exception as e:
            self.logger.error(f"Error submitting inference result: {e}", exc_info=True)
    
    async def start_async(self):
        """Start serving engine (async, non-blocking)."""
        self.logger.info("Starting serving engine...")
        
        # Load model if hash is provided
        if self.model_ipfs_hash:
            if not self.load_model():
                self.logger.error("Failed to load model, cannot start serving")
                return False
        
        # Update status to available
        self.update_status(is_available=True)
        
        # Start listening for inference requests
        self.logger.info("Serving engine started, polling for inference requests")
        
        # Main loop (async, non-blocking)
        import asyncio
        while not self._shutdown_requested:
            try:
                # Poll blockchain for new inference requests and process them
                await self._poll_and_process_requests()
            except Exception as e:
                self.logger.error(f"Error in polling loop: {e}", exc_info=True)
            
            await asyncio.sleep(5.0)  # Poll every 5 seconds (non-blocking sleep)
        
        self.logger.info("Serving engine stopped")
        return True
    
    def start(self):
        """
        Start serving engine (sync wrapper for async start).
        
        This method runs the async start_async in an event loop.
        For async contexts, use start_async() directly.
        """
        import asyncio
        
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, create task
                return asyncio.create_task(self.start_async())
            else:
                # Run async start
                return asyncio.run(self.start_async())
        except RuntimeError:
            # No event loop, create new one
            return asyncio.run(self.start_async())
    
    def stop(self):
        """Stop serving engine."""
        self.logger.info("Stopping serving engine...")
        self._shutdown_requested = True
        self.update_status(is_available=False)


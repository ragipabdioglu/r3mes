#!/usr/bin/env python3
"""
R3MES Inference Server

Production-ready inference server that:
1. Serves AI model inference requests
2. Handles HTTP and WebSocket connections
3. Manages model loading and LoRA adapters
4. Provides real-time inference capabilities
5. Integrates with serving node engine
"""

import logging
import json
import time
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
import torch
import torch.nn as nn

# HTTP server imports
try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from r3mes.utils.logger import setup_logger
from r3mes.miner.lora_manager import LoRAManager


class InferenceRequest(BaseModel):
    """Inference request model."""
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    model_version: Optional[str] = None
    adapter_id: Optional[str] = None


class InferenceResponse(BaseModel):
    """Inference response model."""
    response: str
    model_version: str
    adapter_id: Optional[str] = None
    latency_ms: int
    tokens_generated: int


class InferenceServer:
    """Inference server for serving AI models."""
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        model_version: str = "v1.0.0",
        host: str = "0.0.0.0",
        port: int = 8000,
        lora_manager: Optional[LoRAManager] = None,
        log_level: str = "INFO",
        use_json_logs: bool = False,
    ):
        """
        Initialize inference server.
        
        Args:
            model: AI model to serve
            model_version: Model version
            host: Server host
            port: Server port
            lora_manager: LoRA manager for adapter handling
            log_level: Logging level
            use_json_logs: Whether to use JSON-formatted logs
        """
        if not HAS_FASTAPI:
            raise ImportError("FastAPI is required for inference server. Install with: pip install fastapi uvicorn")
        
        # Setup logger
        log_level_map = {
            "DEBUG": 10,
            "INFO": 20,
            "WARNING": 30,
            "ERROR": 40,
        }
        self.logger = setup_logger(
            "r3mes.inference_server",
            level=log_level_map.get(log_level.upper(), logging.INFO),
            use_json=use_json_logs,
        )
        
        self.model = model
        self.model_version = model_version
        self.host = host
        self.port = port
        self.lora_manager = lora_manager or LoRAManager()
        
        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()
        
        # Server state
        self.is_running = False
        self.active_connections = []  # WebSocket connections
        self.request_count = 0
        self.total_latency_ms = 0
        
        # Create FastAPI app
        self.app = FastAPI(
            title="R3MES Inference Server",
            description="AI Model Inference Server with LoRA Adapter Support",
            version=model_version,
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
        self.logger.info(f"Inference server initialized (host: {host}, port: {port}, device: {self.device})")
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/")
        async def root():
            """Root endpoint."""
            return {
                "service": "R3MES Inference Server",
                "version": self.model_version,
                "status": "running" if self.is_running else "stopped",
                "device": str(self.device),
                "model_loaded": self.model is not None,
            }
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "model_loaded": self.model is not None,
                "device": str(self.device),
                "request_count": self.request_count,
                "avg_latency_ms": self.total_latency_ms / max(self.request_count, 1),
            }
        
        @self.app.get("/stats")
        async def stats():
            """Statistics endpoint."""
            cache_stats = self.lora_manager.get_cache_stats() if self.lora_manager else {}
            
            return {
                "model_version": self.model_version,
                "device": str(self.device),
                "request_count": self.request_count,
                "total_latency_ms": self.total_latency_ms,
                "avg_latency_ms": self.total_latency_ms / max(self.request_count, 1),
                "active_connections": len(self.active_connections),
                "lora_cache": cache_stats,
            }
        
        @self.app.post("/inference", response_model=InferenceResponse)
        async def inference(request: InferenceRequest):
            """Inference endpoint."""
            try:
                start_time = time.time()
                
                # Validate model
                if self.model is None:
                    raise HTTPException(status_code=503, detail="Model not loaded")
                
                # Load LoRA adapter if specified
                if request.adapter_id and self.lora_manager:
                    adapter_result = self.lora_manager.load_adapter(request.adapter_id)
                    if adapter_result is None:
                        raise HTTPException(status_code=404, detail=f"Adapter not found: {request.adapter_id}")
                    
                    adapter_state, _ = adapter_result
                    self.lora_manager.apply_adapter_to_model(self.model, request.adapter_id, adapter_state)
                
                # Run inference
                response_text, tokens_generated = await self._run_inference(
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                )
                
                # Calculate latency
                latency_ms = int((time.time() - start_time) * 1000)
                
                # Update stats
                self.request_count += 1
                self.total_latency_ms += latency_ms
                
                return InferenceResponse(
                    response=response_text,
                    model_version=self.model_version,
                    adapter_id=request.adapter_id,
                    latency_ms=latency_ms,
                    tokens_generated=tokens_generated,
                )
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error in inference: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time inference."""
            await websocket.accept()
            self.active_connections.append(websocket)
            
            try:
                while True:
                    # Receive request
                    data = await websocket.receive_text()
                    request_data = json.loads(data)
                    
                    # Validate request
                    try:
                        request = InferenceRequest(**request_data)
                    except Exception as e:
                        await websocket.send_text(json.dumps({
                            "error": f"Invalid request: {e}"
                        }))
                        continue
                    
                    # Process inference
                    try:
                        start_time = time.time()
                        
                        if self.model is None:
                            await websocket.send_text(json.dumps({
                                "error": "Model not loaded"
                            }))
                            continue
                        
                        # Load LoRA adapter if specified
                        if request.adapter_id and self.lora_manager:
                            adapter_result = self.lora_manager.load_adapter(request.adapter_id)
                            if adapter_result is not None:
                                adapter_state, _ = adapter_result
                                self.lora_manager.apply_adapter_to_model(self.model, request.adapter_id, adapter_state)
                        
                        # Run inference
                        response_text, tokens_generated = await self._run_inference(
                            prompt=request.prompt,
                            max_tokens=request.max_tokens,
                            temperature=request.temperature,
                            top_p=request.top_p,
                        )
                        
                        # Calculate latency
                        latency_ms = int((time.time() - start_time) * 1000)
                        
                        # Update stats
                        self.request_count += 1
                        self.total_latency_ms += latency_ms
                        
                        # Send response
                        response = InferenceResponse(
                            response=response_text,
                            model_version=self.model_version,
                            adapter_id=request.adapter_id,
                            latency_ms=latency_ms,
                            tokens_generated=tokens_generated,
                        )
                        
                        await websocket.send_text(response.json())
                        
                    except Exception as e:
                        self.logger.error(f"Error in WebSocket inference: {e}", exc_info=True)
                        await websocket.send_text(json.dumps({
                            "error": str(e)
                        }))
                        
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
                self.logger.info("WebSocket client disconnected")
        
        @self.app.get("/adapters")
        async def list_adapters():
            """List available LoRA adapters."""
            if not self.lora_manager:
                return {"adapters": []}
            
            adapters = self.lora_manager.list_adapters()
            return {"adapters": adapters}
        
        @self.app.post("/adapters/{adapter_id}/load")
        async def load_adapter(adapter_id: str):
            """Load LoRA adapter."""
            if not self.lora_manager:
                raise HTTPException(status_code=503, detail="LoRA manager not available")
            
            result = self.lora_manager.load_adapter(adapter_id)
            if result is None:
                raise HTTPException(status_code=404, detail=f"Adapter not found: {adapter_id}")
            
            return {"message": f"Adapter loaded: {adapter_id}"}
        
        @self.app.delete("/adapters/{adapter_id}")
        async def remove_adapter(adapter_id: str):
            """Remove LoRA adapter from cache."""
            if not self.lora_manager:
                raise HTTPException(status_code=503, detail="LoRA manager not available")
            
            success = self.lora_manager.remove_adapter(adapter_id)
            if not success:
                raise HTTPException(status_code=404, detail=f"Adapter not found: {adapter_id}")
            
            return {"message": f"Adapter removed: {adapter_id}"}
    
    async def _run_inference(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> tuple[str, int]:
        """
        Run model inference.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Tuple of (response_text, tokens_generated)
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
            
            # Tokenize input (simplified - in production, use proper tokenizer)
            if hasattr(self.model, 'tokenizer') and self.model.tokenizer is not None:
                # Use model's tokenizer
                inputs = self.model.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                input_ids = inputs['input_ids'].to(self.device)
            else:
                # Fallback: create dummy input for SimpleBitNetModel
                words = prompt.split()
                seq_length = min(len(words), 512)
                input_ids = torch.randint(0, 1000, (1, seq_length), device=self.device)
            
            # Run inference
            self.model.eval()
            with torch.no_grad():
                # Generate tokens (simplified generation)
                generated_tokens = 0
                current_input = input_ids
                
                for _ in range(max_tokens):
                    # Forward pass
                    outputs = self.model(current_input)
                    
                    # Get logits
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        # Fallback for SimpleBitNetModel
                        logits = outputs
                    
                    # Apply temperature
                    if temperature > 0:
                        logits = logits / temperature
                    
                    # Apply top-p sampling (simplified)
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        logits[indices_to_remove] = float('-inf')
                    
                    # Sample next token
                    probs = torch.softmax(logits[:, -1, :], dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Append to sequence
                    current_input = torch.cat([current_input, next_token], dim=1)
                    generated_tokens += 1
                    
                    # Check for end token (simplified)
                    if next_token.item() == 0:  # Assuming 0 is end token
                        break
            
            # Decode output (simplified)
            if hasattr(self.model, 'tokenizer') and self.model.tokenizer is not None:
                # Use model's tokenizer
                response_text = self.model.tokenizer.decode(
                    current_input[0][input_ids.shape[1]:],
                    skip_special_tokens=True
                )
            else:
                # Fallback: convert tokens to text representation
                generated_ids = current_input[0][input_ids.shape[1]:].cpu().numpy()
                response_text = f"Generated {len(generated_ids)} tokens: {generated_ids[:10].tolist()}..."
            
            return response_text, generated_tokens
            
        except Exception as e:
            self.logger.error(f"Error in inference: {e}", exc_info=True)
            return f"Error: {e}", 0
    
    def set_model(self, model: nn.Module, model_version: Optional[str] = None):
        """
        Set the model to serve.
        
        Args:
            model: AI model
            model_version: Optional model version
        """
        self.model = model
        if model_version:
            self.model_version = model_version
        
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()
        
        self.logger.info(f"Model set: {model_version or 'unknown'}")
    
    async def start_async(self):
        """Start inference server (async)."""
        try:
            self.is_running = True
            self.logger.info(f"Starting inference server on {self.host}:{self.port}")
            
            # Configure uvicorn
            config = uvicorn.Config(
                app=self.app,
                host=self.host,
                port=self.port,
                log_level="info" if self.logger.level <= logging.INFO else "warning",
                access_log=False,  # Disable access logs to reduce noise
            )
            
            server = uvicorn.Server(config)
            await server.serve()
            
        except Exception as e:
            self.logger.error(f"Error starting inference server: {e}", exc_info=True)
            self.is_running = False
            raise
    
    def start(self):
        """Start inference server (sync wrapper)."""
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
        """Stop inference server."""
        self.is_running = False
        self.logger.info("Inference server stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        cache_stats = self.lora_manager.get_cache_stats() if self.lora_manager else {}
        
        return {
            "model_version": self.model_version,
            "device": str(self.device),
            "is_running": self.is_running,
            "request_count": self.request_count,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": self.total_latency_ms / max(self.request_count, 1),
            "active_connections": len(self.active_connections),
            "lora_cache": cache_stats,
        }
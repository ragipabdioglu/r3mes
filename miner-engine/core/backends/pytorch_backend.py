"""
PyTorch Inference Backend (Phase 1)

Default backend using standard PyTorch operations.
Provides baseline functionality for development and testing.

Features:
- Full PyTorch compatibility
- Easy debugging
- CPU and GPU support
- Serves as fallback for other backends
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
import time
import logging

from ..inference_backend import (
    InferenceBackend,
    BackendCapabilities,
    BackendType,
    InferenceResult,
    register_backend,
)
from ..dora import BitLinearDoRA, DoRAAdapter
from ..bitlinear import BitLinear

logger = logging.getLogger(__name__)


class PyTorchBackend(InferenceBackend):
    """
    PyTorch-based inference backend.
    
    This is the default backend for Phase 1 development.
    Uses standard PyTorch operations for BitLinear + DoRA inference.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize PyTorch backend.
        
        Args:
            device: Target device ('cuda', 'cpu', or None for auto)
        """
        self.device = self._select_device(device)
        self.model: Optional[nn.Module] = None
        self.adapters: Dict[str, DoRAAdapter] = {}
        self.dora_layers: Dict[str, BitLinearDoRA] = {}
        
        logger.info(f"PyTorchBackend initialized on device: {self.device}")
    
    def _select_device(self, device: Optional[str]) -> str:
        """Select best available device."""
        if device:
            return device
        
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def load_model(self, model_path: str) -> bool:
        """
        Load base model from path.
        
        For now, creates a simple test model.
        In production, this would load from IPFS or local path.
        """
        try:
            # TODO: Implement actual model loading from IPFS
            # For now, create a placeholder model structure
            logger.info(f"Loading model from: {model_path}")
            
            # Placeholder: In production, load actual BitNet model
            self.model = self._create_placeholder_model()
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def _create_placeholder_model(self) -> nn.Module:
        """Create placeholder model for testing."""
        class PlaceholderModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(32000, 256)
                self.layers = nn.ModuleList([
                    BitLinear(256, 256) for _ in range(4)
                ])
                self.head = nn.Linear(256, 32000)
            
            def forward(self, x, dora_layers=None):
                h = self.embed(x)
                for i, layer in enumerate(self.layers):
                    if dora_layers and f"layer_{i}" in dora_layers:
                        h = dora_layers[f"layer_{i}"](h)
                    else:
                        h = layer(h)
                    h = F.gelu(h)
                return self.head(h)
        
        return PlaceholderModel()
    
    def load_adapter(self, adapter_id: str, adapter_path: str) -> bool:
        """Load DoRA adapter from file."""
        try:
            adapter = DoRAAdapter.load(adapter_path)
            self.adapters[adapter_id] = adapter
            
            # Create DoRA layer for each model layer
            if self.model:
                for i, layer in enumerate(self.model.layers):
                    if isinstance(layer, BitLinear):
                        dora_layer = BitLinearDoRA(
                            backbone=layer,
                            rank=adapter.rank,
                            alpha=adapter.alpha,
                        )
                        adapter.apply_to_layer(dora_layer)
                        dora_layer.to(self.device)
                        self.dora_layers[f"{adapter_id}_layer_{i}"] = dora_layer
            
            logger.info(f"Loaded adapter: {adapter_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load adapter {adapter_id}: {e}")
            return False
    
    def unload_adapter(self, adapter_id: str) -> bool:
        """Unload adapter from memory."""
        try:
            if adapter_id in self.adapters:
                del self.adapters[adapter_id]
            
            # Remove associated DoRA layers
            keys_to_remove = [k for k in self.dora_layers if k.startswith(adapter_id)]
            for key in keys_to_remove:
                del self.dora_layers[key]
            
            # Clear CUDA cache if on GPU
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            logger.info(f"Unloaded adapter: {adapter_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload adapter {adapter_id}: {e}")
            return False
    
    def inference(
        self,
        input_ids: torch.Tensor,
        adapter_ids: List[str],
        adapter_weights: List[float],
        **kwargs
    ) -> InferenceResult:
        """
        Run inference with specified adapters.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            adapter_ids: List of adapter IDs to use
            adapter_weights: Weights for each adapter
            **kwargs: Additional options (max_length, temperature, etc.)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.perf_counter()
        
        # Move input to device
        input_ids = input_ids.to(self.device)
        
        # Normalize weights
        total_weight = sum(adapter_weights)
        if total_weight > 0:
            adapter_weights = [w / total_weight for w in adapter_weights]
        
        # Prepare DoRA layers for this inference
        active_dora_layers = {}
        for adapter_id, weight in zip(adapter_ids, adapter_weights):
            if adapter_id in self.adapters:
                for key, dora_layer in self.dora_layers.items():
                    if key.startswith(adapter_id):
                        layer_name = key.replace(f"{adapter_id}_", "")
                        if layer_name not in active_dora_layers:
                            active_dora_layers[layer_name] = []
                        active_dora_layers[layer_name].append((dora_layer, weight))
        
        # Run inference
        with torch.no_grad():
            if active_dora_layers:
                # Multi-adapter inference
                output = self._multi_adapter_forward(input_ids, active_dora_layers)
            else:
                # Standard inference without adapters
                output = self.model(input_ids)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return InferenceResult(
            output=output,
            latency_ms=latency_ms,
            backend_used="pytorch",
            metadata={
                'adapters_used': adapter_ids,
                'adapter_weights': adapter_weights,
                'device': self.device,
            }
        )
    
    def _multi_adapter_forward(
        self,
        input_ids: torch.Tensor,
        active_dora_layers: Dict[str, List[tuple]],
    ) -> torch.Tensor:
        """
        Forward pass with multiple weighted adapters.
        
        Combines outputs from multiple DoRA adapters using weighted sum.
        """
        h = self.model.embed(input_ids)
        
        for i, layer in enumerate(self.model.layers):
            layer_name = f"layer_{i}"
            
            if layer_name in active_dora_layers:
                # Weighted combination of adapter outputs
                adapter_outputs = []
                weights = []
                
                for dora_layer, weight in active_dora_layers[layer_name]:
                    adapter_out = dora_layer(h)
                    adapter_outputs.append(adapter_out)
                    weights.append(weight)
                
                # Weighted sum
                h = sum(w * out for w, out in zip(weights, adapter_outputs))
            else:
                h = layer(h)
            
            h = F.gelu(h)
        
        return self.model.head(h)
    
    def get_capabilities(self) -> BackendCapabilities:
        """Get PyTorch backend capabilities."""
        has_gpu = torch.cuda.is_available()
        
        return BackendCapabilities(
            name="PyTorch Backend",
            backend_type=BackendType.PYTORCH,
            supports_gpu=has_gpu,
            supports_quantized=True,
            supports_batching=True,
            max_batch_size=64 if has_gpu else 16,
            estimated_speedup=1.0,  # Baseline
            vram_efficiency=0.6,  # Standard PyTorch efficiency
            precision="fp32",
        )
    
    def is_available(self) -> bool:
        """PyTorch is always available."""
        return True
    
    def get_vram_usage(self) -> Dict[str, float]:
        """Get current VRAM usage."""
        if self.device == 'cuda' and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            
            return {
                'used_mb': allocated,
                'reserved_mb': reserved,
                'total_mb': total,
                'free_mb': total - reserved,
            }
        else:
            # CPU mode - return system memory info
            import psutil
            mem = psutil.virtual_memory()
            return {
                'used_mb': mem.used / (1024 * 1024),
                'total_mb': mem.total / (1024 * 1024),
                'free_mb': mem.available / (1024 * 1024),
                'reserved_mb': 0,
            }
    
    def warmup(self, batch_size: int = 1, seq_len: int = 32):
        """
        Warmup the model with dummy inference.
        
        Useful for CUDA kernel compilation and memory allocation.
        """
        if self.model is None:
            logger.warning("Cannot warmup: model not loaded")
            return
        
        logger.info(f"Warming up with batch_size={batch_size}, seq_len={seq_len}")
        
        dummy_input = torch.randint(0, 1000, (batch_size, seq_len), device=self.device)
        
        with torch.no_grad():
            for _ in range(3):
                _ = self.model(dummy_input)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        logger.info("Warmup complete")


# Register PyTorch backend
register_backend(BackendType.PYTORCH, PyTorchBackend)

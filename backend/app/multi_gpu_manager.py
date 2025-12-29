"""
Multi-GPU Manager

Manages model inference across multiple GPUs for improved performance.
"""

import torch
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MultiGPUManager:
    """
    Manages model distribution across multiple GPUs.
    
    Supports data parallelism and model parallelism strategies.
    """
    
    def __init__(self):
        """Initialize multi-GPU manager."""
        self.devices = self._detect_gpus()
        self.strategy = "data_parallel"  # or "model_parallel"
        
        if len(self.devices) > 1:
            logger.info(f"Multi-GPU detected: {len(self.devices)} GPUs")
            for i, device in enumerate(self.devices):
                logger.info(f"  GPU {i}: {torch.cuda.get_device_name(device)}")
        else:
            logger.info("Single GPU or CPU mode")
    
    def _detect_gpus(self) -> List[torch.device]:
        """Detect available GPUs."""
        if not torch.cuda.is_available():
            return [torch.device("cpu")]
        
        num_gpus = torch.cuda.device_count()
        return [torch.device(f"cuda:{i}") for i in range(num_gpus)]
    
    def distribute_batch(self, batch: Any, model: torch.nn.Module) -> Dict[str, Any]:
        """
        Distribute batch across GPUs.
        
        Args:
            batch: Input batch
            model: Model to distribute
            
        Returns:
            Dictionary with distributed outputs
        """
        if len(self.devices) == 1:
            # Single GPU or CPU
            return {"output": model(batch.to(self.devices[0]))}
        
        if self.strategy == "data_parallel":
            return self._data_parallel_distribute(batch, model)
        else:
            return self._model_parallel_distribute(batch, model)
    
    def _data_parallel_distribute(self, batch: Any, model: torch.nn.Module) -> Dict[str, Any]:
        """Distribute using data parallelism."""
        # Split batch across GPUs
        batch_size = len(batch) if hasattr(batch, "__len__") else 1
        chunk_size = batch_size // len(self.devices)
        
        outputs = []
        for i, device in enumerate(self.devices):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < len(self.devices) - 1 else batch_size
            
            if hasattr(batch, "__getitem__"):
                chunk = batch[start_idx:end_idx]
            else:
                chunk = batch
            
            chunk = chunk.to(device)
            model_device = model.to(device)
            output = model_device(chunk)
            outputs.append(output.cpu())
        
        # Concatenate outputs
        if isinstance(outputs[0], torch.Tensor):
            return {"output": torch.cat(outputs, dim=0)}
        else:
            return {"output": outputs}
    
    def _model_parallel_distribute(self, batch: Any, model: torch.nn.Module) -> Dict[str, Any]:
        """
        Distribute using model parallelism.
        
        Splits model layers across multiple GPUs for very large models.
        """
        if len(self.devices) < 2:
            return self._data_parallel_distribute(batch, model)
        
        # Simple model parallelism: split model into chunks
        # For transformer models, split by layers
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
            # Split transformer layers across GPUs
            num_layers = len(model.transformer.layers)
            layers_per_gpu = num_layers // len(self.devices)
            
            # Move layers to different GPUs
            for i, device in enumerate(self.devices):
                start_layer = i * layers_per_gpu
                end_layer = start_layer + layers_per_gpu if i < len(self.devices) - 1 else num_layers
                
                # Move layers to device
                for j in range(start_layer, end_layer):
                    model.transformer.layers[j] = model.transformer.layers[j].to(device)
            
            # Move input embeddings and output head
            if hasattr(model.transformer, 'wte'):
                model.transformer.wte = model.transformer.wte.to(self.devices[0])
            if hasattr(model, 'lm_head'):
                model.lm_head = model.lm_head.to(self.devices[-1])
            
            # Forward pass with device transfers
            x = batch.to(self.devices[0])
            for i, device in enumerate(self.devices):
                start_layer = i * layers_per_gpu
                end_layer = start_layer + layers_per_gpu if i < len(self.devices) - 1 else num_layers
                
                for j in range(start_layer, end_layer):
                    x = model.transformer.layers[j](x)
                
                # Transfer to next device if not last
                if i < len(self.devices) - 1:
                    x = x.to(self.devices[i + 1])
            
            # Final output on last device
            if hasattr(model, 'lm_head'):
                output = model.lm_head(x)
            else:
                output = x
            
            return {"output": output}
        else:
            # Fallback to data parallelism for non-transformer models
            logger.warning("Model parallelism not supported for this model type, using data parallelism")
            return self._data_parallel_distribute(batch, model)
    
    def get_gpu_utilization(self) -> List[Dict[str, Any]]:
        """Get GPU utilization for all GPUs."""
        if not torch.cuda.is_available():
            return []
        
        utilizations = []
        for i, device in enumerate(self.devices):
            if device.type == "cuda":
                try:
                    # Get GPU memory usage
                    memory_allocated = torch.cuda.memory_allocated(device) / (1024**3)  # GB
                    memory_reserved = torch.cuda.memory_reserved(device) / (1024**3)  # GB
                    memory_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # GB
                    
                    utilizations.append({
                        "gpu_id": i,
                        "device_name": torch.cuda.get_device_name(device),
                        "memory_allocated_gb": memory_allocated,
                        "memory_reserved_gb": memory_reserved,
                        "memory_total_gb": memory_total,
                        "memory_usage_percent": (memory_allocated / memory_total) * 100,
                    })
                except Exception as e:
                    logger.error(f"Failed to get GPU {i} utilization: {e}")
        
        return utilizations


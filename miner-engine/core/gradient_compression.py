"""
Top-k Gradient Compression for Bandwidth Optimization

Compresses gradients by keeping only top-k% of values,
reducing bandwidth by ~90% (if k=0.1).
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple


class CompressedGradient:
    """Represents a compressed gradient."""
    
    def __init__(self, compressed_data: List[Dict[str, Any]]):
        self.compressed_data = compressed_data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "compressed_data": self.compressed_data,
            "compression_ratio": self._calculate_compression_ratio(),
        }
    
    def _calculate_compression_ratio(self) -> float:
        """Calculate compression ratio."""
        total_original = sum(len(d["indices"]) for d in self.compressed_data)
        total_compressed = sum(len(d["values"]) for d in self.compressed_data)
        return total_compressed / total_original if total_original > 0 else 0.0


def compress_gradients(
    gradients: List[torch.Tensor],
    k: float = 0.1,
) -> CompressedGradient:
    """
    Compress gradients by keeping only top-k% of values.
    
    Args:
        gradients: List of gradient tensors
        k: Fraction of values to keep (0.1 = top 10%)
    
    Returns:
        CompressedGradient with indices and values
    """
    compressed = []
    
    for grad in gradients:
        # Flatten gradient
        flat = grad.flatten()
        k_count = int(len(flat) * k)
        
        if k_count == 0:
            k_count = 1  # At least keep 1 value
        
        # Get top-k indices and values (by absolute value)
        topk_values, topk_indices = torch.topk(flat.abs(), k_count)
        
        # Get actual values (not absolute)
        topk_actual_values = flat[topk_indices]
        
        compressed.append({
            "indices": topk_indices.cpu().numpy().tolist(),
            "values": topk_actual_values.cpu().numpy().tolist(),
            "shape": list(grad.shape),
        })
    
    return CompressedGradient(compressed)


def decompress_gradients(
    compressed: CompressedGradient,
    device: str = "cuda",
) -> List[torch.Tensor]:
    """
    Decompress gradients from compressed format.
    
    Args:
        compressed: CompressedGradient object
        device: Device to place tensors on
    
    Returns:
        List of decompressed gradient tensors
    """
    decompressed = []
    
    for data in compressed.compressed_data:
        shape = data["shape"]
        indices = torch.tensor(data["indices"], device=device)
        values = torch.tensor(data["values"], device=device)
        
        # Reconstruct tensor
        flat = torch.zeros(np.prod(shape), device=device)
        flat[indices] = values
        
        # Reshape to original shape
        decompressed.append(flat.reshape(shape))
    
    return decompressed


"""
BitNet b1.58 Quantization Implementation

BitNet b1.58 uses ternary quantization: {-1, 0, +1} for weights.
This module provides quantization and dequantization functions for Llama models.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def quantize_to_bitnet_b158(weights: torch.Tensor, threshold: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize weights to BitNet b1.58 format: {-1, 0, +1}.
    
    BitNet b1.58 quantization:
    - Weights are quantized to ternary values: {-1, 0, +1}
    - Scale factor is stored for dequantization
    - Formula: quantized = sign(weights) if |weights| > threshold else 0
    - Scale = mean(|weights|) for non-zero weights
    
    Args:
        weights: Full precision weights (FP16 or FP32)
        threshold: Threshold for zero quantization (default: 0.0)
        
    Returns:
        Tuple of (quantized_weights, scale_factor)
        - quantized_weights: Tensor with values in {-1, 0, +1}
        - scale_factor: Scale factor for dequantization
    """
    # Calculate scale factor (mean absolute value of non-zero weights)
    abs_weights = torch.abs(weights)
    non_zero_mask = abs_weights > threshold
    if non_zero_mask.any():
        scale_factor = torch.mean(abs_weights[non_zero_mask])
    else:
        scale_factor = torch.tensor(1.0, device=weights.device, dtype=weights.dtype)
    
    # Quantize to {-1, 0, +1}
    # sign(weights) gives -1, 0, or +1
    quantized = torch.sign(weights)
    
    # Set values below threshold to zero
    if threshold > 0:
        quantized[abs_weights <= threshold] = 0.0
    
    return quantized, scale_factor


def dequantize_from_bitnet_b158(quantized_weights: torch.Tensor, scale_factor: torch.Tensor) -> torch.Tensor:
    """
    Dequantize BitNet b1.58 weights back to full precision.
    
    Args:
        quantized_weights: Quantized weights in {-1, 0, +1}
        scale_factor: Scale factor for dequantization
        
    Returns:
        Dequantized weights (approximate full precision)
    """
    return quantized_weights * scale_factor


def apply_bitnet_quantization_to_linear(module: nn.Linear, threshold: float = 0.0) -> nn.Module:
    """
    Apply BitNet b1.58 quantization to a Linear layer.
    
    This replaces the full precision weights with quantized {-1, 0, +1} weights
    and stores scale factors for dequantization.
    
    Args:
        module: nn.Linear module to quantize
        threshold: Threshold for zero quantization
        
    Returns:
        Quantized module (weights are frozen)
    """
    if not isinstance(module, nn.Linear):
        return module
    
    # Get original weights
    original_weights = module.weight.data
    
    # Quantize weights
    quantized_weights, scale_factor = quantize_to_bitnet_b158(original_weights, threshold)
    
    # Replace weights with quantized version
    module.weight.data = quantized_weights
    module.weight.requires_grad = False  # Freeze quantized weights
    
    # Store scale factor as buffer for dequantization
    module.register_buffer('bitnet_scale', scale_factor)
    module.register_buffer('bitnet_quantized', torch.tensor(True))
    
    logger.debug(
        f"Quantized Linear layer: {module.in_features}x{module.out_features}, "
        f"scale={scale_factor.item():.4f}"
    )
    
    return module


def apply_bitnet_quantization_to_model(
    model: nn.Module,
    threshold: float = 0.0,
    target_modules: Optional[list] = None,
) -> nn.Module:
    """
    Apply BitNet b1.58 quantization to all Linear layers in a model.
    
    This function recursively finds all nn.Linear modules and quantizes them.
    LoRA adapters are NOT quantized (they remain trainable).
    
    Args:
        model: Model to quantize
        threshold: Threshold for zero quantization
        target_modules: List of module names to quantize (None = all Linear layers)
        
    Returns:
        Quantized model
    """
    if target_modules is None:
        # Default: Quantize all Linear layers except LoRA adapters
        target_modules = []
    
    quantized_count = 0
    skipped_count = 0
    
    for name, module in model.named_modules():
        # Skip LoRA adapters (they should remain trainable)
        if 'lora' in name.lower():
            skipped_count += 1
            continue
        
        # Skip if target_modules is specified and name not in list
        if target_modules and name not in target_modules:
            continue
        
        if isinstance(module, nn.Linear):
            try:
                apply_bitnet_quantization_to_linear(module, threshold)
                quantized_count += 1
            except Exception as e:
                logger.warning(f"Failed to quantize {name}: {e}")
                skipped_count += 1
    
    logger.info(
        f"BitNet b1.58 quantization applied: {quantized_count} layers quantized, "
        f"{skipped_count} layers skipped"
    )
    
    return model


def get_quantization_stats(model: nn.Module) -> Dict[str, int]:
    """
    Get statistics about BitNet quantization in a model.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary with quantization statistics
    """
    quantized_layers = 0
    total_layers = 0
    total_params = 0
    quantized_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            total_layers += 1
            total_params += module.weight.numel()
            
            # Check if layer is quantized
            if hasattr(module, 'bitnet_quantized') and module.bitnet_quantized:
                quantized_layers += 1
                quantized_params += module.weight.numel()
    
    return {
        "total_layers": total_layers,
        "quantized_layers": quantized_layers,
        "total_parameters": total_params,
        "quantized_parameters": quantized_params,
        "quantization_ratio": quantized_layers / total_layers if total_layers > 0 else 0.0,
    }


class BitNetLinear(nn.Module):
    """
    BitNet b1.58 Linear layer wrapper.
    
    This is a drop-in replacement for nn.Linear that uses BitNet b1.58 quantization.
    The weights are quantized to {-1, 0, +1} and scale factors are stored.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        threshold: float = 0.0,
    ):
        """
        Initialize BitNet Linear layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            bias: Whether to use bias
            threshold: Threshold for zero quantization
        """
        super(BitNetLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        
        # Initialize full precision weights
        self.register_parameter('weight', nn.Parameter(torch.randn(out_features, in_features)))
        
        # Quantize weights
        quantized_weights, scale_factor = quantize_to_bitnet_b158(self.weight.data, threshold)
        self.weight.data = quantized_weights
        self.weight.requires_grad = False  # Freeze quantized weights
        
        # Store scale factor
        self.register_buffer('scale_factor', scale_factor)
        
        # Bias (optional)
        if bias:
            self.register_parameter('bias', nn.Parameter(torch.zeros(out_features)))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quantized weights.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Dequantize weights for computation
        # In production, this would use custom CUDA kernels for efficiency
        dequantized_weights = dequantize_from_bitnet_b158(self.weight, self.scale_factor)
        
        return torch.nn.functional.linear(x, dequantized_weights, self.bias)


"""
BitNet 1.58-bit BitLinear Layer with LoRA (Low-Rank Adaptation)

This module implements BitLinear layers with:
- Frozen backbone weights (quantized {-1, 0, +1})
- Trainable LoRA adapters (rank 4-64)
- Deterministic CUDA kernels for bit-exact reproducibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class BitLinear(nn.Module):
    """
    BitLinear layer with frozen backbone and trainable LoRA adapters.
    
    Architecture:
    - Backbone weights: Frozen, quantized to {-1, 0, +1}
    - LoRA adapters: Trainable matrices A and B with rank r (4-64)
    - Forward: output = backbone(x) + (alpha/rank) * x @ A.T @ B.T
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        bias: bool = False,
        deterministic: bool = True,
    ):
        """
        Initialize BitLinear layer with LoRA adapters.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            lora_rank: LoRA rank (default: 8, range: 4-64)
            lora_alpha: LoRA alpha scaling factor (default: 16.0)
            bias: Whether to use bias (default: False for BitNet)
            deterministic: Enable deterministic operations for reproducibility
        """
        super(BitLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.deterministic = deterministic
        
        # Frozen backbone weights (quantized {-1, 0, +1})
        # These weights are initialized once and never updated
        self.register_buffer('backbone_weight', self._quantize_weights(
            torch.randn(out_features, in_features)
        ))
        
        # LoRA adapters: A and B matrices
        # A: [rank, in_features], B: [out_features, rank]
        # This allows: x @ A.T = [batch, in_features] @ [in_features, rank] = [batch, rank]
        self.lora_A = nn.Parameter(torch.randn(lora_rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank))
        
        # Bias (optional)
        if bias:
            self.register_buffer('bias', torch.zeros(out_features))
        else:
            self.register_buffer('bias', None)
        
        # Mark backbone as frozen (no gradients)
        self.backbone_weight.requires_grad = False
        
        # Configure deterministic operations if enabled
        if deterministic:
            self._configure_deterministic()
    
    def _quantize_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Quantize weights to {-1, 0, +1} using sign function.
        
        Args:
            weights: Full precision weights
            
        Returns:
            Quantized weights in {-1, 0, +1}
        """
        # Sign quantization: -1, 0, or +1
        # Using sign function with threshold for zero
        quantized = torch.sign(weights)
        # Set very small values to zero
        quantized[torch.abs(weights) < 0.1] = 0.0
        return quantized
    
    def _configure_deterministic(self):
        """Configure deterministic operations for reproducibility."""
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: backbone(x) + LoRA(x)
        
        Formula: output = backbone(x) + (alpha/rank) * x @ A.T @ B.T
        
        Args:
            x: Input tensor [batch_size, ..., in_features]
            
        Returns:
            Output tensor [batch_size, ..., out_features]
        """
        # Backbone forward pass (frozen weights)
        backbone_out = F.linear(x, self.backbone_weight, self.bias)
        
        # LoRA forward pass
        # Formula: output = backbone(x) + (alpha/rank) * x @ A.T @ B.T
        # x @ A.T: [batch, ..., in_features] @ [in_features, rank] -> [batch, ..., rank]
        # Then @ B.T: [batch, ..., rank] @ [rank, out_features] -> [batch, ..., out_features]
        # lora_A shape: [rank, in_features], so A.T shape: [in_features, rank]
        lora_intermediate = torch.matmul(x, self.lora_A.t())  # x @ A.T = [batch, ..., in_features] @ [in_features, rank] = [batch, ..., rank]
        # lora_B shape: [out_features, rank], so B.T shape: [rank, out_features]
        lora_out = torch.matmul(lora_intermediate, self.lora_B.t())  # [batch, ..., rank] @ [rank, out_features] = [batch, ..., out_features]
        
        # Scale LoRA output by alpha/rank
        lora_scaled = lora_out * (self.lora_alpha / self.lora_rank)
        
        # Combine backbone and LoRA outputs
        output = backbone_out + lora_scaled
        
        return output
    
    def get_lora_params(self) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Get LoRA adapter parameters for serialization.
        
        Returns:
            Tuple of (lora_A, lora_B, lora_alpha)
        """
        return (self.lora_A.data, self.lora_B.data, self.lora_alpha)
    
    def set_lora_params(self, lora_A: torch.Tensor, lora_B: torch.Tensor, alpha: float):
        """
        Set LoRA adapter parameters (for loading from serialized state).
        
        Args:
            lora_A: LoRA A matrix
            lora_B: LoRA B matrix
            alpha: LoRA alpha scaling factor
        """
        self.lora_A.data = lora_A
        self.lora_B.data = lora_B
        self.lora_alpha = alpha
    
    def get_backbone_hash(self) -> str:
        """
        Get hash of backbone weights for verification.
        
        Returns:
            Hex string hash of backbone weights
        """
        import hashlib
        backbone_bytes = self.backbone_weight.cpu().numpy().tobytes()
        return hashlib.sha256(backbone_bytes).hexdigest()
    
    def estimate_size_mb(self) -> float:
        """
        Estimate size of LoRA adapters in MB.
        
        Returns:
            Size in megabytes
        """
        # LoRA A: [in_features, rank] * 4 bytes (float32)
        # LoRA B: [out_features, rank] * 4 bytes (float32)
        lora_A_size = self.in_features * self.lora_rank * 4
        lora_B_size = self.out_features * self.lora_rank * 4
        total_bytes = lora_A_size + lora_B_size
        return total_bytes / (1024 * 1024)  # Convert to MB


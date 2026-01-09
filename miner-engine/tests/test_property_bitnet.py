"""
Property-Based Tests for BitNet LoRA Architecture

Using Hypothesis for property-based testing with minimum 100 iterations
for statistical confidence.
"""

from hypothesis import given, strategies as st
import torch
import torch.nn as nn
import numpy as np
from core.bitlinear import BitLinear


# Property: Frozen Backbone + LoRA Adapter System
# Validates: Requirements 2.1
# Feature: r3mes-pouw, Property: BitNet LoRA Architecture
@given(
    hidden_size=st.integers(min_value=64, max_value=2048),
    lora_rank=st.integers(min_value=4, max_value=64),
    batch_size=st.integers(min_value=1, max_value=32),
    seq_len=st.integers(min_value=1, max_value=512),
)
def test_property_frozen_backbone_lora_adapters(hidden_size, lora_rank, batch_size, seq_len):
    """
    Property: BitNet layers must have frozen backbone weights (quantized {-1, 0, +1})
    and trainable LoRA adapters.
    
    For any valid configuration:
    - Backbone weights must be in {-1, 0, +1}
    - LoRA adapters must be trainable (requires_grad=True)
    - Backbone weights must be frozen (requires_grad=False)
    """
    # Create BitLinear layer
    layer = BitLinear(hidden_size, hidden_size, lora_rank=lora_rank, deterministic=True)
    
    # Property 1: Backbone weights must be quantized to {-1, 0, +1}
    backbone_weights = layer.weight.data
    unique_values = torch.unique(backbone_weights)
    assert all(v in [-1.0, 0.0, 1.0] for v in unique_values), \
        f"Backbone weights must be in {{-1, 0, +1}}, got {unique_values.tolist()}"
    
    # Property 2: Backbone weights must be frozen
    assert not layer.weight.requires_grad, "Backbone weights must be frozen (requires_grad=False)"
    
    # Property 3: LoRA adapters must be trainable
    assert layer.lora_A.requires_grad, "LoRA adapter A must be trainable"
    assert layer.lora_B.requires_grad, "LoRA adapter B must be trainable"
    
    # Property 4: Forward pass must work correctly
    x = torch.randn(batch_size, seq_len, hidden_size)
    output = layer(x)
    
    assert output.shape == (batch_size, seq_len, hidden_size), \
        f"Output shape must match input shape, got {output.shape}"


# Property: LoRA Adapter Gradient Updates (Backbone Frozen)
# Validates: Requirements 2.3, 2.4
# Feature: r3mes-pouw, Property: LoRA Gradient Updates
@given(
    hidden_size=st.integers(min_value=64, max_value=512),
    lora_rank=st.integers(min_value=4, max_value=32),
    learning_rate=st.floats(min_value=1e-5, max_value=1e-2),
)
def test_property_lora_only_gradient_updates(hidden_size, lora_rank, learning_rate):
    """
    Property: Only LoRA adapter parameters should receive gradient updates,
    backbone weights must remain frozen.
    
    For any training step:
    - LoRA adapter gradients must be non-zero (if loss > 0)
    - Backbone weight gradients must be None (frozen)
    - Backbone weights must remain unchanged after optimizer step
    """
    # Create BitLinear layer
    layer = BitLinear(hidden_size, hidden_size, lora_rank=lora_rank, deterministic=True)
    
    # Store original backbone weights
    original_backbone = layer.weight.data.clone()
    
    # Create optimizer (only for LoRA parameters)
    optimizer = torch.optim.Adam(
        [layer.lora_A, layer.lora_B],
        lr=learning_rate
    )
    
    # Forward and backward pass
    x = torch.randn(1, 10, hidden_size)
    target = torch.randn(1, 10, hidden_size)
    
    output = layer(x)
    loss = nn.MSELoss()(output, target)
    loss.backward()
    
    # Property 1: LoRA adapters must have gradients
    assert layer.lora_A.grad is not None, "LoRA adapter A must have gradients"
    assert layer.lora_B.grad is not None, "LoRA adapter B must have gradients"
    
    # Property 2: Backbone weights must not have gradients
    assert layer.weight.grad is None, "Backbone weights must not have gradients (frozen)"
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    
    # Property 3: Backbone weights must remain unchanged
    assert torch.equal(layer.weight.data, original_backbone), \
        "Backbone weights must remain unchanged after optimizer step"


# Property: LoRA Adapter Transmission (MB scale, 99.6% bandwidth reduction)
# Validates: Requirements 2.5, 2.6
# Feature: r3mes-pouw, Property: LoRA Serialization
@given(
    hidden_size=st.integers(min_value=128, max_value=2048),
    lora_rank=st.integers(min_value=4, max_value=64),
    num_layers=st.integers(min_value=1, max_value=24),
)
def test_property_lora_serialization_bandwidth(hidden_size, lora_rank, num_layers):
    """
    Property: LoRA adapter serialization must result in MB-scale data
    (99.6%+ bandwidth reduction compared to full weights).
    
    For any model configuration:
    - LoRA adapter size must be << full weight size
    - Bandwidth reduction must be >= 99.6%
    """
    # Calculate full weight size (GB scale)
    # Full weights: hidden_size * hidden_size * num_layers * 4 bytes (float32)
    full_weight_size_bytes = hidden_size * hidden_size * num_layers * 4
    full_weight_size_mb = full_weight_size_bytes / (1024 * 1024)
    
    # Calculate LoRA adapter size (MB scale)
    # LoRA: (hidden_size * lora_rank * 2) * num_layers * 4 bytes
    # Matrix A: hidden_size * lora_rank
    # Matrix B: lora_rank * hidden_size
    lora_size_bytes = (hidden_size * lora_rank * 2) * num_layers * 4
    lora_size_mb = lora_size_bytes / (1024 * 1024)
    
    # Property 1: LoRA size must be in MB scale (< 1GB)
    assert lora_size_mb < 1024, \
        f"LoRA size must be < 1GB, got {lora_size_mb:.2f} MB"
    
    # Property 2: Bandwidth reduction must be >= 99.6%
    if full_weight_size_mb > 0:
        bandwidth_reduction = (1 - (lora_size_mb / full_weight_size_mb)) * 100
        assert bandwidth_reduction >= 99.6, \
            f"Bandwidth reduction must be >= 99.6%, got {bandwidth_reduction:.2f}%"
    
    # Property 3: LoRA size must be significantly smaller
    if full_weight_size_mb > 0:
        size_ratio = lora_size_mb / full_weight_size_mb
        assert size_ratio < 0.01, \
            f"LoRA must be < 1% of full weight size, got {size_ratio*100:.2f}%"


# Property: BitNet Weight Quantization
# Validates: Requirements 2.1
# Feature: r3mes-pouw, Property 6: BitNet Weight Quantization
@given(
    layer_size=st.integers(min_value=1, max_value=1000),
)
def test_property_bitnet_weight_quantization(layer_size):
    """
    Property: BitNet weights must be quantized to {-1, 0, +1}.
    
    For any layer size:
    - All weights must be in {-1, 0, +1}
    """
    # Create BitLinear layer
    layer = BitLinear(layer_size, layer_size, lora_rank=8, deterministic=True)
    
    # Property: All weights must be in {-1, 0, +1}
    weights = layer.weight.data.flatten()
    unique_values = torch.unique(weights)
    
    assert all(v in [-1.0, 0.0, 1.0] for v in unique_values), \
        f"All weights must be in {{-1, 0, +1}}, got unique values: {unique_values.tolist()}"


if __name__ == "__main__":
    # Run property tests with Hypothesis
    import pytest
    pytest.main([__file__, "-v"])


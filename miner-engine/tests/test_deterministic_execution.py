"""
Deterministic execution and gradient consistency tests for R3MES miner engine.

These tests are CPU-only and validate that:
- configure_deterministic_execution() correctly locks RNG state
- BitLinear forward passes are bit-exact reproducible given the same global seed
- LoRATrainer produces a stable gradient hash under the same global seed
"""

import torch
import torch.nn as nn

from core.bitlinear import BitLinear
from core.trainer import LoRATrainer
from core.deterministic import configure_deterministic_execution, get_deterministic_config


def test_deterministic_config_basic_flags():
    """configure_deterministic_execution should set deterministic flags consistently."""
    configure_deterministic_execution(global_seed=123)
    cfg = get_deterministic_config()

    assert cfg["torch_use_deterministic_algorithms"] is True
    assert cfg["torch_cudnn_deterministic"] is True
    assert cfg["torch_cudnn_benchmark"] is False
    # CUBLAS_WORKSPACE_CONFIG is set either by environment or default in deterministic module
    assert cfg["cublas_workspace_config"] is not None
    assert cfg["python_hash_seed"] is not None


def _run_bitlinear_forward_once(global_seed: int) -> torch.Tensor:
    """Helper: run a single BitLinear forward pass on CPU with a fixed global seed."""
    # Configure global deterministic execution
    configure_deterministic_execution(global_seed=global_seed)

    # Keep everything on CPU for fully reproducible results
    layer = BitLinear(
        in_features=128,
        out_features=128,
        lora_rank=8,
        deterministic=True,
    )

    x = torch.randn(4, 128)
    with torch.no_grad():
        out = layer(x)
    return out.cpu()


def test_bitlinear_forward_is_deterministic_on_cpu():
    """
    BitLinear forward pass must be bit-exact reproducible on CPU
    when using the same global_seed.
    """
    out1 = _run_bitlinear_forward_once(global_seed=42)
    out2 = _run_bitlinear_forward_once(global_seed=42)

    # Bit-exact equality
    assert torch.equal(out1, out2), "BitLinear forward outputs must match for the same global seed"


class _SimpleBitNetModel(nn.Module):
    """Minimal BitNet-style model for deterministic gradient tests."""

    def __init__(self):
        super().__init__()
        self.layer = BitLinear(64, 64, lora_rank=8, deterministic=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


def _run_lora_training_step_once(global_seed: int):
    """
    Helper: run a single LoRATrainer training step on CPU with a fixed global seed,
    returning (loss, gradient_hash).
    """
    # Global deterministic configuration before model instantiation
    configure_deterministic_execution(global_seed=global_seed)

    model = _SimpleBitNetModel()
    trainer = LoRATrainer(
        model,
        learning_rate=1e-4,
        deterministic=True,
        device=torch.device("cpu"),
    )

    # Additional trainer-local seed for safety
    trainer.set_seed(global_seed)

    inputs = torch.randn(8, 64)
    targets = torch.randn(8, 64)

    loss, gradients = trainer.train_step(inputs, targets)
    grad_hash = trainer.compute_gradient_hash(gradients)

    return loss, grad_hash


def test_lora_trainer_gradient_hash_is_deterministic_on_cpu():
    """
    LoRATrainer must produce the same gradient hash when
    running with the same global_seed on CPU.
    """
    loss1, hash1 = _run_lora_training_step_once(global_seed=1337)
    loss2, hash2 = _run_lora_training_step_once(global_seed=1337)

    assert hash1 == hash2, "Gradient hash must be deterministic for the same global seed on CPU"
    assert loss1 == loss2, "Loss values must match for the same global seed on CPU"



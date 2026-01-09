"""
Unit tests for LoRA Trainer module.

Tests:
- Trainer initialization
- Backbone freezing
- LoRA parameter extraction
- Training step execution
- Gradient hash computation
- Deterministic execution
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import os

# Set environment variables before importing trainer
os.environ.setdefault("LORA_WEIGHT_DECAY", "0.01")
os.environ.setdefault("LORA_GRAD_CLIP_MAX_NORM", "1.0")

from core.trainer import LoRATrainer, DEFAULT_WEIGHT_DECAY, DEFAULT_GRAD_CLIP_MAX_NORM
from core.bitlinear import BitLinear


class SimpleBitLinearModel(nn.Module):
    """Simple model with BitLinear layers for testing."""
    
    def __init__(self, in_features: int = 64, out_features: int = 32, lora_rank: int = 4):
        super().__init__()
        self.layer1 = BitLinear(in_features, out_features, lora_rank=lora_rank)
        self.layer2 = BitLinear(out_features, out_features, lora_rank=lora_rank)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x


class TestLoRATrainerInitialization:
    """Tests for LoRATrainer initialization."""
    
    def test_trainer_initialization(self):
        """Test basic trainer initialization."""
        model = SimpleBitLinearModel()
        trainer = LoRATrainer(model, learning_rate=1e-4)
        
        assert trainer.model is model
        assert trainer.learning_rate == 1e-4
        assert trainer.deterministic is True
        assert trainer.training_step == 0
        assert len(trainer.loss_history) == 0
    
    def test_trainer_with_custom_device(self):
        """Test trainer with custom device."""
        model = SimpleBitLinearModel()
        device = torch.device('cpu')
        trainer = LoRATrainer(model, device=device)
        
        assert trainer.device == device
    
    def test_trainer_non_deterministic(self):
        """Test trainer with deterministic=False."""
        model = SimpleBitLinearModel()
        trainer = LoRATrainer(model, deterministic=False)
        
        assert trainer.deterministic is False


class TestBackboneFreezing:
    """Tests for backbone weight freezing."""
    
    def test_backbone_weights_frozen(self):
        """Test that backbone weights are frozen after initialization."""
        model = SimpleBitLinearModel()
        trainer = LoRATrainer(model)
        
        for module in model.modules():
            if isinstance(module, BitLinear):
                if hasattr(module, 'backbone_weight'):
                    assert module.backbone_weight.requires_grad is False
    
    def test_lora_parameters_trainable(self):
        """Test that LoRA parameters remain trainable."""
        model = SimpleBitLinearModel()
        trainer = LoRATrainer(model)
        
        for module in model.modules():
            if isinstance(module, BitLinear):
                assert module.lora_A.requires_grad is True
                assert module.lora_B.requires_grad is True


class TestLoRAParameterExtraction:
    """Tests for LoRA parameter extraction."""
    
    def test_get_lora_parameters(self):
        """Test extraction of LoRA parameters."""
        model = SimpleBitLinearModel()
        trainer = LoRATrainer(model)
        
        lora_params = trainer.lora_params
        
        # Should have 4 parameters (2 layers x 2 matrices each)
        assert len(lora_params) == 4
        
        # All should be tensors
        for param in lora_params:
            assert isinstance(param, torch.Tensor)
            assert param.requires_grad is True


class TestTrainingStep:
    """Tests for training step execution."""
    
    def test_single_training_step(self):
        """Test single training step."""
        model = SimpleBitLinearModel(in_features=64, out_features=32)
        trainer = LoRATrainer(model, learning_rate=1e-3)
        
        # Create dummy data
        inputs = torch.randn(8, 64)
        targets = torch.randn(8, 32)
        
        # Perform training step
        loss, gradients = trainer.train_step(inputs, targets)
        
        assert isinstance(loss, float)
        assert loss > 0
        assert isinstance(gradients, dict)
        assert len(gradients) > 0
        assert trainer.training_step == 1
        assert len(trainer.loss_history) == 1
    
    def test_multiple_training_steps(self):
        """Test multiple training steps."""
        model = SimpleBitLinearModel(in_features=64, out_features=32)
        trainer = LoRATrainer(model, learning_rate=1e-3)
        
        inputs = torch.randn(8, 64)
        targets = torch.randn(8, 32)
        
        losses = []
        for _ in range(5):
            loss, _ = trainer.train_step(inputs, targets)
            losses.append(loss)
        
        assert trainer.training_step == 5
        assert len(trainer.loss_history) == 5
    
    def test_custom_loss_function(self):
        """Test training with custom loss function."""
        model = SimpleBitLinearModel(in_features=64, out_features=32)
        trainer = LoRATrainer(model)
        
        inputs = torch.randn(8, 64)
        targets = torch.randn(8, 32)
        
        custom_loss = nn.L1Loss()
        loss, gradients = trainer.train_step(inputs, targets, loss_fn=custom_loss)
        
        assert isinstance(loss, float)


class TestGradientHash:
    """Tests for gradient hash computation."""
    
    def test_gradient_hash_computation(self):
        """Test gradient hash computation."""
        model = SimpleBitLinearModel(in_features=64, out_features=32)
        trainer = LoRATrainer(model)
        
        inputs = torch.randn(8, 64)
        targets = torch.randn(8, 32)
        
        _, gradients = trainer.train_step(inputs, targets)
        
        hash_result = trainer.compute_gradient_hash(gradients)
        
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64  # SHA256 hex length
    
    def test_gradient_hash_determinism(self):
        """Test that gradient hash is deterministic."""
        model = SimpleBitLinearModel(in_features=64, out_features=32)
        trainer = LoRATrainer(model, deterministic=True)
        trainer.set_seed(42)
        
        inputs = torch.randn(8, 64)
        targets = torch.randn(8, 32)
        
        _, gradients = trainer.train_step(inputs, targets)
        hash1 = trainer.compute_gradient_hash(gradients)
        
        # Reset and repeat
        model2 = SimpleBitLinearModel(in_features=64, out_features=32)
        trainer2 = LoRATrainer(model2, deterministic=True)
        trainer2.set_seed(42)
        
        inputs2 = torch.randn(8, 64)
        targets2 = torch.randn(8, 32)
        
        _, gradients2 = trainer2.train_step(inputs2, targets2)
        hash2 = trainer2.compute_gradient_hash(gradients2)
        
        assert hash1 == hash2
    
    def test_different_precision_hashes(self):
        """Test that different precisions produce different hashes."""
        model = SimpleBitLinearModel(in_features=64, out_features=32)
        trainer = LoRATrainer(model)
        
        inputs = torch.randn(8, 64)
        targets = torch.randn(8, 32)
        
        _, gradients = trainer.train_step(inputs, targets)
        
        hash_int8 = trainer.compute_gradient_hash(gradients, precision="int8")
        hash_float32 = trainer.compute_gradient_hash(gradients, precision="float32")
        
        # Different precisions should produce different hashes
        assert hash_int8 != hash_float32


class TestDeterministicExecution:
    """Tests for deterministic execution."""
    
    def test_seed_setting(self):
        """Test seed setting for reproducibility."""
        model = SimpleBitLinearModel()
        trainer = LoRATrainer(model, deterministic=True)
        
        trainer.set_seed(42)
        
        # Generate random tensor
        t1 = torch.randn(10)
        
        # Reset seed and generate again
        trainer.set_seed(42)
        t2 = torch.randn(10)
        
        assert torch.allclose(t1, t2)


class TestStateDict:
    """Tests for state dict operations."""
    
    def test_get_lora_state_dict(self):
        """Test getting LoRA state dict."""
        model = SimpleBitLinearModel()
        trainer = LoRATrainer(model)
        
        state_dict = trainer.get_lora_state_dict()
        
        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0
        
        # Check that keys contain lora_A and lora_B
        has_lora_a = any('lora_A' in key for key in state_dict.keys())
        has_lora_b = any('lora_B' in key for key in state_dict.keys())
        
        assert has_lora_a
        assert has_lora_b


class TestMetadata:
    """Tests for training metadata."""
    
    def test_get_training_metadata(self):
        """Test getting training metadata."""
        model = SimpleBitLinearModel()
        trainer = LoRATrainer(model, learning_rate=1e-4)
        
        # Perform a training step first
        inputs = torch.randn(8, 64)
        targets = torch.randn(8, 32)
        trainer.train_step(inputs, targets)
        
        metadata = trainer.get_training_metadata()
        
        assert isinstance(metadata, dict)
        assert 'training_step' in metadata
        assert 'learning_rate' in metadata
        assert 'loss' in metadata
        assert 'lora_size_mb' in metadata
        
        assert metadata['training_step'] == 1
        assert metadata['learning_rate'] == 1e-4
    
    def test_estimate_lora_size(self):
        """Test LoRA size estimation."""
        model = SimpleBitLinearModel(in_features=64, out_features=32, lora_rank=4)
        trainer = LoRATrainer(model)
        
        size_mb = trainer.estimate_lora_size_mb()
        
        assert isinstance(size_mb, float)
        assert size_mb > 0


class TestEnvironmentVariables:
    """Tests for environment variable configuration."""
    
    def test_default_weight_decay(self):
        """Test default weight decay from environment."""
        assert DEFAULT_WEIGHT_DECAY == 0.01
    
    def test_default_grad_clip(self):
        """Test default gradient clipping from environment."""
        assert DEFAULT_GRAD_CLIP_MAX_NORM == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

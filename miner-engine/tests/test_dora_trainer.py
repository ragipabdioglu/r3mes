"""
Tests for DoRATrainer (FAZ 4)

Tests the DoRA training functionality for MinerEngine.
"""

import pytest
import torch
import torch.nn as nn
import os
import sys

# Add miner-engine to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.bitlinear import BitLinear
from core.dora import BitLinearDoRA, DoRAAdapter, create_dora_from_bitlinear
from core.dora_trainer import DoRATrainer


class SimpleDoRAModel(nn.Module):
    """Simple model with DoRA layers for testing."""
    
    def __init__(self, hidden_size: int = 64, num_layers: int = 2, dora_rank: int = 8):
        super().__init__()
        
        # Create BitLinear layers
        bitlinear_layers = [
            BitLinear(hidden_size, hidden_size, lora_rank=dora_rank)
            for _ in range(num_layers)
        ]
        
        # Wrap with DoRA
        self.layers = nn.ModuleList([
            create_dora_from_bitlinear(bl, rank=dora_rank)
            for bl in bitlinear_layers
        ])
        
        # Output layer
        output_bitlinear = BitLinear(hidden_size, hidden_size, lora_rank=dora_rank)
        self.output = create_dora_from_bitlinear(output_bitlinear, rank=dora_rank)
    
    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output(x)


class TestDoRATrainerInitialization:
    """Tests for DoRATrainer initialization."""
    
    def test_trainer_initialization(self):
        """Test basic trainer initialization."""
        model = SimpleDoRAModel(hidden_size=64, num_layers=2, dora_rank=8)
        trainer = DoRATrainer(model, learning_rate=1e-4)
        
        assert trainer.model is not None
        assert trainer.learning_rate == 1e-4
        assert trainer.training_step == 0
        assert len(trainer.loss_history) == 0
    
    def test_backbone_frozen(self):
        """Test that backbone parameters are frozen."""
        model = SimpleDoRAModel(hidden_size=64, num_layers=2, dora_rank=8)
        trainer = DoRATrainer(model, learning_rate=1e-4)
        
        for module in trainer.model.modules():
            if isinstance(module, BitLinearDoRA):
                # Backbone should be frozen
                for param in module.backbone.parameters():
                    assert not param.requires_grad
                # DoRA params should be trainable
                assert module.magnitude.requires_grad
                assert module.direction_A.requires_grad
                assert module.direction_B.requires_grad
    
    def test_parameter_groups(self):
        """Test that parameter groups are correctly separated."""
        model = SimpleDoRAModel(hidden_size=64, num_layers=2, dora_rank=8)
        trainer = DoRATrainer(model, learning_rate=1e-4, magnitude_lr_scale=0.1)
        
        # Should have direction and magnitude params
        assert len(trainer.direction_params) > 0
        assert len(trainer.magnitude_params) > 0
        
        # Direction params should be A and B matrices
        for param in trainer.direction_params:
            assert param.requires_grad
        
        # Magnitude params should be magnitude vectors
        for param in trainer.magnitude_params:
            assert param.requires_grad


class TestDoRATrainerTraining:
    """Tests for DoRATrainer training functionality."""
    
    def test_train_step(self):
        """Test single training step."""
        model = SimpleDoRAModel(hidden_size=64, num_layers=2, dora_rank=8)
        trainer = DoRATrainer(model, learning_rate=1e-4)
        
        # Create dummy data
        inputs = torch.randn(4, 64)
        targets = torch.randn(4, 64)
        
        # Train step
        loss, gradients = trainer.train_step(inputs, targets)
        
        assert isinstance(loss, float)
        assert loss > 0
        assert len(gradients) > 0
        assert trainer.training_step == 1
        assert len(trainer.loss_history) == 1
    
    def test_multiple_train_steps(self):
        """Test multiple training steps."""
        model = SimpleDoRAModel(hidden_size=64, num_layers=2, dora_rank=8)
        trainer = DoRATrainer(model, learning_rate=1e-4)
        
        inputs = torch.randn(4, 64)
        targets = torch.randn(4, 64)
        
        losses = []
        for _ in range(5):
            loss, _ = trainer.train_step(inputs, targets)
            losses.append(loss)
        
        assert trainer.training_step == 5
        assert len(trainer.loss_history) == 5
    
    def test_gradients_collected(self):
        """Test that gradients are correctly collected."""
        model = SimpleDoRAModel(hidden_size=64, num_layers=2, dora_rank=8)
        trainer = DoRATrainer(model, learning_rate=1e-4)
        
        inputs = torch.randn(4, 64)
        targets = torch.randn(4, 64)
        
        _, gradients = trainer.train_step(inputs, targets)
        
        # Should have gradients for magnitude, direction_A, direction_B
        has_magnitude = any('magnitude' in k for k in gradients.keys())
        has_direction_a = any('direction_A' in k for k in gradients.keys())
        has_direction_b = any('direction_B' in k for k in gradients.keys())
        
        assert has_magnitude
        assert has_direction_a
        assert has_direction_b
    
    def test_deterministic_training(self):
        """Test deterministic training with seed."""
        model1 = SimpleDoRAModel(hidden_size=64, num_layers=2, dora_rank=8)
        model2 = SimpleDoRAModel(hidden_size=64, num_layers=2, dora_rank=8)
        
        # Copy weights
        model2.load_state_dict(model1.state_dict())
        
        trainer1 = DoRATrainer(model1, learning_rate=1e-4, deterministic=True)
        trainer2 = DoRATrainer(model2, learning_rate=1e-4, deterministic=True)
        
        trainer1.set_seed(42)
        trainer2.set_seed(42)
        
        inputs = torch.randn(4, 64)
        targets = torch.randn(4, 64)
        
        # Reset seeds before training
        torch.manual_seed(42)
        loss1, grads1 = trainer1.train_step(inputs.clone(), targets.clone())
        
        torch.manual_seed(42)
        loss2, grads2 = trainer2.train_step(inputs.clone(), targets.clone())
        
        # Losses should be very close (may not be exactly equal due to floating point)
        assert abs(loss1 - loss2) < 1e-5


class TestDoRATrainerSerialization:
    """Tests for DoRATrainer serialization."""
    
    def test_get_state_dict(self):
        """Test getting DoRA state dict."""
        model = SimpleDoRAModel(hidden_size=64, num_layers=2, dora_rank=8)
        trainer = DoRATrainer(model, learning_rate=1e-4)
        
        state_dict = trainer.get_dora_state_dict()
        
        assert len(state_dict) > 0
        # Should have magnitude, direction_A, direction_B for each layer
        has_magnitude = any('magnitude' in k for k in state_dict.keys())
        has_direction_a = any('direction_A' in k for k in state_dict.keys())
        has_direction_b = any('direction_B' in k for k in state_dict.keys())
        
        assert has_magnitude
        assert has_direction_a
        assert has_direction_b
    
    def test_load_state_dict(self):
        """Test loading DoRA state dict."""
        model1 = SimpleDoRAModel(hidden_size=64, num_layers=2, dora_rank=8)
        model2 = SimpleDoRAModel(hidden_size=64, num_layers=2, dora_rank=8)
        
        trainer1 = DoRATrainer(model1, learning_rate=1e-4)
        trainer2 = DoRATrainer(model2, learning_rate=1e-4)
        
        # Train model1
        inputs = torch.randn(4, 64)
        targets = torch.randn(4, 64)
        trainer1.train_step(inputs, targets)
        
        # Get state dict and load into model2
        state_dict = trainer1.get_dora_state_dict()
        trainer2.load_dora_state_dict(state_dict)
        
        # Verify parameters match
        for name, module1 in trainer1.model.named_modules():
            if isinstance(module1, BitLinearDoRA):
                for name2, module2 in trainer2.model.named_modules():
                    if name == name2 and isinstance(module2, BitLinearDoRA):
                        assert torch.allclose(module1.magnitude, module2.magnitude)
                        assert torch.allclose(module1.direction_A, module2.direction_A)
                        assert torch.allclose(module1.direction_B, module2.direction_B)
    
    def test_create_adapter(self):
        """Test creating DoRAAdapter from trainer."""
        model = SimpleDoRAModel(hidden_size=64, num_layers=2, dora_rank=8)
        trainer = DoRATrainer(model, learning_rate=1e-4)
        
        # Train a bit
        inputs = torch.randn(4, 64)
        targets = torch.randn(4, 64)
        trainer.train_step(inputs, targets)
        
        # Create adapter
        adapter = trainer.create_adapter("test_adapter", "test_domain")
        
        assert adapter.adapter_id == "test_adapter"
        assert adapter.domain == "test_domain"
        assert adapter.rank == 8
        assert len(adapter.params) > 0
        assert 'magnitude' in adapter.params
        assert 'direction_A' in adapter.params
        assert 'direction_B' in adapter.params


class TestDoRATrainerMetrics:
    """Tests for DoRATrainer metrics."""
    
    def test_estimate_size(self):
        """Test DoRA size estimation."""
        model = SimpleDoRAModel(hidden_size=64, num_layers=2, dora_rank=8)
        trainer = DoRATrainer(model, learning_rate=1e-4)
        
        size_mb = trainer.estimate_dora_size_mb()
        
        assert size_mb > 0
        assert size_mb < 10  # Should be small for test model
    
    def test_training_metadata(self):
        """Test training metadata."""
        model = SimpleDoRAModel(hidden_size=64, num_layers=2, dora_rank=8)
        trainer = DoRATrainer(model, learning_rate=1e-4)
        
        # Train a bit
        inputs = torch.randn(4, 64)
        targets = torch.randn(4, 64)
        trainer.train_step(inputs, targets)
        
        metadata = trainer.get_training_metadata()
        
        assert 'training_step' in metadata
        assert 'learning_rate' in metadata
        assert 'loss' in metadata
        assert 'dora_size_mb' in metadata
        assert 'adapter_type' in metadata
        assert metadata['adapter_type'] == 'dora'
        assert metadata['training_step'] == 1
    
    def test_gradient_hash(self):
        """Test gradient hash computation."""
        model = SimpleDoRAModel(hidden_size=64, num_layers=2, dora_rank=8)
        trainer = DoRATrainer(model, learning_rate=1e-4)
        
        inputs = torch.randn(4, 64)
        targets = torch.randn(4, 64)
        
        _, gradients = trainer.train_step(inputs, targets)
        
        # Compute hash
        try:
            hash_value = trainer.compute_gradient_hash(gradients)
            assert isinstance(hash_value, str)
            assert len(hash_value) == 64  # SHA256 hex
        except ImportError:
            # DeterministicHashVerifier may not be available
            pytest.skip("DeterministicHashVerifier not available")


class TestModelLoaderDoRA:
    """Tests for DoRA model loader."""
    
    def test_load_model_with_enforced_dora(self):
        """Test loading model with enforced DoRA."""
        from r3mes.miner.model_loader import load_model_with_enforced_dora
        
        # Create base model with BitLinear
        class BaseModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = BitLinear(64, 64, lora_rank=8)
                self.layer2 = BitLinear(64, 64, lora_rank=8)
            
            def forward(self, x):
                x = torch.relu(self.layer1(x))
                return self.layer2(x)
        
        base_model = BaseModel()
        dora_model = load_model_with_enforced_dora(base_model, dora_rank=8)
        
        # Check that layers are converted to DoRA
        dora_count = 0
        for module in dora_model.modules():
            if isinstance(module, BitLinearDoRA):
                dora_count += 1
        
        assert dora_count == 2
    
    def test_validate_dora_only_training(self):
        """Test DoRA-only training validation."""
        from r3mes.miner.model_loader import validate_dora_only_training
        
        model = SimpleDoRAModel(hidden_size=64, num_layers=2, dora_rank=8)
        
        # Should pass validation
        result = validate_dora_only_training(model)
        assert result is True
    
    def test_backward_compatibility_lora_function(self):
        """Test backward compatibility with load_model_with_enforced_lora."""
        from r3mes.miner.model_loader import load_model_with_enforced_lora
        
        class BaseModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = BitLinear(64, 64, lora_rank=8)
            
            def forward(self, x):
                return self.layer(x)
        
        base_model = BaseModel()
        
        # Should work but use DoRA internally
        model = load_model_with_enforced_lora(base_model, lora_rank=8)
        
        # Should have DoRA layers
        has_dora = any(isinstance(m, BitLinearDoRA) for m in model.modules())
        assert has_dora


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

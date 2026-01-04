"""
Tests for DoRA (Weight-Decomposed Low-Rank Adaptation) layer.
"""

import pytest
import torch
import tempfile
import os

from core.bitlinear import BitLinear
from core.dora import (
    BitLinearDoRA,
    DoRAAdapter,
    DoRAExpertRegistry,
    create_dora_from_bitlinear,
    merge_dora_experts,
)


class TestBitLinearDoRA:
    """Tests for BitLinearDoRA layer."""
    
    def test_initialization(self):
        """Test DoRA layer initialization."""
        backbone = BitLinear(in_features=256, out_features=128, lora_rank=8)
        dora = BitLinearDoRA(backbone, rank=16, alpha=1.0)
        
        assert dora.in_features == 256
        assert dora.out_features == 128
        assert dora.rank == 16
        assert dora.magnitude.shape == (128,)
        assert dora.direction_A.shape == (16, 256)
        assert dora.direction_B.shape == (128, 16)
    
    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        backbone = BitLinear(in_features=256, out_features=128)
        dora = BitLinearDoRA(backbone, rank=16)
        
        x = torch.randn(4, 256)  # batch_size=4
        output = dora(x)
        
        assert output.shape == (4, 128)
    
    def test_forward_pass_3d(self):
        """Test forward pass with 3D input (sequence)."""
        backbone = BitLinear(in_features=256, out_features=128)
        dora = BitLinearDoRA(backbone, rank=16)
        
        x = torch.randn(4, 32, 256)  # batch=4, seq=32, features=256
        output = dora(x)
        
        assert output.shape == (4, 32, 128)
    
    def test_backbone_frozen(self):
        """Test that backbone parameters are frozen."""
        backbone = BitLinear(in_features=256, out_features=128)
        dora = BitLinearDoRA(backbone, rank=16)
        
        # Backbone should be frozen
        assert not dora.backbone.backbone_weight.requires_grad
        
        # DoRA params should be trainable
        assert dora.magnitude.requires_grad
        assert dora.direction_A.requires_grad
        assert dora.direction_B.requires_grad
    
    def test_trainable_params_count(self):
        """Test trainable parameter counting."""
        backbone = BitLinear(in_features=256, out_features=128)
        dora = BitLinearDoRA(backbone, rank=16)
        
        expected = 128 + (16 * 256) + (128 * 16)  # magnitude + A + B
        assert dora.num_trainable_params() == expected
    
    def test_get_set_params(self):
        """Test parameter get/set."""
        backbone = BitLinear(in_features=256, out_features=128)
        dora = BitLinearDoRA(backbone, rank=16)
        
        # Get params
        params = dora.get_trainable_params()
        assert 'magnitude' in params
        assert 'direction_A' in params
        assert 'direction_B' in params
        
        # Modify and set
        params['magnitude'] = torch.ones(128) * 2.0
        dora.set_trainable_params(params)
        
        assert torch.allclose(dora.magnitude, torch.ones(128) * 2.0)
    
    def test_adapter_hash(self):
        """Test adapter hash computation."""
        backbone = BitLinear(in_features=256, out_features=128)
        dora = BitLinearDoRA(backbone, rank=16)
        
        hash1 = dora.get_adapter_hash()
        assert len(hash1) == 64  # SHA256 hex
        
        # Same params should give same hash
        hash2 = dora.get_adapter_hash()
        assert hash1 == hash2
        
        # Different params should give different hash
        dora.magnitude.data = torch.randn(128)
        hash3 = dora.get_adapter_hash()
        assert hash1 != hash3
    
    def test_size_estimation(self):
        """Test size estimation."""
        backbone = BitLinear(in_features=256, out_features=128)
        dora = BitLinearDoRA(backbone, rank=16)
        
        size_mb = dora.estimate_size_mb()
        assert size_mb > 0
        
        # Expected: (128 + 16*256 + 128*16) * 4 bytes / 1MB
        expected = (128 + 4096 + 2048) * 4 / (1024 * 1024)
        assert abs(size_mb - expected) < 0.001


class TestDoRAAdapter:
    """Tests for DoRAAdapter container."""
    
    def test_initialization(self):
        """Test adapter initialization."""
        adapter = DoRAAdapter(
            adapter_id='test_dora',
            domain='test',
            rank=16,
            alpha=1.0,
            in_features=256,
            out_features=128,
        )
        
        assert adapter.adapter_id == 'test_dora'
        assert adapter.domain == 'test'
        assert adapter.rank == 16
    
    def test_serialization(self):
        """Test adapter serialization to dict."""
        params = {
            'magnitude': torch.ones(128),
            'direction_A': torch.randn(16, 256),
            'direction_B': torch.randn(128, 16),
        }
        
        adapter = DoRAAdapter(
            adapter_id='test_dora',
            domain='test',
            rank=16,
            alpha=1.0,
            in_features=256,
            out_features=128,
            params=params,
        )
        
        data = adapter.to_dict()
        assert data['adapter_id'] == 'test_dora'
        assert 'params' in data
        assert 'magnitude' in data['params']
    
    def test_deserialization(self):
        """Test adapter deserialization from dict."""
        params = {
            'magnitude': torch.ones(128),
            'direction_A': torch.randn(16, 256),
            'direction_B': torch.randn(128, 16),
        }
        
        original = DoRAAdapter(
            adapter_id='test_dora',
            domain='test',
            rank=16,
            alpha=1.0,
            in_features=256,
            out_features=128,
            params=params,
        )
        
        data = original.to_dict()
        restored = DoRAAdapter.from_dict(data)
        
        assert restored.adapter_id == original.adapter_id
        assert restored.rank == original.rank
    
    def test_save_load(self):
        """Test adapter save/load to file."""
        params = {
            'magnitude': torch.ones(128),
            'direction_A': torch.randn(16, 256),
            'direction_B': torch.randn(128, 16),
        }
        
        adapter = DoRAAdapter(
            adapter_id='test_dora',
            domain='test',
            rank=16,
            alpha=1.0,
            in_features=256,
            out_features=128,
            params=params,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_adapter.pt')
            adapter.save(path)
            
            loaded = DoRAAdapter.load(path)
            assert loaded.adapter_id == adapter.adapter_id
            assert loaded.rank == adapter.rank
    
    def test_apply_to_layer(self):
        """Test applying adapter to DoRA layer."""
        backbone = BitLinear(in_features=256, out_features=128)
        dora = BitLinearDoRA(backbone, rank=16)
        
        # Create adapter with specific params
        params = {
            'magnitude': torch.ones(128) * 2.0,
            'direction_A': torch.randn(16, 256),
            'direction_B': torch.randn(128, 16),
        }
        
        adapter = DoRAAdapter(
            adapter_id='test_dora',
            domain='test',
            rank=16,
            alpha=1.0,
            in_features=256,
            out_features=128,
            params=params,
        )
        
        adapter.apply_to_layer(dora)
        
        assert torch.allclose(dora.magnitude, params['magnitude'])


class TestDoRAExpertRegistry:
    """Tests for DoRAExpertRegistry."""
    
    def test_initialization(self):
        """Test registry initialization."""
        registry = DoRAExpertRegistry()
        
        assert 'medical_dora' in registry.DOMAIN_EXPERTS
        assert 'turkish_dora' in registry.LANGUAGE_EXPERTS
        assert 'summarization_dora' in registry.TASK_EXPERTS
    
    def test_register_and_get(self):
        """Test expert registration and retrieval."""
        registry = DoRAExpertRegistry()
        
        adapter = DoRAAdapter(
            adapter_id='medical_dora',
            domain='medical',
            rank=16,
            alpha=1.0,
            in_features=256,
            out_features=128,
        )
        
        registry.register(adapter)
        
        retrieved = registry.get('medical_dora')
        assert retrieved is not None
        assert retrieved.adapter_id == 'medical_dora'
    
    def test_list_experts(self):
        """Test listing experts."""
        registry = DoRAExpertRegistry()
        
        adapter1 = DoRAAdapter('medical_dora', 'medical', 16, 1.0, 256, 128)
        adapter2 = DoRAAdapter('turkish_dora', 'language', 16, 1.0, 256, 128)
        
        registry.register(adapter1)
        registry.register(adapter2)
        
        all_experts = registry.list_experts()
        assert 'medical_dora' in all_experts
        assert 'turkish_dora' in all_experts
    
    def test_get_domain(self):
        """Test domain lookup."""
        registry = DoRAExpertRegistry()
        
        assert registry.get_domain('medical_dora') == 'medical'
        assert registry.get_domain('turkish_dora') == 'language'
        assert registry.get_domain('summarization_dora') == 'task'
        assert registry.get_domain('general_dora') == 'general'


class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_create_dora_from_bitlinear(self):
        """Test DoRA creation from BitLinear."""
        backbone = BitLinear(in_features=256, out_features=128)
        dora = create_dora_from_bitlinear(backbone, rank=16, alpha=1.0)
        
        assert isinstance(dora, BitLinearDoRA)
        assert dora.rank == 16
        assert dora.alpha == 1.0
    
    def test_merge_dora_experts(self):
        """Test merging multiple DoRA experts."""
        backbone = BitLinear(in_features=256, out_features=128)
        base_layer = BitLinearDoRA(backbone, rank=16)
        
        # Create two experts
        expert1 = DoRAAdapter(
            adapter_id='expert1',
            domain='test',
            rank=16,
            alpha=1.0,
            in_features=256,
            out_features=128,
            params={
                'magnitude': torch.ones(128),
                'direction_A': torch.randn(16, 256),
                'direction_B': torch.randn(128, 16),
            }
        )
        
        expert2 = DoRAAdapter(
            adapter_id='expert2',
            domain='test',
            rank=16,
            alpha=1.0,
            in_features=256,
            out_features=128,
            params={
                'magnitude': torch.ones(128) * 2,
                'direction_A': torch.randn(16, 256),
                'direction_B': torch.randn(128, 16),
            }
        )
        
        merged_direction, merged_magnitude = merge_dora_experts(
            experts=[expert1, expert2],
            weights=[0.6, 0.4],
            base_layer=base_layer,
        )
        
        assert merged_direction.shape == (128, 256)
        assert merged_magnitude.shape == (128,)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

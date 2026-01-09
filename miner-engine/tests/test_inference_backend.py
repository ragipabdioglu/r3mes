"""
Tests for Inference Backend Abstraction Layer.
"""

import pytest
import torch

from core.inference_backend import (
    BackendType,
    BackendCapabilities,
    InferenceResult,
    BackendRegistry,
    get_backend_registry,
    get_best_backend,
)
from core.backends import PyTorchBackend


class TestBackendCapabilities:
    """Tests for BackendCapabilities."""
    
    def test_to_dict(self):
        """Test capabilities serialization."""
        caps = BackendCapabilities(
            name="Test Backend",
            backend_type=BackendType.PYTORCH,
            supports_gpu=True,
            supports_quantized=True,
            supports_batching=True,
            max_batch_size=64,
            estimated_speedup=1.0,
            vram_efficiency=0.6,
            precision="fp32",
        )
        
        data = caps.to_dict()
        assert data['name'] == "Test Backend"
        assert data['backend_type'] == "pytorch"
        assert data['supports_gpu'] is True


class TestInferenceResult:
    """Tests for InferenceResult."""
    
    def test_creation(self):
        """Test result creation."""
        output = torch.randn(4, 128)
        result = InferenceResult(
            output=output,
            latency_ms=10.5,
            backend_used="pytorch",
            metadata={'test': True},
        )
        
        assert result.output.shape == (4, 128)
        assert result.latency_ms == 10.5
        assert result.backend_used == "pytorch"
        assert result.metadata['test'] is True


class TestBackendRegistry:
    """Tests for BackendRegistry."""
    
    def test_register_and_get(self):
        """Test backend registration."""
        registry = BackendRegistry()
        registry.register(BackendType.PYTORCH, PyTorchBackend)
        
        backend = registry.get(BackendType.PYTORCH)
        assert backend is not None
        assert isinstance(backend, PyTorchBackend)
    
    def test_list_available(self):
        """Test listing available backends."""
        registry = BackendRegistry()
        registry.register(BackendType.PYTORCH, PyTorchBackend)
        
        available = registry.list_available()
        assert BackendType.PYTORCH in available
    
    def test_get_best_available(self):
        """Test getting best available backend."""
        registry = BackendRegistry()
        registry.register(BackendType.PYTORCH, PyTorchBackend)
        
        backend = registry.get_best_available()
        assert backend is not None


class TestPyTorchBackend:
    """Tests for PyTorchBackend."""
    
    def test_initialization(self):
        """Test backend initialization."""
        backend = PyTorchBackend()
        assert backend.device in ['cuda', 'cpu', 'mps']
    
    def test_is_available(self):
        """Test availability check."""
        backend = PyTorchBackend()
        assert backend.is_available() is True
    
    def test_get_capabilities(self):
        """Test capabilities retrieval."""
        backend = PyTorchBackend()
        caps = backend.get_capabilities()
        
        assert caps.backend_type == BackendType.PYTORCH
        assert caps.supports_batching is True
        assert caps.estimated_speedup == 1.0
    
    def test_load_model(self):
        """Test model loading."""
        backend = PyTorchBackend()
        success = backend.load_model("placeholder")
        
        assert success is True
        assert backend.model is not None
    
    def test_inference_without_adapters(self):
        """Test inference without adapters."""
        backend = PyTorchBackend(device='cpu')
        backend.load_model("placeholder")
        
        input_ids = torch.randint(0, 1000, (2, 16))
        result = backend.inference(
            input_ids=input_ids,
            adapter_ids=[],
            adapter_weights=[],
        )
        
        assert result.output is not None
        assert result.latency_ms > 0
        assert result.backend_used == "pytorch"
    
    def test_get_vram_usage(self):
        """Test VRAM usage reporting."""
        backend = PyTorchBackend()
        usage = backend.get_vram_usage()
        
        assert 'used_mb' in usage
        assert 'total_mb' in usage
        assert 'free_mb' in usage
    
    def test_supports_feature(self):
        """Test feature support check."""
        backend = PyTorchBackend()
        
        assert backend.supports_feature('batching') is True
        assert backend.supports_feature('multi_adapter') is True
    
    def test_warmup(self):
        """Test model warmup."""
        backend = PyTorchBackend(device='cpu')
        backend.load_model("placeholder")
        
        # Should not raise
        backend.warmup(batch_size=1, seq_len=16)


class TestGlobalFunctions:
    """Tests for global helper functions."""
    
    def test_get_backend_registry(self):
        """Test getting global registry."""
        registry = get_backend_registry()
        assert registry is not None
        assert isinstance(registry, BackendRegistry)
    
    def test_get_best_backend(self):
        """Test getting best backend."""
        backend = get_best_backend()
        # Should return PyTorch as it's always available
        assert backend is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

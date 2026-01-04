"""
Inference Backend Abstraction Layer

Provides a unified interface for different inference backends:
- Phase 1: PyTorch Backend (current)
- Phase 2: Triton Backend (future - custom kernels)
- Phase 3: BitNet-cpp Backend (future - native 1.58-bit)

This abstraction allows seamless switching between backends
based on hardware capabilities and availability.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """Available inference backend types."""
    PYTORCH = "pytorch"
    TRITON = "triton"
    BITNET_CPP = "bitnet_cpp"


@dataclass
class BackendCapabilities:
    """Describes backend capabilities and constraints."""
    name: str
    backend_type: BackendType
    supports_gpu: bool
    supports_quantized: bool
    supports_batching: bool
    max_batch_size: int
    estimated_speedup: float  # Relative to PyTorch baseline
    vram_efficiency: float  # 0.0-1.0, higher is better
    precision: str  # "fp32", "fp16", "int8", "1.58bit"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'backend_type': self.backend_type.value,
            'supports_gpu': self.supports_gpu,
            'supports_quantized': self.supports_quantized,
            'supports_batching': self.supports_batching,
            'max_batch_size': self.max_batch_size,
            'estimated_speedup': self.estimated_speedup,
            'vram_efficiency': self.vram_efficiency,
            'precision': self.precision,
        }


@dataclass
class InferenceResult:
    """Result from inference operation."""
    output: torch.Tensor
    latency_ms: float
    backend_used: str
    metadata: Optional[Dict[str, Any]] = None


class InferenceBackend(ABC):
    """
    Abstract base class for inference backends.
    
    All backends must implement:
    - load_model: Load base model from path/IPFS
    - load_adapter: Load DoRA adapter
    - inference: Run inference with adapters
    - get_capabilities: Return backend capabilities
    """
    
    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """
        Load base model from path or IPFS hash.
        
        Args:
            model_path: Local path or IPFS hash
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def load_adapter(self, adapter_id: str, adapter_path: str) -> bool:
        """
        Load DoRA adapter.
        
        Args:
            adapter_id: Unique adapter identifier
            adapter_path: Path to adapter file
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def unload_adapter(self, adapter_id: str) -> bool:
        """
        Unload DoRA adapter from memory.
        
        Args:
            adapter_id: Adapter to unload
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def inference(
        self,
        input_ids: torch.Tensor,
        adapter_ids: List[str],
        adapter_weights: List[float],
        **kwargs
    ) -> InferenceResult:
        """
        Run inference with specified adapters.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            adapter_ids: List of adapter IDs to use
            adapter_weights: Weights for each adapter (sum to 1.0)
            **kwargs: Additional backend-specific options
            
        Returns:
            InferenceResult with output tensor and metadata
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> BackendCapabilities:
        """
        Get backend capabilities.
        
        Returns:
            BackendCapabilities describing this backend
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if backend is available on current system.
        
        Returns:
            True if backend can be used
        """
        pass
    
    @abstractmethod
    def get_vram_usage(self) -> Dict[str, float]:
        """
        Get current VRAM usage.
        
        Returns:
            Dict with 'used_mb', 'total_mb', 'free_mb'
        """
        pass
    
    def supports_feature(self, feature: str) -> bool:
        """
        Check if backend supports a specific feature.
        
        Args:
            feature: Feature name (e.g., "multi_adapter", "streaming")
            
        Returns:
            True if supported
        """
        caps = self.get_capabilities()
        feature_map = {
            'gpu': caps.supports_gpu,
            'quantized': caps.supports_quantized,
            'batching': caps.supports_batching,
            'multi_adapter': True,  # All backends support this
        }
        return feature_map.get(feature, False)


class BackendRegistry:
    """
    Registry for managing available backends.
    
    Provides:
    - Backend registration
    - Auto-selection based on capabilities
    - Fallback handling
    """
    
    def __init__(self):
        self._backends: Dict[BackendType, type] = {}
        self._instances: Dict[BackendType, InferenceBackend] = {}
        self._default: Optional[BackendType] = None
    
    def register(self, backend_type: BackendType, backend_class: type):
        """Register a backend class."""
        self._backends[backend_type] = backend_class
        logger.info(f"Registered backend: {backend_type.value}")
    
    def get(self, backend_type: BackendType) -> Optional[InferenceBackend]:
        """Get or create backend instance."""
        if backend_type not in self._instances:
            if backend_type not in self._backends:
                logger.warning(f"Backend not registered: {backend_type.value}")
                return None
            
            backend_class = self._backends[backend_type]
            instance = backend_class()
            
            if not instance.is_available():
                logger.warning(f"Backend not available: {backend_type.value}")
                return None
            
            self._instances[backend_type] = instance
        
        return self._instances[backend_type]
    
    def get_best_available(self) -> Optional[InferenceBackend]:
        """
        Get the best available backend based on capabilities.
        
        Priority: BitNet-cpp > Triton > PyTorch
        """
        priority = [
            BackendType.BITNET_CPP,
            BackendType.TRITON,
            BackendType.PYTORCH,
        ]
        
        for backend_type in priority:
            backend = self.get(backend_type)
            if backend is not None:
                logger.info(f"Selected backend: {backend_type.value}")
                return backend
        
        logger.error("No available backend found!")
        return None
    
    def list_available(self) -> List[BackendType]:
        """List all available backends."""
        available = []
        for backend_type in self._backends:
            backend = self.get(backend_type)
            if backend is not None:
                available.append(backend_type)
        return available
    
    def set_default(self, backend_type: BackendType):
        """Set default backend."""
        self._default = backend_type
    
    def get_default(self) -> Optional[InferenceBackend]:
        """Get default backend or best available."""
        if self._default:
            backend = self.get(self._default)
            if backend:
                return backend
        return self.get_best_available()


# Global registry instance
_registry = BackendRegistry()


def get_backend_registry() -> BackendRegistry:
    """Get global backend registry."""
    return _registry


def get_best_backend() -> Optional[InferenceBackend]:
    """Get best available backend."""
    return _registry.get_best_available()


def register_backend(backend_type: BackendType, backend_class: type):
    """Register a backend class."""
    _registry.register(backend_type, backend_class)

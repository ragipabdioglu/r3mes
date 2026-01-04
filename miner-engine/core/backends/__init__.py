"""
Inference Backend Implementations

Available backends:
- PyTorchBackend: Default backend using PyTorch (Phase 1)
- TritonBackend: Custom Triton kernels (Phase 2 - planned)
- BitNetCppBackend: Native 1.58-bit inference (Phase 3 - planned)
"""

from .pytorch_backend import PyTorchBackend
from .triton_backend import TritonBackend
from .bitnet_cpp_backend import BitNetCppBackend

__all__ = [
    'PyTorchBackend',
    'TritonBackend',
    'BitNetCppBackend',
]

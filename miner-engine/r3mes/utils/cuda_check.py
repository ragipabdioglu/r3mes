"""
CUDA Availability and PyTorch CUDA Support Check

Provides functions to check CUDA availability and PyTorch CUDA support.
"""

import subprocess
import sys
from typing import Tuple, Optional


def check_nvidia_smi() -> Tuple[bool, Optional[str]]:
    """
    Check if nvidia-smi is available.
    
    Returns:
        Tuple of (is_available, version_string)
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Extract version from output
            version_line = result.stdout.split('\n')[0]
            return True, version_line
        return False, None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, None


def check_cuda_availability() -> Tuple[bool, Optional[str]]:
    """
    Check if CUDA is available on the system.
    
    Returns:
        Tuple of (is_available, cuda_version)
    """
    # First check nvidia-smi
    nvidia_available, nvidia_version = check_nvidia_smi()
    if not nvidia_available:
        return False, None
    
    # Try to detect CUDA version from nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Look for CUDA Version in output
            for line in result.stdout.split('\n'):
                if 'CUDA Version' in line:
                    # Extract version number
                    parts = line.split('CUDA Version:')
                    if len(parts) > 1:
                        version = parts[1].strip().split()[0]
                        return True, version
    except Exception:
        pass
    
    return True, None  # NVIDIA driver available but CUDA version unknown


def check_pytorch_cuda() -> Tuple[bool, Optional[str]]:
    """
    Check if PyTorch has CUDA support.
    
    Returns:
        Tuple of (is_available, cuda_version)
    """
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            return True, cuda_version
        return False, None
    except ImportError:
        return False, None


def check_cuda_compatibility() -> dict:
    """
    Comprehensive CUDA compatibility check.
    
    Returns:
        Dictionary with compatibility information
    """
    result = {
        'nvidia_driver_available': False,
        'nvidia_driver_version': None,
        'cuda_available': False,
        'cuda_version': None,
        'pytorch_cuda_available': False,
        'pytorch_cuda_version': None,
        'compatible': False,
    }
    
    # Check NVIDIA driver
    nvidia_available, nvidia_version = check_nvidia_smi()
    result['nvidia_driver_available'] = nvidia_available
    result['nvidia_driver_version'] = nvidia_version
    
    # Check CUDA
    cuda_available, cuda_version = check_cuda_availability()
    result['cuda_available'] = cuda_available
    result['cuda_version'] = cuda_version
    
    # Check PyTorch CUDA
    pytorch_cuda_available, pytorch_cuda_version = check_pytorch_cuda()
    result['pytorch_cuda_available'] = pytorch_cuda_available
    result['pytorch_cuda_version'] = pytorch_cuda_version
    
    # Overall compatibility
    result['compatible'] = (
        nvidia_available and
        cuda_available and
        pytorch_cuda_available
    )
    
    return result


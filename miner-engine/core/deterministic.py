"""
Deterministic execution configuration for R3MES miner engine.

Ensures bit-exact reproducibility across different runs and environments.
"""

import os
import random
from typing import Optional
import numpy as np
import torch


def configure_deterministic_execution(global_seed: Optional[int] = None):
    """
    Configure PyTorch and NumPy for deterministic execution.
    
    This function must be called before any model initialization or training.
    
    Args:
        global_seed: Global seed from blockchain (derived from block hash + training round).
                     If None, uses fixed seed 42 for testing.
    """
    # Use global seed from blockchain if provided, otherwise use fixed seed for testing
    seed = global_seed if global_seed is not None else 42
    
    # Set Python hash seed for deterministic hashing
    os.environ['PYTHONHASHSEED'] = str(seed % (2**32))  # Python hash seed must be 32-bit
    
    # Lock all random number generators with the global seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Set CUDA seeds if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Configure PyTorch for deterministic operations
    # Note: This may reduce performance but ensures reproducibility
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Set CuDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set CUBLAS workspace configuration (for deterministic matrix operations)
    # This matches the Docker environment variable
    # Can be overridden via CUBLAS_WORKSPACE_CONFIG environment variable
    cublas_config = os.getenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', cublas_config)
    
    # Enable CUDA deterministic algorithms if available
    if torch.cuda.is_available():
        # PyTorch 2.1.0+ supports deterministic algorithms
        try:
            torch.backends.cudnn.allow_tf32 = False  # Disable TF32 for determinism
            torch.backends.cuda.matmul.allow_tf32 = False
        except AttributeError:
            pass  # Older PyTorch versions may not have these attributes
        
        # Enable NVIDIA cuEvol/Deterministic Algorithms if available
        # This requires CUDA 11.8+ and PyTorch 2.1.0+
        try:
            # Set environment variable for deterministic CUDA kernels
            os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '1')  # Synchronous execution for determinism
            cublas_config = os.getenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', cublas_config)
            
            # Try to enable deterministic algorithms (may not be available on all GPUs)
            # This is a best-effort attempt - actual determinism depends on GPU architecture
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = False
            if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                torch.backends.cuda.matmul.allow_tf32 = False
        except Exception as e:
            print(f"⚠️  Warning: Could not enable all deterministic CUDA settings: {e}")


def get_deterministic_config() -> dict:
    """
    Get deterministic configuration dictionary.
    
    Returns:
        Dictionary with deterministic settings
    """
    return {
        "torch_use_deterministic_algorithms": True,
        "torch_cudnn_deterministic": True,
        "torch_cudnn_benchmark": False,
        "cublas_workspace_config": os.environ.get('CUBLAS_WORKSPACE_CONFIG', ':4096:8'),
        "python_hash_seed": os.environ.get('PYTHONHASHSEED', '0'),
    }


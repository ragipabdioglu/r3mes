"""
CUDA Installer Utility
Handles Windows-specific CUDA installation for PyTorch.
"""

import platform
import subprocess
import sys
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# PyTorch CUDA index URLs
PYTORCH_CUDA_URLS = {
    "cu118": "https://download.pytorch.org/whl/cu118",
    "cu121": "https://download.pytorch.org/whl/cu121",
    "cu124": "https://download.pytorch.org/whl/cu124",
}


def detect_cuda_version() -> Optional[str]:
    """
    Detect installed CUDA version.
    
    Returns:
        CUDA version string (e.g., "11.8", "12.1") or None
    """
    try:
        # Try nvidia-smi
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        
        if result.returncode == 0:
            # Driver version detected, map to CUDA version
            # This is approximate - actual CUDA toolkit version may differ
            driver_version = result.stdout.strip()
            major_version = int(driver_version.split(".")[0])
            
            # Approximate mapping (can be improved)
            if major_version >= 535:
                return "12.4"
            elif major_version >= 525:
                return "12.1"
            elif major_version >= 520:
                return "11.8"
            else:
                return "11.8"  # Default to 11.8 for older drivers
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
        # nvidia-smi not available or failed
        pass
    except Exception as e:
        # Unexpected error, log for debugging
        import logging
        logging.getLogger(__name__).debug(f"Error detecting CUDA version via nvidia-smi: {e}")
    
    # Try nvcc (if CUDA toolkit is installed)
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        
        if result.returncode == 0:
            # Parse version from output
            for line in result.stdout.split("\n"):
                if "release" in line.lower():
                    version_part = line.split("release")[-1].strip().split(",")[0]
                    version = version_part.strip()
                    # Extract major.minor
                    parts = version.split(".")
                    if len(parts) >= 2:
                        return f"{parts[0]}.{parts[1]}"
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        # nvcc not available
        pass
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug(f"Error detecting CUDA version via nvcc: {e}")
    
    return None


def install_pytorch_cuda(cuda_version: Optional[str] = None) -> Tuple[bool, str]:
    """
    Install PyTorch with CUDA support.
    
    Args:
        cuda_version: CUDA version (e.g., "11.8", "12.1"). If None, auto-detect.
    
    Returns:
        (success, message)
    """
    if platform.system() != "Windows":
        return False, "CUDA installer is only for Windows"
    
    # Detect CUDA version if not provided
    if cuda_version is None:
        cuda_version = detect_cuda_version()
        if cuda_version is None:
            # Default to 11.8 (most compatible)
            cuda_version = "11.8"
            logger.warning("Could not detect CUDA version, defaulting to 11.8")
    
    # Map to PyTorch index URL
    cuda_key = None
    if cuda_version.startswith("11.8"):
        cuda_key = "cu118"
    elif cuda_version.startswith("12.1"):
        cuda_key = "cu121"
    elif cuda_version.startswith("12.4"):
        cuda_key = "cu124"
    else:
        # Default to 11.8
        cuda_key = "cu118"
        logger.warning(f"Unknown CUDA version {cuda_version}, using 11.8")
    
    index_url = PYTORCH_CUDA_URLS[cuda_key]
    
    try:
        logger.info(f"Installing PyTorch with CUDA {cuda_version} support...")
        logger.info(f"Using index: {index_url}")
        
        # Uninstall CPU-only PyTorch first (if installed)
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"],
            capture_output=True,
        )
        
        # Install CUDA-enabled PyTorch
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                index_url,
            ],
            capture_output=True,
            text=True,
        )
        
        if result.returncode == 0:
            # Verify installation
            try:
                import torch
                if torch.cuda.is_available():
                    return True, f"PyTorch with CUDA {cuda_version} installed successfully"
                else:
                    return False, "PyTorch installed but CUDA not available. Check GPU drivers."
            except ImportError:
                return False, "PyTorch installation failed"
        else:
            return False, f"Installation failed: {result.stderr}"
            
    except Exception as e:
        return False, f"Error installing PyTorch: {str(e)}"


def check_pytorch_cuda() -> Tuple[bool, Optional[str]]:
    """
    Check if PyTorch has CUDA support.
    
    Returns:
        (has_cuda, cuda_version)
    """
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            return True, cuda_version
        else:
            return False, None
    except ImportError:
        return False, None


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    print("Checking PyTorch CUDA support...")
    has_cuda, version = check_pytorch_cuda()
    if has_cuda:
        print(f"✅ PyTorch has CUDA support: {version}")
    else:
        print("❌ PyTorch does not have CUDA support")
        
        if platform.system() == "Windows":
            print("\nAttempting to install CUDA-enabled PyTorch...")
            success, message = install_pytorch_cuda()
            print(f"{'✅' if success else '❌'} {message}")


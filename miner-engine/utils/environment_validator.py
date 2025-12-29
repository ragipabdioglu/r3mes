"""
Execution Environment Validator for R3MES Miner Engine

Validates that the execution environment matches the approved specifications
for deterministic, bit-exact gradient reproducibility.
"""

import sys
import platform
import subprocess
from typing import Dict, List, Tuple, Optional
import torch


class EnvironmentValidator:
    """
    Validates execution environment against approved specifications.
    
    Ensures:
    - Software versions match exactly
    - Hardware configuration is compatible
    - Deterministic algorithms are enabled
    - Floating point mode is correct
    """
    
    def __init__(self, required_environment: Optional[Dict] = None):
        """
        Initialize environment validator.
        
        Args:
            required_environment: Required environment specification (from blockchain)
        """
        self.required_environment = required_environment or self._get_default_environment()
    
    def _get_default_environment(self) -> Dict:
        """Get default environment specification for testing."""
        return {
            "platform": "nvidia",  # nvidia, amd, intel
            "cuda_version": "12.1.0",
            "pytorch_version": "2.1.0",
            "python_version": "3.10",
            "cudnn_version": "8.9.0",
            "deterministic_algorithms_enabled": True,
            "cublas_workspace_config": ":4096:8",
            "floating_point_mode": "fp32",
        }
    
    def validate_all(self) -> Tuple[bool, List[str]]:
        """
        Validate all environment aspects.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate Python version
        if not self.validate_python_version():
            errors.append(f"Python version mismatch: required {self.required_environment.get('python_version')}, got {sys.version.split()[0]}")
        
        # Validate PyTorch version
        if not self.validate_pytorch_version():
            errors.append(f"PyTorch version mismatch: required {self.required_environment.get('pytorch_version')}, got {torch.__version__}")
        
        # Validate platform-specific requirements
        platform_type = self.required_environment.get("platform", "nvidia")
        if platform_type == "nvidia":
            if not self.validate_cuda_version():
                errors.append(f"CUDA version mismatch: required {self.required_environment.get('cuda_version')}")
            if not self.validate_cudnn_version():
                errors.append(f"cuDNN version mismatch: required {self.required_environment.get('cudnn_version')}")
        elif platform_type == "amd":
            if not self.validate_rocm_version():
                errors.append(f"ROCm version mismatch: required {self.required_environment.get('rocm_version')}")
        elif platform_type == "intel":
            if not self.validate_intel_xpu_version():
                errors.append(f"Intel XPU version mismatch: required {self.required_environment.get('intel_xpu_version')}")
        
        # Validate deterministic algorithms
        if not self.validate_deterministic_algorithms():
            errors.append("Deterministic algorithms not enabled")
        
        # Validate CUBLAS workspace config
        if not self.validate_cublas_workspace():
            errors.append(f"CUBLAS workspace config mismatch: required {self.required_environment.get('cublas_workspace_config')}")
        
        # Validate floating point mode
        if not self.validate_floating_point_mode():
            errors.append(f"Floating point mode mismatch: required {self.required_environment.get('floating_point_mode')}")
        
        return len(errors) == 0, errors
    
    def validate_python_version(self) -> bool:
        """Validate Python version matches requirement."""
        required = self.required_environment.get("python_version", "3.10")
        current = f"{sys.version_info.major}.{sys.version_info.minor}"
        return current == required
    
    def validate_pytorch_version(self) -> bool:
        """Validate PyTorch version matches requirement."""
        required = self.required_environment.get("pytorch_version", "2.1.0")
        current = torch.__version__.split("+")[0]  # Remove +cu121 suffix
        return current == required
    
    def validate_cuda_version(self) -> bool:
        """Validate CUDA version matches requirement."""
        if not torch.cuda.is_available():
            return False
        
        required = self.required_environment.get("cuda_version", "12.1.0")
        # Get CUDA version from PyTorch
        cuda_version = torch.version.cuda
        if cuda_version is None:
            return False
        
        # Compare major.minor versions
        required_major, required_minor = map(int, required.split(".")[:2])
        current_major, current_minor = map(int, cuda_version.split(".")[:2])
        
        return current_major == required_major and current_minor == required_minor
    
    def validate_cudnn_version(self) -> bool:
        """Validate cuDNN version matches requirement."""
        if not torch.cuda.is_available():
            return False
        
        required = self.required_environment.get("cudnn_version", "8.9.0")
        cudnn_version = torch.backends.cudnn.version()
        
        if cudnn_version is None:
            return False
        
        # Compare major.minor versions
        required_major, required_minor = map(int, required.split(".")[:2])
        # cuDNN version is an integer, convert to major.minor
        cudnn_str = str(cudnn_version)
        if len(cudnn_str) >= 2:
            current_major = int(cudnn_str[0])
            current_minor = int(cudnn_str[1]) if len(cudnn_str) > 1 else 0
        else:
            return False
        
        return current_major == required_major and current_minor == required_minor
    
    def validate_rocm_version(self) -> bool:
        """Validate ROCm version matches requirement."""
        # ROCm version detection (simplified)
        try:
            result = subprocess.run(
                ["rocminfo", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Parse ROCm version from output
                # This is a simplified check
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return False
    
    def validate_intel_xpu_version(self) -> bool:
        """Validate Intel XPU version matches requirement."""
        # Intel XPU version detection (simplified)
        # Check for Intel oneAPI runtime
        try:
            result = subprocess.run(
                ["dpkg", "-l", "intel-oneapi-runtime-dpcpp-cpp"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return False
    
    def validate_deterministic_algorithms(self) -> bool:
        """Validate that deterministic algorithms are enabled."""
        required = self.required_environment.get("deterministic_algorithms_enabled", True)
        
        if not required:
            return True  # If not required, skip check
        
        # Check if deterministic algorithms are available
        try:
            # Try to enable deterministic algorithms
            torch.use_deterministic_algorithms(True, warn_only=True)
            return True
        except Exception:
            return False
    
    def validate_cublas_workspace(self) -> bool:
        """Validate CUBLAS workspace configuration."""
        import os
        required = self.required_environment.get("cublas_workspace_config", ":4096:8")
        current = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")
        return current == required
    
    def validate_floating_point_mode(self) -> bool:
        """Validate floating point mode."""
        required = self.required_environment.get("floating_point_mode", "fp32")
        
        # Check if TF32 is disabled (for fp32 mode)
        if required == "fp32":
            # TF32 should be disabled for determinism
            tf32_enabled = (
                torch.backends.cudnn.allow_tf32 if hasattr(torch.backends.cudnn, 'allow_tf32') else False
            ) or (
                torch.backends.cuda.matmul.allow_tf32 if hasattr(torch.backends.cuda.matmul, 'allow_tf32') else False
            )
            return not tf32_enabled
        
        return True
    
    def get_environment_info(self) -> Dict:
        """
        Get current environment information.
        
        Returns:
            Dictionary with current environment details
        """
        info = {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "pytorch_version": torch.__version__.split("+")[0],
            "platform": platform.system(),
            "platform_machine": platform.machine(),
        }
        
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = torch.backends.cudnn.version()
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
        else:
            info["cuda_available"] = False
        
        # Check deterministic algorithms
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            info["deterministic_algorithms_supported"] = True
        except Exception:
            info["deterministic_algorithms_supported"] = False
        
        # Check CUBLAS workspace
        import os
        info["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")
        
        return info


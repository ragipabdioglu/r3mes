"""
GPU Architecture Detection and Handling

Detects GPU architecture and provides architecture-specific handling
for deterministic training and verification.
"""

import torch
import subprocess
import re
from typing import Optional, Dict


class GPUArchitectureDetector:
    """Detects and handles GPU architecture information."""
    
    # Supported architectures
    SUPPORTED_ARCHITECTURES = {
        "Pascal": ["sm_60", "sm_61", "sm_62"],
        "Volta": ["sm_70", "sm_72"],
        "Turing": ["sm_75"],
        "Ampere": ["sm_80", "sm_86"],
        "Ada": ["sm_89"],
        "Blackwell": ["sm_100"],
    }
    
    def __init__(self):
        self.architecture = None
        self.compute_capability = None
        self.device_name = None
        self._detect_architecture()
    
    def _detect_architecture(self):
        """Detect GPU architecture using CUDA."""
        if not torch.cuda.is_available():
            self.architecture = "CPU"
            return
        
        # Get device properties
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        self.device_name = props.name
        self.compute_capability = f"sm_{props.major}{props.minor}"
        
        # Map compute capability to architecture
        for arch, capabilities in self.SUPPORTED_ARCHITECTURES.items():
            if self.compute_capability in capabilities:
                self.architecture = arch
                return
        
        # Unknown architecture
        self.architecture = f"Unknown_{self.compute_capability}"
    
    def get_architecture(self) -> str:
        """Get detected GPU architecture."""
        return self.architecture
    
    def get_compute_capability(self) -> Optional[str]:
        """Get compute capability string."""
        return self.compute_capability
    
    def get_device_name(self) -> Optional[str]:
        """Get GPU device name."""
        return self.device_name
    
    def is_supported(self) -> bool:
        """Check if GPU architecture is supported."""
        return self.architecture in self.SUPPORTED_ARCHITECTURES or self.architecture == "CPU"
    
    def get_metadata(self) -> Dict[str, str]:
        """
        Get GPU architecture metadata for gradient submission.
        
        Returns:
            Dictionary with architecture information
        """
        metadata = {
            "architecture": self.architecture,
            "compute_capability": self.compute_capability or "N/A",
            "device_name": self.device_name or "CPU",
            "cuda_available": str(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda or "N/A",
        }
        
        # Add deterministic execution info
        if torch.cuda.is_available():
            try:
                metadata["cudnn_deterministic"] = str(torch.backends.cudnn.deterministic)
                metadata["cudnn_benchmark"] = str(torch.backends.cudnn.benchmark)
                metadata["deterministic_algorithms"] = str(torch.are_deterministic_algorithms_enabled())
            except AttributeError:
                pass
        
        return metadata
    
    def configure_deterministic_cuda(self):
        """
        Configure CUDA for deterministic execution based on GPU architecture.
        
        This ensures that deterministic algorithms are used when available.
        """
        if not torch.cuda.is_available():
            return
        
        # Enable deterministic algorithms (best effort)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as e:
            print(f"⚠️  Warning: Could not enable deterministic algorithms: {e}")
        
        # Architecture-specific deterministic settings
        if self.architecture in ["Ampere", "Ada", "Blackwell"]:
            # Newer architectures support better deterministic algorithms
            try:
                # Disable TF32 for determinism (TF32 is non-deterministic)
                if hasattr(torch.backends.cudnn, 'allow_tf32'):
                    torch.backends.cudnn.allow_tf32 = False
                if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                    torch.backends.cuda.matmul.allow_tf32 = False
            except AttributeError:
                pass
    
    @staticmethod
    def can_compare_directly(arch1: str, arch2: str) -> bool:
        """
        Check if two architectures can be compared directly.
        
        Args:
            arch1: First architecture
            arch2: Second architecture
            
        Returns:
            True if architectures are the same and can be compared directly
        """
        # Same architectures can be compared directly
        if arch1 == arch2:
            return True
        
        # CPU can always be used for verification
        if arch1 == "CPU" or arch2 == "CPU":
            return True
        
        # Different GPU architectures require CPU fallback
        return False


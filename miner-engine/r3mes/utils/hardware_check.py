"""
Hardware Check Utility
Checks system requirements and warns users about insufficient hardware.
"""

import platform
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# System requirements
MIN_VRAM_GB = 8  # Minimum VRAM in GB
RECOMMENDED_VRAM_GB = 12  # Recommended VRAM in GB
MIN_RAM_GB = 16  # Minimum RAM in GB
RECOMMENDED_RAM_GB = 32  # Recommended RAM in GB

# GPU recommendations
GPU_RECOMMENDATIONS = {
    "minimum": {
        "models": ["RTX 3060", "RTX 3070", "RTX 4060"],
        "vram_gb": 12,
        "description": "Minimum for mining (may be slow)",
    },
    "recommended": {
        "models": ["RTX 3090", "RTX 4090", "A100"],
        "vram_gb": 24,
        "description": "Recommended for optimal performance",
    },
}


def get_gpu_info() -> Dict[str, Any]:
    """
    Get GPU information.
    
    Returns:
        Dict with 'name', 'vram_gb', 'cuda_available', etc.
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return {
                "available": False,
                "name": "CPU (No GPU)",
                "vram_gb": 0,
                "cuda_available": False,
            }
        
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        
        # Get VRAM in GB
        vram_bytes = torch.cuda.get_device_properties(device).total_memory
        vram_gb = vram_bytes / (1024 ** 3)
        
        return {
            "available": True,
            "name": props.name,
            "vram_gb": round(vram_gb, 2),
            "cuda_available": True,
            "compute_capability": f"{props.major}.{props.minor}",
        }
    except ImportError:
        return {
            "available": False,
            "name": "Unknown (PyTorch not installed)",
            "vram_gb": 0,
            "cuda_available": False,
        }
    except Exception as e:
        logger.error(f"Error getting GPU info: {e}")
        return {
            "available": False,
            "name": "Unknown (Error)",
            "vram_gb": 0,
            "cuda_available": False,
        }


def get_ram_info() -> Dict[str, Any]:
    """
    Get RAM information.
    
    Returns:
        Dict with 'total_gb', etc.
    """
    try:
        import psutil
        ram_bytes = psutil.virtual_memory().total
        ram_gb = ram_bytes / (1024 ** 3)
        return {
            "total_gb": round(ram_gb, 2),
        }
    except ImportError:
        # Fallback for systems without psutil
        try:
            if platform.system() == "Linux":
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            ram_kb = int(line.split()[1])
                            ram_gb = ram_kb / (1024 ** 2)
                            return {"total_gb": round(ram_gb, 2)}
        except (IOError, OSError, ValueError) as e:
            # File not accessible or parsing error on non-Linux systems
            logger.debug(f"Could not read /proc/meminfo: {e}")
        
        return {"total_gb": 0}
    except Exception as e:
        logger.error(f"Error getting RAM info: {e}")
        return {"total_gb": 0}


def check_hardware_requirements() -> Dict[str, Any]:
    """
    Check if system meets hardware requirements.
    
    Returns:
        Dict with 'meets_minimum', 'meets_recommended', 'warnings', etc.
    """
    gpu_info = get_gpu_info()
    ram_info = get_ram_info()
    
    # Check GPU
    gpu_meets_minimum = gpu_info.get("vram_gb", 0) >= MIN_VRAM_GB or not gpu_info.get("available", False)
    gpu_meets_recommended = gpu_info.get("vram_gb", 0) >= RECOMMENDED_VRAM_GB
    
    # Check RAM
    ram_meets_minimum = ram_info.get("total_gb", 0) >= MIN_RAM_GB
    ram_meets_recommended = ram_info.get("total_gb", 0) >= RECOMMENDED_RAM_GB
    
    # Generate warnings
    warnings = []
    
    if not gpu_info.get("available", False):
        warnings.append("⚠️  No GPU detected. Mining will use CPU (very slow).")
    elif not gpu_meets_minimum:
        warnings.append(
            f"⚠️  GPU VRAM ({gpu_info.get('vram_gb', 0)}GB) is below minimum ({MIN_VRAM_GB}GB). "
            f"Mining may fail with 'Out of Memory' errors."
        )
    elif not gpu_meets_recommended:
        warnings.append(
            f"⚠️  GPU VRAM ({gpu_info.get('vram_gb', 0)}GB) is below recommended ({RECOMMENDED_VRAM_GB}GB). "
            f"Performance may be suboptimal."
        )
    
    if not ram_meets_minimum:
        warnings.append(
            f"⚠️  RAM ({ram_info.get('total_gb', 0)}GB) is below minimum ({MIN_RAM_GB}GB). "
            f"System may be slow or unstable."
        )
    elif not ram_meets_recommended:
        warnings.append(
            f"⚠️  RAM ({ram_info.get('total_gb', 0)}GB) is below recommended ({RECOMMENDED_RAM_GB}GB). "
            f"Consider upgrading for better performance."
        )
    
    return {
        "meets_minimum": gpu_meets_minimum and ram_meets_minimum,
        "meets_recommended": gpu_meets_recommended and ram_meets_recommended,
        "gpu": gpu_info,
        "ram": ram_info,
        "warnings": warnings,
        "recommendations": GPU_RECOMMENDATIONS,
    }


def print_hardware_summary() -> None:
    """Print hardware summary to console."""
    result = check_hardware_requirements()
    
    print("\n" + "=" * 60)
    print("Hardware Check Summary")
    print("=" * 60)
    
    # GPU Info
    gpu = result["gpu"]
    print(f"\nGPU: {gpu.get('name', 'Unknown')}")
    if gpu.get("available"):
        print(f"  VRAM: {gpu.get('vram_gb', 0)} GB")
        print(f"  CUDA: {'Available' if gpu.get('cuda_available') else 'Not Available'}")
    else:
        print("  Status: Not Available (CPU mode)")
    
    # RAM Info
    ram = result["ram"]
    print(f"\nRAM: {ram.get('total_gb', 0)} GB")
    
    # Status
    print(f"\nStatus:")
    if result["meets_recommended"]:
        print("  ✅ Meets recommended requirements")
    elif result["meets_minimum"]:
        print("  ⚠️  Meets minimum requirements (performance may be suboptimal)")
    else:
        print("  ❌ Does not meet minimum requirements")
    
    # Warnings
    if result["warnings"]:
        print("\nWarnings:")
        for warning in result["warnings"]:
            print(f"  {warning}")
    
    # Recommendations
    print("\nRecommendations:")
    print(f"  Minimum: {GPU_RECOMMENDATIONS['minimum']['description']}")
    print(f"  Recommended: {GPU_RECOMMENDATIONS['recommended']['description']}")
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    print_hardware_summary()


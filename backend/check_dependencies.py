"""
Dependency Checker - Checks for Docker and CUDA installation

Provides user-friendly guidance for missing dependencies.
"""

import os
import sys
import subprocess
import platform
import webbrowser
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def check_docker():
    """Check if Docker is installed."""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return False, None


def check_cuda():
    """Check if CUDA is available."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return True, result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return False, None


def get_docker_install_url():
    """Get Docker installation URL based on OS."""
    system = platform.system()
    if system == "Windows":
        return "https://www.docker.com/products/docker-desktop"
    elif system == "Darwin":  # macOS
        return "https://www.docker.com/products/docker-desktop"
    elif system == "Linux":
        return "https://docs.docker.com/engine/install/"
    return "https://www.docker.com/get-started"


def check_dependencies():
    """Check all dependencies and provide guidance."""
    logger.info("Checking dependencies...\n")
    
    issues = []
    
    # Check Docker
    docker_installed, docker_version = check_docker()
    if docker_installed:
        logger.info(f"Docker: {docker_version}")
    else:
        logger.warning("Docker: Not installed")
        issues.append({
            "name": "Docker",
            "url": get_docker_install_url(),
            "description": "Docker is required for containerized model execution"
        })
    
    # Check CUDA
    cuda_available, cuda_info = check_cuda()
    if cuda_available:
        logger.info("CUDA: Available")
        # Extract GPU info
        if cuda_info:
            lines = cuda_info.split('\n')
            for line in lines:
                if 'NVIDIA' in line or 'GeForce' in line or 'RTX' in line:
                    logger.info(f"GPU: {line.strip()}")
    else:
        logger.warning("CUDA: Not detected (CPU mode will be used)")
        issues.append({
            "name": "CUDA",
            "url": "https://developer.nvidia.com/cuda-downloads",
            "description": "CUDA is optional but recommended for GPU acceleration"
        })
    
    # Check Python version
    python_version = sys.version_info
    logger.info(f"Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if issues:
        logger.warning("\nMissing Dependencies:")
        for issue in issues:
            logger.warning(f"\n{issue['name']}:")
            logger.warning(f"Description: {issue['description']}")
            logger.warning(f"Install: {issue['url']}")
        
        logger.info("\nWould you like to open the installation pages? (y/n): ", end="")
        response = input().strip().lower()
        if response == 'y':
            for issue in issues:
                webbrowser.open(issue['url'])
    
    return len(issues) == 0


if __name__ == "__main__":
    all_ok = check_dependencies()
    sys.exit(0 if all_ok else 1)


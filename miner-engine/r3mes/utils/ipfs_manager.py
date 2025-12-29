"""
IPFS Manager - Embedded IPFS Daemon Management
Automatically downloads and manages IPFS binary for cross-platform support.
"""

import os
import sys
import platform
import subprocess
import shutil
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# IPFS version and download URLs
IPFS_VERSION = "0.24.0"
IPFS_BASE_URL = "https://dist.ipfs.tech/kubo"

# Platform-specific download URLs
IPFS_DOWNLOADS = {
    "Windows": {
        "url": f"{IPFS_BASE_URL}/v{IPFS_VERSION}/kubo_v{IPFS_VERSION}_windows-amd64.zip",
        "binary": "ipfs.exe",
    },
    "Darwin": {
        "url": f"{IPFS_BASE_URL}/v{IPFS_VERSION}/kubo_v{IPFS_VERSION}_darwin-amd64.tar.gz",
        "binary": "ipfs",
    },
    "Linux": {
        "url": f"{IPFS_BASE_URL}/v{IPFS_VERSION}/kubo_v{IPFS_VERSION}_linux-amd64.tar.gz",
        "binary": "ipfs",
    },
}


def get_ipfs_dir() -> Path:
    """Get IPFS directory in user's home."""
    home = Path.home()
    ipfs_dir = home / ".r3mes" / "ipfs"
    ipfs_dir.mkdir(parents=True, exist_ok=True)
    return ipfs_dir


def get_ipfs_binary_path() -> Path:
    """Get path to IPFS binary."""
    ipfs_dir = get_ipfs_dir()
    system = platform.system()
    
    if system == "Windows":
        return ipfs_dir / "ipfs.exe"
    else:
        return ipfs_dir / "ipfs"


def is_ipfs_installed() -> bool:
    """Check if IPFS is installed (either in PATH or embedded)."""
    # Check embedded binary
    if get_ipfs_binary_path().exists():
        return True
    
    # Check system PATH
    return shutil.which("ipfs") is not None


def download_ipfs() -> bool:
    """
    Download IPFS binary for current platform.
    
    Returns:
        True if successful, False otherwise
    """
    system = platform.system()
    
    if system not in IPFS_DOWNLOADS:
        logger.error(f"Unsupported platform: {system}")
        return False
    
    download_info = IPFS_DOWNLOADS[system]
    url = download_info["url"]
    binary_name = download_info["binary"]
    
    ipfs_dir = get_ipfs_dir()
    download_path = ipfs_dir / f"ipfs_download.{url.split('.')[-1]}"
    binary_path = get_ipfs_binary_path()
    
    try:
        logger.info(f"Downloading IPFS from {url}...")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        # Download file
        with open(download_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info("Extracting IPFS binary...")
        
        # Extract based on file type
        if download_path.suffix == ".zip":
            with zipfile.ZipFile(download_path, "r") as zip_ref:
                # Find the binary in the zip
                for member in zip_ref.namelist():
                    if member.endswith(binary_name):
                        with zip_ref.open(member) as source:
                            with open(binary_path, "wb") as target:
                                target.write(source.read())
                        break
        elif download_path.suffix in [".tar.gz", ".tgz"]:
            with tarfile.open(download_path, "r:gz") as tar_ref:
                # Find the binary in the tar
                for member in tar_ref.getmembers():
                    if member.name.endswith(binary_name):
                        tar_ref.extract(member, ipfs_dir)
                        # Move to final location
                        extracted_path = ipfs_dir / member.name
                        if extracted_path != binary_path:
                            shutil.move(extracted_path, binary_path)
                        break
        
        # Make executable (Unix)
        if system != "Windows":
            os.chmod(binary_path, 0o755)
        
        # Cleanup download file
        download_path.unlink()
        
        logger.info(f"IPFS binary installed at {binary_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download IPFS: {e}")
        if download_path.exists():
            download_path.unlink()
        return False


def initialize_ipfs() -> bool:
    """
    Initialize IPFS repository if not already initialized.
    
    Returns:
        True if successful, False otherwise
    """
    ipfs_binary = get_ipfs_binary_path()
    
    if not ipfs_binary.exists():
        logger.error("IPFS binary not found")
        return False
    
    try:
        # Check if already initialized
        ipfs_repo = Path.home() / ".ipfs"
        if ipfs_repo.exists() and (ipfs_repo / "config").exists():
            logger.info("IPFS repository already initialized")
            return True
        
        # Initialize IPFS
        logger.info("Initializing IPFS repository...")
        result = subprocess.run(
            [str(ipfs_binary), "init"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            logger.info("IPFS repository initialized")
            return True
        else:
            logger.error(f"IPFS init failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to initialize IPFS: {e}")
        return False


def start_ipfs_daemon() -> Optional[subprocess.Popen]:
    """
    Start IPFS daemon in background.
    
    Returns:
        Popen process if successful, None otherwise
    """
    ipfs_binary = get_ipfs_binary_path()
    
    if not ipfs_binary.exists():
        logger.error("IPFS binary not found")
        return None
    
    # Check if already running
    if is_ipfs_running():
        logger.info("IPFS daemon is already running")
        return None
    
    try:
        logger.info("Starting IPFS daemon...")
        process = subprocess.Popen(
            [str(ipfs_binary), "daemon"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        
        # Wait a bit to check if it started successfully
        import time
        time.sleep(2)
        
        if process.poll() is None:
            logger.info("IPFS daemon started")
            return process
        else:
            logger.error("IPFS daemon failed to start")
            return None
            
    except Exception as e:
        logger.error(f"Failed to start IPFS daemon: {e}")
        return None


def is_ipfs_running() -> bool:
    """Check if IPFS daemon is running."""
    try:
        response = requests.get("http://localhost:5001/api/v0/version", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        # Connection failed - IPFS not running
        return False


def ensure_ipfs_available() -> Tuple[bool, Optional[str]]:
    """
    Ensure IPFS is available (download if needed, start daemon).
    
    Returns:
        (success, message)
    """
    # Check if IPFS is installed
    if not is_ipfs_installed():
        logger.info("IPFS not found, downloading...")
        if not download_ipfs():
            return False, "Failed to download IPFS binary"
    
    # Initialize if needed
    if not initialize_ipfs():
        return False, "Failed to initialize IPFS repository"
    
    # Start daemon if not running
    if not is_ipfs_running():
        process = start_ipfs_daemon()
        if process is None:
            return False, "Failed to start IPFS daemon"
        return True, "IPFS daemon started"
    
    return True, "IPFS is running"


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    success, message = ensure_ipfs_available()
    print(f"Success: {success}")
    print(f"Message: {message}")


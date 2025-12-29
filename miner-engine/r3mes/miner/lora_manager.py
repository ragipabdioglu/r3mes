"""
LoRA Manager for Miner

Handles LoRA adapter selection, download from IPFS, and loading into the model.
"""

import os
import json
import logging
import requests
from pathlib import Path
from typing import List, Dict, Optional, Any

# Try to import PEFT (optional dependency)
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    PeftModel = None

from ..utils.ipfs_client import IPFSClient

logger = logging.getLogger(__name__)


class LoRAManager:
    """Manages LoRA adapters for miner serving nodes."""
    
    def __init__(
        self,
        backend_url: str,
        ipfs_client: Optional[IPFSClient] = None,
        lora_cache_dir: Optional[str] = None
    ):
        """
        Initialize LoRA manager.
        
        Args:
            backend_url: Backend API URL (e.g., "https://api.r3mes.network")
            ipfs_client: IPFS client instance (optional, will create if not provided)
            lora_cache_dir: Directory to cache LoRA adapters (default: ~/.r3mes/loras)
        """
        self.backend_url = backend_url.rstrip('/')
        self.ipfs_client = ipfs_client
        
        # Set up LoRA cache directory
        if lora_cache_dir:
            self.lora_cache_dir = Path(lora_cache_dir)
        else:
            home = Path.home()
            self.lora_cache_dir = home / ".r3mes" / "loras"
        
        self.lora_cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"LoRA cache directory: {self.lora_cache_dir}")
    
    def get_available_loras(self) -> List[Dict[str, Any]]:
        """
        Get list of available LoRA adapters from backend.
        
        Returns:
            List of LoRA dictionaries with name, ipfs_hash, etc.
        """
        try:
            import os
            timeout = int(os.getenv("R3MES_LORA_FETCH_TIMEOUT", "10"))
            response = requests.get(
                f"{self.backend_url}/api/lora/list",
                params={"active_only": True},
                timeout=timeout
            )
            response.raise_for_status()
            data = response.json()
            # Backend returns a list directly, not wrapped in "loras" key
            if isinstance(data, list):
                return data
            # Fallback: try to get from "loras" key if it's a dict
            return data.get("loras", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch LoRA list from backend: {e}")
            return []
    
    def select_loras(self, min_count: int = 1) -> List[str]:
        """
        Select LoRA adapters to download and use.
        
        Currently selects the first available LoRA(s). In the future, this could
        be made configurable or use a selection strategy.
        
        Args:
            min_count: Minimum number of LoRAs to select (default: 1)
            
        Returns:
            List of selected LoRA adapter names
        """
        available_loras = self.get_available_loras()
        
        if not available_loras:
            logger.warning("No LoRA adapters available from backend")
            return []
        
        # Select at least min_count LoRAs (or all available if fewer)
        selected = []
        for lora in available_loras[:max(min_count, len(available_loras))]:
            selected.append(lora["name"])
        
        logger.info(f"Selected {len(selected)} LoRA adapter(s): {', '.join(selected)}")
        return selected
    
    def download_lora_from_ipfs(
        self,
        lora_name: str,
        ipfs_hash: str
    ) -> Optional[Path]:
        """
        Download LoRA adapter from IPFS.
        
        Args:
            lora_name: LoRA adapter name
            ipfs_hash: IPFS hash of the LoRA adapter
            
        Returns:
            Path to downloaded LoRA directory, or None if failed
        """
        # Check if already cached
        lora_dir = self.lora_cache_dir / lora_name
        if lora_dir.exists() and (lora_dir / "adapter_config.json").exists():
            logger.info(f"LoRA {lora_name} already cached at {lora_dir}")
            return lora_dir
        
        if not self.ipfs_client:
            logger.error("IPFS client not available, cannot download LoRA")
            return None
        
        if not self.ipfs_client.is_connected():
            logger.error("IPFS client not connected, cannot download LoRA")
            return None
        
        try:
            logger.info(f"Downloading LoRA {lora_name} from IPFS: {ipfs_hash}")
            
            # Create directory for this LoRA
            lora_dir.mkdir(parents=True, exist_ok=True)
            
            # Download from IPFS
            # IPFS can return either a directory or a single file
            # For LoRA adapters, we expect a directory with adapter_config.json and adapter_model.bin
            content = self.ipfs_client.retrieve_content(ipfs_hash)
            
            if content is None:
                logger.error(f"Failed to retrieve LoRA {lora_name} from IPFS")
                return None
            
            # Try to extract if it's a tar/zip archive
            # For now, assume it's a directory structure that IPFS returns
            # Save as adapter files
            adapter_config_path = lora_dir / "adapter_config.json"
            adapter_model_path = lora_dir / "adapter_model.bin"
            
            # If content is JSON (adapter_config), save it
            try:
                config_data = json.loads(content)
                with open(adapter_config_path, 'w') as f:
                    json.dump(config_data, f, indent=2)
                logger.info(f"Saved adapter config for {lora_name}")
            except json.JSONDecodeError:
                # If not JSON, might be binary model data
                with open(adapter_model_path, 'wb') as f:
                    f.write(content)
                logger.info(f"Saved adapter model for {lora_name}")
            
            # Verify download
            if adapter_config_path.exists() or adapter_model_path.exists():
                logger.info(f"Successfully downloaded LoRA {lora_name} to {lora_dir}")
                return lora_dir
            else:
                logger.error(f"Downloaded LoRA {lora_name} but files are missing")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading LoRA {lora_name} from IPFS: {e}", exc_info=True)
            # Clean up on failure
            if lora_dir.exists():
                import shutil
                shutil.rmtree(lora_dir, ignore_errors=True)
            return None
    
    def load_lora_adapters(
        self,
        base_model,
        lora_names: List[str]
    ) -> Dict[str, Any]:
        """
        Load LoRA adapters into base model.
        
        Args:
            base_model: Base model (PyTorch model)
            lora_names: List of LoRA adapter names to load
            
        Returns:
            Dictionary with loaded adapters and status
        """
        loaded_adapters = {}
        
        for lora_name in lora_names:
            lora_dir = self.lora_cache_dir / lora_name
            
            if not lora_dir.exists():
                logger.warning(f"LoRA directory not found: {lora_dir}")
                continue
            
            try:
                if not PEFT_AVAILABLE:
                    logger.error(f"PEFT library not available, cannot load LoRA adapter {lora_name}")
                    continue
                
                # Load LoRA adapter using PEFT
                adapter = PeftModel.from_pretrained(
                    base_model,
                    str(lora_dir),
                    adapter_name=lora_name
                )
                loaded_adapters[lora_name] = adapter
                logger.info(f"Loaded LoRA adapter: {lora_name}")
            except Exception as e:
                logger.error(f"Failed to load LoRA adapter {lora_name}: {e}", exc_info=True)
        
        return {
            "loaded": list(loaded_adapters.keys()),
            "failed": [name for name in lora_names if name not in loaded_adapters],
            "adapters": loaded_adapters
        }
    
    def get_lora_path(self, lora_name: str) -> Optional[Path]:
        """
        Get path to cached LoRA adapter.
        
        Args:
            lora_name: LoRA adapter name
            
        Returns:
            Path to LoRA directory, or None if not found
        """
        lora_dir = self.lora_cache_dir / lora_name
        if lora_dir.exists() and (lora_dir / "adapter_config.json").exists():
            return lora_dir
        return None


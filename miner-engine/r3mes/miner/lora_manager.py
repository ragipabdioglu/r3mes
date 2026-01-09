#!/usr/bin/env python3
"""
R3MES LoRA Manager

Production-ready LoRA adapter manager that:
1. Manages LoRA adapter caching and storage
2. Handles adapter loading and saving
3. Optimizes memory usage for multiple adapters
4. Provides adapter versioning and metadata
5. Implements efficient adapter swapping
"""

import logging
import json
import hashlib
import time
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import torch
import torch.nn as nn

from r3mes.utils.logger import setup_logger
from core.serialization import LoRASerializer


class LoRAManager:
    """LoRA adapter manager for efficient adapter handling."""
    
    def __init__(
        self,
        cache_dir: str = "lora_cache",
        max_cache_size_mb: int = 1024,  # 1GB default
        max_adapters_in_memory: int = 10,
        log_level: str = "INFO",
        use_json_logs: bool = False,
    ):
        """
        Initialize LoRA manager.
        
        Args:
            cache_dir: Directory for caching LoRA adapters
            max_cache_size_mb: Maximum cache size in MB
            max_adapters_in_memory: Maximum number of adapters to keep in memory
            log_level: Logging level
            use_json_logs: Whether to use JSON-formatted logs
        """
        # Setup logger
        log_level_map = {
            "DEBUG": 10,
            "INFO": 20,
            "WARNING": 30,
            "ERROR": 40,
        }
        self.logger = setup_logger(
            "r3mes.lora_manager",
            level=log_level_map.get(log_level.upper(), logging.INFO),
            use_json=use_json_logs,
        )
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_cache_size_mb = max_cache_size_mb
        self.max_adapters_in_memory = max_adapters_in_memory
        
        # In-memory adapter cache
        self.memory_cache = {}  # adapter_id -> (adapter_state, metadata, last_access)
        self.cache_order = []   # LRU order
        
        # Disk cache metadata
        self.disk_cache_metadata = {}  # adapter_id -> metadata
        self.serializer = LoRASerializer()
        
        # Load existing cache metadata
        self._load_cache_metadata()
        
        self.logger.info(f"LoRA manager initialized (cache_dir: {cache_dir}, max_cache: {max_cache_size_mb}MB)")
    
    def _load_cache_metadata(self):
        """Load cache metadata from disk."""
        try:
            metadata_file = self.cache_dir / "cache_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.disk_cache_metadata = json.load(f)
                self.logger.info(f"Loaded cache metadata: {len(self.disk_cache_metadata)} adapters")
            else:
                self.logger.info("No existing cache metadata found")
        except Exception as e:
            self.logger.error(f"Error loading cache metadata: {e}")
            self.disk_cache_metadata = {}
    
    def _save_cache_metadata(self):
        """Save cache metadata to disk."""
        try:
            metadata_file = self.cache_dir / "cache_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(self.disk_cache_metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving cache metadata: {e}")
    
    def _generate_adapter_id(self, adapter_state: Dict[str, torch.Tensor], metadata: Dict[str, Any]) -> str:
        """
        Generate unique adapter ID based on state and metadata.
        
        Args:
            adapter_state: LoRA adapter state dict
            metadata: Adapter metadata
            
        Returns:
            Unique adapter ID (hash)
        """
        try:
            # Create hash from adapter parameters
            hash_input = ""
            
            # Add parameter shapes and some values
            for name, param in sorted(adapter_state.items()):
                hash_input += f"{name}:{param.shape}:{param.sum().item():.6f}:"
            
            # Add relevant metadata
            for key in ["model_version", "lora_rank", "lora_alpha"]:
                if key in metadata:
                    hash_input += f"{key}:{metadata[key]}:"
            
            # Generate SHA-256 hash
            adapter_id = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
            return adapter_id
            
        except Exception as e:
            self.logger.error(f"Error generating adapter ID: {e}")
            return f"adapter_{int(time.time())}"
    
    def _update_memory_cache_lru(self, adapter_id: str):
        """Update LRU order for memory cache."""
        if adapter_id in self.cache_order:
            self.cache_order.remove(adapter_id)
        self.cache_order.append(adapter_id)
    
    def _evict_memory_cache(self):
        """Evict least recently used adapters from memory cache."""
        while len(self.memory_cache) > self.max_adapters_in_memory:
            if not self.cache_order:
                break
            
            # Remove least recently used
            lru_adapter_id = self.cache_order.pop(0)
            if lru_adapter_id in self.memory_cache:
                del self.memory_cache[lru_adapter_id]
                self.logger.debug(f"Evicted adapter from memory: {lru_adapter_id}")
    
    def _get_cache_size_mb(self) -> float:
        """Get current disk cache size in MB."""
        try:
            total_size = 0
            for file_path in self.cache_dir.glob("*.lora"):
                total_size += file_path.stat().st_size
            return total_size / (1024 * 1024)
        except Exception as e:
            self.logger.error(f"Error calculating cache size: {e}")
            return 0.0
    
    def _cleanup_disk_cache(self):
        """Clean up disk cache if it exceeds size limit."""
        try:
            current_size_mb = self._get_cache_size_mb()
            if current_size_mb <= self.max_cache_size_mb:
                return
            
            self.logger.info(f"Disk cache size ({current_size_mb:.2f}MB) exceeds limit ({self.max_cache_size_mb}MB), cleaning up")
            
            # Sort adapters by last access time (oldest first)
            adapter_files = []
            for adapter_id, metadata in self.disk_cache_metadata.items():
                file_path = self.cache_dir / f"{adapter_id}.lora"
                if file_path.exists():
                    last_access = metadata.get("last_access", 0)
                    adapter_files.append((last_access, adapter_id, file_path))
            
            adapter_files.sort()  # Sort by last_access (oldest first)
            
            # Remove oldest adapters until under size limit
            for last_access, adapter_id, file_path in adapter_files:
                try:
                    file_path.unlink()
                    del self.disk_cache_metadata[adapter_id]
                    self.logger.debug(f"Removed cached adapter: {adapter_id}")
                    
                    current_size_mb = self._get_cache_size_mb()
                    if current_size_mb <= self.max_cache_size_mb:
                        break
                except Exception as e:
                    self.logger.error(f"Error removing cached adapter {adapter_id}: {e}")
            
            # Save updated metadata
            self._save_cache_metadata()
            
            final_size_mb = self._get_cache_size_mb()
            self.logger.info(f"Disk cache cleaned up: {final_size_mb:.2f}MB")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up disk cache: {e}")
    
    def save_adapter(
        self,
        adapter_state: Dict[str, torch.Tensor],
        metadata: Optional[Dict[str, Any]] = None,
        adapter_id: Optional[str] = None,
    ) -> str:
        """
        Save LoRA adapter to cache.
        
        Args:
            adapter_state: LoRA adapter state dict
            metadata: Optional metadata
            adapter_id: Optional adapter ID (will generate if not provided)
            
        Returns:
            Adapter ID
        """
        try:
            # Prepare metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                "saved_at": time.time(),
                "last_access": time.time(),
                "num_parameters": sum(p.numel() for p in adapter_state.values()),
                "parameter_names": list(adapter_state.keys()),
            })
            
            # Generate adapter ID if not provided
            if adapter_id is None:
                adapter_id = self._generate_adapter_id(adapter_state, metadata)
            
            self.logger.debug(f"Saving adapter: {adapter_id}")
            
            # Save to memory cache
            self.memory_cache[adapter_id] = (adapter_state.copy(), metadata.copy(), time.time())
            self._update_memory_cache_lru(adapter_id)
            self._evict_memory_cache()
            
            # Save to disk cache
            file_path = self.cache_dir / f"{adapter_id}.lora"
            adapter_bytes = self.serializer.serialize_gradients(adapter_state, metadata)
            
            with open(file_path, 'wb') as f:
                f.write(adapter_bytes)
            
            # Update disk cache metadata
            self.disk_cache_metadata[adapter_id] = metadata
            self._save_cache_metadata()
            
            # Clean up disk cache if needed
            self._cleanup_disk_cache()
            
            self.logger.info(f"Adapter saved: {adapter_id} ({len(adapter_bytes) / 1024:.2f} KB)")
            return adapter_id
            
        except Exception as e:
            self.logger.error(f"Error saving adapter: {e}", exc_info=True)
            return ""
    
    def load_adapter(self, adapter_id: str) -> Optional[Tuple[Dict[str, torch.Tensor], Dict[str, Any]]]:
        """
        Load LoRA adapter from cache.
        
        Args:
            adapter_id: Adapter ID
            
        Returns:
            Tuple of (adapter_state, metadata) or None if not found
        """
        try:
            self.logger.debug(f"Loading adapter: {adapter_id}")
            
            # Check memory cache first
            if adapter_id in self.memory_cache:
                adapter_state, metadata, _ = self.memory_cache[adapter_id]
                
                # Update access time
                self.memory_cache[adapter_id] = (adapter_state, metadata, time.time())
                self._update_memory_cache_lru(adapter_id)
                
                self.logger.debug(f"Adapter loaded from memory: {adapter_id}")
                return adapter_state.copy(), metadata.copy()
            
            # Check disk cache
            file_path = self.cache_dir / f"{adapter_id}.lora"
            if not file_path.exists():
                self.logger.warning(f"Adapter not found: {adapter_id}")
                return None
            
            # Load from disk
            with open(file_path, 'rb') as f:
                adapter_bytes = f.read()
            
            adapter_state, metadata = self.serializer.deserialize_gradients(adapter_bytes)
            
            # Update access time
            if adapter_id in self.disk_cache_metadata:
                self.disk_cache_metadata[adapter_id]["last_access"] = time.time()
                self._save_cache_metadata()
            
            # Add to memory cache
            self.memory_cache[adapter_id] = (adapter_state.copy(), metadata.copy(), time.time())
            self._update_memory_cache_lru(adapter_id)
            self._evict_memory_cache()
            
            self.logger.debug(f"Adapter loaded from disk: {adapter_id}")
            return adapter_state, metadata
            
        except Exception as e:
            self.logger.error(f"Error loading adapter: {e}", exc_info=True)
            return None
    
    def has_adapter(self, adapter_id: str) -> bool:
        """
        Check if adapter exists in cache.
        
        Args:
            adapter_id: Adapter ID
            
        Returns:
            True if adapter exists
        """
        # Check memory cache
        if adapter_id in self.memory_cache:
            return True
        
        # Check disk cache
        file_path = self.cache_dir / f"{adapter_id}.lora"
        return file_path.exists()
    
    def remove_adapter(self, adapter_id: str) -> bool:
        """
        Remove adapter from cache.
        
        Args:
            adapter_id: Adapter ID
            
        Returns:
            True if adapter was removed
        """
        try:
            removed = False
            
            # Remove from memory cache
            if adapter_id in self.memory_cache:
                del self.memory_cache[adapter_id]
                removed = True
            
            if adapter_id in self.cache_order:
                self.cache_order.remove(adapter_id)
            
            # Remove from disk cache
            file_path = self.cache_dir / f"{adapter_id}.lora"
            if file_path.exists():
                file_path.unlink()
                removed = True
            
            if adapter_id in self.disk_cache_metadata:
                del self.disk_cache_metadata[adapter_id]
                self._save_cache_metadata()
            
            if removed:
                self.logger.info(f"Adapter removed: {adapter_id}")
            else:
                self.logger.warning(f"Adapter not found for removal: {adapter_id}")
            
            return removed
            
        except Exception as e:
            self.logger.error(f"Error removing adapter: {e}", exc_info=True)
            return False
    
    def list_adapters(self) -> List[Dict[str, Any]]:
        """
        List all cached adapters.
        
        Returns:
            List of adapter information dicts
        """
        try:
            adapters = []
            
            # Get all adapter IDs
            all_adapter_ids = set()
            all_adapter_ids.update(self.memory_cache.keys())
            all_adapter_ids.update(self.disk_cache_metadata.keys())
            
            for adapter_id in all_adapter_ids:
                adapter_info = {
                    "adapter_id": adapter_id,
                    "in_memory": adapter_id in self.memory_cache,
                    "on_disk": (self.cache_dir / f"{adapter_id}.lora").exists(),
                }
                
                # Get metadata
                if adapter_id in self.memory_cache:
                    _, metadata, _ = self.memory_cache[adapter_id]
                    adapter_info.update(metadata)
                elif adapter_id in self.disk_cache_metadata:
                    adapter_info.update(self.disk_cache_metadata[adapter_id])
                
                adapters.append(adapter_info)
            
            return adapters
            
        except Exception as e:
            self.logger.error(f"Error listing adapters: {e}", exc_info=True)
            return []
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        try:
            disk_size_mb = self._get_cache_size_mb()
            
            # Calculate memory usage
            memory_size_mb = 0
            for adapter_state, _, _ in self.memory_cache.values():
                for param in adapter_state.values():
                    memory_size_mb += param.numel() * param.element_size()
            memory_size_mb /= (1024 * 1024)
            
            return {
                "memory_cache_count": len(self.memory_cache),
                "disk_cache_count": len(self.disk_cache_metadata),
                "memory_size_mb": memory_size_mb,
                "disk_size_mb": disk_size_mb,
                "max_cache_size_mb": self.max_cache_size_mb,
                "max_adapters_in_memory": self.max_adapters_in_memory,
                "cache_dir": str(self.cache_dir),
            }
            
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {}
    
    def clear_cache(self, memory_only: bool = False) -> bool:
        """
        Clear adapter cache.
        
        Args:
            memory_only: If True, only clear memory cache
            
        Returns:
            True if successful
        """
        try:
            # Clear memory cache
            self.memory_cache.clear()
            self.cache_order.clear()
            self.logger.info("Memory cache cleared")
            
            if not memory_only:
                # Clear disk cache
                for file_path in self.cache_dir.glob("*.lora"):
                    file_path.unlink()
                
                self.disk_cache_metadata.clear()
                self._save_cache_metadata()
                self.logger.info("Disk cache cleared")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}", exc_info=True)
            return False
    
    def apply_adapter_to_model(
        self,
        model: nn.Module,
        adapter_id: str,
        adapter_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> bool:
        """
        Apply LoRA adapter to model.
        
        Args:
            model: Model to apply adapter to
            adapter_id: Adapter ID
            adapter_state: Optional adapter state (will load if not provided)
            
        Returns:
            True if successful
        """
        try:
            # Load adapter if not provided
            if adapter_state is None:
                result = self.load_adapter(adapter_id)
                if result is None:
                    self.logger.error(f"Failed to load adapter: {adapter_id}")
                    return False
                adapter_state, _ = result
            
            # Apply adapter to model
            model_state = model.state_dict()
            
            for name, param in adapter_state.items():
                if name in model_state:
                    # Update model parameter
                    model_state[name].copy_(param)
                else:
                    self.logger.warning(f"Parameter not found in model: {name}")
            
            self.logger.info(f"Adapter applied to model: {adapter_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying adapter to model: {e}", exc_info=True)
            return False
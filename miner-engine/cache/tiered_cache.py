"""
Tiered DoRA Cache System

Implements 3-tier caching for DoRA adapters:
- Tier 1: VRAM (GPU memory) - Hot cache, instant access
- Tier 2: RAM (CPU memory) - Warm cache, ~5ms access
- Tier 3: Disk (local storage) - Cold cache, ~50-100ms access

Features:
- LRU eviction policy
- Predictive loading based on router hints
- Automatic tier promotion/demotion
- IPFS integration for cold storage
"""

import torch
import asyncio
import time
import os
import json
from typing import Optional, Dict, List, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict
from pathlib import Path
import logging
import threading

logger = logging.getLogger(__name__)


class CacheTier(Enum):
    """Cache tier levels."""
    VRAM = 1  # GPU memory
    RAM = 2   # CPU memory
    DISK = 3  # Local disk
    IPFS = 4  # Remote IPFS (not cached locally)


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    vram_hits: int = 0
    ram_hits: int = 0
    disk_hits: int = 0
    evictions: int = 0
    promotions: int = 0
    demotions: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hit_rate,
            'vram_hits': self.vram_hits,
            'ram_hits': self.ram_hits,
            'disk_hits': self.disk_hits,
            'evictions': self.evictions,
            'promotions': self.promotions,
            'demotions': self.demotions,
        }


@dataclass
class CachedAdapter:
    """Cached adapter with metadata."""
    adapter_id: str
    tier: CacheTier
    data: Any  # DoRAAdapter or tensor data
    size_mb: float
    last_access: float = field(default_factory=time.time)
    access_count: int = 0
    
    def touch(self):
        """Update access time and count."""
        self.last_access = time.time()
        self.access_count += 1


class TieredDoRACache:
    """
    3-tier cache for DoRA adapters.
    
    Tier 1 (VRAM): Hot adapters loaded on GPU
    Tier 2 (RAM): Warm adapters in CPU memory
    Tier 3 (Disk): Cold adapters on local disk
    
    Usage:
        cache = TieredDoRACache(vram_capacity_mb=2048)
        adapter = await cache.get("medical_dora")
        await cache.predictive_load(["coding_dora", "legal_dora"])
    """
    
    # Default hot adapters to preload
    DEFAULT_HOT_ADAPTERS = ["turkish_dora", "general_dora"]
    
    def __init__(
        self,
        vram_capacity_mb: int = 2048,
        ram_capacity_mb: int = 8192,
        disk_cache_dir: str = ".r3mes/dora_cache",
        enable_predictive: bool = False,  # MVP: disabled
    ):
        """
        Initialize tiered cache.
        
        Args:
            vram_capacity_mb: Max VRAM for adapters (MB)
            ram_capacity_mb: Max RAM for adapters (MB)
            disk_cache_dir: Directory for disk cache
            enable_predictive: Enable predictive loading (MVP: False)
        """
        self.vram_capacity_mb = vram_capacity_mb
        self.ram_capacity_mb = ram_capacity_mb
        self.disk_cache_dir = Path(disk_cache_dir)
        self.enable_predictive = enable_predictive
        
        # Cache storage (OrderedDict for LRU)
        self._vram_cache: OrderedDict[str, CachedAdapter] = OrderedDict()
        self._ram_cache: OrderedDict[str, CachedAdapter] = OrderedDict()
        
        # Current usage tracking
        self._vram_used_mb: float = 0.0
        self._ram_used_mb: float = 0.0
        
        # Statistics
        self.stats = CacheStats()
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Pending predictive loads
        self._pending_loads: Set[str] = set()
        
        # Ensure disk cache directory exists
        self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Device for VRAM cache
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(
            f"TieredDoRACache initialized: VRAM={vram_capacity_mb}MB, "
            f"RAM={ram_capacity_mb}MB, disk={disk_cache_dir}"
        )
    
    async def get(self, adapter_id: str) -> Optional[Any]:
        """
        Get adapter from cache, checking all tiers.
        
        Args:
            adapter_id: Adapter identifier
            
        Returns:
            Adapter data or None if not found
        """
        with self._lock:
            # Tier 1: VRAM
            if adapter_id in self._vram_cache:
                cached = self._vram_cache[adapter_id]
                cached.touch()
                # Move to end (most recently used)
                self._vram_cache.move_to_end(adapter_id)
                self.stats.hits += 1
                self.stats.vram_hits += 1
                logger.debug(f"VRAM hit: {adapter_id}")
                return cached.data
            
            # Tier 2: RAM
            if adapter_id in self._ram_cache:
                cached = self._ram_cache[adapter_id]
                cached.touch()
                self._ram_cache.move_to_end(adapter_id)
                self.stats.hits += 1
                self.stats.ram_hits += 1
                logger.debug(f"RAM hit: {adapter_id}")
                
                # Promote to VRAM if space available
                await self._try_promote_to_vram(adapter_id, cached)
                
                return cached.data
            
            # Tier 3: Disk
            disk_path = self._get_disk_path(adapter_id)
            if disk_path.exists():
                adapter_data = await self._load_from_disk(adapter_id)
                if adapter_data is not None:
                    self.stats.hits += 1
                    self.stats.disk_hits += 1
                    logger.debug(f"Disk hit: {adapter_id}")
                    return adapter_data
            
            # Cache miss
            self.stats.misses += 1
            logger.debug(f"Cache miss: {adapter_id}")
            return None
    
    async def put(
        self,
        adapter_id: str,
        data: Any,
        size_mb: float,
        tier: CacheTier = CacheTier.RAM,
    ):
        """
        Put adapter into cache.
        
        Args:
            adapter_id: Adapter identifier
            data: Adapter data
            size_mb: Size in MB
            tier: Target tier (default: RAM)
        """
        with self._lock:
            cached = CachedAdapter(
                adapter_id=adapter_id,
                tier=tier,
                data=data,
                size_mb=size_mb,
            )
            
            if tier == CacheTier.VRAM:
                await self._put_vram(adapter_id, cached)
            elif tier == CacheTier.RAM:
                await self._put_ram(adapter_id, cached)
            elif tier == CacheTier.DISK:
                await self._save_to_disk(adapter_id, data)
    
    async def _put_vram(self, adapter_id: str, cached: CachedAdapter):
        """Put adapter into VRAM cache."""
        # Evict if necessary
        while self._vram_used_mb + cached.size_mb > self.vram_capacity_mb:
            if not self._vram_cache:
                logger.warning(f"Cannot fit {adapter_id} in VRAM ({cached.size_mb}MB)")
                # Fall back to RAM
                await self._put_ram(adapter_id, cached)
                return
            
            # Evict LRU (first item)
            evict_id, evict_cached = self._vram_cache.popitem(last=False)
            self._vram_used_mb -= evict_cached.size_mb
            self.stats.evictions += 1
            
            # Demote to RAM
            evict_cached.tier = CacheTier.RAM
            await self._put_ram(evict_id, evict_cached)
            self.stats.demotions += 1
            logger.debug(f"Evicted {evict_id} from VRAM to RAM")
        
        # Move data to GPU if needed
        if hasattr(cached.data, 'to') and self._device == 'cuda':
            cached.data = cached.data.to(self._device)
        
        self._vram_cache[adapter_id] = cached
        self._vram_used_mb += cached.size_mb
        logger.debug(f"Cached {adapter_id} in VRAM ({cached.size_mb:.2f}MB)")
    
    async def _put_ram(self, adapter_id: str, cached: CachedAdapter):
        """Put adapter into RAM cache."""
        # Evict if necessary
        while self._ram_used_mb + cached.size_mb > self.ram_capacity_mb:
            if not self._ram_cache:
                logger.warning(f"Cannot fit {adapter_id} in RAM ({cached.size_mb}MB)")
                # Save to disk
                await self._save_to_disk(adapter_id, cached.data)
                return
            
            # Evict LRU
            evict_id, evict_cached = self._ram_cache.popitem(last=False)
            self._ram_used_mb -= evict_cached.size_mb
            self.stats.evictions += 1
            
            # Save to disk
            await self._save_to_disk(evict_id, evict_cached.data)
            logger.debug(f"Evicted {evict_id} from RAM to disk")
        
        # Move data to CPU if needed
        if hasattr(cached.data, 'to'):
            cached.data = cached.data.to('cpu')
        
        cached.tier = CacheTier.RAM
        self._ram_cache[adapter_id] = cached
        self._ram_used_mb += cached.size_mb
        logger.debug(f"Cached {adapter_id} in RAM ({cached.size_mb:.2f}MB)")
    
    async def _try_promote_to_vram(self, adapter_id: str, cached: CachedAdapter):
        """Try to promote adapter from RAM to VRAM."""
        if self._vram_used_mb + cached.size_mb <= self.vram_capacity_mb:
            # Remove from RAM
            if adapter_id in self._ram_cache:
                del self._ram_cache[adapter_id]
                self._ram_used_mb -= cached.size_mb
            
            # Add to VRAM
            cached.tier = CacheTier.VRAM
            await self._put_vram(adapter_id, cached)
            self.stats.promotions += 1
            logger.debug(f"Promoted {adapter_id} to VRAM")
    
    def _get_disk_path(self, adapter_id: str) -> Path:
        """Get disk cache path for adapter."""
        return self.disk_cache_dir / f"{adapter_id}.pt"
    
    async def _save_to_disk(self, adapter_id: str, data: Any):
        """Save adapter to disk cache."""
        path = self._get_disk_path(adapter_id)
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: torch.save(data, path))
        
        logger.debug(f"Saved {adapter_id} to disk: {path}")
    
    async def _load_from_disk(self, adapter_id: str) -> Optional[Any]:
        """Load adapter from disk cache."""
        path = self._get_disk_path(adapter_id)
        
        if not path.exists():
            return None
        
        try:
            # Run in thread pool
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None, lambda: torch.load(path, map_location='cpu')
            )
            
            # Cache in RAM
            size_mb = self._estimate_size(data)
            await self.put(adapter_id, data, size_mb, CacheTier.RAM)
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load {adapter_id} from disk: {e}")
            return None
    
    def _estimate_size(self, data: Any) -> float:
        """Estimate size of data in MB."""
        if hasattr(data, 'estimate_size_mb'):
            return data.estimate_size_mb()
        elif isinstance(data, torch.Tensor):
            return data.numel() * data.element_size() / (1024 * 1024)
        elif isinstance(data, dict):
            total = 0
            for v in data.values():
                if isinstance(v, torch.Tensor):
                    total += v.numel() * v.element_size()
            return total / (1024 * 1024)
        else:
            return 1.0  # Default estimate
    
    async def predictive_load(self, likely_adapters: List[str]):
        """
        Predictively load adapters based on router hints.
        
        Args:
            likely_adapters: List of adapters likely to be needed
        """
        if not self.enable_predictive:
            return
        
        for adapter_id in likely_adapters:
            if adapter_id in self._pending_loads:
                continue
            
            if adapter_id in self._vram_cache or adapter_id in self._ram_cache:
                continue
            
            self._pending_loads.add(adapter_id)
            asyncio.create_task(self._background_load(adapter_id))
    
    async def _background_load(self, adapter_id: str):
        """Background task to load adapter."""
        try:
            await self._load_from_disk(adapter_id)
        finally:
            self._pending_loads.discard(adapter_id)
    
    async def preload_hot_adapters(self, adapter_ids: Optional[List[str]] = None):
        """
        Preload hot adapters into VRAM at startup.
        
        Args:
            adapter_ids: Adapters to preload (default: DEFAULT_HOT_ADAPTERS)
        """
        adapters = adapter_ids or self.DEFAULT_HOT_ADAPTERS
        
        for adapter_id in adapters:
            data = await self._load_from_disk(adapter_id)
            if data:
                size_mb = self._estimate_size(data)
                await self.put(adapter_id, data, size_mb, CacheTier.VRAM)
                logger.info(f"Preloaded hot adapter: {adapter_id}")
    
    def evict(self, adapter_id: str) -> bool:
        """
        Evict adapter from all cache tiers.
        
        Args:
            adapter_id: Adapter to evict
            
        Returns:
            True if evicted
        """
        with self._lock:
            evicted = False
            
            if adapter_id in self._vram_cache:
                cached = self._vram_cache.pop(adapter_id)
                self._vram_used_mb -= cached.size_mb
                evicted = True
            
            if adapter_id in self._ram_cache:
                cached = self._ram_cache.pop(adapter_id)
                self._ram_used_mb -= cached.size_mb
                evicted = True
            
            if evicted:
                self.stats.evictions += 1
            
            return evicted
    
    def clear(self):
        """Clear all caches."""
        with self._lock:
            self._vram_cache.clear()
            self._ram_cache.clear()
            self._vram_used_mb = 0.0
            self._ram_used_mb = 0.0
            
            if self._device == 'cuda':
                torch.cuda.empty_cache()
            
            logger.info("Cache cleared")
    
    def get_usage(self) -> Dict[str, Any]:
        """Get cache usage statistics."""
        return {
            'vram': {
                'used_mb': self._vram_used_mb,
                'capacity_mb': self.vram_capacity_mb,
                'utilization': self._vram_used_mb / self.vram_capacity_mb,
                'count': len(self._vram_cache),
            },
            'ram': {
                'used_mb': self._ram_used_mb,
                'capacity_mb': self.ram_capacity_mb,
                'utilization': self._ram_used_mb / self.ram_capacity_mb,
                'count': len(self._ram_cache),
            },
            'stats': self.stats.to_dict(),
        }
    
    def list_cached(self) -> Dict[str, List[str]]:
        """List all cached adapters by tier."""
        return {
            'vram': list(self._vram_cache.keys()),
            'ram': list(self._ram_cache.keys()),
            'disk': [
                p.stem for p in self.disk_cache_dir.glob("*.pt")
            ],
        }

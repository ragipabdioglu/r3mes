"""
Cache Module for R3MES Miner Engine

Provides tiered caching for DoRA adapters:
- Tier 1: VRAM (hot cache, ~0ms latency)
- Tier 2: RAM (warm cache, ~5ms latency)
- Tier 3: Disk (cold cache, ~50-100ms latency)
"""

from .tiered_cache import TieredDoRACache, CacheTier, CacheStats
from .vram_manager import VRAMManager, VRAMAllocation

__all__ = [
    'TieredDoRACache',
    'CacheTier',
    'CacheStats',
    'VRAMManager',
    'VRAMAllocation',
]

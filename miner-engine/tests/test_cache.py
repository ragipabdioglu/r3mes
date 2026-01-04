"""
Tests for Tiered Cache System.
"""

import pytest
import torch
import tempfile
import asyncio
from pathlib import Path

from cache.tiered_cache import TieredDoRACache, CacheTier, CacheStats
from cache.vram_manager import VRAMManager, VRAMTier


class TestCacheStats:
    """Tests for CacheStats."""
    
    def test_hit_rate(self):
        """Test hit rate calculation."""
        stats = CacheStats(hits=80, misses=20)
        assert stats.hit_rate == 0.8
    
    def test_hit_rate_zero(self):
        """Test hit rate with no accesses."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0
    
    def test_to_dict(self):
        """Test stats serialization."""
        stats = CacheStats(hits=10, misses=5, vram_hits=8)
        data = stats.to_dict()
        
        assert data['hits'] == 10
        assert data['misses'] == 5
        assert data['vram_hits'] == 8


class TestTieredDoRACache:
    """Tests for TieredDoRACache."""
    
    @pytest.fixture
    def cache(self, tmp_path):
        """Create cache with temp directory."""
        return TieredDoRACache(
            vram_capacity_mb=100,
            ram_capacity_mb=200,
            disk_cache_dir=str(tmp_path / "cache"),
        )
    
    @pytest.mark.asyncio
    async def test_put_and_get_ram(self, cache):
        """Test putting and getting from RAM cache."""
        data = torch.randn(100, 100)
        await cache.put("test_adapter", data, size_mb=0.04, tier=CacheTier.RAM)
        
        result = await cache.get("test_adapter")
        assert result is not None
        # Compare on same device
        assert torch.allclose(result.cpu(), data.cpu())
    
    @pytest.mark.asyncio
    async def test_cache_miss(self, cache):
        """Test cache miss returns None."""
        result = await cache.get("nonexistent")
        assert result is None
        assert cache.stats.misses == 1
    
    @pytest.mark.asyncio
    async def test_cache_hit_stats(self, cache):
        """Test cache hit updates stats."""
        data = torch.randn(10, 10)
        await cache.put("test", data, size_mb=0.001, tier=CacheTier.RAM)
        
        await cache.get("test")
        assert cache.stats.hits == 1
        assert cache.stats.ram_hits == 1
    
    @pytest.mark.asyncio
    async def test_lru_eviction(self, cache):
        """Test LRU eviction when cache is full."""
        # Fill RAM cache
        for i in range(10):
            data = torch.randn(100, 100)
            await cache.put(f"adapter_{i}", data, size_mb=25, tier=CacheTier.RAM)
        
        # First adapters should be evicted
        assert "adapter_0" not in cache._ram_cache
        assert cache.stats.evictions > 0
    
    @pytest.mark.asyncio
    async def test_disk_cache(self, cache):
        """Test disk caching."""
        data = torch.randn(50, 50)
        await cache.put("disk_test", data, size_mb=0.01, tier=CacheTier.DISK)
        
        # Clear RAM cache
        cache._ram_cache.clear()
        cache._ram_used_mb = 0
        
        # Should load from disk
        result = await cache.get("disk_test")
        assert result is not None
        assert cache.stats.disk_hits == 1
    
    @pytest.mark.asyncio
    async def test_evict(self, cache):
        """Test manual eviction."""
        data = torch.randn(10, 10)
        await cache.put("to_evict", data, size_mb=0.001, tier=CacheTier.RAM)
        
        result = cache.evict("to_evict")
        assert result is True
        
        result = await cache.get("to_evict")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_clear(self, cache):
        """Test clearing all caches."""
        data = torch.randn(10, 10)
        await cache.put("test1", data, size_mb=0.001, tier=CacheTier.RAM)
        await cache.put("test2", data, size_mb=0.001, tier=CacheTier.RAM)
        
        cache.clear()
        
        assert len(cache._ram_cache) == 0
        assert cache._ram_used_mb == 0
    
    def test_get_usage(self, cache):
        """Test usage statistics."""
        usage = cache.get_usage()
        
        assert 'vram' in usage
        assert 'ram' in usage
        assert 'stats' in usage
        assert usage['vram']['capacity_mb'] == 100
        assert usage['ram']['capacity_mb'] == 200
    
    def test_list_cached(self, cache):
        """Test listing cached adapters."""
        cached = cache.list_cached()
        
        assert 'vram' in cached
        assert 'ram' in cached
        assert 'disk' in cached


class TestVRAMManager:
    """Tests for VRAMManager."""
    
    @pytest.fixture
    def manager(self):
        """Create VRAM manager."""
        return VRAMManager()
    
    def test_tier_detection(self, manager):
        """Test VRAM tier detection."""
        assert manager.tier in [VRAMTier.LOW, VRAMTier.MEDIUM, VRAMTier.HIGH]
    
    def test_max_experts(self, manager):
        """Test max experts based on tier."""
        assert manager.max_experts in [1, 2, 3]
        
        if manager.tier == VRAMTier.LOW:
            assert manager.max_experts == 1
        elif manager.tier == VRAMTier.MEDIUM:
            assert manager.max_experts == 2
        else:
            assert manager.max_experts == 3
    
    def test_allocate_deallocate(self, manager):
        """Test allocation and deallocation."""
        # Allocate
        result = manager.allocate("test_adapter", size_mb=10, priority=1)
        assert result is True
        assert "test_adapter" in manager._allocations
        
        # Deallocate
        result = manager.deallocate("test_adapter")
        assert result is True
        assert "test_adapter" not in manager._allocations
    
    def test_duplicate_allocation(self, manager):
        """Test duplicate allocation returns True."""
        manager.allocate("test", size_mb=10)
        result = manager.allocate("test", size_mb=10)
        assert result is True
    
    def test_select_experts_for_vram(self, manager):
        """Test expert selection based on VRAM."""
        expert_scores = [
            ("medical_dora", 0.9),
            ("coding_dora", 0.7),
            ("legal_dora", 0.5),
            ("general_dora", 0.3),
        ]
        
        expert_sizes = {
            "medical_dora": 10,
            "coding_dora": 10,
            "legal_dora": 10,
            "general_dora": 10,
        }
        
        selected = manager.select_experts_for_vram(
            expert_scores=expert_scores,
            expert_sizes=expert_sizes,
        )
        
        assert len(selected) <= manager.max_experts
        assert all(isinstance(s, tuple) and len(s) == 2 for s in selected)
    
    def test_select_experts_fallback(self, manager):
        """Test fallback when confidence is low."""
        expert_scores = [
            ("medical_dora", 0.3),  # Below threshold
        ]
        
        expert_sizes = {"medical_dora": 10, "general_dora": 10}
        
        selected = manager.select_experts_for_vram(
            expert_scores=expert_scores,
            expert_sizes=expert_sizes,
            confidence_threshold=0.5,
        )
        
        # Should include fallback
        expert_ids = [eid for eid, _ in selected]
        assert "general_dora" in expert_ids
    
    def test_get_status(self, manager):
        """Test status reporting."""
        status = manager.get_status()
        
        assert 'tier' in status
        assert 'max_experts' in status
        assert 'total_vram_mb' in status
        assert 'allocations' in status


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

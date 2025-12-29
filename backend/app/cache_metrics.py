"""
Cache Metrics and Monitoring

Tracks cache performance metrics (hit rate, miss rate, etc.)
"""

import time
import logging
from typing import Dict, Optional
from collections import defaultdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CacheMetrics:
    """
    Tracks cache performance metrics.
    
    Metrics tracked:
    - Hits: Number of successful cache retrievals
    - Misses: Number of cache misses
    - Hit rate: Percentage of requests served from cache
    - Average response time: Average time to retrieve from cache
    - Keys by tag: Number of keys per tag
    """
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.total_requests = 0
        self.response_times: List[float] = []
        self.hits_by_key: Dict[str, int] = defaultdict(int)
        self.misses_by_key: Dict[str, int] = defaultdict(int)
        self._start_time = datetime.now()
    
    def record_hit(self, key: str, response_time: float = 0.0):
        """Record a cache hit."""
        self.hits += 1
        self.total_requests += 1
        self.hits_by_key[key] += 1
        if response_time > 0:
            self.response_times.append(response_time)
    
    def record_miss(self, key: str, response_time: float = 0.0):
        """Record a cache miss."""
        self.misses += 1
        self.total_requests += 1
        self.misses_by_key[key] += 1
        if response_time > 0:
            self.response_times.append(response_time)
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.hits / self.total_requests) * 100.0
    
    def get_miss_rate(self) -> float:
        """Get cache miss rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.misses / self.total_requests) * 100.0
    
    def get_avg_response_time(self) -> float:
        """Get average response time in milliseconds."""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times) * 1000  # Convert to ms
    
    def get_stats(self) -> Dict:
        """Get comprehensive cache statistics."""
        uptime = (datetime.now() - self._start_time).total_seconds()
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": self.total_requests,
            "hit_rate": round(self.get_hit_rate(), 2),
            "miss_rate": round(self.get_miss_rate(), 2),
            "avg_response_time_ms": round(self.get_avg_response_time(), 2),
            "uptime_seconds": round(uptime, 2),
            "requests_per_second": round(self.total_requests / uptime if uptime > 0 else 0, 2),
            "top_hit_keys": dict(sorted(self.hits_by_key.items(), key=lambda x: x[1], reverse=True)[:10]),
            "top_miss_keys": dict(sorted(self.misses_by_key.items(), key=lambda x: x[1], reverse=True)[:10]),
        }
    
    def reset(self):
        """Reset all metrics."""
        self.hits = 0
        self.misses = 0
        self.total_requests = 0
        self.response_times.clear()
        self.hits_by_key.clear()
        self.misses_by_key.clear()
        self._start_time = datetime.now()


# Global cache metrics instance
_cache_metrics: Optional[CacheMetrics] = None


def get_cache_metrics() -> CacheMetrics:
    """Get global cache metrics instance."""
    global _cache_metrics
    if _cache_metrics is None:
        _cache_metrics = CacheMetrics()
    return _cache_metrics


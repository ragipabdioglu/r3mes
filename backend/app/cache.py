"""
Redis Cache Manager - Async Redis client wrapper

Provides caching functionality for API responses and session management.
"""

import redis.asyncio as redis
from typing import Optional, Dict, Any, List
import json
import logging
import os
import time

from .cache_metrics import get_cache_metrics

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Redis cache manager for API response caching.
    
    Provides async operations for get/set/delete with TTL support.
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize cache manager.
        
        Args:
            redis_url: Redis connection URL (defaults to REDIS_URL env var)
        """
        # Check for Docker secrets (REDIS_PASSWORD_FILE)
        redis_password_file = os.getenv("REDIS_PASSWORD_FILE")
        redis_password = None
        
        if redis_password_file:
            if os.path.exists(redis_password_file):
                try:
                    with open(redis_password_file, 'r') as f:
                        redis_password = f.read().strip()
                    logger.info("Using Docker secrets for Redis password")
                except Exception as e:
                    logger.warning(f"Failed to read Redis password file: {e}")
        
        # Redis URL - in production, should be set via environment variable
        # Note: localhost is acceptable for Redis if running in same container/pod
        is_production = os.getenv("R3MES_ENV", "development").lower() == "production"
        redis_url_env = redis_url or os.getenv("REDIS_URL")
        
        if redis_password:
            # Construct REDIS_URL from components
            redis_host = os.getenv("REDIS_HOST", "redis")
            redis_port = os.getenv("REDIS_PORT", "6379")
            redis_db = os.getenv("REDIS_DB", "0")
            self.redis_url = f"redis://:{redis_password}@{redis_host}:{redis_port}/{redis_db}"
        elif redis_url_env:
            self.redis_url = redis_url_env
        else:
            if is_production:
                # In production, warn but allow localhost for Redis (common in containerized deployments)
                logger.warning(
                    "REDIS_URL not set in production, using localhost fallback. "
                    "Consider setting REDIS_URL for production deployments."
                )
            self.redis_url = "redis://localhost:6379/0"
        
        self.redis: Optional[redis.Redis] = None
        self._connected = False
    
    async def connect(self):
        """Connect to Redis server with production-optimized settings."""
        if not self._connected:
            try:
                # Production connection pool settings
                is_production = os.getenv("R3MES_ENV", "development").lower() == "production"
                
                connection_kwargs = {
                    "encoding": "utf-8",
                    "decode_responses": True,
                }
                
                if is_production:
                    # Production: connection pooling and retry logic
                    connection_kwargs.update({
                        "max_connections": 50,
                        "retry_on_timeout": True,
                        "socket_keepalive": True,
                        "socket_keepalive_options": {
                            1: 1,  # TCP_KEEPIDLE
                            2: 3,  # TCP_KEEPINTVL
                            3: 3,  # TCP_KEEPCNT
                        },
                        "health_check_interval": 30,
                    })
                
                self.redis = await redis.from_url(
                    self.redis_url,
                    **connection_kwargs
                )
                # Test connection
                await self.redis.ping()
                self._connected = True
                logger.info(f"Connected to Redis at {self.redis_url} (production={'yes' if is_production else 'no'})")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                logger.warning("Cache will be disabled. Continuing without cache...")
                self._connected = False
                self.redis = None
    
    async def close(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
            self._connected = False
            logger.info("Redis connection closed")
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get value from cache with metrics tracking.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value as dict, or None if not found
        """
        if not self._connected:
            await self.connect()
        
        if not self.redis:
            metrics = get_cache_metrics()
            metrics.record_miss(key)
            return None
        
        start_time = time.time()
        try:
            value = await self.redis.get(key)
            response_time = time.time() - start_time
            
            metrics = get_cache_metrics()
            if value:
                metrics.record_hit(key, response_time)
                # Record Prometheus metrics
                from .metrics import record_cache_hit
                record_cache_hit()
                return json.loads(value)
            else:
                metrics.record_miss(key, response_time)
                # Record Prometheus metrics
                from .metrics import record_cache_miss
                record_cache_miss()
                return None
        except Exception as e:
            metrics = get_cache_metrics()
            metrics.record_miss(key)
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Dict[str, Any], ttl: int = 3600) -> bool:
        """
        Set value in cache with TTL.
        
        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl: Time to live in seconds (default: 1 hour)
            
        Returns:
            True if successful, False otherwise
        """
        if not self._connected:
            await self.connect()
        
        if not self.redis:
            return False
        
        try:
            serialized = json.dumps(value)
            await self.redis.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if deleted, False otherwise
        """
        if not self._connected:
            await self.connect()
        
        if not self.redis:
            return False
        
        try:
            result = await self.redis.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key to check
            
        Returns:
            True if exists, False otherwise
        """
        if not self._connected:
            await self.connect()
        
        if not self.redis:
            return False
        
        try:
            result = await self.redis.exists(key)
            return result > 0
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Increment a counter in cache.
        
        Args:
            key: Cache key
            amount: Amount to increment (default: 1)
            
        Returns:
            New value, or None if error
        """
        if not self._connected:
            await self.connect()
        
        if not self.redis:
            return None
        
        try:
            return await self.redis.incrby(key, amount)
        except Exception as e:
            logger.error(f"Cache increment error for key {key}: {e}")
            return None
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.
        
        Returns:
            Dictionary with cache stats (hits, misses, Redis info, etc.)
        """
        if not self._connected:
            await self.connect()
        
        metrics = get_cache_metrics()
        app_metrics = metrics.get_stats()
        
        if not self.redis:
            return {
                "connected": False,
                **app_metrics
            }
        
        try:
            info = await self.redis.info("stats")
            memory_info = await self.redis.info("memory")
            
            return {
                "connected": True,
                "redis_hits": int(info.get("keyspace_hits", 0)),
                "redis_misses": int(info.get("keyspace_misses", 0)),
                "total_keys": await self.redis.dbsize(),
                "memory_used_mb": round(int(memory_info.get("used_memory", 0)) / 1024 / 1024, 2),
                "memory_peak_mb": round(int(memory_info.get("used_memory_peak", 0)) / 1024 / 1024, 2),
                **app_metrics
            }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {
                "connected": False,
                **app_metrics
            }


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


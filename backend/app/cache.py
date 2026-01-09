"""
Cache manager for R3MES Backend using Redis
"""

import os
import json
import logging
from typing import Optional, Any
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class CacheManager:
    """Redis cache manager with async support."""
    
    def __init__(self):
        """Initialize Redis connection."""
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis_client: Optional[redis.Redis] = None
        self._connected = False
    
    async def connect(self):
        """Connect to Redis."""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            await self.redis_client.ping()
            self._connected = True
            logger.info("Connected to Redis")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Cache will be disabled.")
            self._connected = False
    
    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis_client:
            await self.redis_client.close()
            self._connected = False
            logger.info("Disconnected from Redis")
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if not self._connected or not self.redis_client:
            return None
        
        try:
            value = await self.redis_client.get(key)
            if value:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return None
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
        """
        if not self._connected or not self.redis_client:
            return
        
        try:
            # Serialize value if not string
            if not isinstance(value, str):
                value = json.dumps(value)
            
            if ttl:
                await self.redis_client.setex(key, ttl, value)
            else:
                await self.redis_client.set(key, value)
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
    
    async def delete(self, key: str):
        """
        Delete key from cache.
        
        Args:
            key: Cache key
        """
        if not self._connected or not self.redis_client:
            return
        
        try:
            await self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists
        """
        if not self._connected or not self.redis_client:
            return False
        
        try:
            return await self.redis_client.exists(key) > 0
        except Exception as e:
            logger.error(f"Error checking cache key {key}: {e}")
            return False


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get cache manager singleton."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager

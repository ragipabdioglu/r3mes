"""
Cache Optimizer - Multi-Level Caching Strategy

Implements intelligent caching with Redis backend, cache warming,
invalidation strategies, and performance monitoring.
"""

import asyncio
import json
import logging
import pickle
import zlib
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import hashlib
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache levels for multi-level caching."""
    MEMORY = "memory"
    REDIS = "redis"
    DATABASE = "database"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    size_bytes: int = 0
    level: CacheLevel = CacheLevel.MEMORY


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    memory_usage_bytes: int = 0
    total_entries: int = 0
    
    @property
    def hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class CacheSerializer:
    """Handles serialization/deserialization for cache values."""
    
    @staticmethod
    def serialize(value: Any, compress: bool = True) -> bytes:
        """
        Serialize value for caching.
        
        Args:
            value: Value to serialize
            compress: Whether to compress the data
            
        Returns:
            Serialized bytes
        """
        try:
            # Use pickle for Python objects
            data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Compress if requested and beneficial
            if compress and len(data) > 1024:  # Only compress larger data
                compressed = zlib.compress(data, level=6)
                if len(compressed) < len(data) * 0.9:  # Only if significant compression
                    return b'compressed:' + compressed
            
            return b'raw:' + data
            
        except Exception as e:
            logger.error(f"Failed to serialize cache value: {e}")
            # Fallback to JSON for simple types
            try:
                json_data = json.dumps(value, default=str).encode('utf-8')
                return b'json:' + json_data
            except:
                raise ValueError(f"Cannot serialize value of type {type(value)}")
    
    @staticmethod
    def deserialize(data: bytes) -> Any:
        """
        Deserialize cached value.
        
        Args:
            data: Serialized bytes
            
        Returns:
            Deserialized value
        """
        try:
            if data.startswith(b'compressed:'):
                compressed_data = data[11:]  # Remove 'compressed:' prefix
                decompressed = zlib.decompress(compressed_data)
                return pickle.loads(decompressed)
            elif data.startswith(b'raw:'):
                raw_data = data[4:]  # Remove 'raw:' prefix
                return pickle.loads(raw_data)
            elif data.startswith(b'json:'):
                json_data = data[5:]  # Remove 'json:' prefix
                return json.loads(json_data.decode('utf-8'))
            else:
                # Legacy format - assume pickle
                return pickle.loads(data)
                
        except Exception as e:
            logger.error(f"Failed to deserialize cache value: {e}")
            return None


class MemoryCache:
    """In-memory cache with LRU eviction."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        """
        Initialize memory cache.
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []  # For LRU
        self.stats = CacheStats()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        if key not in self.cache:
            self.stats.misses += 1
            return None
        
        entry = self.cache[key]
        
        # Check expiration
        if entry.expires_at and datetime.now() > entry.expires_at:
            await self.delete(key)
            self.stats.misses += 1
            return None
        
        # Update access info
        entry.access_count += 1
        entry.last_accessed = datetime.now()
        
        # Move to end of access order (most recently used)
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        
        self.stats.hits += 1
        return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory cache."""
        try:
            # Calculate size
            serialized = CacheSerializer.serialize(value, compress=False)
            size_bytes = len(serialized)
            
            # Check if we need to evict
            await self._ensure_capacity(size_bytes)
            
            # Create entry
            expires_at = datetime.now() + timedelta(seconds=ttl) if ttl else None
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                expires_at=expires_at,
                size_bytes=size_bytes,
                level=CacheLevel.MEMORY
            )
            
            # Remove old entry if exists
            if key in self.cache:
                old_entry = self.cache[key]
                self.stats.memory_usage_bytes -= old_entry.size_bytes
            
            # Add new entry
            self.cache[key] = entry
            self.stats.memory_usage_bytes += size_bytes
            self.stats.sets += 1
            self.stats.total_entries = len(self.cache)
            
            # Update access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set memory cache entry: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from memory cache."""
        if key not in self.cache:
            return False
        
        entry = self.cache[key]
        self.stats.memory_usage_bytes -= entry.size_bytes
        del self.cache[key]
        
        if key in self.access_order:
            self.access_order.remove(key)
        
        self.stats.deletes += 1
        self.stats.total_entries = len(self.cache)
        return True
    
    async def _ensure_capacity(self, new_entry_size: int):
        """Ensure cache has capacity for new entry."""
        # Check memory limit
        while (self.stats.memory_usage_bytes + new_entry_size > self.max_memory_bytes and 
               self.cache):
            await self._evict_lru()
        
        # Check size limit
        while len(self.cache) >= self.max_size and self.cache:
            await self._evict_lru()
    
    async def _evict_lru(self):
        """Evict least recently used entry."""
        if not self.access_order:
            return
        
        lru_key = self.access_order[0]
        await self.delete(lru_key)
        self.stats.evictions += 1
    
    async def clear(self):
        """Clear all entries."""
        self.cache.clear()
        self.access_order.clear()
        self.stats = CacheStats()


class RedisCache:
    """Redis-based cache with advanced features."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", prefix: str = "r3mes:"):
        """
        Initialize Redis cache.
        
        Args:
            redis_url: Redis connection URL
            prefix: Key prefix for namespacing
        """
        self.redis_url = redis_url
        self.prefix = prefix
        self.redis_client: Optional[redis.Redis] = None
        self.stats = CacheStats()
    
    async def connect(self):
        """Connect to Redis."""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False,  # We handle bytes manually
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Connected to Redis cache")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
    
    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.prefix}{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not self.redis_client:
            await self.connect()
            if not self.redis_client:
                self.stats.misses += 1
                return None
        
        try:
            redis_key = self._make_key(key)
            data = await self.redis_client.get(redis_key)
            
            if data is None:
                self.stats.misses += 1
                return None
            
            # Deserialize
            value = CacheSerializer.deserialize(data)
            if value is None:
                self.stats.misses += 1
                return None
            
            self.stats.hits += 1
            return value
            
        except Exception as e:
            logger.error(f"Failed to get from Redis cache: {e}")
            self.stats.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache."""
        if not self.redis_client:
            await self.connect()
            if not self.redis_client:
                return False
        
        try:
            redis_key = self._make_key(key)
            data = CacheSerializer.serialize(value, compress=True)
            
            if ttl:
                result = await self.redis_client.setex(redis_key, ttl, data)
            else:
                result = await self.redis_client.set(redis_key, data)
            
            if result:
                self.stats.sets += 1
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to set Redis cache entry: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache."""
        if not self.redis_client:
            return False
        
        try:
            redis_key = self._make_key(key)
            result = await self.redis_client.delete(redis_key)
            
            if result > 0:
                self.stats.deletes += 1
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete from Redis cache: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        if not self.redis_client:
            return False
        
        try:
            redis_key = self._make_key(key)
            result = await self.redis_client.exists(redis_key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Failed to check Redis key existence: {e}")
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for existing key."""
        if not self.redis_client:
            return False
        
        try:
            redis_key = self._make_key(key)
            result = await self.redis_client.expire(redis_key, ttl)
            return result
            
        except Exception as e:
            logger.error(f"Failed to set Redis key expiration: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern."""
        if not self.redis_client:
            return 0
        
        try:
            redis_pattern = self._make_key(pattern)
            keys = await self.redis_client.keys(redis_pattern)
            
            if keys:
                result = await self.redis_client.delete(*keys)
                self.stats.deletes += result
                return result
            return 0
            
        except Exception as e:
            logger.error(f"Failed to clear Redis pattern: {e}")
            return 0


class MultiLevelCache:
    """
    Multi-level cache with memory and Redis backends.
    
    Implements intelligent cache hierarchy with automatic promotion/demotion
    and cache warming strategies.
    """
    
    def __init__(
        self,
        memory_cache: Optional[MemoryCache] = None,
        redis_cache: Optional[RedisCache] = None,
        default_ttl: int = 300
    ):
        """
        Initialize multi-level cache.
        
        Args:
            memory_cache: Memory cache instance
            redis_cache: Redis cache instance
            default_ttl: Default TTL in seconds
        """
        self.memory_cache = memory_cache or MemoryCache()
        self.redis_cache = redis_cache
        self.default_ttl = default_ttl
        self.warming_tasks: Dict[str, asyncio.Task] = {}
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from multi-level cache.
        
        Checks memory first, then Redis, with automatic promotion.
        """
        # Try memory cache first
        value = await self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try Redis cache
        if self.redis_cache:
            value = await self.redis_cache.get(key)
            if value is not None:
                # Promote to memory cache
                await self.memory_cache.set(key, value, self.default_ttl)
                return value
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in multi-level cache.
        
        Sets in both memory and Redis caches.
        """
        ttl = ttl or self.default_ttl
        
        # Set in memory cache
        memory_success = await self.memory_cache.set(key, value, ttl)
        
        # Set in Redis cache
        redis_success = True
        if self.redis_cache:
            redis_success = await self.redis_cache.set(key, value, ttl)
        
        return memory_success or redis_success
    
    async def delete(self, key: str) -> bool:
        """Delete value from all cache levels."""
        memory_success = await self.memory_cache.delete(key)
        
        redis_success = True
        if self.redis_cache:
            redis_success = await self.redis_cache.delete(key)
        
        return memory_success or redis_success
    
    async def warm_cache(self, keys: List[str], loader_fn: Callable[[List[str]], List[Any]]):
        """
        Warm cache with data for given keys.
        
        Args:
            keys: Keys to warm
            loader_fn: Function to load data for keys
        """
        try:
            # Load data
            values = await loader_fn(keys)
            
            # Cache loaded values
            for key, value in zip(keys, values):
                if value is not None:
                    await self.set(key, value)
            
            logger.info(f"Warmed cache for {len([v for v in values if v is not None])} keys")
            
        except Exception as e:
            logger.error(f"Failed to warm cache: {e}")
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern."""
        total_deleted = 0
        
        # Clear from Redis (supports patterns)
        if self.redis_cache:
            total_deleted += await self.redis_cache.clear_pattern(pattern)
        
        # Clear from memory (need to check each key)
        memory_keys_to_delete = []
        for key in self.memory_cache.cache.keys():
            if self._matches_pattern(key, pattern):
                memory_keys_to_delete.append(key)
        
        for key in memory_keys_to_delete:
            if await self.memory_cache.delete(key):
                total_deleted += 1
        
        return total_deleted
    
    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Check if key matches pattern (simple wildcard support)."""
        if '*' not in pattern:
            return key == pattern
        
        # Simple wildcard matching
        parts = pattern.split('*')
        if len(parts) == 2:
            prefix, suffix = parts
            return key.startswith(prefix) and key.endswith(suffix)
        
        # More complex patterns would need proper regex
        return False
    
    def get_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all cache levels."""
        stats = {"memory": self.memory_cache.stats}
        
        if self.redis_cache:
            stats["redis"] = self.redis_cache.stats
        
        return stats


class CacheManager:
    """
    High-level cache manager with intelligent caching strategies.
    
    Provides decorators and utilities for easy cache integration.
    """
    
    def __init__(self, cache: MultiLevelCache):
        """
        Initialize cache manager.
        
        Args:
            cache: Multi-level cache instance
        """
        self.cache = cache
        self.key_generators: Dict[str, Callable] = {}
    
    def register_key_generator(self, name: str, generator: Callable) -> None:
        """Register a key generator function."""
        self.key_generators[name] = generator
    
    def generate_key(self, generator_name: str, *args, **kwargs) -> str:
        """Generate cache key using registered generator."""
        if generator_name not in self.key_generators:
            # Fallback to simple key generation
            key_parts = [generator_name] + [str(arg) for arg in args]
            if kwargs:
                key_parts.extend([f"{k}:{v}" for k, v in sorted(kwargs.items())])
            return ":".join(key_parts)
        
        return self.key_generators[generator_name](*args, **kwargs)
    
    def cached(
        self,
        key_generator: str,
        ttl: Optional[int] = None,
        cache_none: bool = False
    ):
        """
        Decorator for caching function results.
        
        Args:
            key_generator: Name of key generator to use
            ttl: Cache TTL in seconds
            cache_none: Whether to cache None results
        """
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self.generate_key(key_generator, *args, **kwargs)
                
                # Try to get from cache
                cached_result = await self.cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Cache result
                if result is not None or cache_none:
                    await self.cache.set(cache_key, result, ttl)
                
                return result
            
            return wrapper
        return decorator
    
    async def get_or_set(
        self,
        key: str,
        loader_fn: Callable,
        ttl: Optional[int] = None
    ) -> Any:
        """
        Get value from cache or load and set if not found.
        
        Args:
            key: Cache key
            loader_fn: Function to load value if not cached
            ttl: Cache TTL in seconds
            
        Returns:
            Cached or loaded value
        """
        # Try cache first
        value = await self.cache.get(key)
        if value is not None:
            return value
        
        # Load value
        value = await loader_fn()
        
        # Cache if not None
        if value is not None:
            await self.cache.set(key, value, ttl)
        
        return value
    
    async def invalidate_user_cache(self, wallet_address: str):
        """Invalidate all cache entries for a user."""
        patterns = [
            f"user:{wallet_address}:*",
            f"credits:{wallet_address}",
            f"api_keys:{wallet_address}:*",
            f"reservations:{wallet_address}:*"
        ]
        
        total_deleted = 0
        for pattern in patterns:
            deleted = await self.cache.invalidate_pattern(pattern)
            total_deleted += deleted
        
        logger.debug(f"Invalidated {total_deleted} cache entries for user {wallet_address}")
        return total_deleted
    
    async def warm_user_cache(self, wallet_addresses: List[str], database_instance):
        """Warm cache for multiple users."""
        from .batch_loader import DatabaseBatchLoader
        
        batch_loader = DatabaseBatchLoader(database_instance)
        
        # Warm user data
        users = await batch_loader.load_users(wallet_addresses)
        for wallet, user in zip(wallet_addresses, users):
            if user:
                user_key = f"user:{wallet}"
                await self.cache.set(user_key, user)
        
        logger.info(f"Warmed user cache for {len([u for u in users if u])} users")


# Default cache instances
def create_default_cache_manager(redis_url: Optional[str] = None) -> CacheManager:
    """Create default cache manager with standard configuration."""
    memory_cache = MemoryCache(max_size=1000, max_memory_mb=100)
    
    redis_cache = None
    if redis_url:
        redis_cache = RedisCache(redis_url)
    
    multi_cache = MultiLevelCache(memory_cache, redis_cache)
    manager = CacheManager(multi_cache)
    
    # Register common key generators
    manager.register_key_generator(
        "user",
        lambda wallet_address: f"user:{wallet_address}"
    )
    manager.register_key_generator(
        "user_credits",
        lambda wallet_address: f"credits:{wallet_address}"
    )
    manager.register_key_generator(
        "api_keys",
        lambda wallet_address: f"api_keys:{wallet_address}"
    )
    
    return manager
"""
Cache Invalidation Strategy

Provides tag-based and pattern-based cache invalidation.
"""

import asyncio
import logging
from typing import List, Set, Optional, Pattern, Dict
import re

from .cache import get_cache_manager

logger = logging.getLogger(__name__)


class CacheInvalidator:
    """
    Manages cache invalidation using tags and patterns.
    
    Supports:
    - Tag-based invalidation (invalidate all keys with a tag)
    - Pattern-based invalidation (invalidate keys matching a pattern)
    - Key-based invalidation (invalidate specific keys)
    """
    
    def __init__(self):
        self.cache = get_cache_manager()
        # Tag registry: tag -> set of keys
        self._tag_registry: Dict[str, Set[str]] = {}
        # Key registry: key -> set of tags
        self._key_tags: Dict[str, Set[str]] = {}
    
    async def invalidate_by_tag(self, tag: str) -> int:
        """
        Invalidate all keys with a specific tag.
        
        Args:
            tag: Tag to invalidate
            
        Returns:
            Number of keys invalidated
        """
        if tag not in self._tag_registry:
            logger.debug(f"No keys found for tag: {tag}")
            return 0
        
        keys = self._tag_registry[tag].copy()
        invalidated = 0
        
        for key in keys:
            if await self.cache.delete(key):
                invalidated += 1
                # Remove from tag registry
                self._tag_registry[tag].discard(key)
                # Remove from key tags
                if key in self._key_tags:
                    self._key_tags[key].discard(tag)
                    if not self._key_tags[key]:
                        del self._key_tags[key]
        
        if not self._tag_registry[tag]:
            del self._tag_registry[tag]
        
        logger.info(f"Invalidated {invalidated} keys for tag: {tag}")
        return invalidated
    
    async def invalidate_by_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching a pattern.
        
        Args:
            pattern: Pattern to match (supports wildcards: *, ?)
            
        Returns:
            Number of keys invalidated
        """
        if not self.cache.redis:
            return 0
        
        # Convert pattern to regex
        regex_pattern = pattern.replace("*", ".*").replace("?", ".")
        compiled_pattern = re.compile(regex_pattern)
        
        invalidated = 0
        
        # Scan all keys (use SCAN for production to avoid blocking)
        cursor = 0
        while True:
            cursor, keys = await self.cache.redis.scan(cursor, match="*", count=100)
            
            for key in keys:
                if compiled_pattern.match(key):
                    if await self.cache.delete(key):
                        invalidated += 1
                        # Clean up tag registry
                        if key in self._key_tags:
                            for tag in self._key_tags[key]:
                                if tag in self._tag_registry:
                                    self._tag_registry[tag].discard(key)
                            del self._key_tags[key]
            
            if cursor == 0:
                break
        
        logger.info(f"Invalidated {invalidated} keys matching pattern: {pattern}")
        return invalidated
    
    async def invalidate_key(self, key: str) -> bool:
        """
        Invalidate a specific key.
        
        Args:
            key: Key to invalidate
            
        Returns:
            True if invalidated, False otherwise
        """
        result = await self.cache.delete(key)
        
        # Clean up tag registry
        if key in self._key_tags:
            for tag in self._key_tags[key]:
                if tag in self._tag_registry:
                    self._tag_registry[tag].discard(key)
            del self._key_tags[key]
        
        return result
    
    def register_tag(self, key: str, tags: List[str]):
        """
        Register tags for a cache key.
        
        Args:
            key: Cache key
            tags: List of tags
        """
        if key not in self._key_tags:
            self._key_tags[key] = set()
        
        for tag in tags:
            self._key_tags[key].add(tag)
            if tag not in self._tag_registry:
                self._tag_registry[tag] = set()
            self._tag_registry[tag].add(key)
    
    async def invalidate_user_cache(self, wallet_address: str):
        """
        Invalidate all cache entries for a specific user.
        
        Args:
            wallet_address: User wallet address
        """
        patterns = [
            f"user:info:{wallet_address}",
            f"user:credits:{wallet_address}",
            f"miner:stats:{wallet_address}",
            f"miner:earnings:{wallet_address}",
            f"miner:hashrate:{wallet_address}",
        ]
        
        invalidated = 0
        for pattern in patterns:
            if await self.cache.delete(pattern):
                invalidated += 1
        
        # Also invalidate by tag
        await self.invalidate_by_tag(f"user:{wallet_address}")
        
        logger.info(f"Invalidated {invalidated} cache entries for user: {wallet_address}")
    
    async def invalidate_blockchain_cache(self):
        """Invalidate all blockchain-related cache entries."""
        patterns = [
            "block:*",
            "blockchain:*",
            "network:*",
            "rpc:*",
        ]
        
        invalidated = 0
        for pattern in patterns:
            invalidated += await self.invalidate_by_pattern(pattern)
        
        logger.info(f"Invalidated {invalidated} blockchain cache entries")


# Global cache invalidator instance
_cache_invalidator: Optional[CacheInvalidator] = None


def get_cache_invalidator() -> CacheInvalidator:
    """Get global cache invalidator instance."""
    global _cache_invalidator
    if _cache_invalidator is None:
        _cache_invalidator = CacheInvalidator()
    return _cache_invalidator


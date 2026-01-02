"""
Cache Invalidation Strategy

Provides intelligent cache invalidation patterns and strategies.
"""

import os
import logging
from typing import Dict, List, Set, Optional, Any
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class InvalidationStrategy(Enum):
    """Cache invalidation strategies."""
    TTL = "ttl"  # Time-based expiration
    DEPENDENCY = "dependency"  # Dependency-based invalidation
    TAG = "tag"  # Tag-based invalidation
    PATTERN = "pattern"  # Pattern-based invalidation
    MANUAL = "manual"  # Manual invalidation


class CacheInvalidationManager:
    """
    Manages cache invalidation strategies and dependencies.
    
    Provides intelligent cache invalidation based on data relationships
    and business logic requirements.
    """
    
    def __init__(self, cache_manager):
        """
        Initialize cache invalidation manager.
        
        Args:
            cache_manager: Cache manager instance
        """
        self.cache = cache_manager
        self.dependencies: Dict[str, Set[str]] = {}  # key -> dependent keys
        self.tags: Dict[str, Set[str]] = {}  # tag -> keys with that tag
        self.key_tags: Dict[str, Set[str]] = {}  # key -> tags
        self.patterns: Dict[str, str] = {}  # pattern -> description
        
        # Configuration
        self.enable_dependency_tracking = os.getenv("CACHE_DEPENDENCY_TRACKING", "true").lower() == "true"
        self.enable_tag_tracking = os.getenv("CACHE_TAG_TRACKING", "true").lower() == "true"
        self.default_ttl = int(os.getenv("CACHE_DEFAULT_TTL", "300"))  # 5 minutes
    
    async def set_with_dependencies(
        self,
        key: str,
        value: Any,
        ttl: int = None,
        dependencies: List[str] = None,
        tags: List[str] = None
    ) -> bool:
        """
        Set cache value with dependency and tag tracking.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            dependencies: List of keys this cache depends on
            tags: List of tags for this cache entry
            
        Returns:
            True if set successfully
        """
        try:
            # Set the cache value
            ttl = ttl or self.default_ttl
            success = await self.cache.set(key, value, ttl=ttl)
            
            if not success:
                return False
            
            # Track dependencies
            if self.enable_dependency_tracking and dependencies:
                await self._track_dependencies(key, dependencies)
            
            # Track tags
            if self.enable_tag_tracking and tags:
                await self._track_tags(key, tags)
            
            logger.debug(f"Cache set with tracking: {key} (deps: {dependencies}, tags: {tags})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set cache with dependencies: {e}")
            return False
    
    async def invalidate_by_dependency(self, dependency_key: str) -> int:
        """
        Invalidate all cache entries that depend on a specific key.
        
        Args:
            dependency_key: Key that changed
            
        Returns:
            Number of invalidated entries
        """
        if not self.enable_dependency_tracking:
            return 0
        
        try:
            dependent_keys = self.dependencies.get(dependency_key, set())
            
            if not dependent_keys:
                return 0
            
            # Invalidate all dependent keys
            invalidated_count = 0
            for key in dependent_keys:
                if await self.cache.delete(key):
                    invalidated_count += 1
                    # Clean up tracking
                    await self._cleanup_key_tracking(key)
            
            # Clean up dependency tracking
            if dependency_key in self.dependencies:
                del self.dependencies[dependency_key]
            
            logger.info(f"Invalidated {invalidated_count} cache entries dependent on {dependency_key}")
            return invalidated_count
            
        except Exception as e:
            logger.error(f"Failed to invalidate by dependency {dependency_key}: {e}")
            return 0
    
    async def invalidate_by_tag(self, tag: str) -> int:
        """
        Invalidate all cache entries with a specific tag.
        
        Args:
            tag: Tag to invalidate
            
        Returns:
            Number of invalidated entries
        """
        if not self.enable_tag_tracking:
            return 0
        
        try:
            tagged_keys = self.tags.get(tag, set())
            
            if not tagged_keys:
                return 0
            
            # Invalidate all tagged keys
            invalidated_count = 0
            for key in tagged_keys:
                if await self.cache.delete(key):
                    invalidated_count += 1
                    # Clean up tracking
                    await self._cleanup_key_tracking(key)
            
            # Clean up tag tracking
            if tag in self.tags:
                del self.tags[tag]
            
            logger.info(f"Invalidated {invalidated_count} cache entries with tag {tag}")
            return invalidated_count
            
        except Exception as e:
            logger.error(f"Failed to invalidate by tag {tag}: {e}")
            return 0
    
    async def invalidate_by_pattern(self, pattern: str) -> int:
        """
        Invalidate all cache entries matching a pattern.
        
        Args:
            pattern: Pattern to match (supports wildcards)
            
        Returns:
            Number of invalidated entries
        """
        try:
            # Get all keys matching pattern
            matching_keys = await self._get_keys_by_pattern(pattern)
            
            if not matching_keys:
                return 0
            
            # Invalidate matching keys
            invalidated_count = 0
            for key in matching_keys:
                if await self.cache.delete(key):
                    invalidated_count += 1
                    # Clean up tracking
                    await self._cleanup_key_tracking(key)
            
            logger.info(f"Invalidated {invalidated_count} cache entries matching pattern {pattern}")
            return invalidated_count
            
        except Exception as e:
            logger.error(f"Failed to invalidate by pattern {pattern}: {e}")
            return 0
    
    async def invalidate_user_cache(self, wallet_address: str) -> int:
        """
        Invalidate all cache entries for a specific user.
        
        Args:
            wallet_address: User wallet address
            
        Returns:
            Number of invalidated entries
        """
        try:
            # Define user-related cache patterns
            user_patterns = [
                f"user_info:{wallet_address}",
                f"user_stats:{wallet_address}",
                f"api_keys:{wallet_address}",
                f"api_key_stats:{wallet_address}",
                f"chat_stats:{wallet_address}",
                f"miner_stats:{wallet_address}",
                f"earnings:{wallet_address}*",
                f"hashrate:{wallet_address}*"
            ]
            
            total_invalidated = 0
            for pattern in user_patterns:
                count = await self.invalidate_by_pattern(pattern)
                total_invalidated += count
            
            logger.info(f"Invalidated {total_invalidated} cache entries for user {wallet_address}")
            return total_invalidated
            
        except Exception as e:
            logger.error(f"Failed to invalidate user cache for {wallet_address}: {e}")
            return 0
    
    async def invalidate_system_cache(self) -> int:
        """
        Invalidate system-wide cache entries.
        
        Returns:
            Number of invalidated entries
        """
        try:
            # Define system cache patterns
            system_patterns = [
                "network_stats*",
                "block_*",
                "leaderboard*",
                "serving_nodes*",
                "available_models*",
                "system_metrics*"
            ]
            
            total_invalidated = 0
            for pattern in system_patterns:
                count = await self.invalidate_by_pattern(pattern)
                total_invalidated += count
            
            logger.info(f"Invalidated {total_invalidated} system cache entries")
            return total_invalidated
            
        except Exception as e:
            logger.error(f"Failed to invalidate system cache: {e}")
            return 0
    
    async def _track_dependencies(self, key: str, dependencies: List[str]):
        """Track cache dependencies."""
        for dep in dependencies:
            if dep not in self.dependencies:
                self.dependencies[dep] = set()
            self.dependencies[dep].add(key)
    
    async def _track_tags(self, key: str, tags: List[str]):
        """Track cache tags."""
        # Track tags -> keys mapping
        for tag in tags:
            if tag not in self.tags:
                self.tags[tag] = set()
            self.tags[tag].add(key)
        
        # Track key -> tags mapping
        self.key_tags[key] = set(tags)
    
    async def _cleanup_key_tracking(self, key: str):
        """Clean up tracking information for a key."""
        # Clean up tag tracking
        if key in self.key_tags:
            tags = self.key_tags[key]
            for tag in tags:
                if tag in self.tags:
                    self.tags[tag].discard(key)
                    if not self.tags[tag]:  # Remove empty tag sets
                        del self.tags[tag]
            del self.key_tags[key]
        
        # Clean up dependency tracking
        for dep_key, dependent_keys in list(self.dependencies.items()):
            dependent_keys.discard(key)
            if not dependent_keys:  # Remove empty dependency sets
                del self.dependencies[dep_key]
    
    async def _get_keys_by_pattern(self, pattern: str) -> List[str]:
        """
        Get cache keys matching a pattern.
        
        This is a simplified implementation. In production, you might want
        to use Redis SCAN or similar for better performance.
        """
        try:
            # This is a placeholder - implement based on your cache backend
            # For Redis: use SCAN command
            # For in-memory: iterate through keys
            
            # Simple wildcard matching for demonstration
            import fnmatch
            
            # Get all keys (this is expensive - use SCAN in production)
            all_keys = await self._get_all_cache_keys()
            
            matching_keys = []
            for key in all_keys:
                if fnmatch.fnmatch(key, pattern):
                    matching_keys.append(key)
            
            return matching_keys
            
        except Exception as e:
            logger.error(f"Failed to get keys by pattern {pattern}: {e}")
            return []
    
    async def _get_all_cache_keys(self) -> List[str]:
        """
        Get all cache keys.
        
        This is a placeholder implementation.
        """
        try:
            # This should be implemented based on your cache backend
            # For Redis: use SCAN
            # For in-memory: return list of keys
            
            # Placeholder - return empty list
            return []
            
        except Exception as e:
            logger.error(f"Failed to get all cache keys: {e}")
            return []
    
    def get_invalidation_stats(self) -> Dict[str, Any]:
        """
        Get cache invalidation statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "dependency_tracking_enabled": self.enable_dependency_tracking,
            "tag_tracking_enabled": self.enable_tag_tracking,
            "tracked_dependencies": len(self.dependencies),
            "tracked_tags": len(self.tags),
            "tracked_keys": len(self.key_tags),
            "default_ttl": self.default_ttl
        }


# Global cache invalidation manager instance
_invalidation_manager: Optional[CacheInvalidationManager] = None


def get_cache_invalidation_manager(cache_manager) -> CacheInvalidationManager:
    """
    Get or create global cache invalidation manager.
    
    Args:
        cache_manager: Cache manager instance
        
    Returns:
        Cache invalidation manager instance
    """
    global _invalidation_manager
    
    if _invalidation_manager is None:
        _invalidation_manager = CacheInvalidationManager(cache_manager)
    
    return _invalidation_manager


# Convenience functions for common invalidation patterns

async def invalidate_user_data(wallet_address: str, cache_manager) -> int:
    """
    Invalidate all cache data for a user.
    
    Args:
        wallet_address: User wallet address
        cache_manager: Cache manager instance
        
    Returns:
        Number of invalidated entries
    """
    invalidation_manager = get_cache_invalidation_manager(cache_manager)
    return await invalidation_manager.invalidate_user_cache(wallet_address)


async def invalidate_system_data(cache_manager) -> int:
    """
    Invalidate system-wide cache data.
    
    Args:
        cache_manager: Cache manager instance
        
    Returns:
        Number of invalidated entries
    """
    invalidation_manager = get_cache_invalidation_manager(cache_manager)
    return await invalidation_manager.invalidate_system_cache()


async def invalidate_by_tags(tags: List[str], cache_manager) -> int:
    """
    Invalidate cache entries by tags.
    
    Args:
        tags: List of tags to invalidate
        cache_manager: Cache manager instance
        
    Returns:
        Number of invalidated entries
    """
    invalidation_manager = get_cache_invalidation_manager(cache_manager)
    
    total_invalidated = 0
    for tag in tags:
        count = await invalidation_manager.invalidate_by_tag(tag)
        total_invalidated += count
    
    return total_invalidated
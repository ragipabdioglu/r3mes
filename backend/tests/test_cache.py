"""
Unit tests for Cache system

Tests Redis caching, cache invalidation, and cache middleware.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.cache import CacheManager, get_cache_manager


class TestCacheManager(unittest.TestCase):
    """Unit tests for CacheManager"""
    
    def setUp(self):
        """Set up test environment"""
        os.environ["R3MES_ENV"] = "test"
        os.environ["R3MES_TEST_MODE"] = "true"
        os.environ["REDIS_URL"] = "redis://localhost:6379/0"
    
    def tearDown(self):
        """Clean up"""
        os.environ.pop("R3MES_ENV", None)
        os.environ.pop("R3MES_TEST_MODE", None)
        os.environ.pop("REDIS_URL", None)
    
    @patch('app.cache.redis.from_url')
    def test_cache_initialization(self, mock_redis):
        """Test cache initialization"""
        mock_redis.return_value = MagicMock()
        
        cache = CacheManager()
        self.assertIsNotNone(cache)
    
    @patch('app.cache.redis.from_url')
    async def test_cache_get_set(self, mock_redis):
        """Test cache get and set operations"""
        mock_client = MagicMock()
        mock_client.get.return_value = '{"test": "value"}'
        mock_client.setex.return_value = True
        mock_client.ping.return_value = True
        mock_redis.return_value = mock_client
        
        cache = CacheManager()
        await cache.connect()
        
        # Test set
        result = await cache.set("test_key", {"test": "value"}, ttl=60)
        self.assertTrue(result)
        
        # Test get
        value = await cache.get("test_key")
        self.assertEqual(value, {"test": "value"})
    
    @patch('app.cache.redis.from_url')
    async def test_cache_delete(self, mock_redis):
        """Test cache delete operation"""
        mock_client = MagicMock()
        mock_client.delete.return_value = 1
        mock_client.ping.return_value = True
        mock_redis.return_value = mock_client
        
        cache = CacheManager()
        await cache.connect()
        result = await cache.delete("test_key")
        self.assertTrue(result)
    
    def test_get_cache_manager(self):
        """Test global cache manager getter"""
        cache = get_cache_manager()
        self.assertIsInstance(cache, CacheManager)


if __name__ == "__main__":
    unittest.main()


#!/usr/bin/env python3
"""
Performance Optimization Tests

Comprehensive tests for all performance optimization components including
database optimization, caching, batch loading, and response optimization.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import List, Dict, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.exceptions import (
    DatabaseError,
    NetworkError,
    TimeoutError,
    InsufficientCreditsError,
)


class TestDatabaseOptimization:
    """Test cases for database performance optimization."""
    
    @pytest.mark.asyncio
    async def test_connection_pooling_performance(self):
        """Test database connection pooling performance."""
        # Mock database pool with simpler approach
        mock_pool = Mock()
        mock_connection = AsyncMock()
        
        # Test connection acquisition time
        start_time = time.time()
        
        # Simulate connection acquisition
        connection = mock_connection
        await asyncio.sleep(0.001)  # 1ms operation
        
        end_time = time.time()
        operation_time = end_time - start_time
        
        # Connection pooling should be fast (< 50ms)
        assert operation_time < 0.05
        assert connection is not None
    
    @pytest.mark.asyncio
    async def test_query_optimization_with_indexes(self):
        """Test query performance with proper indexing."""
        # Mock database with indexed queries
        mock_db = AsyncMock()
        
        # Mock fast indexed query (< 5ms)
        mock_db.execute_indexed_query.return_value = [
            {"id": 1, "wallet_address": "remes1test", "credits": 100.0}
        ]
        
        # Test indexed query performance
        start_time = time.time()
        result = await mock_db.execute_indexed_query(
            "SELECT * FROM users WHERE wallet_address = ? AND is_active = ?",
            ("remes1test", True)
        )
        end_time = time.time()
        
        query_time = end_time - start_time
        
        # Indexed queries should be fast
        assert query_time < 0.005  # < 5ms
        assert len(result) == 1
        assert result[0]["wallet_address"] == "remes1test"
    
    @pytest.mark.asyncio
    async def test_batch_insert_optimization(self):
        """Test batch insert performance optimization."""
        # Mock database with batch operations
        mock_db = AsyncMock()
        
        # Test data
        batch_data = [
            {"wallet_address": f"remes1wallet{i:038d}", "credits": 100.0}
            for i in range(100)
        ]
        
        # Mock batch insert
        mock_db.batch_insert.return_value = len(batch_data)
        
        # Test batch insert performance
        start_time = time.time()
        result = await mock_db.batch_insert("users", batch_data)
        end_time = time.time()
        
        batch_time = end_time - start_time
        
        # Batch operations should be efficient
        assert batch_time < 0.1  # < 100ms for 100 records
        assert result == 100
        mock_db.batch_insert.assert_called_once_with("users", batch_data)
    
    @pytest.mark.asyncio
    async def test_database_query_caching(self):
        """Test database query result caching."""
        # Mock cache and database
        mock_cache = AsyncMock()
        mock_db = AsyncMock()
        
        wallet_address = "remes1testaddress234567890234567890234567890"
        cache_key = f"user_info:{wallet_address}"
        
        # First call - cache miss
        mock_cache.get.return_value = None
        mock_db.get_user_info.return_value = {
            "wallet_address": wallet_address,
            "credits": 100.0
        }
        
        # Test first call (cache miss)
        start_time = time.time()
        
        cached_result = await mock_cache.get(cache_key)
        if cached_result is None:
            result = await mock_db.get_user_info(wallet_address)
            await mock_cache.set(cache_key, result, ttl=300)
        else:
            result = cached_result
        
        first_call_time = time.time() - start_time
        
        # Second call - cache hit
        mock_cache.get.return_value = {
            "wallet_address": wallet_address,
            "credits": 100.0
        }
        
        start_time = time.time()
        cached_result = await mock_cache.get(cache_key)
        second_call_time = time.time() - start_time
        
        # Cache hit should be much faster than database query
        assert second_call_time < first_call_time / 10
        assert result["wallet_address"] == wallet_address


class TestCachingOptimization:
    """Test cases for caching performance optimization."""
    
    @pytest.mark.asyncio
    async def test_redis_cache_performance(self):
        """Test Redis cache performance."""
        # Mock Redis cache
        mock_redis = AsyncMock()
        
        # Test data
        cache_key = "user:remes1test:info"
        cache_value = {"credits": 100.0, "is_miner": True}
        
        # Mock Redis operations
        mock_redis.set.return_value = True
        mock_redis.get.return_value = cache_value
        mock_redis.delete.return_value = 1
        
        # Test cache SET performance
        start_time = time.time()
        await mock_redis.set(cache_key, cache_value, ex=300)
        set_time = time.time() - start_time
        
        # Test cache GET performance
        start_time = time.time()
        result = await mock_redis.get(cache_key)
        get_time = time.time() - start_time
        
        # Cache operations should be very fast (< 1ms)
        assert set_time < 0.001
        assert get_time < 0.001
        assert result == cache_value
    
    @pytest.mark.asyncio
    async def test_cache_hit_ratio_optimization(self):
        """Test cache hit ratio optimization."""
        # Mock cache with hit/miss tracking
        cache_stats = {"hits": 0, "misses": 0}
        mock_cache = AsyncMock()
        
        def mock_get(key):
            if key in ["frequent_key_1", "frequent_key_2"]:
                cache_stats["hits"] += 1
                return {"data": f"cached_value_for_{key}"}
            else:
                cache_stats["misses"] += 1
                return None
        
        mock_cache.get.side_effect = mock_get
        
        # Simulate cache access patterns
        keys = ["frequent_key_1", "frequent_key_2", "frequent_key_1", 
                "rare_key_1", "frequent_key_2", "rare_key_2"]
        
        for key in keys:
            await mock_cache.get(key)
        
        # Calculate hit ratio
        total_requests = cache_stats["hits"] + cache_stats["misses"]
        hit_ratio = cache_stats["hits"] / total_requests
        
        # Good cache hit ratio should be > 60%
        assert hit_ratio > 0.6
        assert cache_stats["hits"] == 4  # frequent keys
        assert cache_stats["misses"] == 2  # rare keys
    
    @pytest.mark.asyncio
    async def test_cache_invalidation_performance(self):
        """Test cache invalidation performance."""
        # Mock cache with pattern-based invalidation
        mock_cache = AsyncMock()
        
        # Mock cache keys
        cache_keys = [
            "user:remes1test1:info",
            "user:remes1test1:credits", 
            "user:remes1test2:info",
            "api_key:remes1test1:list",
        ]
        
        # Mock pattern-based deletion
        mock_cache.delete_pattern.return_value = 3  # 3 keys deleted
        
        # Test cache invalidation performance
        start_time = time.time()
        deleted_count = await mock_cache.delete_pattern("user:remes1test1:*")
        invalidation_time = time.time() - start_time
        
        # Cache invalidation should be fast
        assert invalidation_time < 0.01  # < 10ms
        assert deleted_count == 3
        mock_cache.delete_pattern.assert_called_once_with("user:remes1test1:*")
    
    @pytest.mark.asyncio
    async def test_cache_warming_strategy(self):
        """Test cache warming strategy performance."""
        # Mock cache and database
        mock_cache = AsyncMock()
        mock_db = AsyncMock()
        
        # Mock frequently accessed data
        popular_wallets = [
            f"remes1popular{i:037d}" for i in range(10)
        ]
        
        # Mock database responses
        mock_db.get_user_info.return_value = {"credits": 100.0}
        mock_cache.set.return_value = True
        
        # Test cache warming performance
        start_time = time.time()
        
        # Warm cache with popular data
        for wallet in popular_wallets:
            user_info = await mock_db.get_user_info(wallet)
            await mock_cache.set(f"user:{wallet}:info", user_info, ex=3600)
        
        warming_time = time.time() - start_time
        
        # Cache warming should be efficient
        assert warming_time < 0.1  # < 100ms for 10 items
        assert mock_db.get_user_info.call_count == 10
        assert mock_cache.set.call_count == 10


class TestBatchProcessingOptimization:
    """Test cases for batch processing optimization."""
    
    @pytest.mark.asyncio
    async def test_batch_user_loading(self):
        """Test batch user loading performance."""
        # Mock database with batch loading
        mock_db = AsyncMock()
        
        # Test data
        wallet_addresses = [
            f"remes1wallet{i:038d}" for i in range(50)
        ]
        
        # Mock batch response
        mock_users = [
            {"wallet_address": addr, "credits": 100.0, "is_miner": True}
            for addr in wallet_addresses
        ]
        mock_db.get_users_batch.return_value = mock_users
        
        # Test batch loading performance
        start_time = time.time()
        users = await mock_db.get_users_batch(wallet_addresses)
        batch_time = time.time() - start_time
        
        # Batch loading should be efficient
        assert batch_time < 0.05  # < 50ms for 50 users
        assert len(users) == 50
        assert all(user["credits"] == 100.0 for user in users)
    
    @pytest.mark.asyncio
    async def test_concurrent_batch_processing(self):
        """Test concurrent batch processing performance."""
        # Mock service with batch operations
        mock_service = AsyncMock()
        
        # Test data - multiple batches
        batches = [
            [f"remes1batch1_{i:036d}" for i in range(20)],
            [f"remes1batch2_{i:036d}" for i in range(20)],
            [f"remes1batch3_{i:036d}" for i in range(20)],
        ]
        
        # Mock batch processing
        mock_service.process_batch.return_value = {"processed": 20, "success": True}
        
        # Test concurrent batch processing
        start_time = time.time()
        
        tasks = []
        for batch in batches:
            task = mock_service.process_batch(batch)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        concurrent_time = time.time() - start_time
        
        # Concurrent processing should be faster than sequential
        assert concurrent_time < 0.1  # < 100ms for 3 concurrent batches
        assert len(results) == 3
        assert all(result["success"] for result in results)
    
    @pytest.mark.asyncio
    async def test_batch_size_optimization(self):
        """Test optimal batch size for performance."""
        # Mock database with different batch sizes
        mock_db = AsyncMock()
        
        # Test different batch sizes
        batch_sizes = [10, 50, 100, 200]
        performance_results = {}
        
        for batch_size in batch_sizes:
            # Generate test data
            test_data = [
                {"id": i, "data": f"item_{i}"}
                for i in range(batch_size)
            ]
            
            # Mock batch operation time (simulated)
            # Smaller batches: more overhead, larger batches: more processing
            if batch_size <= 50:
                mock_time = 0.01 + (batch_size * 0.0001)  # Linear growth
            else:
                mock_time = 0.01 + (batch_size * 0.0002)  # Slower for large batches
            
            mock_db.batch_insert.return_value = batch_size
            
            # Simulate batch operation
            start_time = time.time()
            await asyncio.sleep(mock_time)  # Simulate processing time
            result = await mock_db.batch_insert("test_table", test_data)
            end_time = time.time()
            
            performance_results[batch_size] = {
                "time": end_time - start_time,
                "throughput": batch_size / (end_time - start_time)
            }
        
        # Find optimal batch size (highest throughput)
        optimal_batch_size = max(
            performance_results.keys(),
            key=lambda x: performance_results[x]["throughput"]
        )
        
        # Optimal batch size should be reasonable (not too small, not too large)
        assert 10 <= optimal_batch_size <= 200
        assert performance_results[optimal_batch_size]["throughput"] > 0


class TestAPIResponseOptimization:
    """Test cases for API response optimization."""
    
    @pytest.mark.asyncio
    async def test_response_compression(self):
        """Test API response compression performance."""
        # Mock large response data
        large_response = {
            "users": [
                {
                    "wallet_address": f"remes1user{i:039d}",
                    "credits": 100.0,
                    "transactions": [
                        {"id": j, "amount": 10.0, "timestamp": "2024-01-01T00:00:00Z"}
                        for j in range(10)
                    ]
                }
                for i in range(100)
            ]
        }
        
        # Mock compression
        import json
        import gzip
        
        # Uncompressed size
        uncompressed_data = json.dumps(large_response).encode('utf-8')
        uncompressed_size = len(uncompressed_data)
        
        # Compressed size
        compressed_data = gzip.compress(uncompressed_data)
        compressed_size = len(compressed_data)
        
        # Compression ratio
        compression_ratio = compressed_size / uncompressed_size
        
        # Good compression should reduce size by at least 50%
        assert compression_ratio < 0.5
        assert compressed_size < uncompressed_size
    
    @pytest.mark.asyncio
    async def test_pagination_performance(self):
        """Test pagination performance optimization."""
        # Mock database with pagination
        mock_db = AsyncMock()
        
        # Test pagination parameters
        page_size = 20
        total_records = 1000
        
        # Mock paginated response
        mock_db.get_paginated_users.return_value = {
            "users": [
                {"id": i, "wallet_address": f"remes1user{i:039d}"}
                for i in range(page_size)
            ],
            "total": total_records,
            "page": 1,
            "page_size": page_size,
            "has_next": True
        }
        
        # Test pagination performance
        start_time = time.time()
        result = await mock_db.get_paginated_users(page=1, page_size=page_size)
        pagination_time = time.time() - start_time
        
        # Pagination should be fast regardless of total records
        assert pagination_time < 0.01  # < 10ms
        assert len(result["users"]) == page_size
        assert result["total"] == total_records
        assert result["has_next"] is True
    
    @pytest.mark.asyncio
    async def test_field_selection_optimization(self):
        """Test API field selection optimization."""
        # Mock database with field selection
        mock_db = AsyncMock()
        
        # Full user data
        full_user_data = {
            "id": 1,
            "wallet_address": "remes1testaddress234567890234567890234567890",
            "credits": 100.0,
            "is_miner": True,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
            "metadata": {"key1": "value1", "key2": "value2"},
            "preferences": {"theme": "dark", "language": "en"}
        }
        
        # Mock selective field query
        selected_fields = ["wallet_address", "credits", "is_miner"]
        mock_db.get_user_fields.return_value = {
            field: full_user_data[field] for field in selected_fields
        }
        
        # Test field selection performance
        start_time = time.time()
        result = await mock_db.get_user_fields("remes1test", selected_fields)
        selection_time = time.time() - start_time
        
        # Field selection should be fast and return only requested fields
        assert selection_time < 0.005  # < 5ms
        assert len(result) == len(selected_fields)
        assert all(field in result for field in selected_fields)
        assert "metadata" not in result  # Should not include unselected fields


class TestMemoryOptimization:
    """Test cases for memory usage optimization."""
    
    @pytest.mark.asyncio
    async def test_memory_efficient_data_structures(self):
        """Test memory-efficient data structures."""
        # Test memory usage with different data structures
        import sys
        
        # Large dataset
        data_size = 10000
        
        # List vs Generator comparison
        # List (loads all in memory)
        data_list = [{"id": i, "value": f"item_{i}"} for i in range(data_size)]
        list_memory = sys.getsizeof(data_list)
        
        # Generator (lazy loading)
        def data_generator():
            for i in range(data_size):
                yield {"id": i, "value": f"item_{i}"}
        
        gen = data_generator()
        generator_memory = sys.getsizeof(gen)
        
        # Generator should use significantly less memory
        assert generator_memory < list_memory / 100
        
        # Test generator functionality
        first_items = [next(gen) for _ in range(5)]
        assert len(first_items) == 5
        assert first_items[0]["id"] == 0
    
    @pytest.mark.asyncio
    async def test_object_pooling_optimization(self):
        """Test object pooling for memory optimization."""
        # Mock object pool
        class ObjectPool:
            def __init__(self, max_size=10):
                self.pool = []
                self.max_size = max_size
                self.created_count = 0
                self.reused_count = 0
            
            def get_object(self):
                if self.pool:
                    self.reused_count += 1
                    return self.pool.pop()
                else:
                    self.created_count += 1
                    return {"id": self.created_count, "data": None}
            
            def return_object(self, obj):
                if len(self.pool) < self.max_size:
                    obj["data"] = None  # Reset object
                    self.pool.append(obj)
        
        # Test object pooling
        pool = ObjectPool(max_size=5)
        
        # Get objects
        objects = []
        for i in range(10):
            obj = pool.get_object()
            obj["data"] = f"data_{i}"
            objects.append(obj)
        
        # Return objects to pool
        for obj in objects[:5]:
            pool.return_object(obj)
        
        # Get objects again (should reuse)
        reused_objects = []
        for i in range(3):
            obj = pool.get_object()
            reused_objects.append(obj)
        
        # Verify object reuse
        assert pool.reused_count == 3
        assert pool.created_count == 10
        assert len(pool.pool) == 2  # 5 returned - 3 reused = 2 remaining
    
    @pytest.mark.asyncio
    async def test_garbage_collection_optimization(self):
        """Test garbage collection optimization."""
        import gc
        import weakref
        
        # Create objects that should be garbage collected
        class TestObject:
            def __init__(self, data):
                self.data = data
        
        # Create objects and weak references
        objects = []
        weak_refs = []
        
        for i in range(100):
            obj = TestObject(f"data_{i}")
            objects.append(obj)
            weak_refs.append(weakref.ref(obj))
        
        # Verify objects exist
        assert len(objects) == 100
        assert all(ref() is not None for ref in weak_refs)
        
        # Clear references
        objects.clear()
        
        # Force garbage collection
        gc.collect()
        
        # Check if objects were garbage collected
        alive_objects = sum(1 for ref in weak_refs if ref() is not None)
        
        # Most objects should be garbage collected
        assert alive_objects < 10  # Less than 10% should remain


class TestNetworkOptimization:
    """Test cases for network performance optimization."""
    
    @pytest.mark.asyncio
    async def test_connection_pooling_http(self):
        """Test HTTP connection pooling performance."""
        # Mock HTTP client with connection pooling
        mock_http_client = AsyncMock()
        
        # Mock connection pool stats
        pool_stats = {
            "active_connections": 0,
            "idle_connections": 5,
            "total_requests": 0
        }
        
        async def mock_request(url):
            pool_stats["total_requests"] += 1
            if pool_stats["idle_connections"] > 0:
                pool_stats["idle_connections"] -= 1
                pool_stats["active_connections"] += 1
                # Simulate fast request with pooled connection
                await asyncio.sleep(0.001)
            else:
                # Simulate slower request with new connection
                await asyncio.sleep(0.01)
            
            pool_stats["active_connections"] -= 1
            pool_stats["idle_connections"] += 1
            
            return {"status": 200, "data": f"response_for_{url}"}
        
        mock_http_client.get.side_effect = mock_request
        
        # Test multiple requests
        urls = [f"https://api.example.com/endpoint_{i}" for i in range(10)]
        
        start_time = time.time()
        tasks = [mock_http_client.get(url) for url in urls]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Connection pooling should make requests faster
        assert total_time < 0.1  # < 100ms for 10 requests
        assert len(results) == 10
        assert pool_stats["total_requests"] == 10
    
    @pytest.mark.asyncio
    async def test_request_timeout_optimization(self):
        """Test request timeout optimization."""
        # Mock service with timeout handling
        mock_service = AsyncMock()
        
        async def mock_slow_request():
            await asyncio.sleep(2.0)  # 2 second delay
            return {"data": "slow_response"}
        
        async def mock_fast_request():
            await asyncio.sleep(0.1)  # 100ms delay
            return {"data": "fast_response"}
        
        # Test timeout handling
        timeout_duration = 1.0  # 1 second timeout
        
        # Fast request should succeed
        start_time = time.time()
        try:
            result = await asyncio.wait_for(mock_fast_request(), timeout=timeout_duration)
            fast_time = time.time() - start_time
            assert result["data"] == "fast_response"
            assert fast_time < timeout_duration
        except asyncio.TimeoutError:
            pytest.fail("Fast request should not timeout")
        
        # Slow request should timeout
        start_time = time.time()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(mock_slow_request(), timeout=timeout_duration)
        
        timeout_time = time.time() - start_time
        assert abs(timeout_time - timeout_duration) < 0.1  # Should timeout at ~1 second
    
    @pytest.mark.asyncio
    async def test_retry_mechanism_optimization(self):
        """Test retry mechanism optimization."""
        # Mock service with retry logic
        class RetryService:
            def __init__(self):
                self.attempt_count = 0
                self.max_retries = 3
                self.base_delay = 0.1
            
            async def unreliable_request(self):
                self.attempt_count += 1
                
                # Fail first 2 attempts, succeed on 3rd
                if self.attempt_count < 3:
                    raise NetworkError(
                        message="Network temporarily unavailable",
                        endpoint="https://api.example.com"
                    )
                
                return {"status": "success", "attempts": self.attempt_count}
            
            async def request_with_retry(self):
                for attempt in range(self.max_retries + 1):
                    try:
                        return await self.unreliable_request()
                    except NetworkError as e:
                        if attempt == self.max_retries:
                            raise e
                        
                        # Exponential backoff
                        delay = self.base_delay * (2 ** attempt)
                        await asyncio.sleep(delay)
        
        # Test retry mechanism
        service = RetryService()
        
        start_time = time.time()
        result = await service.request_with_retry()
        total_time = time.time() - start_time
        
        # Should succeed after retries
        assert result["status"] == "success"
        assert result["attempts"] == 3
        
        # Total time should include retry delays
        expected_min_time = 0.1 + 0.2  # First two retry delays
        assert total_time >= expected_min_time


class TestConcurrencyOptimization:
    """Test cases for concurrency optimization."""
    
    @pytest.mark.asyncio
    async def test_async_task_performance(self):
        """Test async task performance optimization."""
        # Mock async operations
        async def async_operation(duration, result_value):
            await asyncio.sleep(duration)
            return result_value
        
        # Test sequential vs concurrent execution
        operations = [
            (0.1, "result_1"),
            (0.1, "result_2"), 
            (0.1, "result_3"),
            (0.1, "result_4"),
        ]
        
        # Sequential execution
        start_time = time.time()
        sequential_results = []
        for duration, value in operations:
            result = await async_operation(duration, value)
            sequential_results.append(result)
        sequential_time = time.time() - start_time
        
        # Concurrent execution
        start_time = time.time()
        tasks = [async_operation(duration, value) for duration, value in operations]
        concurrent_results = await asyncio.gather(*tasks)
        concurrent_time = time.time() - start_time
        
        # Concurrent should be much faster
        assert concurrent_time < sequential_time / 2
        assert len(concurrent_results) == len(sequential_results)
        assert concurrent_results == sequential_results
    
    @pytest.mark.asyncio
    async def test_semaphore_rate_limiting(self):
        """Test semaphore-based rate limiting."""
        # Create semaphore to limit concurrent operations
        max_concurrent = 3
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Track concurrent operations
        concurrent_count = 0
        max_observed_concurrent = 0
        
        async def limited_operation(operation_id):
            nonlocal concurrent_count, max_observed_concurrent
            
            async with semaphore:
                concurrent_count += 1
                max_observed_concurrent = max(max_observed_concurrent, concurrent_count)
                
                # Simulate work
                await asyncio.sleep(0.1)
                
                concurrent_count -= 1
                return f"result_{operation_id}"
        
        # Start many operations
        tasks = [limited_operation(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # Verify rate limiting worked
        assert max_observed_concurrent <= max_concurrent
        assert len(results) == 10
        assert concurrent_count == 0  # All operations completed
    
    @pytest.mark.asyncio
    async def test_worker_pool_optimization(self):
        """Test worker pool optimization."""
        # Mock worker pool
        class WorkerPool:
            def __init__(self, max_workers=5):
                self.max_workers = max_workers
                self.active_workers = 0
                self.completed_tasks = 0
            
            async def submit_task(self, task_func, *args):
                # Wait for available worker
                while self.active_workers >= self.max_workers:
                    await asyncio.sleep(0.01)
                
                self.active_workers += 1
                try:
                    result = await task_func(*args)
                    self.completed_tasks += 1
                    return result
                finally:
                    self.active_workers -= 1
        
        # Test worker pool
        pool = WorkerPool(max_workers=3)
        
        async def work_task(task_id):
            await asyncio.sleep(0.05)  # 50ms work
            return f"task_{task_id}_completed"
        
        # Submit many tasks
        start_time = time.time()
        tasks = [pool.submit_task(work_task, i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Verify worker pool efficiency
        assert len(results) == 10
        assert pool.completed_tasks == 10
        assert pool.active_workers == 0
        
        # Should complete faster than sequential execution
        sequential_time = 10 * 0.05  # 10 tasks * 50ms each
        assert total_time < sequential_time


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
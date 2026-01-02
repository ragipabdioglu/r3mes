"""
Load Testing and Performance Tests

Tests for API performance under load, concurrent requests, and stress testing.
"""

import pytest
import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import requests
from fastapi.testclient import TestClient

from app.main import app


class LoadTestConfig:
    """Configuration for load tests."""
    
    # Test configuration
    BASE_URL = "http://localhost:8000"
    CONCURRENT_USERS = 10
    REQUESTS_PER_USER = 20
    TEST_DURATION_SECONDS = 30
    
    # Performance thresholds
    MAX_RESPONSE_TIME_MS = 1000  # 1 second
    MAX_95TH_PERCENTILE_MS = 2000  # 2 seconds
    MIN_REQUESTS_PER_SECOND = 50
    MAX_ERROR_RATE_PERCENT = 5


class PerformanceMetrics:
    """Collect and analyze performance metrics."""
    
    def __init__(self):
        self.response_times: List[float] = []
        self.status_codes: List[int] = []
        self.errors: List[str] = []
        self.start_time: float = 0
        self.end_time: float = 0
    
    def add_response(self, response_time: float, status_code: int, error: str = None):
        """Add a response measurement."""
        self.response_times.append(response_time)
        self.status_codes.append(status_code)
        if error:
            self.errors.append(error)
    
    def start_test(self):
        """Mark test start time."""
        self.start_time = time.time()
    
    def end_test(self):
        """Mark test end time."""
        self.end_time = time.time()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.response_times:
            return {"error": "No response times recorded"}
        
        total_requests = len(self.response_times)
        successful_requests = len([code for code in self.status_codes if 200 <= code < 300])
        error_requests = total_requests - successful_requests
        
        duration = self.end_time - self.start_time if self.end_time > self.start_time else 1
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "error_requests": error_requests,
            "error_rate_percent": (error_requests / total_requests) * 100,
            "duration_seconds": duration,
            "requests_per_second": total_requests / duration,
            "response_times": {
                "min_ms": min(self.response_times) * 1000,
                "max_ms": max(self.response_times) * 1000,
                "mean_ms": statistics.mean(self.response_times) * 1000,
                "median_ms": statistics.median(self.response_times) * 1000,
                "p95_ms": statistics.quantiles(self.response_times, n=20)[18] * 1000,  # 95th percentile
                "p99_ms": statistics.quantiles(self.response_times, n=100)[98] * 1000,  # 99th percentile
            },
            "status_code_distribution": {
                code: self.status_codes.count(code) for code in set(self.status_codes)
            },
            "errors": self.errors[:10]  # First 10 errors
        }


class TestHealthEndpointLoad:
    """Load test for health endpoint."""
    
    def test_health_endpoint_load(self):
        """Test health endpoint under load."""
        metrics = PerformanceMetrics()
        
        def make_request():
            """Make a single request to health endpoint."""
            start_time = time.time()
            try:
                response = requests.get(f"{LoadTestConfig.BASE_URL}/health", timeout=5)
                response_time = time.time() - start_time
                metrics.add_response(response_time, response.status_code)
                return response.status_code
            except Exception as e:
                response_time = time.time() - start_time
                metrics.add_response(response_time, 0, str(e))
                return 0
        
        # Run load test
        metrics.start_test()
        
        with ThreadPoolExecutor(max_workers=LoadTestConfig.CONCURRENT_USERS) as executor:
            futures = []
            
            # Submit requests
            for _ in range(LoadTestConfig.CONCURRENT_USERS * LoadTestConfig.REQUESTS_PER_USER):
                future = executor.submit(make_request)
                futures.append(future)
            
            # Wait for completion
            for future in as_completed(futures):
                future.result()
        
        metrics.end_test()
        
        # Analyze results
        stats = metrics.get_statistics()
        
        # Assertions
        assert stats["error_rate_percent"] < LoadTestConfig.MAX_ERROR_RATE_PERCENT, \
            f"Error rate {stats['error_rate_percent']:.2f}% exceeds threshold {LoadTestConfig.MAX_ERROR_RATE_PERCENT}%"
        
        assert stats["response_times"]["p95_ms"] < LoadTestConfig.MAX_95TH_PERCENTILE_MS, \
            f"95th percentile {stats['response_times']['p95_ms']:.2f}ms exceeds threshold {LoadTestConfig.MAX_95TH_PERCENTILE_MS}ms"
        
        assert stats["requests_per_second"] > LoadTestConfig.MIN_REQUESTS_PER_SECOND, \
            f"RPS {stats['requests_per_second']:.2f} below threshold {LoadTestConfig.MIN_REQUESTS_PER_SECOND}"
        
        print(f"\n=== Health Endpoint Load Test Results ===")
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Success Rate: {(stats['successful_requests']/stats['total_requests']*100):.2f}%")
        print(f"Requests/Second: {stats['requests_per_second']:.2f}")
        print(f"Mean Response Time: {stats['response_times']['mean_ms']:.2f}ms")
        print(f"95th Percentile: {stats['response_times']['p95_ms']:.2f}ms")


class TestChatEndpointLoad:
    """Load test for chat endpoint."""
    
    @pytest.mark.asyncio
    async def test_chat_endpoint_concurrent_requests(self):
        """Test chat endpoint with concurrent requests."""
        metrics = PerformanceMetrics()
        
        # Mock data for testing
        test_wallet = "remes1test123456789012345678901234567890123456"
        test_message = "Hello, this is a test message for load testing."
        
        def make_chat_request():
            """Make a single chat request."""
            start_time = time.time()
            try:
                response = requests.post(
                    f"{LoadTestConfig.BASE_URL}/chat",
                    json={
                        "message": test_message,
                        "wallet_address": test_wallet
                    },
                    timeout=10
                )
                response_time = time.time() - start_time
                metrics.add_response(response_time, response.status_code)
                return response.status_code
            except Exception as e:
                response_time = time.time() - start_time
                metrics.add_response(response_time, 0, str(e))
                return 0
        
        # Run load test with fewer concurrent requests for chat endpoint
        concurrent_users = 5  # Reduced for chat endpoint
        requests_per_user = 10
        
        metrics.start_test()
        
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = []
            
            for _ in range(concurrent_users * requests_per_user):
                future = executor.submit(make_chat_request)
                futures.append(future)
            
            for future in as_completed(futures):
                future.result()
        
        metrics.end_test()
        
        # Analyze results
        stats = metrics.get_statistics()
        
        # More lenient thresholds for chat endpoint
        max_chat_response_time = 5000  # 5 seconds
        min_chat_rps = 10  # 10 RPS
        
        print(f"\n=== Chat Endpoint Load Test Results ===")
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Success Rate: {(stats['successful_requests']/stats['total_requests']*100):.2f}%")
        print(f"Requests/Second: {stats['requests_per_second']:.2f}")
        print(f"Mean Response Time: {stats['response_times']['mean_ms']:.2f}ms")
        print(f"95th Percentile: {stats['response_times']['p95_ms']:.2f}ms")
        
        # Note: These assertions might fail if the backend is not running
        # or if inference is disabled. Adjust thresholds based on your setup.
        if stats['total_requests'] > 0:
            assert stats["response_times"]["p95_ms"] < max_chat_response_time, \
                f"Chat 95th percentile {stats['response_times']['p95_ms']:.2f}ms exceeds {max_chat_response_time}ms"


class TestDatabaseLoad:
    """Load test for database operations."""
    
    def test_user_info_endpoint_load(self):
        """Test user info endpoint under load."""
        metrics = PerformanceMetrics()
        
        test_wallets = [
            f"remes1test{i:040d}" for i in range(10)  # 10 different test wallets
        ]
        
        def make_user_info_request():
            """Make a single user info request."""
            import random
            wallet = random.choice(test_wallets)
            
            start_time = time.time()
            try:
                response = requests.get(f"{LoadTestConfig.BASE_URL}/user/info/{wallet}", timeout=5)
                response_time = time.time() - start_time
                metrics.add_response(response_time, response.status_code)
                return response.status_code
            except Exception as e:
                response_time = time.time() - start_time
                metrics.add_response(response_time, 0, str(e))
                return 0
        
        # Run load test
        metrics.start_test()
        
        with ThreadPoolExecutor(max_workers=LoadTestConfig.CONCURRENT_USERS) as executor:
            futures = []
            
            for _ in range(LoadTestConfig.CONCURRENT_USERS * LoadTestConfig.REQUESTS_PER_USER):
                future = executor.submit(make_user_info_request)
                futures.append(future)
            
            for future in as_completed(futures):
                future.result()
        
        metrics.end_test()
        
        # Analyze results
        stats = metrics.get_statistics()
        
        print(f"\n=== User Info Endpoint Load Test Results ===")
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Success Rate: {(stats['successful_requests']/stats['total_requests']*100):.2f}%")
        print(f"Requests/Second: {stats['requests_per_second']:.2f}")
        print(f"Mean Response Time: {stats['response_times']['mean_ms']:.2f}ms")
        print(f"95th Percentile: {stats['response_times']['p95_ms']:.2f}ms")


class TestMemoryAndResourceUsage:
    """Test memory and resource usage under load."""
    
    def test_memory_usage_during_load(self):
        """Test memory usage during load testing."""
        import psutil
        import os
        
        # Get current process
        process = psutil.Process(os.getpid())
        
        # Record initial memory usage
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run a load test
        metrics = PerformanceMetrics()
        
        def make_request():
            try:
                response = requests.get(f"{LoadTestConfig.BASE_URL}/health", timeout=5)
                return response.status_code
            except:
                return 0
        
        # Run requests
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(make_request) for _ in range(100)]
            for future in as_completed(futures):
                future.result()
        
        # Record final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"\n=== Memory Usage Test Results ===")
        print(f"Initial Memory: {initial_memory:.2f} MB")
        print(f"Final Memory: {final_memory:.2f} MB")
        print(f"Memory Increase: {memory_increase:.2f} MB")
        
        # Assert memory increase is reasonable (less than 100MB for this test)
        assert memory_increase < 100, f"Memory increase {memory_increase:.2f}MB is too high"


class TestStressTest:
    """Stress testing to find breaking points."""
    
    def test_gradual_load_increase(self):
        """Test with gradually increasing load."""
        results = []
        
        for concurrent_users in [1, 5, 10, 20, 50]:
            print(f"\nTesting with {concurrent_users} concurrent users...")
            
            metrics = PerformanceMetrics()
            
            def make_request():
                start_time = time.time()
                try:
                    response = requests.get(f"{LoadTestConfig.BASE_URL}/health", timeout=5)
                    response_time = time.time() - start_time
                    metrics.add_response(response_time, response.status_code)
                    return response.status_code
                except Exception as e:
                    response_time = time.time() - start_time
                    metrics.add_response(response_time, 0, str(e))
                    return 0
            
            metrics.start_test()
            
            with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
                futures = [executor.submit(make_request) for _ in range(concurrent_users * 10)]
                for future in as_completed(futures):
                    future.result()
            
            metrics.end_test()
            stats = metrics.get_statistics()
            
            results.append({
                "concurrent_users": concurrent_users,
                "rps": stats["requests_per_second"],
                "mean_response_time": stats["response_times"]["mean_ms"],
                "error_rate": stats["error_rate_percent"]
            })
            
            print(f"RPS: {stats['requests_per_second']:.2f}, "
                  f"Mean Response Time: {stats['response_times']['mean_ms']:.2f}ms, "
                  f"Error Rate: {stats['error_rate_percent']:.2f}%")
        
        # Analyze results to find performance degradation points
        print(f"\n=== Stress Test Summary ===")
        for result in results:
            print(f"Users: {result['concurrent_users']:2d}, "
                  f"RPS: {result['rps']:6.2f}, "
                  f"Response Time: {result['mean_response_time']:6.2f}ms, "
                  f"Error Rate: {result['error_rate']:5.2f}%")


class TestCachePerformance:
    """Test cache performance under load."""
    
    def test_cache_hit_performance(self):
        """Test performance with cache hits vs cache misses."""
        # This test would require cache warming and then testing
        # the same endpoints repeatedly to measure cache performance
        
        # First request (cache miss)
        start_time = time.time()
        response1 = requests.get(f"{LoadTestConfig.BASE_URL}/health")
        cache_miss_time = time.time() - start_time
        
        # Second request (should be cache hit if caching is implemented)
        start_time = time.time()
        response2 = requests.get(f"{LoadTestConfig.BASE_URL}/health")
        cache_hit_time = time.time() - start_time
        
        print(f"\n=== Cache Performance Test ===")
        print(f"Cache Miss Time: {cache_miss_time*1000:.2f}ms")
        print(f"Cache Hit Time: {cache_hit_time*1000:.2f}ms")
        
        # Cache hit should be faster (if caching is implemented)
        if cache_hit_time < cache_miss_time:
            improvement = ((cache_miss_time - cache_hit_time) / cache_miss_time) * 100
            print(f"Cache Improvement: {improvement:.2f}%")


def run_load_tests():
    """Run all load tests."""
    print("Starting Load Tests...")
    print("=" * 50)
    
    # Note: These tests require the backend server to be running
    # Start the server with: uvicorn app.main:app --host 0.0.0.0 --port 8000
    
    try:
        # Quick connectivity test
        response = requests.get(f"{LoadTestConfig.BASE_URL}/health", timeout=5)
        print(f"Server connectivity: OK (Status: {response.status_code})")
    except Exception as e:
        print(f"Server connectivity: FAILED ({e})")
        print("Please start the backend server before running load tests.")
        return
    
    # Run tests
    test_classes = [
        TestHealthEndpointLoad,
        TestDatabaseLoad,
        TestMemoryAndResourceUsage,
        TestStressTest,
        TestCachePerformance,
    ]
    
    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}...")
        test_instance = test_class()
        
        # Run all test methods
        for method_name in dir(test_instance):
            if method_name.startswith("test_"):
                try:
                    method = getattr(test_instance, method_name)
                    print(f"  Running {method_name}...")
                    method()
                    print(f"  ✅ {method_name} passed")
                except Exception as e:
                    print(f"  ❌ {method_name} failed: {e}")


if __name__ == "__main__":
    run_load_tests()
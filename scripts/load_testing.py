#!/usr/bin/env python3
"""
R3MES Load Testing Suite

Comprehensive load testing for R3MES production deployment.
Tests API performance, database load, and system scalability.
"""

import asyncio
import aiohttp
import time
import json
import statistics
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class LoadTestConfig:
    """Load test configuration."""
    base_url: str = "http://localhost:8000"
    concurrent_users: int = 100
    test_duration_minutes: int = 10
    ramp_up_seconds: int = 60
    api_key: Optional[str] = None
    wallet_address: str = "remes1test123456789012345678901234567890"


@dataclass
class TestResult:
    """Individual test result."""
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    timestamp: datetime
    error: Optional[str] = None
    payload_size: int = 0


class LoadTester:
    """Comprehensive load testing for R3MES API."""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results: List[TestResult] = []
        self.active_sessions = 0
        self.start_time = None
        self.end_time = None
    
    async def create_session(self) -> aiohttp.ClientSession:
        """Create HTTP session with proper headers."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "R3MES-LoadTester/1.0"
        }
        
        if self.config.api_key:
            headers["X-API-Key"] = self.config.api_key
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        return aiohttp.ClientSession(
            headers=headers,
            timeout=timeout,
            connector=aiohttp.TCPConnector(limit=200, limit_per_host=50)
        )
    
    async def test_health_endpoint(self, session: aiohttp.ClientSession) -> TestResult:
        """Test health endpoint."""
        start_time = time.time()
        
        try:
            async with session.get(f"{self.config.base_url}/health") as response:
                end_time = time.time()
                response_time_ms = (end_time - start_time) * 1000
                
                return TestResult(
                    endpoint="/health",
                    method="GET",
                    status_code=response.status,
                    response_time_ms=response_time_ms,
                    timestamp=datetime.now(),
                    payload_size=len(await response.text())
                )
        except Exception as e:
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            return TestResult(
                endpoint="/health",
                method="GET",
                status_code=0,
                response_time_ms=response_time_ms,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    async def test_user_info_endpoint(self, session: aiohttp.ClientSession) -> TestResult:
        """Test user info endpoint."""
        start_time = time.time()
        
        try:
            url = f"{self.config.base_url}/user/info/{self.config.wallet_address}"
            async with session.get(url) as response:
                end_time = time.time()
                response_time_ms = (end_time - start_time) * 1000
                
                return TestResult(
                    endpoint="/user/info/{wallet}",
                    method="GET",
                    status_code=response.status,
                    response_time_ms=response_time_ms,
                    timestamp=datetime.now(),
                    payload_size=len(await response.text())
                )
        except Exception as e:
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            return TestResult(
                endpoint="/user/info/{wallet}",
                method="GET",
                status_code=0,
                response_time_ms=response_time_ms,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    async def test_chat_endpoint(self, session: aiohttp.ClientSession) -> TestResult:
        """Test chat endpoint with inference."""
        start_time = time.time()
        
        payload = {
            "message": "Hello, this is a load test message",
            "wallet_address": self.config.wallet_address
        }
        
        try:
            async with session.post(
                f"{self.config.base_url}/chat",
                json=payload
            ) as response:
                end_time = time.time()
                response_time_ms = (end_time - start_time) * 1000
                
                # For streaming responses, read the full stream
                response_text = ""
                if response.headers.get('content-type', '').startswith('text/plain'):
                    async for chunk in response.content.iter_chunked(1024):
                        response_text += chunk.decode('utf-8')
                else:
                    response_text = await response.text()
                
                return TestResult(
                    endpoint="/chat",
                    method="POST",
                    status_code=response.status,
                    response_time_ms=response_time_ms,
                    timestamp=datetime.now(),
                    payload_size=len(response_text)
                )
        except Exception as e:
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            return TestResult(
                endpoint="/chat",
                method="POST",
                status_code=0,
                response_time_ms=response_time_ms,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    async def test_network_stats_endpoint(self, session: aiohttp.ClientSession) -> TestResult:
        """Test network stats endpoint."""
        start_time = time.time()
        
        try:
            async with session.get(f"{self.config.base_url}/network/stats") as response:
                end_time = time.time()
                response_time_ms = (end_time - start_time) * 1000
                
                return TestResult(
                    endpoint="/network/stats",
                    method="GET",
                    status_code=response.status,
                    response_time_ms=response_time_ms,
                    timestamp=datetime.now(),
                    payload_size=len(await response.text())
                )
        except Exception as e:
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            return TestResult(
                endpoint="/network/stats",
                method="GET",
                status_code=0,
                response_time_ms=response_time_ms,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    async def simulate_user_session(self, user_id: int, session: aiohttp.ClientSession) -> List[TestResult]:
        """Simulate a complete user session."""
        results = []
        
        # Typical user flow
        test_scenarios = [
            self.test_health_endpoint,
            self.test_user_info_endpoint,
            self.test_network_stats_endpoint,
            self.test_chat_endpoint,  # Most resource-intensive
        ]
        
        for scenario in test_scenarios:
            try:
                result = await scenario(session)
                results.append(result)
                
                # Small delay between requests (realistic user behavior)
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"User {user_id} scenario failed: {e}")
        
        return results
    
    async def run_load_test(self) -> Dict:
        """Run comprehensive load test."""
        logger.info(f"Starting load test with {self.config.concurrent_users} concurrent users")
        logger.info(f"Test duration: {self.config.test_duration_minutes} minutes")
        logger.info(f"Ramp-up time: {self.config.ramp_up_seconds} seconds")
        
        self.start_time = datetime.now()
        end_time = self.start_time + timedelta(minutes=self.config.test_duration_minutes)
        
        # Create semaphore to limit concurrent connections
        semaphore = asyncio.Semaphore(self.config.concurrent_users)
        
        async def user_worker(user_id: int):
            """Worker function for individual user simulation."""
            async with semaphore:
                session = await self.create_session()
                try:
                    while datetime.now() < end_time:
                        self.active_sessions += 1
                        user_results = await self.simulate_user_session(user_id, session)
                        self.results.extend(user_results)
                        self.active_sessions -= 1
                        
                        # Wait before next session (simulate user think time)
                        await asyncio.sleep(5)
                        
                finally:
                    await session.close()
        
        # Start users with ramp-up
        tasks = []
        ramp_up_delay = self.config.ramp_up_seconds / self.config.concurrent_users
        
        for user_id in range(self.config.concurrent_users):
            task = asyncio.create_task(user_worker(user_id))
            tasks.append(task)
            
            # Ramp-up delay
            if user_id < self.config.concurrent_users - 1:
                await asyncio.sleep(ramp_up_delay)
        
        # Wait for all tasks to complete or timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.test_duration_minutes * 60 + 60
            )
        except asyncio.TimeoutError:
            logger.warning("Load test timed out, cancelling remaining tasks")
            for task in tasks:
                task.cancel()
        
        self.end_time = datetime.now()
        
        # Generate results
        return self.analyze_results()
    
    def analyze_results(self) -> Dict:
        """Analyze load test results."""
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Filter successful requests
        successful_results = [r for r in self.results if r.status_code == 200]
        failed_results = [r for r in self.results if r.status_code != 200]
        
        # Calculate response time statistics
        response_times = [r.response_time_ms for r in successful_results]
        
        if response_times:
            response_time_stats = {
                "min": min(response_times),
                "max": max(response_times),
                "mean": statistics.mean(response_times),
                "median": statistics.median(response_times),
                "p95": statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times),
                "p99": statistics.quantiles(response_times, n=100)[98] if len(response_times) > 100 else max(response_times)
            }
        else:
            response_time_stats = {}
        
        # Calculate throughput
        test_duration_seconds = (self.end_time - self.start_time).total_seconds()
        total_requests = len(self.results)
        successful_requests = len(successful_results)
        
        # Group results by endpoint
        endpoint_stats = {}
        for result in self.results:
            endpoint = result.endpoint
            if endpoint not in endpoint_stats:
                endpoint_stats[endpoint] = {
                    "total": 0,
                    "successful": 0,
                    "failed": 0,
                    "response_times": []
                }
            
            endpoint_stats[endpoint]["total"] += 1
            if result.status_code == 200:
                endpoint_stats[endpoint]["successful"] += 1
                endpoint_stats[endpoint]["response_times"].append(result.response_time_ms)
            else:
                endpoint_stats[endpoint]["failed"] += 1
        
        # Calculate endpoint-specific stats
        for endpoint, stats in endpoint_stats.items():
            if stats["response_times"]:
                stats["avg_response_time"] = statistics.mean(stats["response_times"])
                stats["p95_response_time"] = statistics.quantiles(stats["response_times"], n=20)[18] if len(stats["response_times"]) > 20 else max(stats["response_times"])
            else:
                stats["avg_response_time"] = 0
                stats["p95_response_time"] = 0
            
            stats["success_rate"] = (stats["successful"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        
        # System resource usage during test
        system_stats = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
        
        return {
            "test_config": {
                "concurrent_users": self.config.concurrent_users,
                "test_duration_minutes": self.config.test_duration_minutes,
                "base_url": self.config.base_url
            },
            "summary": {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": len(failed_results),
                "success_rate": (successful_requests / total_requests) * 100 if total_requests > 0 else 0,
                "requests_per_second": total_requests / test_duration_seconds if test_duration_seconds > 0 else 0,
                "test_duration_seconds": test_duration_seconds
            },
            "response_time_stats": response_time_stats,
            "endpoint_stats": endpoint_stats,
            "system_stats": system_stats,
            "errors": [{"endpoint": r.endpoint, "error": r.error, "timestamp": r.timestamp.isoformat()} 
                      for r in failed_results if r.error][:10]  # First 10 errors
        }
    
    def generate_report(self, results: Dict) -> str:
        """Generate load test report."""
        report = f"""
# R3MES Load Test Report

**Test Date**: {datetime.now().isoformat()}
**Test Duration**: {results['test_config']['test_duration_minutes']} minutes
**Concurrent Users**: {results['test_config']['concurrent_users']}
**Target URL**: {results['test_config']['base_url']}

## Summary

- **Total Requests**: {results['summary']['total_requests']:,}
- **Successful Requests**: {results['summary']['successful_requests']:,}
- **Failed Requests**: {results['summary']['failed_requests']:,}
- **Success Rate**: {results['summary']['success_rate']:.2f}%
- **Requests/Second**: {results['summary']['requests_per_second']:.2f}

## Response Time Statistics

"""
        
        if results['response_time_stats']:
            stats = results['response_time_stats']
            report += f"""
- **Min Response Time**: {stats['min']:.2f}ms
- **Max Response Time**: {stats['max']:.2f}ms
- **Mean Response Time**: {stats['mean']:.2f}ms
- **Median Response Time**: {stats['median']:.2f}ms
- **95th Percentile**: {stats['p95']:.2f}ms
- **99th Percentile**: {stats['p99']:.2f}ms

"""
        
        report += "## Endpoint Performance\n\n"
        
        for endpoint, stats in results['endpoint_stats'].items():
            report += f"""### {endpoint}
- **Total Requests**: {stats['total']:,}
- **Success Rate**: {stats['success_rate']:.2f}%
- **Average Response Time**: {stats['avg_response_time']:.2f}ms
- **95th Percentile**: {stats['p95_response_time']:.2f}ms

"""
        
        report += f"""## System Resources During Test

- **CPU Usage**: {results['system_stats']['cpu_percent']:.1f}%
- **Memory Usage**: {results['system_stats']['memory_percent']:.1f}%
- **Disk Usage**: {results['system_stats']['disk_percent']:.1f}%

## Performance Assessment

"""
        
        # Performance assessment
        assessments = []
        
        if results['response_time_stats']:
            p95 = results['response_time_stats']['p95']
            if p95 < 200:
                assessments.append("✅ **Excellent**: 95th percentile response time under 200ms")
            elif p95 < 500:
                assessments.append("⚠️ **Good**: 95th percentile response time under 500ms")
            else:
                assessments.append("❌ **Poor**: 95th percentile response time over 500ms")
        
        success_rate = results['summary']['success_rate']
        if success_rate >= 99.9:
            assessments.append("✅ **Excellent**: Success rate above 99.9%")
        elif success_rate >= 99:
            assessments.append("⚠️ **Good**: Success rate above 99%")
        else:
            assessments.append("❌ **Poor**: Success rate below 99%")
        
        rps = results['summary']['requests_per_second']
        if rps >= 100:
            assessments.append("✅ **High Throughput**: Over 100 requests/second")
        elif rps >= 50:
            assessments.append("⚠️ **Medium Throughput**: 50-100 requests/second")
        else:
            assessments.append("❌ **Low Throughput**: Under 50 requests/second")
        
        for assessment in assessments:
            report += f"{assessment}\n"
        
        if results.get('errors'):
            report += "\n## Recent Errors\n\n"
            for error in results['errors'][:5]:
                report += f"- **{error['endpoint']}**: {error['error']} at {error['timestamp']}\n"
        
        return report


async def main():
    """Main load testing workflow."""
    parser = argparse.ArgumentParser(description="R3MES Load Testing Suite")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL to test")
    parser.add_argument("--users", type=int, default=50, help="Number of concurrent users")
    parser.add_argument("--duration", type=int, default=5, help="Test duration in minutes")
    parser.add_argument("--ramp-up", type=int, default=30, help="Ramp-up time in seconds")
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument("--wallet", default="remes1test123456789012345678901234567890", help="Test wallet address")
    
    args = parser.parse_args()
    
    config = LoadTestConfig(
        base_url=args.url,
        concurrent_users=args.users,
        test_duration_minutes=args.duration,
        ramp_up_seconds=args.ramp_up,
        api_key=args.api_key,
        wallet_address=args.wallet
    )
    
    tester = LoadTester(config)
    
    print(f"🚀 Starting load test against {config.base_url}")
    print(f"📊 Configuration: {config.concurrent_users} users, {config.test_duration_minutes} minutes")
    
    try:
        results = await tester.run_load_test()
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"load_test_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate and save report
        report = tester.generate_report(results)
        report_file = f"load_test_report_{timestamp}.md"
        with open(report_file, "w") as f:
            f.write(report)
        
        print(f"📋 Results saved to {results_file}")
        print(f"📊 Report saved to {report_file}")
        
        # Print summary
        print("\n" + "="*50)
        print("LOAD TEST SUMMARY")
        print("="*50)
        print(f"Total Requests: {results['summary']['total_requests']:,}")
        print(f"Success Rate: {results['summary']['success_rate']:.2f}%")
        print(f"Requests/Second: {results['summary']['requests_per_second']:.2f}")
        
        if results['response_time_stats']:
            print(f"Mean Response Time: {results['response_time_stats']['mean']:.2f}ms")
            print(f"95th Percentile: {results['response_time_stats']['p95']:.2f}ms")
        
        # Exit code based on performance
        if (results['summary']['success_rate'] >= 99 and 
            results['response_time_stats'].get('p95', 1000) < 500):
            print("✅ Load test PASSED")
            return 0
        else:
            print("❌ Load test FAILED - Performance targets not met")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ Load test interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Load test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
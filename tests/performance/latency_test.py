"""
Latency Testing for R3MES Backend

Tests API endpoint latency under various conditions.
"""

import asyncio
import aiohttp
import time
import statistics
from typing import List, Dict
import json


async def test_endpoint_latency(
    session: aiohttp.ClientSession,
    url: str,
    method: str = "GET",
    payload: Dict = None,
    iterations: int = 100
) -> Dict:
    """Test endpoint latency."""
    latencies = []
    errors = 0
    
    for i in range(iterations):
        start_time = time.time()
        
        try:
            if method == "GET":
                async with session.get(url) as response:
                    await response.read()
            elif method == "POST":
                async with session.post(url, json=payload) as response:
                    await response.read()
            
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)
            
        except Exception as e:
            errors += 1
            print(f"Error on iteration {i}: {e}")
    
    if not latencies:
        return {
            "url": url,
            "method": method,
            "errors": errors,
            "success_rate": 0.0,
        }
    
    return {
        "url": url,
        "method": method,
        "iterations": iterations,
        "errors": errors,
        "success_rate": (iterations - errors) / iterations * 100,
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "avg_latency_ms": statistics.mean(latencies),
        "median_latency_ms": statistics.median(latencies),
        "p95_latency_ms": statistics.quantiles(latencies, n=20)[18],  # 95th percentile
        "p99_latency_ms": statistics.quantiles(latencies, n=100)[98],  # 99th percentile
    }


async def run_latency_tests(base_url: str = "http://localhost:8000"):
    """Run latency tests for all endpoints."""
    print("🚀 Starting Latency Tests...")
    print(f"   Base URL: {base_url}")
    print("")
    
    async with aiohttp.ClientSession() as session:
        tests = [
            {
                "name": "Health Check",
                "url": f"{base_url}/health",
                "method": "GET",
            },
            {
                "name": "Network Stats",
                "url": f"{base_url}/network/stats",
                "method": "GET",
            },
            {
                "name": "User Info",
                "url": f"{base_url}/user/info/remes1test123",
                "method": "GET",
            },
            {
                "name": "Chat Message",
                "url": f"{base_url}/chat",
                "method": "POST",
                "payload": {
                    "message": "What is R3MES?",
                    "wallet_address": "remes1test123",
                    "adapter": "general",
                },
            },
            {
                "name": "Leaderboard",
                "url": f"{base_url}/leaderboard",
                "method": "GET",
            },
            {
                "name": "Metrics",
                "url": f"{base_url}/metrics",
                "method": "GET",
            },
        ]
        
        results = []
        
        for test in tests:
            print(f"📊 Testing {test['name']}...")
            result = await test_endpoint_latency(
                session,
                test["url"],
                test["method"],
                test.get("payload"),
                iterations=100
            )
            result["name"] = test["name"]
            results.append(result)
            
            print(f"   Success Rate: {result['success_rate']:.1f}%")
            print(f"   Avg Latency: {result['avg_latency_ms']:.2f} ms")
            print(f"   P95 Latency: {result['p95_latency_ms']:.2f} ms")
            print(f"   P99 Latency: {result['p99_latency_ms']:.2f} ms")
            print("")
        
        # Save results
        with open("reports/latency_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("✅ Latency tests completed!")
        print("📊 Results saved to reports/latency_test_results.json")
        
        # Summary
        print("")
        print("📋 Summary:")
        print(f"   Total endpoints tested: {len(results)}")
        avg_success = statistics.mean([r["success_rate"] for r in results])
        print(f"   Average success rate: {avg_success:.1f}%")
        avg_latency = statistics.mean([r["avg_latency_ms"] for r in results])
        print(f"   Average latency: {avg_latency:.2f} ms")


if __name__ == "__main__":
    import sys
    
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    asyncio.run(run_latency_tests(base_url))


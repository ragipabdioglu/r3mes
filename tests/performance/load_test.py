#!/usr/bin/env python3
"""
Load Testing for R3MES Backend

Tests backend with 1000+ concurrent miners.
"""

import asyncio
import aiohttp
import time
from typing import List
import statistics

# Configuration
BACKEND_URL = "http://localhost:8000"
CONCURRENT_MINERS = 1000
REQUESTS_PER_MINER = 10
TOTAL_REQUESTS = CONCURRENT_MINERS * REQUESTS_PER_MINER


async def simulate_miner(session: aiohttp.ClientSession, miner_id: int) -> List[float]:
    """Simulate a single miner making requests."""
    latencies = []
    
    for i in range(REQUESTS_PER_MINER):
        start_time = time.time()
        
        try:
            async with session.get(f"{BACKEND_URL}/health") as response:
                await response.json()
                latency = (time.time() - start_time) * 1000  # Convert to ms
                latencies.append(latency)
        except Exception as e:
            print(f"Miner {miner_id} request {i} failed: {e}")
            latencies.append(float('inf'))
    
    return latencies


async def run_load_test():
    """Run load test with concurrent miners."""
    print(f"Starting load test: {CONCURRENT_MINERS} concurrent miners, {REQUESTS_PER_MINER} requests each")
    print(f"Total requests: {TOTAL_REQUESTS}")
    
    connector = aiohttp.TCPConnector(limit=CONCURRENT_MINERS)
    async with aiohttp.ClientSession(connector=connector) as session:
        start_time = time.time()
        
        # Create tasks for all miners
        tasks = [
            simulate_miner(session, miner_id)
            for miner_id in range(CONCURRENT_MINERS)
        ]
        
        # Run all miners concurrently
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Aggregate results
        all_latencies = [lat for miner_latencies in results for lat in miner_latencies if lat != float('inf')]
        
        if all_latencies:
            print(f"\nResults:")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Requests per second: {TOTAL_REQUESTS / total_time:.2f}")
            print(f"  Successful requests: {len(all_latencies)}/{TOTAL_REQUESTS}")
            print(f"  Average latency: {statistics.mean(all_latencies):.2f}ms")
            print(f"  Median latency: {statistics.median(all_latencies):.2f}ms")
            print(f"  P95 latency: {sorted(all_latencies)[int(len(all_latencies) * 0.95)]:.2f}ms")
            print(f"  P99 latency: {sorted(all_latencies)[int(len(all_latencies) * 0.99)]:.2f}ms")
        else:
            print("No successful requests!")


if __name__ == "__main__":
    asyncio.run(run_load_test())


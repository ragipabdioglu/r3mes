# Caching Strategy

This document outlines the caching strategy for the R3MES backend, including cache invalidation, metrics, and warming strategies.

## 1. Cache Architecture

The R3MES backend uses Redis for caching with the following components:

- **CacheManager**: Core caching operations (get/set/delete)
- **CacheInvalidator**: Tag-based and pattern-based invalidation
- **CacheMetrics**: Performance tracking (hit rate, miss rate, response times)
- **CacheWarmer**: Pre-loading frequently accessed data

## 2. Cache Keys

Cache keys follow a consistent naming pattern:

- `user:info:{wallet_address}` - User information
- `user:credits:{wallet_address}` - User credits
- `miner:stats:{wallet_address}` - Miner statistics
- `miner:earnings:{wallet_address}` - Miner earnings history
- `miner:hashrate:{wallet_address}` - Miner hashrate history
- `network:stats` - Network statistics
- `block:height` - Current block height
- `block:recent` - Recent blocks
- `api:key:{api_key_hash}` - API key validation

## 3. Cache TTL (Time To Live)

Different data types have different TTL values:

- **Network Stats**: 60 seconds (frequently updated)
- **Block Height**: 10 seconds (very frequently updated)
- **Recent Blocks**: 30 seconds (moderately updated)
- **User Info**: 300 seconds (5 minutes, relatively stable)
- **Miner Stats**: 60 seconds (frequently updated)
- **Miner History**: 600 seconds (10 minutes, historical data)
- **API Keys**: 3600 seconds (1 hour, rarely changes)

## 4. Cache Invalidation

### 4.1. Tag-Based Invalidation

Tags allow invalidating related cache entries together:

```python
from .cache_invalidation import get_cache_invalidator

invalidator = get_cache_invalidator()
await invalidator.invalidate_by_tag("user:wallet123")
```

### 4.2. Pattern-Based Invalidation

Invalidate keys matching a pattern:

```python
# Invalidate all user-related cache
await invalidator.invalidate_by_pattern("user:*")

# Invalidate all miner stats
await invalidator.invalidate_by_pattern("miner:stats:*")
```

### 4.3. Key-Based Invalidation

Invalidate specific keys:

```python
await invalidator.invalidate_key("user:info:wallet123")
```

### 4.4. Automatic Invalidation

Cache is automatically invalidated when:

- User credits are updated (`add_credits`, `deduct_credit`)
- User information is modified
- Miner statistics are updated
- Blockchain events occur (blocks, transactions)

## 5. Cache Metrics

Cache performance is tracked with the following metrics:

- **Hits**: Number of successful cache retrievals
- **Misses**: Number of cache misses
- **Hit Rate**: Percentage of requests served from cache
- **Miss Rate**: Percentage of requests that missed cache
- **Average Response Time**: Average time to retrieve from cache
- **Requests Per Second**: Cache request throughput
- **Top Hit Keys**: Most frequently accessed cache keys
- **Top Miss Keys**: Keys that frequently miss cache

### 5.1. Accessing Metrics

```python
from .cache_metrics import get_cache_metrics

metrics = get_cache_metrics()
stats = metrics.get_stats()

print(f"Hit rate: {stats['hit_rate']}%")
print(f"Average response time: {stats['avg_response_time_ms']}ms")
```

## 6. Cache Warming

Cache warming pre-loads frequently accessed data to improve response times.

### 6.1. Startup Warming

On application startup, the following data is warmed:

- Network statistics
- Recent blocks
- Current block height

### 6.2. Periodic Warming

Cache is periodically refreshed to ensure data freshness:

- Network stats: Every 5 minutes
- Recent blocks: Every 30 seconds
- Block height: Every 10 seconds

### 6.3. Event-Driven Warming

Cache is warmed when specific events occur:

- User login: Warm user-specific data
- Miner registration: Warm miner statistics
- Block creation: Warm network stats and recent blocks

## 7. Best Practices

### 7.1. When to Cache

Cache data that is:
- Frequently accessed
- Expensive to compute/query
- Relatively stable (doesn't change too often)
- Not user-specific (or cache per-user with appropriate TTL)

### 7.2. When NOT to Cache

Don't cache:
- Real-time data that must be accurate
- User-specific sensitive data (unless encrypted)
- Data that changes on every request
- Very large objects (consider size limits)

### 7.3. Cache Key Design

- Use consistent naming patterns
- Include relevant identifiers (wallet_address, block_height, etc.)
- Keep keys concise but descriptive
- Use prefixes to group related keys

### 7.4. TTL Selection

- Short TTL (10-60s): Frequently changing data (block height, network stats)
- Medium TTL (5-10 min): Moderately changing data (user info, miner stats)
- Long TTL (1+ hour): Rarely changing data (API keys, configuration)

## 8. Monitoring and Alerts

### 8.1. Key Metrics to Monitor

- **Hit Rate**: Should be > 70% for optimal performance
- **Miss Rate**: Should be < 30%
- **Response Time**: Should be < 10ms for cache hits
- **Memory Usage**: Monitor Redis memory usage
- **Key Count**: Monitor total number of cached keys

### 8.2. Alerts

Set up alerts for:
- Hit rate drops below 50%
- Cache memory usage exceeds 80%
- Cache connection failures
- Average response time exceeds 50ms

## 9. Cache Invalidation Patterns

### 9.1. Write-Through

Update cache immediately when data changes:

```python
# Update database
await database.add_credits(wallet, amount)

# Invalidate cache
await invalidator.invalidate_user_cache(wallet)
```

### 9.2. Write-Behind

Update cache asynchronously after database update:

```python
# Update database
await database.add_credits(wallet, amount)

# Invalidate cache asynchronously
asyncio.create_task(invalidator.invalidate_user_cache(wallet))
```

## 10. Troubleshooting

### 10.1. Low Hit Rate

- Check TTL values (may be too short)
- Verify cache keys are consistent
- Check if data is being invalidated too frequently
- Review cache warming strategy

### 10.2. High Memory Usage

- Review TTL values (may be too long)
- Check for memory leaks (keys not expiring)
- Consider implementing cache size limits
- Review what data is being cached

### 10.3. Stale Data

- Reduce TTL for frequently changing data
- Implement more aggressive invalidation
- Review cache warming frequency
- Check invalidation logic

---

**Son Güncelleme**: 2025-12-24


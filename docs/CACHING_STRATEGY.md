# Caching Strategy

This document outlines the caching strategy for the R3MES backend, including cache invalidation, metrics, and warming strategies.

## 1. Cache Architecture

The R3MES backend uses Redis for caching with the following components:

- **CacheManager**: Core caching operations (get/set/delete)
- **CacheInvalidator**: Tag-based and pattern-based invalidation
- **CacheMetrics**: Performance tracking (hit rate, miss rate, response times)
- **CacheWarmer**: Pre-loading frequently accessed data

## 2. Production Configuration

### Redis Settings

Production Redis configuration (in `docker-compose.prod.yml`):
- **Persistence**: AOF (Append-Only File) enabled
- **Memory Limit**: 2GB
- **Eviction Policy**: allkeys-lru (Least Recently Used)
- **Save Points**: 
  - 900 seconds if 1 key changed
  - 300 seconds if 10 keys changed
  - 60 seconds if 10000 keys changed

### Connection Pooling

Production connection settings:
- **Max Connections**: 50
- **Keep-Alive**: Enabled
- **Health Check Interval**: 30 seconds
- **Retry on Timeout**: Enabled

## 3. Cache Keys

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

## 4. Cache TTL (Time To Live)

Different data types have different TTL values:

- **Network Stats**: 60 seconds (frequently updated)
- **Block Height**: 10 seconds (very frequently updated)
- **Recent Blocks**: 30 seconds (moderately updated)
- **User Info**: 300 seconds (5 minutes, relatively stable)
- **Miner Stats**: 60 seconds (frequently updated)
- **Miner History**: 600 seconds (10 minutes, historical data)
- **API Keys**: 3600 seconds (1 hour, rarely changes)

## 5. Cache Invalidation

### 5.1. Tag-Based Invalidation

Tags allow invalidating related cache entries together:

```python
from .cache_invalidation import get_cache_invalidator

invalidator = get_cache_invalidator()
await invalidator.invalidate_by_tag("user:wallet123")
```

### 5.2. Pattern-Based Invalidation

Invalidate keys matching a pattern:

```python
# Invalidate all user-related cache
await invalidator.invalidate_by_pattern("user:*")

# Invalidate all miner stats
await invalidator.invalidate_by_pattern("miner:stats:*")
```

### 5.3. Automatic Invalidation

Cache is automatically invalidated when:
- User credits are updated
- Miner stats are updated
- API keys are created/revoked

## 6. Cache Headers

API responses include cache headers:

- `X-Cache: HIT` - Response served from cache
- `X-Cache: MISS` - Response generated fresh
- `Cache-Control: public, max-age=60` - Cache control directive

## 7. Cache Metrics

Monitor cache performance via Prometheus:

- `cache_hits_total` - Total cache hits
- `cache_misses_total` - Total cache misses
- `cache_hit_rate` - Cache hit rate (hits / (hits + misses))
- `cache_response_time_seconds` - Cache operation duration

## 8. Cache Warming

Pre-load frequently accessed data on startup:

```python
from .cache_warmer import warm_cache

# Warm cache on application startup
await warm_cache()
```

## 9. Best Practices

1. **TTL Selection**: Choose TTL based on data update frequency
2. **Key Naming**: Use consistent, hierarchical key naming
3. **Invalidation**: Invalidate cache when data changes
4. **Monitoring**: Monitor cache hit rate (target: >80%)
5. **Memory Management**: Set appropriate memory limits and eviction policies
6. **Connection Pooling**: Use connection pooling in production

## 10. Troubleshooting

### Low Cache Hit Rate

- Check TTL values (may be too short)
- Verify cache invalidation is not too aggressive
- Monitor cache memory usage

### High Memory Usage

- Review cache TTL values
- Check for memory leaks
- Adjust eviction policy if needed

### Cache Connection Issues

- Verify Redis is running
- Check connection pool settings
- Review network connectivity


# ADR-0006: Redis Caching Strategy

**Status**: Accepted  
**Date**: 2024-03-15  
**Deciders**: Backend Team, DevOps Team

## Context

We need a caching layer to:
- Reduce database load
- Improve API response times
- Cache expensive computations (blockchain queries, analytics)
- Support distributed caching (multiple backend instances)
- Invalidate cache when data changes

Options include:
- In-memory caching (simple, but not distributed)
- Redis (distributed, proven, feature-rich)
- Memcached (simpler, but fewer features)

## Decision

We will use **Redis** for distributed caching with:
1. Tag-based invalidation for related data
2. TTL-based expiration for time-sensitive data
3. Pattern-based invalidation for bulk operations
4. Cache warming on startup for critical data
5. Metrics tracking for cache performance

## Consequences

### Positive
- **Distributed**: Works across multiple backend instances
- **Feature-Rich**: Supports many data structures and operations
- **Performance**: Very fast (in-memory)
- **Persistence**: Can persist to disk (optional)
- **Pub/Sub**: Can use for cache invalidation notifications
- **Proven**: Widely used, well-tested

### Negative
- **Dependency**: Additional service to maintain
- **Memory**: Uses server memory (cost)
- **Complexity**: More complex than in-memory cache
- **Network**: Adds network latency (though minimal)

### Neutral
- Can be disabled for development (fallback to no cache)
- Supports both standalone and cluster modes
- Can use Redis Cloud for managed service

## Implementation Details

- Use `redis` Python library with `hiredis` for performance
- Implement tag-based invalidation system
- Cache key naming convention: `{component}:{resource}:{id}`
- TTL configuration per cache key type
- Cache metrics (hit rate, miss rate, response time)
- Cache warming for frequently accessed data

## Alternatives Considered

1. **In-Memory Caching (Python dict)**:
   - Pros: Simple, no dependencies, fast
   - Cons: Not distributed, lost on restart, limited features

2. **Memcached**:
   - Pros: Simple, fast, distributed
   - Cons: Fewer features, no persistence, no pub/sub

3. **Database Query Cache**:
   - Pros: No additional service
   - Cons: Slower than Redis, database load

---

**Related ADRs**: None


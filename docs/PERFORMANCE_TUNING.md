# Performance Tuning Guide

This guide explains how to optimize R3MES performance for production deployment.

## Overview

Performance tuning involves optimizing:
- Database queries and indexes
- Application-level caching
- API response times
- Resource utilization (CPU, memory, disk I/O)

## Database Tuning

### PostgreSQL Configuration

Production PostgreSQL settings (in `docker-compose.prod.yml`):

```yaml
postgresql:
  environment:
    POSTGRES_SHARED_BUFFERS: 256MB
    POSTGRES_EFFECTIVE_CACHE_SIZE: 1GB
    POSTGRES_MAINTENANCE_WORK_MEM: 64MB
    POSTGRES_CHECKPOINT_COMPLETION_TARGET: 0.9
    POSTGRES_WAL_BUFFERS: 16MB
    POSTGRES_DEFAULT_STATISTICS_TARGET: 100
    POSTGRES_RANDOM_PAGE_COST: 1.1
    POSTGRES_EFFECTIVE_IO_CONCURRENCY: 200
```

### Query Optimization

1. **Analyze Slow Queries**:
   ```bash
   python scripts/analyze_queries.py
   ```

2. **Add Missing Indexes**:
   - Use `IndexAuditor.get_missing_indexes()` to find candidates
   - Create indexes for frequently queried columns
   - Use composite indexes for multi-column queries

3. **Remove Unused Indexes**:
   - Use `IndexAuditor.get_unused_indexes()` to find candidates
   - Drop indexes that are never used

### Connection Pooling

Production pool settings:
- **Min Size**: 10 connections
- **Max Size**: 50 connections
- **Max Lifetime**: 3600 seconds (1 hour)
- **Idle Timeout**: 300 seconds (5 minutes)

## Application Tuning

### Caching Strategy

1. **Enable Redis Caching**:
   - Set `REDIS_URL` environment variable
   - Configure appropriate TTL values
   - Monitor cache hit rate (target: >80%)

2. **API Response Caching**:
   - Use `@cache_response()` decorator for endpoints
   - Set appropriate TTL based on data freshness requirements
   - Invalidate cache when data changes

### Worker Configuration

1. **GPU Workers**:
   ```bash
   export MAX_WORKERS=2  # Number of GPU workers
   ```

2. **Task Queue**:
   - Adjust queue size based on load
   - Monitor queue depth
   - Scale workers based on demand

## Infrastructure Tuning

### CPU Optimization

1. **Process Affinity**:
   - Pin workers to specific CPU cores
   - Reserve cores for system processes

2. **Threading**:
   - Use async/await for I/O-bound operations
   - Use thread pool for CPU-bound operations

### Memory Optimization

1. **Model Loading**:
   - Use model quantization (8-bit, 4-bit)
   - Load models on-demand
   - Share model memory across workers

2. **Cache Limits**:
   - Set Redis memory limits
   - Configure eviction policies
   - Monitor memory usage

### Disk I/O Optimization

1. **Database**:
   - Use SSD storage
   - Configure WAL archiving
   - Optimize checkpoint frequency

2. **Logs**:
   - Use log rotation
   - Archive old logs
   - Use centralized logging

## Monitoring and Profiling

### Performance Metrics

Monitor via Prometheus:
- API request duration (p50, p95, p99)
- Database query duration
- Cache hit rate
- CPU and memory usage
- GPU utilization

### Profiling Tools

1. **Python Profiling**:
   ```python
   import cProfile
   cProfile.run('your_function()')
   ```

2. **Database Profiling**:
   ```sql
   EXPLAIN ANALYZE SELECT ...;
   ```

3. **Application Profiling**:
   - Use OpenTelemetry for distributed tracing
   - Monitor with Grafana dashboards

## Load Testing

### Test Scenarios

1. **Baseline Test**:
   - Measure current performance
   - Identify bottlenecks

2. **Stress Test**:
   - Test under high load
   - Identify breaking points

3. **Endurance Test**:
   - Test over extended period
   - Check for memory leaks

### Tools

- **Locust**: Python-based load testing
- **Apache Bench**: Simple HTTP load testing
- **k6**: Modern load testing tool

## Best Practices

1. **Measure First**: Always measure before optimizing
2. **Profile**: Use profiling tools to identify bottlenecks
3. **Incremental**: Make small, incremental changes
4. **Monitor**: Continuously monitor performance metrics
5. **Document**: Document all tuning changes

## Troubleshooting

### High Latency

1. Check database query performance
2. Review cache hit rate
3. Monitor network latency
4. Check resource utilization

### High Memory Usage

1. Review cache configuration
2. Check for memory leaks
3. Optimize model loading
4. Adjust worker count

### High CPU Usage

1. Profile application code
2. Optimize hot paths
3. Scale horizontally
4. Use async operations


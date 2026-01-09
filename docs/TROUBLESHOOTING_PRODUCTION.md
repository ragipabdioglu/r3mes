# Production Troubleshooting Guide

This guide provides troubleshooting procedures for common production issues.

## Common Issues

### Backend Service Issues

#### Service Won't Start

**Symptoms**: Service fails to start

**Diagnosis**:
```bash
# Check service status
systemctl status r3mes-backend

# Check logs
journalctl -u r3mes-backend -n 100

# Check configuration
python scripts/validate_production_env.py
```

**Common Causes**:
- Missing environment variables
- Database connection failure
- Port already in use
- Invalid configuration

**Resolution**:
1. Verify environment variables
2. Check database connectivity
3. Verify port availability
4. Review configuration files

#### High Error Rate

**Symptoms**: High 5xx error rate

**Diagnosis**:
```bash
# Check error logs
journalctl -u r3mes-backend | grep ERROR

# Check Prometheus metrics
curl http://localhost:9090/api/v1/query?query=rate(api_requests_total{status_code=~"5.."}[5m])
```

**Common Causes**:
- Database connection issues
- Redis connection issues
- Resource exhaustion
- Application bugs

**Resolution**:
1. Check database/Redis connectivity
2. Review resource usage
3. Check recent deployments
4. Review application logs

### Database Issues

#### Connection Pool Exhausted

**Symptoms**: "too many connections" errors

**Diagnosis**:
```sql
-- Check active connections
SELECT count(*) FROM pg_stat_activity;

-- Check connection pool usage
SELECT * FROM pg_stat_database;
```

**Resolution**:
1. Increase connection pool size
2. Review connection leak
3. Restart PostgreSQL
4. Optimize connection usage

#### Slow Queries

**Symptoms**: High query latency

**Diagnosis**:
```bash
# Analyze slow queries
python scripts/analyze_queries.py

# Check PostgreSQL slow query log
tail -f /var/log/postgresql/postgresql-*.log
```

**Resolution**:
1. Add missing indexes
2. Optimize queries
3. Review query plans
4. Consider query caching

### Redis Issues

#### Connection Failures

**Symptoms**: Cache not working, connection errors

**Diagnosis**:
```bash
# Test Redis connection
redis-cli -h localhost -p 6379 ping

# Check Redis status
systemctl status redis
```

**Resolution**:
1. Verify Redis is running
2. Check network connectivity
3. Review Redis configuration
4. Check Redis logs

#### Memory Issues

**Symptoms**: Redis OOM errors

**Diagnosis**:
```bash
# Check Redis memory
redis-cli INFO memory

# Check memory usage
redis-cli --bigkeys
```

**Resolution**:
1. Increase memory limit
2. Adjust eviction policy
3. Review cache TTL values
4. Clear unused keys

### SSL/TLS Issues

#### Certificate Expiration

**Symptoms**: SSL certificate errors

**Diagnosis**:
```bash
# Check certificate expiration
certbot certificates

# Test certificate
openssl s_client -connect r3mes.network:443 -servername r3mes.network
```

**Resolution**:
1. Renew certificates:
   ```bash
   ./scripts/renew_certificates.sh
   ```
2. Reload Nginx:
   ```bash
   systemctl reload nginx
   ```

#### Certificate Chain Issues

**Symptoms**: Browser shows certificate warnings

**Diagnosis**:
```bash
# Check certificate chain
openssl s_client -connect r3mes.network:443 -showcerts
```

**Resolution**:
1. Verify certificate chain
2. Update intermediate certificates
3. Reload Nginx

### Monitoring Issues

#### Metrics Not Appearing

**Symptoms**: No metrics in Prometheus

**Diagnosis**:
```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Check backend metrics endpoint
curl http://localhost:8000/metrics
```

**Resolution**:
1. Verify Prometheus configuration
2. Check scrape targets
3. Verify metrics endpoint
4. Review network connectivity

#### Alerts Not Firing

**Symptoms**: Alerts not triggering

**Diagnosis**:
```bash
# Check Alertmanager
curl http://localhost:9093/api/v2/alerts

# Check alert rules
curl http://localhost:9090/api/v1/rules
```

**Resolution**:
1. Verify alert rules
2. Check Alertmanager configuration
3. Test alert manually
4. Review notification channels

## Debug Procedures

### Enable Debug Logging

```bash
# Backend
export LOG_LEVEL=DEBUG
systemctl restart r3mes-backend

# Frontend
export NODE_ENV=development
npm run dev
```

### Database Debugging

```sql
-- Enable query logging
ALTER DATABASE r3mes SET log_statement = 'all';
ALTER DATABASE r3mes SET log_duration = on;

-- Check active queries
SELECT * FROM pg_stat_activity WHERE state = 'active';

-- Check locks
SELECT * FROM pg_locks WHERE NOT granted;
```

### Network Debugging

```bash
# Check connections
netstat -tuln | grep :8000

# Test connectivity
curl -v https://api.r3mes.network/health

# Check DNS
dig r3mes.network
```

## Performance Debugging

### CPU Profiling

```python
# Python profiling
import cProfile
cProfile.run('your_function()')

# Or use py-spy
py-spy record -o profile.svg --pid <pid>
```

### Memory Profiling

```python
# Memory profiling
from memory_profiler import profile

@profile
def your_function():
    # Your code
    pass
```

### Database Profiling

```sql
-- Enable query timing
\timing on

-- Analyze query
EXPLAIN ANALYZE SELECT ...;
```

## Recovery Procedures

### Service Recovery

1. **Stop Service**:
   ```bash
   systemctl stop r3mes-backend
   ```

2. **Check Configuration**:
   ```bash
   python scripts/validate_production_env.py
   ```

3. **Start Service**:
   ```bash
   systemctl start r3mes-backend
   ```

4. **Verify**:
   ```bash
   curl http://localhost:8000/health
   ```

### Database Recovery

1. **Stop Writes**:
   ```bash
   # Set database to read-only
   ```

2. **Restore Backup**:
   ```bash
   ./scripts/restore_database.sh /backups/r3mes_YYYYMMDD.sql.gz
   ```

3. **Verify**:
   ```bash
   python scripts/verify_migration.py
   ```

## Best Practices

1. **Document Issues**: Document all issues and resolutions
2. **Monitor Continuously**: Use monitoring to detect issues early
3. **Test Procedures**: Regularly test recovery procedures
4. **Keep Logs**: Retain logs for troubleshooting
5. **Update Runbooks**: Keep runbooks up to date


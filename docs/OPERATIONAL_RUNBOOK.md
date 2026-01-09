# R3MES Operational Runbook

## Overview

This runbook provides step-by-step procedures for operating R3MES in production. It covers common operational tasks, troubleshooting, and emergency procedures.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Monitoring and Alerting](#monitoring-and-alerting)
3. [Common Operations](#common-operations)
4. [Troubleshooting](#troubleshooting)
5. [Emergency Procedures](#emergency-procedures)
6. [Maintenance Tasks](#maintenance-tasks)
7. [Performance Optimization](#performance-optimization)

## System Architecture

### Components Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │     Frontend    │    │     Backend     │
│   (Nginx/ALB)   │────│   (Next.js)     │────│   (FastAPI)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                       ┌─────────────────┐    ┌─────────────────┐
                       │     Redis       │    │   PostgreSQL    │
                       │    (Cache)      │────│   (Database)    │
                       └─────────────────┘    └─────────────────┘
                                                        │
                                              ┌─────────────────┐
                                              │   Blockchain    │
                                              │     (Cosmos)    │
                                              └─────────────────┘
```

### Key Services

- **Frontend**: Next.js application serving the web interface
- **Backend**: FastAPI application providing REST API and WebSocket endpoints
- **Database**: PostgreSQL for persistent data storage
- **Cache**: Redis for session management and API response caching
- **Blockchain**: Cosmos SDK-based blockchain for consensus and mining

## Monitoring and Alerting

### Monitoring Stack

- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards
- **AlertManager**: Alert routing and notification
- **Jaeger**: Distributed tracing (optional)

### Key Metrics to Monitor

#### Backend API
```bash
# Response time (target: <200ms p95)
http_request_duration_seconds{quantile="0.95"}

# Error rate (target: <1%)
rate(http_requests_total{status=~"5.."}[5m])

# Request rate
rate(http_requests_total[5m])
```

#### Database
```bash
# Connection pool usage (target: <80%)
db_connection_pool_usage

# Query duration (target: <100ms avg)
db_query_duration_seconds

# Active connections
db_connections_active
```

#### System Resources
```bash
# CPU usage (target: <80%)
cpu_usage_percent

# Memory usage (target: <85%)
memory_usage_percent

# Disk usage (target: <90%)
disk_usage_percent
```

### Alert Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| API Response Time (p95) | >200ms | >500ms |
| Error Rate | >1% | >5% |
| CPU Usage | >70% | >85% |
| Memory Usage | >80% | >90% |
| Disk Usage | >85% | >95% |
| Database Connections | >80% | >95% |

## Common Operations

### 1. Checking System Health

```bash
# Check all services status
kubectl get pods -n r3mes

# Check service endpoints
kubectl get services -n r3mes

# Check ingress status
kubectl get ingress -n r3mes

# Run health check
python scripts/performance_monitor.py --single-check
```

### 2. Viewing Logs

```bash
# Backend logs
kubectl logs -f deployment/r3mes-backend -n r3mes

# Frontend logs
kubectl logs -f deployment/r3mes-frontend -n r3mes

# Database logs
kubectl logs -f statefulset/postgres -n r3mes

# All logs with labels
kubectl logs -l app=r3mes -n r3mes --tail=100
```

### 3. Scaling Services

```bash
# Scale backend horizontally
kubectl scale deployment r3mes-backend --replicas=5 -n r3mes

# Scale frontend
kubectl scale deployment r3mes-frontend --replicas=3 -n r3mes

# Check HPA status
kubectl get hpa -n r3mes
```

### 4. Database Operations

```bash
# Connect to database
kubectl exec -it statefulset/postgres -n r3mes -- psql -U $POSTGRES_USER -d $POSTGRES_DB

# Check database size
kubectl exec -it statefulset/postgres -n r3mes -- psql -U $POSTGRES_USER -d $POSTGRES_DB -c "SELECT pg_size_pretty(pg_database_size('r3mes'));"

# Check active connections
kubectl exec -it statefulset/postgres -n r3mes -- psql -U $POSTGRES_USER -d $POSTGRES_DB -c "SELECT * FROM connection_stats;"
```

### 5. Cache Operations

```bash
# Connect to Redis
kubectl exec -it deployment/redis -n r3mes -- redis-cli

# Check Redis info
kubectl exec -it deployment/redis -n r3mes -- redis-cli info

# Clear cache (use with caution)
kubectl exec -it deployment/redis -n r3mes -- redis-cli flushall
```

## Troubleshooting

### High Response Times

**Symptoms**: API response times >500ms, user complaints about slow performance

**Investigation Steps**:
1. Check system resources (CPU, memory, disk)
2. Check database query performance
3. Check Redis cache hit rate
4. Review recent deployments

```bash
# Check system resources
kubectl top pods -n r3mes

# Check database performance
kubectl exec -it statefulset/postgres -n r3mes -- psql -U $POSTGRES_USER -d $POSTGRES_DB -c "SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"

# Check Redis performance
kubectl exec -it deployment/redis -n r3mes -- redis-cli info stats
```

**Common Solutions**:
- Scale backend pods if CPU/memory high
- Optimize slow database queries
- Clear Redis cache if hit rate is low
- Check for memory leaks in application

### High Error Rates

**Symptoms**: 5xx errors >5%, error alerts firing

**Investigation Steps**:
1. Check application logs for errors
2. Check database connectivity
3. Check external service dependencies
4. Verify configuration

```bash
# Check recent errors
kubectl logs --tail=100 deployment/r3mes-backend -n r3mes | grep ERROR

# Check database connectivity
kubectl exec -it deployment/r3mes-backend -n r3mes -- python -c "
import asyncpg
import asyncio
async def test():
    conn = await asyncpg.connect('$DATABASE_URL')
    await conn.close()
    print('Database OK')
asyncio.run(test())
"

# Check Redis connectivity
kubectl exec -it deployment/r3mes-backend -n r3mes -- python -c "
import redis
r = redis.from_url('$REDIS_URL')
r.ping()
print('Redis OK')
"
```

### Database Connection Issues

**Symptoms**: Database connection errors, connection pool exhausted

**Investigation Steps**:
1. Check database pod status
2. Check connection pool metrics
3. Check for long-running queries
4. Verify database configuration

```bash
# Check database pod
kubectl get pod -l component=database -n r3mes

# Check connection pool
kubectl exec -it statefulset/postgres -n r3mes -- psql -U $POSTGRES_USER -d $POSTGRES_DB -c "SELECT * FROM pg_stat_activity WHERE state = 'active';"

# Check for long-running queries
kubectl exec -it statefulset/postgres -n r3mes -- psql -U $POSTGRES_USER -d $POSTGRES_DB -c "SELECT pid, now() - pg_stat_activity.query_start AS duration, query FROM pg_stat_activity WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes';"
```

**Solutions**:
- Restart database pod if necessary
- Kill long-running queries
- Increase connection pool size
- Optimize slow queries

### Memory Leaks

**Symptoms**: Gradually increasing memory usage, OOMKilled pods

**Investigation Steps**:
1. Monitor memory usage over time
2. Check for memory leaks in application code
3. Review garbage collection metrics
4. Check for resource leaks

```bash
# Check memory usage trends
kubectl top pods -n r3mes --sort-by=memory

# Check pod restart history
kubectl get pods -n r3mes -o wide

# Get detailed pod metrics
kubectl describe pod <pod-name> -n r3mes
```

## Emergency Procedures

### Service Outage

**Immediate Actions**:
1. Check system status and identify affected components
2. Notify stakeholders via communication channels
3. Implement emergency response plan
4. Document incident timeline

```bash
# Quick health check
kubectl get pods -n r3mes
kubectl get services -n r3mes
kubectl get ingress -n r3mes

# Check recent events
kubectl get events -n r3mes --sort-by='.lastTimestamp'
```

### Database Failure

**Immediate Actions**:
1. Check database pod status
2. Attempt to restart database service
3. If restart fails, restore from backup
4. Verify data integrity after recovery

```bash
# Check database status
kubectl get statefulset postgres -n r3mes
kubectl describe statefulset postgres -n r3mes

# Restart database (if safe)
kubectl rollout restart statefulset/postgres -n r3mes

# Check database logs
kubectl logs statefulset/postgres -n r3mes --tail=100
```

### Security Incident

**Immediate Actions**:
1. Isolate affected systems
2. Preserve evidence
3. Notify security team
4. Implement containment measures

```bash
# Check for suspicious activity
kubectl logs -l app=r3mes -n r3mes | grep -i "unauthorized\|attack\|breach"

# Review recent configuration changes
kubectl get events -n r3mes --field-selector type=Warning

# Check network policies
kubectl get networkpolicy -n r3mes
```

## Maintenance Tasks

### Daily Tasks

- [ ] Check system health dashboard
- [ ] Review error logs and alerts
- [ ] Monitor resource usage trends
- [ ] Verify backup completion

### Weekly Tasks

- [ ] Review performance metrics
- [ ] Check security scan results
- [ ] Update documentation if needed
- [ ] Review capacity planning

### Monthly Tasks

- [ ] Security updates and patches
- [ ] Performance optimization review
- [ ] Disaster recovery testing
- [ ] Cost optimization review

### Quarterly Tasks

- [ ] Full security audit
- [ ] Load testing
- [ ] Disaster recovery drill
- [ ] Architecture review

## Performance Optimization

### Database Optimization

```sql
-- Check slow queries
SELECT query, mean_time, calls, total_time 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;

-- Check index usage
SELECT schemaname, tablename, attname, n_distinct, correlation 
FROM pg_stats 
WHERE schemaname = 'public';

-- Analyze table statistics
ANALYZE;

-- Vacuum tables
VACUUM ANALYZE;
```

### Cache Optimization

```bash
# Check Redis memory usage
kubectl exec -it deployment/redis -n r3mes -- redis-cli info memory

# Check cache hit rate
kubectl exec -it deployment/redis -n r3mes -- redis-cli info stats | grep hit

# Monitor key expiration
kubectl exec -it deployment/redis -n r3mes -- redis-cli info keyspace
```

### Application Optimization

- Monitor API endpoint performance
- Optimize database queries
- Implement proper caching strategies
- Use connection pooling effectively
- Monitor memory usage and garbage collection

## Contact Information

### On-Call Rotation

- **Primary**: DevOps Team Lead
- **Secondary**: Backend Developer
- **Escalation**: CTO

### Communication Channels

- **Slack**: #r3mes-alerts
- **Email**: ops@r3mes.network
- **Phone**: Emergency hotline

### External Contacts

- **Cloud Provider**: AWS/GCP Support
- **DNS Provider**: Cloudflare Support
- **Monitoring**: Datadog/New Relic Support

---

**Last Updated**: January 1, 2026
**Version**: 1.0
**Next Review**: February 1, 2026
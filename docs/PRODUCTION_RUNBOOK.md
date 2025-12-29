# Production Runbook

This document provides operational procedures for R3MES production deployment.

## Table of Contents

1. [Deployment Procedures](#deployment-procedures)
2. [Rollback Procedures](#rollback-procedures)
3. [Common Issues](#common-issues)
4. [Emergency Contacts](#emergency-contacts)

## Deployment Procedures

### Pre-Deployment Checklist

- [ ] Run security scans
- [ ] Run test suite
- [ ] Review changelog
- [ ] Backup database
- [ ] Notify team
- [ ] Verify monitoring is active

### Deployment Steps

1. **Validate Environment**:
   ```bash
   python scripts/validate_production_env.py
   ```

2. **Backup Database**:
   ```bash
   ./scripts/backup_database.sh
   ```

3. **Deploy Backend**:
   ```bash
   # Using Docker Compose
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d backend
   
   # Or using systemd
   systemctl restart r3mes-backend
   ```

4. **Deploy Frontend**:
   ```bash
   cd web-dashboard
   npm run build
   systemctl restart r3mes-frontend
   ```

5. **Verify Deployment**:
   ```bash
   # Check health endpoints
   curl https://api.r3mes.network/health
   curl https://r3mes.network/api/health
   
   # Check logs
   journalctl -u r3mes-backend -f
   ```

6. **Run Smoke Tests**:
   ```bash
   # Test critical endpoints
   curl https://api.r3mes.network/health
   curl https://api.r3mes.network/network/stats
   ```

### Post-Deployment

- [ ] Monitor error rates
- [ ] Check response times
- [ ] Verify metrics collection
- [ ] Review logs for errors
- [ ] Notify team of completion

## Rollback Procedures

### Quick Rollback

1. **Stop Current Version**:
   ```bash
   systemctl stop r3mes-backend
   systemctl stop r3mes-frontend
   ```

2. **Restore Previous Version**:
   ```bash
   # Using Docker
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --scale backend=0
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d backend:previous-version
   
   # Or using systemd
   systemctl start r3mes-backend@previous-version
   ```

3. **Verify Rollback**:
   ```bash
   curl https://api.r3mes.network/health
   ```

### Database Rollback

If database migration needs rollback:

```bash
# Restore from backup
./scripts/restore_database.sh /backups/r3mes_YYYYMMDD_HHMMSS.sql.gz

# Or rollback Alembic migration
cd backend
alembic downgrade -1
```

## Common Issues

### Issue: High Error Rate

**Symptoms**: High 5xx error rate in monitoring

**Resolution**:
1. Check application logs
2. Check database connectivity
3. Check Redis connectivity
4. Review recent deployments
5. Scale resources if needed

### Issue: Slow Response Times

**Symptoms**: High p95/p99 latency

**Resolution**:
1. Check database query performance
2. Review cache hit rate
3. Check resource utilization (CPU, memory)
4. Review slow query log
5. Optimize queries or add indexes

### Issue: Database Connection Errors

**Symptoms**: Database connection failures

**Resolution**:
1. Check PostgreSQL status
2. Verify connection pool settings
3. Check network connectivity
4. Review connection pool metrics
5. Restart PostgreSQL if needed

### Issue: Certificate Expiration

**Symptoms**: SSL certificate errors

**Resolution**:
1. Check certificate expiration:
   ```bash
   certbot certificates
   ```
2. Renew certificates:
   ```bash
   ./scripts/renew_certificates.sh
   ```
3. Reload Nginx:
   ```bash
   systemctl reload nginx
   ```

### Issue: High Memory Usage

**Symptoms**: High memory usage, potential OOM kills

**Resolution**:
1. Check memory usage:
   ```bash
   free -h
   ```
2. Review cache configuration
3. Check for memory leaks
4. Adjust worker count
5. Scale resources if needed

## Emergency Contacts

### On-Call Rotation

- **Primary**: [Contact Info]
- **Secondary**: [Contact Info]
- **Escalation**: [Contact Info]

### Service Providers

- **Hosting Provider**: [Contact Info]
- **DNS Provider**: [Contact Info]
- **CDN Provider**: [Contact Info]

## Monitoring

### Key Metrics

- API error rate (target: <0.1%)
- API latency p95 (target: <2s)
- Database connection pool usage
- Cache hit rate (target: >80%)
- CPU usage (target: <80%)
- Memory usage (target: <80%)

### Alert Thresholds

- **Critical**: Service down, database connection failure
- **Warning**: High error rate, high latency, low cache hit rate

## Maintenance Windows

### Scheduled Maintenance

- **Weekly**: Database backup verification
- **Monthly**: Security updates, dependency updates
- **Quarterly**: Security audit, performance review

### Maintenance Procedures

1. Notify users (if needed)
2. Enable maintenance mode
3. Perform maintenance
4. Verify services
5. Disable maintenance mode
6. Monitor for issues

## Troubleshooting Commands

```bash
# Check service status
systemctl status r3mes-backend
systemctl status r3mes-frontend
systemctl status nginx
systemctl status postgresql

# View logs
journalctl -u r3mes-backend -f
journalctl -u r3mes-frontend -f
tail -f /var/log/nginx/error.log

# Check database
psql $DATABASE_URL -c "SELECT version();"

# Check Redis
redis-cli ping

# Check disk space
df -h

# Check memory
free -h

# Check network
netstat -tuln
```


# R3MES Infrastructure - Quick Reference Guide

## Critical Issues at a Glance

### 🔴 CRITICAL (Fix Immediately)
1. **Backend runs as root** → Add USER directive to Dockerfile
2. **No K8s security contexts** → Add securityContext to all pods
3. **No Pod Security Policy** → Create and enforce PSP
4. **Database not encrypted** → Enable PostgreSQL SSL
5. **Backups not encrypted** → Encrypt backup files
6. **No centralized logging** → Deploy Loki + Promtail
7. **No distributed tracing** → Deploy Jaeger
8. **Missing alert rules** → Add critical alerts to Prometheus
9. **No audit logging** → Enable K8s audit logs
10. **No CI/CD pipeline** → Create GitHub Actions workflows
11. **No off-site backups** → Set up S3/cloud backup
12. **No backup encryption** → Encrypt backups with GPG/OpenSSL

### 🟠 HIGH (Fix Within 1 Week)
1. Pin base image versions with SHA256
2. Implement secret rotation mechanism
3. Set up database replication
4. Set up Redis replication
5. Implement global load balancing
6. Add DDoS protection (rate limiting)
7. Enforce firewall configuration
8. Add application metrics to backend
9. Tune alert thresholds
10. Implement automated testing in CI/CD
11. Add security scanning to CI/CD
12. Define maintenance windows
13. Tune HPA configuration
14. Add network policy for ingress
15. Implement WAF
16. Set up VPN for admin access
17. Implement MFA
18. Implement automated rollback

### 🟡 MEDIUM (Fix Within 1 Month)
1. Create operational runbooks
2. Implement automated performance testing
3. Optimize PostgreSQL configuration
4. Set up query performance monitoring
5. Implement certificate pinning
6. Add structured logging (JSON)
7. Set up log aggregation
8. Create troubleshooting guide
9. Implement vertical pod autoscaler
10. Add health check documentation
11. Create disaster recovery plan
12. Define RTO/RPO objectives
13. Implement data retention policy
14. Add infrastructure metrics
15. Implement certificate monitoring

---

## Quick Fix Commands

### Fix Backend Container (5 minutes)
```bash
# Update backend/Dockerfile
# Add after FROM line:
RUN useradd -m -u 1001 -s /sbin/nologin appuser
# Add before CMD:
USER appuser

# Rebuild
docker build -t r3mes/backend:latest backend/
```

### Add K8s Security Contexts (10 minutes)
```bash
# Add to k8s/production-deployment.yaml in spec.securityContext:
securityContext:
  runAsNonRoot: true
  runAsUser: 1001
  fsGroup: 1001

# Apply
kubectl apply -f k8s/production-deployment.yaml
```

### Enable Database Encryption (15 minutes)
```bash
# Generate certificate
openssl req -new -x509 -days 365 -nodes \
  -out docker/certs/server.crt \
  -keyout docker/certs/server.key

# Update docker-compose.prod.yml postgres service
# Add POSTGRES_INITDB_ARGS with SSL options
```

### Deploy Loki Logging (20 minutes)
```bash
# Start Loki
docker-compose -f docker/docker-compose.logging.yml up -d

# Add Loki datasource to Grafana
# Create dashboards for log analysis
```

### Deploy Jaeger Tracing (20 minutes)
```bash
# Start Jaeger
docker-compose -f docker/docker-compose.tracing.yml up -d

# Update backend to send traces
# Access at http://localhost:16686
```

### Add Critical Alerts (15 minutes)
```bash
# Add to monitoring/prometheus/alerts.prod.yml:
# - DatabaseConnectionPoolExhausted
# - RedisMemoryExhausted
# - DiskSpaceCritical
# - PodRestartLoop
# - DeploymentReplicasMismatch
# - StatefulSetReplicasMismatch

# Reload Prometheus
curl -X POST http://localhost:9090/-/reload
```

---

## Monitoring Dashboard

### Key Metrics to Monitor
- Backend API response time (target: <200ms p95)
- Database query time (target: <100ms avg)
- Redis cache hit rate (target: >85%)
- Pod restart count (target: 0)
- Disk usage (alert: >80%)
- Memory usage (alert: >85%)
- CPU usage (alert: >80%)
- SSL certificate expiration (alert: <30 days)

### Alert Channels
- **Critical**: Slack + Email + PagerDuty
- **Warning**: Slack + Email
- **Info**: Slack only

---

## Deployment Checklist

Before going to production:
- [ ] All critical issues fixed
- [ ] Security audit completed
- [ ] Load testing passed
- [ ] Disaster recovery drill completed
- [ ] Monitoring operational
- [ ] Alerting tested
- [ ] Backup/restore tested
- [ ] Documentation complete
- [ ] Team trained
- [ ] Stakeholder approval

---

## Emergency Procedures

### Service Down
1. Check pod status: `kubectl get pods -n r3mes`
2. Check logs: `kubectl logs -f pod/name -n r3mes`
3. Check events: `kubectl describe pod/name -n r3mes`
4. Restart pod: `kubectl delete pod/name -n r3mes`

### Database Issues
1. Check connection: `docker exec r3mes-postgres-prod pg_isready`
2. Check logs: `docker logs r3mes-postgres-prod`
3. Check disk: `docker exec r3mes-postgres-prod df -h`
4. Restart: `docker restart r3mes-postgres-prod`

### High Memory Usage
1. Check usage: `docker stats`
2. Check processes: `docker top container-name`
3. Increase limits in docker-compose.prod.yml
4. Restart service

### SSL Certificate Issues
1. Check expiration: `docker exec r3mes-certbot-prod certbot certificates`
2. Manual renewal: `docker exec r3mes-certbot-prod certbot renew --force-renewal`
3. Check logs: `docker logs r3mes-certbot-prod`

---

## Performance Tuning

### PostgreSQL
```sql
-- Check slow queries
SELECT query, calls, mean_time FROM pg_stat_statements 
ORDER BY mean_time DESC LIMIT 10;

-- Analyze table
ANALYZE table_name;

-- Reindex
REINDEX TABLE table_name;
```

### Redis
```bash
# Check memory usage
redis-cli INFO memory

# Check key distribution
redis-cli --scan | head -100

# Monitor commands
redis-cli MONITOR
```

### Backend
```python
# Add metrics
from prometheus_client import Counter, Histogram

request_count = Counter('requests_total', 'Total requests')
request_duration = Histogram('request_duration_seconds', 'Request duration')
```

---

## Useful Commands

```bash
# View all services
docker-compose -f docker/docker-compose.prod.yml ps

# View logs
docker-compose -f docker/docker-compose.prod.yml logs -f backend

# Restart service
docker-compose -f docker/docker-compose.prod.yml restart backend

# Execute command in container
docker exec -it r3mes-backend-prod bash

# Check resource usage
docker stats

# View network
docker network inspect docker_r3mes-network

# Backup database
docker exec r3mes-postgres-prod pg_dump -U r3mes r3mes > backup.sql

# Restore database
docker exec -i r3mes-postgres-prod psql -U r3mes r3mes < backup.sql

# Check SSL certificate
docker exec r3mes-nginx-prod certbot certificates

# View Prometheus targets
curl http://localhost:9090/api/v1/targets

# View Grafana dashboards
curl http://localhost:3000/api/dashboards/home
```

---

## Resources

- **Docker Docs**: https://docs.docker.com/
- **Kubernetes Docs**: https://kubernetes.io/docs/
- **Prometheus Docs**: https://prometheus.io/docs/
- **Grafana Docs**: https://grafana.com/docs/
- **PostgreSQL Docs**: https://www.postgresql.org/docs/
- **Redis Docs**: https://redis.io/documentation
- **Nginx Docs**: https://nginx.org/en/docs/

---

**Last Updated**: January 2025  
**Maintained by**: DevOps Team

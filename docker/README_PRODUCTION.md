# R3MES Production Docker Deployment Guide

## Overview

This guide explains how to deploy the complete R3MES stack using Docker Compose in production.

**For Testnet Deployment**: See [Testnet Deployment Guide](../../docs/TESTNET_DEPLOYMENT.md) for testnet-specific instructions.

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- At least 8GB RAM
- 50GB+ disk space
- (Optional) NVIDIA GPU with nvidia-container-toolkit for mining

## Quick Start

### 1. Prepare Environment

#### Create Docker Secrets

**IMPORTANT**: R3MES uses Docker Secrets for secure password management. You must create secrets before deployment.

**Option A: Interactive (Recommended for first-time setup)**
```bash
bash scripts/create_secrets.sh
```

**Option B: From .env.production file**
```bash
cd docker
cp env.production.example .env.production
# Edit .env.production and set passwords (they will be moved to secrets)
bash ../scripts/create_secrets_from_env.sh
```

This creates secure secret files in `docker/secrets/`:
- `postgres_password.txt` - PostgreSQL password
- `redis_password.txt` - Redis password
- `grafana_admin_password.txt` - Grafana admin password

#### Configure Environment Variables

```bash
cd docker
cp env.production.example .env.production
# Edit .env.production and set non-sensitive values
```

**Critical Variables to Set:**
- `CHAIN_ID` - Blockchain chain ID (e.g., `remes-mainnet-1` for mainnet, `remes-testnet-1` for testnet)
- `DOMAIN` - Your domain name (e.g., r3mes.network for mainnet, testnet.r3mes.network for testnet)
- `EMAIL` - Email for Let's Encrypt certificates
- `NEXT_PUBLIC_API_URL` - Frontend API URL
- `CORS_ALLOWED_ORIGINS` - Allowed CORS origins

**Note**: Passwords are now managed via Docker Secrets, not in `.env.production`!

### 2. Deploy

**Option A: Using Deployment Script**
```bash
bash scripts/deploy_production_docker.sh
```

**Option B: Using Makefile**
```bash
make docker-prod-up
```

**Option C: Manual Docker Compose**
```bash
cd docker
docker-compose -f docker-compose.prod.yml up -d
```

### 3. Enable GPU Mining (Optional)

If you have NVIDIA GPU and want to run mining:

```bash
cd docker
docker-compose -f docker-compose.prod.yml --profile miner up -d
```

Or using Makefile:
```bash
make docker-prod-up-miner
```

## Services

The production stack includes:

1. **PostgreSQL** - Database (port 5432 internal)
2. **Redis** - Cache (port 6379 internal)
3. **IPFS** - Distributed storage (ports 4001, 5001, 8080)
4. **remesd** - Blockchain node (ports 26656, 26657, 9090, 1317)
5. **Backend** - FastAPI API (port 8000 internal)
6. **Frontend** - Next.js dashboard (port 3000 internal)
7. **Nginx** - Reverse proxy + SSL (ports 80, 443)
8. **Certbot** - Let's Encrypt certificate renewal
9. **Miner** - GPU mining engine (optional, profile: miner)
10. **Prometheus** - Metrics collection (port 9090 internal)
11. **Grafana** - Monitoring dashboards (port 3001 internal)
12. **Alertmanager** - Alert routing (port 9093 internal)
13. **Node Exporter** - System metrics
14. **Redis Exporter** - Redis metrics
15. **PostgreSQL Exporter** - Database metrics
16. **Postgres Backup** - Automated daily backups

## Networking

All services communicate via Docker service names:

- Backend → Blockchain: `http://remesd:26657` (RPC), `remesd:9090` (gRPC)
- Backend → IPFS: `http://ipfs:5001`
- Backend → PostgreSQL: `postgres:5432`
- Backend → Redis: `redis:6379`
- Nginx → Backend: `backend:8000`
- Nginx → Frontend: `frontend:3000`
- Miner → Blockchain: `remesd:9090`
- Miner → IPFS: `http://ipfs:5001`

## SSL/HTTPS Setup

### Automatic (Let's Encrypt)

1. Set `DOMAIN` and `EMAIL` in `.env.production`
2. Ensure port 80 is accessible from internet
3. Deploy stack - certificates will be obtained automatically
4. Certbot renews certificates every 12 hours

### Manual SSL

If you have your own certificates:

1. Mount certificate files to nginx volume
2. Update `docker/nginx/nginx.conf` with certificate paths
3. Restart nginx service

## Health Checks

All services have health checks. View status:

```bash
cd docker
docker-compose -f docker-compose.prod.yml ps
```

## Testing Networking

Test connectivity between services:

```bash
bash scripts/test_docker_networking.sh
```

Or using Makefile:
```bash
make docker-prod-test
```

## Logs

View logs for all services:

```bash
cd docker
docker-compose -f docker-compose.prod.yml logs -f
```

View logs for specific service:

```bash
docker-compose -f docker-compose.prod.yml logs -f backend
```

Or using Makefile:
```bash
make docker-prod-logs
```

## Management Commands

### Start Services
```bash
make docker-prod-up
# or
cd docker && docker-compose -f docker-compose.prod.yml up -d
```

### Stop Services
```bash
make docker-prod-down
# or
cd docker && docker-compose -f docker-compose.prod.yml down
```

### Restart Services
```bash
make docker-prod-restart
# or
cd docker && docker-compose -f docker-compose.prod.yml restart
```

### Restart Specific Service
```bash
cd docker
docker-compose -f docker-compose.prod.yml restart backend
```

### View Service Status
```bash
cd docker
docker-compose -f docker-compose.prod.yml ps
```

### Execute Command in Container
```bash
# Backend container
docker exec -it r3mes-backend-prod bash

# Blockchain container
docker exec -it r3mes-blockchain-prod sh
```

## Volumes

Data is persisted in Docker volumes:

- `postgres_data` - Database data
- `redis_data` - Redis data
- `ipfs_data` - IPFS repository
- `blockchain_data` - Blockchain node data
- `miner_data` - Miner data (if using)
- `nginx_certs` - SSL certificates
- `nginx_www` - Certbot webroot

Backup volumes:
```bash
docker run --rm -v r3mes_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup.tar.gz /data
```

## Troubleshooting

### Services Not Starting

1. Check logs: `docker-compose -f docker-compose.prod.yml logs [service]`
2. Check health: `docker-compose -f docker-compose.prod.yml ps`
3. Verify environment variables: `cat docker/.env.production`

### Backend Cannot Connect to Blockchain

1. Verify blockchain is healthy: `docker exec r3mes-blockchain-prod wget -q -O- http://localhost:26657/status`
2. Check network: `docker network inspect docker_r3mes-network`
3. Verify environment: `docker exec r3mes-backend-prod env | grep BLOCKCHAIN`

### SSL Certificate Issues

1. Check certbot logs: `docker logs r3mes-certbot-prod`
2. Verify port 80 is accessible
3. Check DNS points to server
4. Manually obtain certificate:
   ```bash
   docker exec r3mes-certbot-prod certbot certonly --webroot -w /var/www/certbot -d yourdomain.com
   ```

### GPU Mining Not Working

1. Verify nvidia-container-toolkit: `docker info | grep nvidia`
2. Check GPU: `nvidia-smi`
3. Check miner logs: `docker logs r3mes-miner-prod`
4. Verify miner is using profile: `docker-compose -f docker-compose.prod.yml --profile miner ps`

## Security Notes

1. **Never commit `.env.production`** to git
2. **Never commit `docker/secrets/`** to git (already in `.gitignore`)
3. **Docker Secrets** - Passwords are stored securely in `docker/secrets/` files
4. Use strong passwords (auto-generated by `create_secrets.sh`)
5. Keep Docker and images updated
6. Regularly renew SSL certificates (automatic with certbot)
7. Monitor logs for suspicious activity
8. Use firewall to restrict access to ports 80, 443 only
9. Secrets files have 600 permissions (owner read/write only)

### Docker Secrets Management

For detailed information on managing Docker secrets, see:
- **[Docker Secrets Guide](DOCKER_SECRETS_GUIDE.md)** - Complete secrets management guide
- **[Docker Secrets Implementation](DOCKER_SECRETS_IMPLEMENTATION.md)** - Technical implementation details

## Performance Tuning

### PostgreSQL
- Adjust `shared_buffers`, `effective_cache_size` in docker-compose.prod.yml based on available RAM

### Redis
- Adjust `maxmemory` based on available RAM
- Tune `maxmemory-policy` based on use case

### Backend
- Adjust `DATABASE_POOL_MIN_SIZE` and `DATABASE_POOL_MAX_SIZE` based on load
- Monitor with Prometheus metrics endpoint

## Monitoring

The production stack includes a complete monitoring solution:

### Services

- **Prometheus** (port 9090 internal) - Metrics collection and storage
- **Grafana** (port 3001 internal) - Visualization dashboards  
- **Alertmanager** (port 9093 internal) - Alert routing and notification
- **Node Exporter** - System metrics (CPU, memory, disk, network)
- **Redis Exporter** - Redis performance metrics
- **PostgreSQL Exporter** - Database metrics

### Accessing Monitoring

**Grafana:**
- Default URL: `http://your-domain:3001` (configure via Nginx)
- Default username: `admin`
- Password: Set via Docker Secret `grafana_admin_password`

**Prometheus:**
- Internal: `http://prometheus:9090`
- External: Configure via Nginx if needed

**Metrics Endpoints:**
- Backend: `http://backend:8000/metrics` (internal)
- Node Exporter: `http://node-exporter:9100/metrics`
- Redis Exporter: `http://redis-exporter:9121/metrics`
- PostgreSQL Exporter: `http://postgres-exporter:9187/metrics`

### Pre-configured Dashboards

Grafana includes pre-configured dashboards for:
- Backend API metrics
- Database performance
- Redis cache statistics
- System resources

See `docker/grafana/dashboards/` for dashboard definitions.

### Alert Notifications

Alertmanager is configured to send alerts via:
- **Slack** - Team notifications (configure `SLACK_WEBHOOK_URL`)
- **Email** - Critical alerts (configure SMTP settings)
- **PagerDuty** - On-call escalation (optional)

**Setup**:
1. Add notification settings to `.env.production`:
   ```bash
   SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
   ALERT_EMAIL_TO=alerts@your-domain.com
   SMTP_HOST=smtp.gmail.com:587
   SMTP_USERNAME=your-email@gmail.com
   SMTP_PASSWORD=your-app-password
   ```

2. Restart Alertmanager:
   ```bash
   docker-compose -f docker-compose.prod.yml restart alertmanager
   ```

For detailed setup instructions, see **[Alertmanager Setup Guide](ALERTMANAGER_SETUP.md)**.

## Backup

### Automated Backups

The production stack includes an automated backup service (`postgres-backup`) that:
- Runs daily at 2:00 AM UTC
- Creates PostgreSQL dumps in custom format
- Retains backups for 7 days
- Stores backups in `postgres_backup` volume

**Backup Location:** `/backups` inside the backup container

**Manual Backup:**
```bash
docker exec r3mes-postgres-backup-prod ls -lh /backups
```

### Manual Database Backup
```bash
docker exec r3mes-postgres-prod pg_dump -U r3mes r3mes > backup.sql
```

### Volume Backup
```bash
docker run --rm -v r3mes_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup.tar.gz /data
```

### Restore from Backup
```bash
# Copy backup file to container
docker cp backup.sql r3mes-postgres-prod:/tmp/

# Restore
docker exec -i r3mes-postgres-prod psql -U r3mes r3mes < /tmp/backup.sql
```

## Updates

To update services:

1. Pull latest code
2. Rebuild images: `docker-compose -f docker-compose.prod.yml build`
3. Restart services: `docker-compose -f docker-compose.prod.yml up -d`

## Support

For issues, check:
- Service logs
- Health check status
- Network connectivity tests
- Documentation in `docs/` directory


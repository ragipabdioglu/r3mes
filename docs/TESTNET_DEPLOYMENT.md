# R3MES Testnet Deployment Guide

This guide explains how to deploy R3MES to a testnet environment.

## Overview

Testnet deployment follows the same architecture as mainnet, but with testnet-specific configuration:
- **Chain ID**: `remes-testnet-1` (or your testnet chain ID)
- **Domain**: `testnet.r3mes.network` (or your testnet domain)
- **URLs**: All public URLs should point to testnet endpoints

## Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- Domain configured for testnet (DNS pointing to your server)
- Port 80 and 443 open in firewall
- (Optional) NVIDIA GPU for mining

## Quick Start

### Option 1: Quick Deploy (Recommended - One Command)

```bash
git clone <your-repo-url> R3MES
cd R3MES
bash scripts/quick_deploy.sh --domain testnet.r3mes.network --email admin@r3mes.network
```

That's it! The script automatically:
- ✅ Checks/installs Docker and Docker Compose
- ✅ Creates Docker secrets with random passwords
- ✅ Configures environment variables
- ✅ Deploys all services

**Custom Chain ID:**
```bash
bash scripts/quick_deploy.sh \
  --domain testnet.r3mes.network \
  --email admin@r3mes.network \
  --chain-id remes-testnet-1
```

### Option 2: Manual Deploy (Step by Step)

#### 1. Prepare Environment

#### Create Docker Secrets

```bash
bash scripts/create_secrets.sh
```

This creates secure password files:
- `docker/secrets/postgres_password.txt`
- `docker/secrets/redis_password.txt`
- `docker/secrets/grafana_admin_password.txt`

#### Configure Environment Variables

```bash
cd docker
cp env.testnet.example .env.production
nano .env.production  # Edit and set testnet-specific values
```

**Critical Variables for Testnet:**
```bash
# Blockchain Configuration
CHAIN_ID=remes-testnet-1  # Your testnet chain ID

# Domain Configuration
DOMAIN=testnet.r3mes.network  # Your testnet domain
CORS_ALLOWED_ORIGINS=https://testnet.r3mes.network,https://testnet-www.r3mes.network

# Frontend URLs
NEXT_PUBLIC_CHAIN_ID=remes-testnet-1  # Must match CHAIN_ID
NEXT_PUBLIC_SITE_URL=https://testnet.r3mes.network
NEXT_PUBLIC_API_URL=https://testnet.r3mes.network/api
NEXT_PUBLIC_RPC_URL=https://testnet-rpc.r3mes.network
NEXT_PUBLIC_REST_URL=https://testnet-api.r3mes.network

# Email for Let's Encrypt
EMAIL=admin@r3mes.network
```

#### 2. Deploy

```bash
bash scripts/deploy_production_docker.sh
```

Or manually:

```bash
cd docker
docker-compose -f docker-compose.prod.yml up -d
```

### 3. Verify Deployment

#### Check Service Status

```bash
cd docker
docker-compose -f docker-compose.prod.yml ps
```

All services should show `Up` status with healthy healthchecks.

#### Check Health Endpoints

- Frontend: `https://testnet.r3mes.network/health`
- Backend: `https://testnet.r3mes.network/api/health`
- Blockchain: `https://testnet.r3mes.network/api/blockchain/health`

#### Check Logs

```bash
# Backend logs
docker logs -f r3mes-backend-prod

# Blockchain logs
docker logs -f r3mes-blockchain-prod
```

## Serving Node Configuration (Optional)

If you want to run miners as serving nodes on testnet to provide distributed LoRA inference:

### Miner Environment Variables

Add to your miner's environment configuration:

```bash
# Enable serving node mode
R3MES_ENABLE_SERVING_NODE=true

# Backend URL (testnet)
R3MES_BACKEND_URL=https://testnet.r3mes.network/api

# Serving node endpoint (your miner's public IP)
R3MES_SERVING_NODE_HOST=0.0.0.0
R3MES_SERVING_NODE_PORT=8081

# Or set full URL explicitly:
# R3MES_SERVING_NODE_URL=http://your-public-ip:8081
```

### Register LoRA Adapters

Before miners can serve LoRAs, register them in the backend:

```bash
curl -X POST https://testnet.r3mes.network/api/lora/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "legal",
    "ipfs_hash": "QmYourIPFSHashHere",
    "description": "Legal domain LoRA adapter",
    "category": "legal",
    "version": "1.0"
  }'
```

### Verify Serving Nodes

Check registered serving nodes:

```bash
# List all serving nodes
curl https://testnet.r3mes.network/api/serving-node/list

# List serving nodes for a specific LoRA
curl https://testnet.r3mes.network/api/serving-node/list?lora=legal
```

### Detailed Setup

For complete serving node setup instructions, see [Serving Node Setup Guide](../SERVING_NODE_SETUP.md).

# Frontend logs
docker logs -f r3mes-frontend-prod
```

## Genesis Initialization

### First-Time Setup

If this is the first time deploying the blockchain node, you need to initialize genesis:

#### Option 1: Default Genesis (Quick Start)

The blockchain node will auto-initialize with default genesis on first start if the volume is empty.

#### Option 2: Custom Genesis (Production)

For production testnet, initialize with custom genesis:

```bash
# 1. Generate genesis trap jobs
python scripts/generate_genesis_traps.py --count 50 --output genesis_vault_entries.json

# 2. Finalize genesis
python scripts/finalize_genesis.py \
  --model-hash "YOUR_IPFS_CID" \
  --model-version "v1.0.0" \
  --chain-id "remes-testnet-1" \
  --output remes/config/genesis.json \
  --vault-entries genesis_vault_entries.json

# 3. Validate genesis
python scripts/validate_genesis.py remes/config/genesis.json

# 4. Initialize blockchain node with custom genesis
docker-compose -f docker/docker-compose.prod.yml exec remesd remesd init testnet-node --chain-id remes-testnet-1 --home /app/.remesd

# 5. Copy genesis.json to container
docker cp remes/config/genesis.json r3mes-blockchain-prod:/app/.remesd/config/genesis.json

# 6. Restart blockchain node
docker-compose -f docker/docker-compose.prod.yml restart remesd
```

## Database Migrations

Database migrations run automatically when the backend container starts. The `init_db.sh` script:

1. Waits for PostgreSQL to be ready
2. Reads PostgreSQL password from Docker secrets
3. Runs Alembic migrations (`alembic upgrade head`)
4. Starts the backend application

To check migration status:

```bash
docker logs r3mes-backend-prod | grep -i "migration"
```

## Monitoring

### Grafana Dashboard

Access Grafana at `https://testnet.r3mes.network:3001`

- Default username: `admin`
- Password: From `docker/secrets/grafana_admin_password.txt`

### Prometheus Metrics

Access Prometheus at `https://testnet.r3mes.network:9090`

### Alertmanager

Access Alertmanager at `https://testnet.r3mes.network:9093`

Configure alerts in `.env.production`:
```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
ALERT_EMAIL_TO=alerts@r3mes.network
SMTP_HOST=smtp.gmail.com:587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
```

## Backups

Automated daily backups are configured via the `postgres-backup` service:

- **Location**: `docker/postgres_backup/`
- **Retention**: 7 days
- **Schedule**: Daily at 02:00 UTC

To restore from backup:

```bash
bash scripts/restore_database.sh /path/to/backup.sql
```

## SSL/HTTPS

SSL certificates are automatically obtained and renewed via Let's Encrypt:

1. Ensure `DOMAIN` is set correctly in `.env.production`
2. Ensure port 80 is accessible (for ACME challenge)
3. Deploy stack - Certbot will obtain certificates automatically
4. Certificates renew every 12 hours automatically

## Troubleshooting

### Backend Container Fails to Start

**Problem**: Backend container exits with error about database connection.

**Solution**: Check PostgreSQL logs and ensure migrations completed:
```bash
docker logs r3mes-postgres-prod
docker logs r3mes-backend-prod | grep -i "migration\|postgres"
```

### Blockchain Node Not Syncing

**Problem**: Blockchain node shows "not syncing" or "catching up" for extended period.

**Solution**: 
1. Check if genesis.json is correctly configured
2. Check if CHAIN_ID matches in all services
3. Check blockchain logs: `docker logs -f r3mes-blockchain-prod`

### SSL Certificate Issues

**Problem**: SSL certificate not obtained or expired.

**Solution**:
1. Check Certbot logs: `docker logs r3mes-certbot-prod`
2. Ensure port 80 is accessible
3. Ensure DNS is correctly configured
4. Manually renew: `docker-compose -f docker/docker-compose.prod.yml exec certbot certbot renew`

### CORS Errors

**Problem**: Frontend cannot connect to backend API due to CORS errors.

**Solution**: 
1. Check `CORS_ALLOWED_ORIGINS` in `.env.production` matches your testnet domain
2. Ensure no localhost URLs in production mode
3. Restart backend: `docker-compose -f docker/docker-compose.prod.yml restart backend`

## Differences from Mainnet

1. **Chain ID**: Testnet uses `remes-testnet-1` instead of `remes-mainnet-1`
2. **Domain**: Testnet uses `testnet.r3mes.network` instead of `r3mes.network`
3. **Faucet Limits**: Testnet may have higher faucet limits for testing
4. **Monitoring**: Separate Grafana dashboards recommended for testnet vs mainnet

## Production Checklist

Before going live on testnet:

- [ ] All environment variables set correctly
- [ ] Docker secrets created
- [ ] CHAIN_ID matches testnet chain ID
- [ ] Domain DNS configured correctly
- [ ] SSL certificates obtained
- [ ] Database migrations completed successfully
- [ ] Blockchain node syncing correctly
- [ ] Health checks passing
- [ ] Monitoring configured (Grafana, Alertmanager)
- [ ] Backups configured
- [ ] Firewall rules configured (only 22, 80, 443 open)
- [ ] Faucet wallet funded (if using faucet)

## Support

For issues or questions:
- Check logs: `docker logs -f <container-name>`
- Check health endpoints
- Review monitoring dashboards
- Check `docs/TROUBLESHOOTING.md` for common issues


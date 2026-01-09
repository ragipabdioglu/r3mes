# R3MES Migration Guide

**Version**: 1.0.0  
**Last Updated**: 2025-12-24

---

## Overview

This guide provides step-by-step instructions for migrating R3MES components between environments (development, staging, production) and upgrading between versions.

---

## Table of Contents

1. [Environment Migration](#environment-migration)
2. [Version Upgrades](#version-upgrades)
3. [Database Migrations](#database-migrations)
4. [Configuration Migration](#configuration-migration)
5. [Data Migration](#data-migration)
6. [Rollback Procedures](#rollback-procedures)

---

## Environment Migration

### Development → Staging

#### 1. Backend Service

**Prerequisites**:
- Staging environment configured
- Database backup completed
- Environment variables set

**Steps**:

```bash
# 1. Backup development database
pg_dump -h localhost -U r3mes_user r3mes_dev > backup_dev_$(date +%Y%m%d).sql

# 2. Update environment variables
export R3MES_ENV=staging
export DATABASE_URL=postgresql://user:pass@staging-db:5432/r3mes_staging
export REDIS_URL=redis://staging-redis:6379/0
export BLOCKCHAIN_RPC_URL=http://staging-node:26657
export CORS_ALLOWED_ORIGINS=https://staging.r3mes.io

# 3. Run database migrations
cd backend
alembic upgrade head

# 4. Deploy backend service
docker-compose -f docker/docker-compose.staging.yml up -d backend

# 5. Verify deployment
curl https://staging-api.r3mes.io/health
```

#### 2. Blockchain Node

**Steps**:

```bash
# 1. Export genesis state (if needed)
remesd export > genesis_export.json

# 2. Update node configuration
export R3MES_ENV=staging
export R3MES_GRPC_ADDR=staging-node:9090
export R3MES_TENDERMINT_RPC_ADDR=http://staging-node:26657
export CORS_ALLOWED_ORIGINS=https://staging.r3mes.io

# 3. Start node
remesd start --home ~/.remesd-staging
```

#### 3. Miner Engine

**Steps**:

```bash
# 1. Update environment variables
export R3MES_ENV=staging
export R3MES_NODE_GRPC_URL=staging-node:9090
export R3MES_BACKEND_URL=https://staging-api.r3mes.io
export R3MES_IPFS_URL=http://staging-ipfs:5001

# 2. Restart miner
systemctl restart r3mes-miner
```

#### 4. Web Dashboard

**Steps**:

```bash
# 1. Update environment variables
export NEXT_PUBLIC_API_URL=https://staging-api.r3mes.io
export NEXT_PUBLIC_BACKEND_URL=https://staging-api.r3mes.io
export NEXT_PUBLIC_BLOCKCHAIN_RPC_URL=http://staging-node:26657
export NEXT_PUBLIC_WS_URL=wss://staging-api.r3mes.io

# 2. Build and deploy
cd web-dashboard
npm run build
npm run start
```

### Staging → Production

**Critical Checklist**:

- [ ] All secrets migrated to secret management service
- [ ] CORS configuration verified (no localhost)
- [ ] SSL/TLS certificates configured
- [ ] Monitoring and alerting set up
- [ ] Backup procedures tested
- [ ] Rollback plan documented
- [ ] Load testing completed
- [ ] Security audit passed

**Steps**:

```bash
# 1. Final staging backup
pg_dump -h staging-db -U user r3mes_staging > backup_staging_$(date +%Y%m%d).sql

# 2. Update production secrets
# Use secret management service (AWS Secrets Manager, Vault, etc.)
aws secretsmanager update-secret \
  --secret-id r3mes/production/database_url \
  --secret-string "postgresql://user:pass@prod-db:5432/r3mes_prod"

# 3. Deploy with zero-downtime strategy
# Use blue-green deployment or rolling updates

# 4. Verify production deployment
curl https://api.r3mes.io/health
```

---

## Version Upgrades

### Backend Service Upgrade

#### From v0.x to v1.0.0

**Breaking Changes**:
- Environment variable names standardized
- CORS configuration requires explicit origins in production
- Input validation strengthened (may reject previously valid inputs)

**Migration Steps**:

```bash
# 1. Review breaking changes
# See CHANGELOG.md for details

# 2. Backup current version
docker tag r3mes-backend:latest r3mes-backend:v0.x-backup

# 3. Update environment variables
# Rename old variables to new standardized names
export DATABASE_URL=${OLD_DB_URL}
export REDIS_URL=${OLD_REDIS_URL}
export CORS_ALLOWED_ORIGINS=${OLD_CORS_ORIGINS}

# 4. Run database migrations
cd backend
alembic upgrade head

# 5. Deploy new version
docker-compose up -d backend

# 6. Verify upgrade
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/status
```

### Blockchain Node Upgrade

#### From v0.x to v1.0.0

**Breaking Changes**:
- Module parameters structure changed
- Query endpoints updated
- gRPC service definitions updated

**Migration Steps**:

```bash
# 1. Stop node
systemctl stop remesd

# 2. Backup state
cp -r ~/.remesd/data ~/.remesd/data.backup
remesd export > state_export_$(date +%Y%m%d).json

# 3. Update binary
# Download new version or build from source
go install ./cmd/remesd

# 4. Run migration (if needed)
remesd migrate

# 5. Start node
systemctl start remesd

# 6. Verify upgrade
remesd status
curl http://localhost:26657/status
```

### Miner Engine Upgrade

#### From v0.x to v1.0.0

**Breaking Changes**:
- Environment variable names changed
- gRPC client updated
- IPFS integration improved

**Migration Steps**:

```bash
# 1. Stop miner
systemctl stop r3mes-miner

# 2. Update environment variables
# Update .env file with new variable names
export R3MES_NODE_GRPC_URL=${OLD_NODE_URL}
export R3MES_BACKEND_URL=${OLD_BACKEND_URL}

# 3. Update code
git pull origin main
pip install -r requirements.txt

# 4. Restart miner
systemctl start r3mes-miner

# 5. Verify upgrade
journalctl -u r3mes-miner -f
```

---

## Database Migrations

### PostgreSQL Migrations

**Using Alembic (Backend)**:

```bash
# 1. Create migration
cd backend
alembic revision --autogenerate -m "description"

# 2. Review migration file
# Edit alembic/versions/xxx_description.py if needed

# 3. Apply migration
alembic upgrade head

# 4. Verify migration
psql -h localhost -U user -d r3mes -c "\d table_name"
```

**Manual Migration**:

```sql
-- Example: Add new column
ALTER TABLE users ADD COLUMN api_key_hash VARCHAR(255);

-- Example: Create index
CREATE INDEX idx_users_wallet_address ON users(wallet_address);

-- Example: Update data
UPDATE users SET api_key_hash = encode(digest(api_key, 'sha256'), 'hex') WHERE api_key IS NOT NULL;
```

### State Migration (Blockchain)

**Export State**:

```bash
# Export current state
remesd export > state_export.json

# Export specific module state
remesd query remes params --output json > params_export.json
```

**Import State**:

```bash
# Import state (requires genesis)
remesd init validator --chain-id r3mes-testnet
remesd import state_export.json
```

---

## Configuration Migration

### Environment Variables

**Standardization**:

Old variable names → New standardized names:

```
OLD_DB_URL → DATABASE_URL
OLD_REDIS_URL → REDIS_URL
OLD_NODE_URL → R3MES_NODE_GRPC_URL
OLD_BACKEND_URL → R3MES_BACKEND_URL
OLD_CORS_ORIGINS → CORS_ALLOWED_ORIGINS
```

**Migration Script**:

```bash
#!/bin/bash
# migrate_env.sh

# Read old .env file
source .env.old

# Write new .env file
cat > .env.new << EOF
DATABASE_URL=${OLD_DB_URL}
REDIS_URL=${OLD_REDIS_URL}
R3MES_NODE_GRPC_URL=${OLD_NODE_URL}
R3MES_BACKEND_URL=${OLD_BACKEND_URL}
CORS_ALLOWED_ORIGINS=${OLD_CORS_ORIGINS}
EOF
```

### Secret Management Migration

**From Environment Variables to Secret Management Service**:

```python
# Old: Environment variables
import os
database_url = os.getenv("DATABASE_URL")

# New: Secret management service
import boto3
import json

def get_secret(secret_name: str) -> dict:
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

secrets = get_secret('r3mes/production')
database_url = secrets['database_url']
```

---

## Data Migration

### User Data Migration

**Export Users**:

```sql
-- Export users to CSV
COPY (
    SELECT wallet_address, credits, created_at, updated_at
    FROM users
) TO '/tmp/users_export.csv' WITH CSV HEADER;
```

**Import Users**:

```sql
-- Import users from CSV
COPY users(wallet_address, credits, created_at, updated_at)
FROM '/tmp/users_import.csv' WITH CSV HEADER;
```

### Blockchain State Migration

**Export Task Pools**:

```bash
# Query all task pools
remesd query remes task-pools --output json > task_pools_export.json
```

**Import Task Pools**:

```bash
# Create task pools via governance or admin commands
# (Requires appropriate permissions)
```

---

## Rollback Procedures

### Backend Service Rollback

**Quick Rollback**:

```bash
# 1. Stop current version
docker-compose stop backend

# 2. Start previous version
docker-compose -f docker-compose.rollback.yml up -d backend

# 3. Verify rollback
curl http://localhost:8000/health
```

**Database Rollback**:

```bash
# 1. Restore database backup
psql -h localhost -U user -d r3mes < backup_$(date +%Y%m%d).sql

# 2. Rollback Alembic migration
cd backend
alembic downgrade -1

# 3. Verify rollback
psql -h localhost -U user -d r3mes -c "SELECT version_num FROM alembic_version;"
```

### Blockchain Node Rollback

**State Rollback**:

```bash
# 1. Stop node
systemctl stop remesd

# 2. Restore state backup
rm -r ~/.remesd/data
cp -r ~/.remesd/data.backup ~/.remesd/data

# 3. Restore binary (if needed)
cp remesd.v0.x /usr/local/bin/remesd

# 4. Start node
systemctl start remesd

# 5. Verify rollback
remesd status
```

### Miner Engine Rollback

**Code Rollback**:

```bash
# 1. Stop miner
systemctl stop r3mes-miner

# 2. Checkout previous version
git checkout v0.x
pip install -r requirements.txt

# 3. Restart miner
systemctl start r3mes-miner

# 4. Verify rollback
journalctl -u r3mes-miner -f
```

---

## Best Practices

### Pre-Migration Checklist

- [ ] Backup all data (database, state, files)
- [ ] Document current configuration
- [ ] Test migration in staging environment
- [ ] Prepare rollback plan
- [ ] Notify users (if applicable)
- [ ] Schedule maintenance window (if needed)

### During Migration

- [ ] Monitor logs for errors
- [ ] Verify each step before proceeding
- [ ] Keep backups accessible
- [ ] Document any issues encountered

### Post-Migration

- [ ] Verify all services are running
- [ ] Test critical functionality
- [ ] Monitor metrics and alerts
- [ ] Update documentation
- [ ] Archive old backups

---

## Troubleshooting

### Common Issues

**Issue**: Database migration fails

**Solution**:
```bash
# Check migration status
alembic current

# Review migration file
cat alembic/versions/xxx_description.py

# Manually fix if needed
psql -h localhost -U user -d r3mes -c "ALTER TABLE ..."
```

**Issue**: Environment variables not loading

**Solution**:
```bash
# Verify .env file exists
ls -la .env

# Check variable names
cat .env | grep R3MES

# Reload environment
source .env
```

**Issue**: Service won't start after migration

**Solution**:
```bash
# Check logs
docker-compose logs backend
journalctl -u r3mes-miner -n 100

# Verify configuration
docker-compose config

# Rollback if needed
# Follow rollback procedures above
```

---

## Support

For migration assistance:

1. **Check Documentation**: Review relevant docs in `docs/`
2. **Review Logs**: Check service logs for errors
3. **Open Issue**: Create GitHub issue with details
4. **Contact Support**: Reach out to development team

---

**Last Updated**: 2025-12-24


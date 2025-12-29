# R3MES Disaster Recovery Plan

**Version**: 1.0.0  
**Last Updated**: 2025-12-24  
**Review Frequency**: Quarterly

---

## Overview

This document outlines the disaster recovery procedures for the R3MES platform, including backup strategies, recovery procedures, and business continuity plans.

---

## Table of Contents

1. [Recovery Objectives](#recovery-objectives)
2. [Backup Strategy](#backup-strategy)
3. [Recovery Procedures](#recovery-procedures)
4. [Incident Response](#incident-response)
5. [Testing & Validation](#testing--validation)
6. [Contact Information](#contact-information)

---

## Recovery Objectives

### Recovery Time Objectives (RTO)

| Component | RTO | Priority |
|-----------|-----|----------|
| Blockchain Node | 1 hour | Critical |
| Backend API | 30 minutes | Critical |
| Database | 15 minutes | Critical |
| Web Dashboard | 1 hour | High |
| Miner Engine | 2 hours | Medium |

### Recovery Point Objectives (RPO)

| Component | RPO | Backup Frequency |
|-----------|-----|------------------|
| Database | 5 minutes | Continuous (WAL) + Hourly snapshots |
| Blockchain State | 1 block | Every block (automatic) |
| Configuration | 1 hour | Hourly |
| Logs | 1 day | Daily |

---

## Backup Strategy

### Database Backups

#### PostgreSQL (Backend)

**Automated Backups**:

Use the provided backup script with systemd timer:

```bash
# Install backup timer
sudo ./scripts/setup_backup_cron.sh

# Manual backup
./scripts/backup_database.sh

# Backup with S3 upload
export S3_BUCKET=r3mes-backups
./scripts/backup_database.sh
```

**Backup Schedule**:
- Daily backups at 2 AM (retained for 7 days locally)
- Weekly full backup on Sunday (retained for 4 weeks)
- Monthly archive (uploaded to S3, retained for 12 months)

**WAL Archiving** (Continuous):

```postgresql
-- postgresql.conf
wal_level = replica
archive_mode = on
archive_command = 'cp %p /backups/wal/%f'
```

**Backup Verification**:

```bash
# Verify backup integrity
pg_restore --list /backups/db/daily/r3mes_20251224.dump | head -20

# Test restore (to temporary database)
createdb r3mes_test
pg_restore -d r3mes_test /backups/db/daily/r3mes_20251224.dump
```

#### Redis Backups

**Snapshot Backups**:

```bash
# Redis RDB snapshot (configured in redis.conf)
save 900 1      # Save if at least 1 key changed in 900 seconds
save 300 10     # Save if at least 10 keys changed in 300 seconds
save 60 10000   # Save if at least 10000 keys changed in 60 seconds

# Manual backup
redis-cli BGSAVE
cp /var/lib/redis/dump.rdb /backups/redis/redis_$(date +%Y%m%d_%H%M%S).rdb
```

**AOF (Append-Only File)**:

```bash
# redis.conf
appendonly yes
appendfsync everysec
```

### Blockchain State Backups

#### Node State

**Automated Backups**:

```bash
# Daily state export
0 3 * * * remesd export > /backups/blockchain/state/state_$(date +\%Y\%m\%d).json

# Block height tracking
remesd status | jq '.sync_info.latest_block_height' > /backups/blockchain/height_$(date +\%Y\%m\%d).txt
```

**Data Directory Backup**:

```bash
# Full data directory backup (weekly)
0 4 * * 0 tar -czf /backups/blockchain/data/data_$(date +\%Y\%m\%d).tar.gz ~/.remesd/data
```

#### Genesis & Configuration

```bash
# Backup genesis file
cp ~/.remesd/config/genesis.json /backups/blockchain/genesis/genesis_$(date +%Y%m%d).json

# Backup configuration
cp ~/.remesd/config/config.toml /backups/blockchain/config/config_$(date +%Y%m%d).toml
cp ~/.remesd/config/app.toml /backups/blockchain/config/app_$(date +%Y%m%d).toml
```

### Application Backups

#### Code & Configuration

**Git Repository**:

```bash
# Regular git backups (automated via CI/CD)
# All code is version controlled in Git
git remote add backup https://backup-repo.example.com/r3mes.git
git push backup main
```

**Configuration Files**:

```bash
# Backup environment files (without secrets)
cp .env.example /backups/config/env.example_$(date +%Y%m%d)
cp docker-compose.yml /backups/config/docker-compose_$(date +%Y%m%d).yml
```

#### Docker Images

```bash
# Save Docker images
docker save r3mes-backend:latest | gzip > /backups/docker/backend_$(date +%Y%m%d).tar.gz
docker save r3mes-miner:latest | gzip > /backups/docker/miner_$(date +%Y%m%d).tar.gz
```

### Secret Backups

**⚠️ CRITICAL: Never backup secrets in plain text**

**Secret Management Service**:

```bash
# Export secrets from secret management service (encrypted)
# AWS Secrets Manager
aws secretsmanager get-secret-value --secret-id r3mes/production/database_url > /backups/secrets/db_url_encrypted.json

# HashiCorp Vault
vault kv get -format=json r3mes/production > /backups/secrets/vault_encrypted.json
```

**Key Rotation Records**:

```bash
# Document key rotation dates (not the keys themselves)
echo "Database password rotated: $(date)" >> /backups/secrets/rotation_log.txt
```

---

## Recovery Procedures

### Database Recovery

#### Full Database Restore

```bash
# 1. Stop application
docker-compose stop backend

# 2. Drop existing database (if needed)
dropdb -h localhost -U postgres r3mes_prod

# 3. Create new database
createdb -h localhost -U postgres r3mes_prod

# 4. Restore from backup
pg_restore -h localhost -U r3mes_user -d r3mes_prod /backups/db/daily/r3mes_20251224.dump

# 5. Verify restore
psql -h localhost -U r3mes_user -d r3mes_prod -c "SELECT COUNT(*) FROM users;"

# 6. Start application
docker-compose start backend
```

#### Point-in-Time Recovery (PITR)

```bash
# 1. Restore base backup
pg_restore -h localhost -U r3mes_user -d r3mes_prod /backups/db/daily/r3mes_20251224.dump

# 2. Restore WAL files up to target time
pg_basebackup -D /var/lib/postgresql/data -Ft -z -P

# 3. Configure recovery
cat > /var/lib/postgresql/data/recovery.conf << EOF
restore_command = 'cp /backups/wal/%f %p'
recovery_target_time = '2025-12-24 14:30:00'
EOF

# 4. Start PostgreSQL (will automatically recover)
systemctl start postgresql
```

#### Redis Recovery

```bash
# 1. Stop Redis
systemctl stop redis

# 2. Restore RDB file
cp /backups/redis/redis_20251224_120000.rdb /var/lib/redis/dump.rdb
chown redis:redis /var/lib/redis/dump.rdb

# 3. Start Redis
systemctl start redis

# 4. Verify
redis-cli PING
redis-cli DBSIZE
```

### Blockchain Node Recovery

#### Full Node Recovery

```bash
# 1. Stop node
systemctl stop remesd

# 2. Restore data directory
rm -rf ~/.remesd/data
tar -xzf /backups/blockchain/data/data_20251224.tar.gz -C ~/.remesd/

# 3. Restore configuration (if needed)
cp /backups/blockchain/config/config_20251224.toml ~/.remesd/config/config.toml
cp /backups/blockchain/config/app_20251224.toml ~/.remesd/config/app.toml

# 4. Start node
systemctl start remesd

# 5. Verify
remesd status
curl http://localhost:26657/status
```

#### State Import (from Export)

```bash
# 1. Stop node
systemctl stop remesd

# 2. Initialize new node (if needed)
remesd init validator --chain-id r3mes-mainnet

# 3. Import state
remesd import /backups/blockchain/state/state_20251224.json

# 4. Start node
systemctl start remesd
```

### Backend Service Recovery

#### Application Recovery

```bash
# 1. Restore code
git clone https://github.com/your-org/r3mes.git
cd r3mes/backend

# 2. Restore configuration
cp /backups/config/env.example .env
# Manually set secrets from secret management service

# 3. Restore database (see Database Recovery above)

# 4. Start services
docker-compose up -d

# 5. Verify
curl http://localhost:8000/health
```

#### Docker Image Recovery

```bash
# 1. Load Docker image
docker load < /backups/docker/backend_20251224.tar.gz

# 2. Tag image
docker tag r3mes-backend:latest r3mes-backend:restored

# 3. Update docker-compose.yml to use restored image

# 4. Start service
docker-compose up -d backend
```

### Web Dashboard Recovery

```bash
# 1. Restore code
git clone https://github.com/your-org/r3mes.git
cd r3mes/web-dashboard

# 2. Install dependencies
npm install

# 3. Restore configuration
cp /backups/config/env.example .env.local
# Set environment variables

# 4. Build
npm run build

# 5. Start
npm run start
```

---

## Incident Response

### Severity Levels

#### Critical (P0)
- **Definition**: Complete service outage, data loss, security breach
- **Response Time**: Immediate (< 15 minutes)
- **Escalation**: On-call engineer + CTO

#### High (P1)
- **Definition**: Major feature unavailable, performance degradation
- **Response Time**: 1 hour
- **Escalation**: On-call engineer

#### Medium (P2)
- **Definition**: Minor feature unavailable, non-critical issues
- **Response Time**: 4 hours
- **Escalation**: Regular business hours

#### Low (P3)
- **Definition**: Cosmetic issues, minor bugs
- **Response Time**: Next business day
- **Escalation**: Regular business hours

### Incident Response Procedure

1. **Detection**: Monitor alerts, user reports, logs
2. **Assessment**: Determine severity and impact
3. **Containment**: Isolate affected systems
4. **Recovery**: Execute recovery procedures
5. **Verification**: Test and validate recovery
6. **Communication**: Notify stakeholders
7. **Post-Mortem**: Document incident and lessons learned

### Communication Plan

**Internal**:
- Slack: #r3mes-incidents
- Email: incidents@r3mes.io
- Phone: On-call rotation

**External**:
- Status page: https://status.r3mes.io
- Twitter: @r3mes_status
- Email: support@r3mes.io

---

## Testing & Validation

### Backup Testing

**Monthly Tests**:

```bash
# Test database restore
createdb r3mes_test
pg_restore -d r3mes_test /backups/db/daily/r3mes_$(date +%Y%m%d).dump
psql -d r3mes_test -c "SELECT COUNT(*) FROM users;"
dropdb r3mes_test

# Test blockchain state import
remesd export > test_export.json
diff test_export.json /backups/blockchain/state/state_$(date +%Y%m%d).json
```

**Quarterly Disaster Recovery Drills**:

1. Simulate complete system failure
2. Execute recovery procedures
3. Measure RTO and RPO
4. Document findings
5. Update procedures

### Recovery Validation Checklist

- [ ] Database restored and verified
- [ ] Blockchain node synced
- [ ] Backend API responding
- [ ] Web dashboard accessible
- [ ] All services healthy
- [ ] Data integrity verified
- [ ] Performance metrics normal
- [ ] User access restored

---

## Backup Storage

### On-Site Storage

- **Location**: Primary data center
- **Retention**: 30 days
- **Format**: Encrypted backups

### Off-Site Storage

- **Location**: Secondary data center / Cloud storage
- **Retention**: 90 days
- **Format**: Encrypted, compressed
- **Replication**: Daily sync

### Cloud Storage

- **Provider**: AWS S3 / Google Cloud Storage
- **Retention**: 1 year
- **Format**: Encrypted, versioned
- **Access**: IAM-controlled

---

## Contact Information

### On-Call Rotation

**Primary On-Call**:
- Name: [Engineer Name]
- Phone: +1-XXX-XXX-XXXX
- Email: oncall@r3mes.io

**Secondary On-Call**:
- Name: [Engineer Name]
- Phone: +1-XXX-XXX-XXXX
- Email: oncall-backup@r3mes.io

### Escalation Contacts

**CTO**:
- Name: [CTO Name]
- Phone: +1-XXX-XXX-XXXX
- Email: cto@r3mes.io

**Infrastructure Lead**:
- Name: [Lead Name]
- Phone: +1-XXX-XXX-XXXX
- Email: infra@r3mes.io

### External Vendors

**Cloud Provider Support**:
- AWS: [Support Plan Details]
- Google Cloud: [Support Plan Details]

**Database Support**:
- PostgreSQL: [Support Contact]

---

## Appendix

### Backup Locations

```
/backups/
├── db/
│   ├── daily/          # Daily database backups
│   ├── weekly/         # Weekly database backups
│   ├── monthly/        # Monthly database backups
│   └── wal/            # WAL archive files
├── blockchain/
│   ├── state/          # State exports
│   ├── data/           # Data directory backups
│   ├── genesis/        # Genesis file backups
│   └── config/         # Configuration backups
├── redis/              # Redis RDB backups
├── docker/             # Docker image backups
├── config/             # Configuration file backups
└── secrets/            # Encrypted secret backups
```

### Recovery Scripts

All recovery scripts are located in `/scripts/recovery/`:

- `restore_database.sh`: Database restore script
- `restore_blockchain.sh`: Blockchain node restore script
- `restore_backend.sh`: Backend service restore script
- `verify_backup.sh`: Backup verification script

---

**Last Updated**: 2025-12-24  
**Next Review**: 2026-03-24  
**Document Owner**: Infrastructure Team


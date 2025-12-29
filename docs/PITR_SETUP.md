# PostgreSQL Point-in-Time Recovery (PITR) Setup

This guide explains how to set up PostgreSQL Point-in-Time Recovery for R3MES.

## Overview

PITR allows you to restore the database to any point in time, not just to the last backup. This is achieved through:
- WAL (Write-Ahead Log) archiving
- Base backups
- Recovery configuration

## Setup

### 1. Run Setup Script

```bash
sudo ./scripts/setup_pitr.sh
```

This script:
- Creates WAL archive directory
- Configures PostgreSQL for WAL archiving
- Sets up archive command
- Restarts PostgreSQL

### 2. Create Initial Base Backup

```bash
sudo -u postgres pg_basebackup \
  -D /backups/base/base_$(date +%Y%m%d) \
  -Ft -z -P
```

### 3. Schedule Base Backups

Add to crontab:

```bash
# Weekly base backup (Sunday at 1 AM)
0 1 * * 0 sudo -u postgres pg_basebackup -D /backups/base/base_$(date +\%Y\%m\%d) -Ft -z -P
```

## PITR Restore Procedure

### 1. Stop PostgreSQL

```bash
sudo systemctl stop postgresql
```

### 2. Restore Base Backup

```bash
# Remove current data directory
sudo rm -rf /var/lib/postgresql/data/*

# Restore base backup
sudo -u postgres tar -xzf /backups/base/base_20241224/base.tar.gz -C /var/lib/postgresql/data/
```

### 3. Configure Recovery

Create `recovery.conf`:

```conf
restore_command = 'cp /backups/wal/%f %p'
recovery_target_time = '2025-12-24 12:00:00'
recovery_target_action = 'promote'
```

### 4. Start PostgreSQL

```bash
sudo systemctl start postgresql
```

PostgreSQL will automatically replay WAL files until the target time.

## Monitoring

### Check WAL Archive Status

```sql
SELECT * FROM pg_stat_archiver;
```

### Check Archive Lag

```bash
# Find latest archived WAL
ls -t /backups/wal/ | head -1

# Compare with current WAL
sudo -u postgres psql -c "SELECT pg_current_wal_lsn();"
```

## Best Practices

1. **Regular Base Backups**: Create base backups weekly
2. **WAL Archive Monitoring**: Monitor archive directory size
3. **S3 Backup**: Upload WAL files to S3 for off-site storage
4. **Test Restores**: Regularly test PITR restore procedure
5. **Retention Policy**: Keep WAL files for at least 7 days

## Troubleshooting

### Archive Command Failing

Check PostgreSQL logs:
```bash
sudo tail -f /var/log/postgresql/postgresql-*.log
```

### WAL Archive Directory Full

Set up automatic cleanup:
```bash
# Remove WAL files older than 7 days
find /backups/wal -name "*.wal" -mtime +7 -delete
```


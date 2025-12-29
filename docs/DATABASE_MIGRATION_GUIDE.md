# Database Migration Guide

This guide explains how to migrate from SQLite to PostgreSQL for production deployment.

## Overview

R3MES uses Alembic for database schema versioning and migrations. This guide covers:
- Initial migration setup
- Production migration procedure
- Data migration from SQLite to PostgreSQL
- Verification and rollback

## Prerequisites

- PostgreSQL 12+ installed and running
- Python 3.10+ with dependencies installed
- Access to SQLite database file
- Backup of SQLite database

## Initial Setup

### 1. Install Alembic

```bash
cd backend
pip install alembic
```

### 2. Initialize Alembic (if not already done)

```bash
alembic init alembic
```

### 3. Configure Alembic

Edit `alembic/env.py` to use the database models (already configured).

## Running Migrations

### Development

```bash
# Create a new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1
```

### Production

```bash
# Set DATABASE_URL
export DATABASE_URL=postgresql://user:password@host:5432/r3mes

# Run migrations
alembic upgrade head
```

## Production Migration (SQLite → PostgreSQL)

### 1. Backup SQLite Database

```bash
cp /var/lib/r3mes/database.db /backups/database_backup_$(date +%Y%m%d).db
```

### 2. Run Migration Script

```bash
python scripts/migrate_to_postgresql_production.py \
  --source-sqlite /var/lib/r3mes/database.db \
  --target-postgresql postgresql://user:password@host:5432/r3mes \
  --backup-path /backups/
```

This script:
- Creates SQLite backup
- Verifies PostgreSQL connection
- Runs Alembic migrations
- Migrates data from SQLite to PostgreSQL
- Verifies migration

### 3. Verify Migration

```bash
python scripts/verify_migration.py \
  --source-sqlite /var/lib/r3mes/database.db \
  --target-postgresql postgresql://user:password@host:5432/r3mes
```

### 4. Update Configuration

Update environment variables:
```bash
export DATABASE_TYPE=postgresql
export DATABASE_URL=postgresql://user:password@host:5432/r3mes
```

### 5. Restart Services

```bash
systemctl restart r3mes-backend
```

## Migration Files

- `backend/alembic/versions/001_initial_migration.py` - Initial schema
- `backend/alembic/versions/002_add_indexes.py` - Performance indexes

## Rollback

If migration fails, you can rollback:

```bash
# Rollback Alembic migrations
alembic downgrade base

# Restore from SQLite backup
cp /backups/database_backup_YYYYMMDD.db /var/lib/r3mes/database.db
```

## Troubleshooting

### Migration Fails

1. Check PostgreSQL connection
2. Verify database user permissions
3. Check disk space
4. Review Alembic logs

### Data Mismatch

1. Compare row counts
2. Verify sample records
3. Check foreign key constraints
4. Review migration logs

## Best Practices

1. **Always Backup**: Create backups before migration
2. **Test First**: Test migration in staging environment
3. **Verify**: Always verify migration results
4. **Monitor**: Monitor database performance after migration
5. **Document**: Document any custom migration steps


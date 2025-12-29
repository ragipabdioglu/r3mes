#!/bin/bash
# Database Backup Script
#
# Creates a compressed backup of PostgreSQL database with timestamp naming.
# Optionally uploads to S3 for cloud storage.
#
# Usage:
#   ./scripts/backup_database.sh
#
# Environment Variables:
#   DATABASE_URL: PostgreSQL connection URL (required)
#   BACKUP_DIR: Backup directory (default: /backups)
#   S3_BUCKET: S3 bucket for backup storage (optional)
#   AWS_REGION: AWS region for S3 (optional)
#   RETENTION_DAYS: Number of days to keep local backups (default: 7)

set -e

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/backups}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/r3mes_${DATE}.sql"
BACKUP_FILE_GZ="${BACKUP_FILE}.gz"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
    log_error "DATABASE_URL environment variable is not set"
    exit 1
fi

# Create backup directory
mkdir -p "$BACKUP_DIR"

log_info "Starting database backup..."
log_info "Backup directory: $BACKUP_DIR"
log_info "Timestamp: $DATE"

# Create PostgreSQL dump
log_info "Creating database dump..."
if pg_dump "$DATABASE_URL" > "$BACKUP_FILE"; then
    log_info "✅ Database dump created: $BACKUP_FILE"
else
    log_error "Failed to create database dump"
    exit 1
fi

# Compress backup
log_info "Compressing backup..."
if gzip "$BACKUP_FILE"; then
    log_info "✅ Backup compressed: $BACKUP_FILE_GZ"
    BACKUP_SIZE=$(du -h "$BACKUP_FILE_GZ" | cut -f1)
    log_info "Backup size: $BACKUP_SIZE"
else
    log_error "Failed to compress backup"
    exit 1
fi

# Upload to S3 if configured
if [ -n "$S3_BUCKET" ]; then
    log_info "Uploading to S3 bucket: $S3_BUCKET"
    
    if command -v aws &> /dev/null; then
        S3_PATH="s3://${S3_BUCKET}/database-backups/r3mes_${DATE}.sql.gz"
        
        if aws s3 cp "$BACKUP_FILE_GZ" "$S3_PATH" ${AWS_REGION:+--region $AWS_REGION}; then
            log_info "✅ Backup uploaded to S3: $S3_PATH"
        else
            log_warn "Failed to upload to S3 (backup file still exists locally)"
        fi
    else
        log_warn "AWS CLI not found, skipping S3 upload"
    fi
fi

# Clean up old backups (local retention)
log_info "Cleaning up old backups (retention: $RETENTION_DAYS days)..."
find "$BACKUP_DIR" -name "r3mes_*.sql.gz" -type f -mtime +$RETENTION_DAYS -delete
REMAINING=$(find "$BACKUP_DIR" -name "r3mes_*.sql.gz" -type f | wc -l)
log_info "Remaining backups: $REMAINING"

log_info "✅ Backup completed successfully: $BACKUP_FILE_GZ"

# Exit with success
exit 0


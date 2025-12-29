#!/bin/bash
# Database Restore Script
#
# Restores PostgreSQL database from a backup file.
#
# Usage:
#   ./scripts/restore_database.sh [backup_file]
#
# Environment Variables:
#   DATABASE_URL: PostgreSQL connection URL (required)
#   BACKUP_DIR: Backup directory (default: /backups)

set -e

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/backups}"

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

# Get backup file
if [ -n "$1" ]; then
    BACKUP_FILE="$1"
else
    # List available backups
    log_info "Available backups:"
    ls -lh "$BACKUP_DIR"/r3mes_*.sql.gz 2>/dev/null | tail -10 || {
        log_error "No backups found in $BACKUP_DIR"
        exit 1
    }
    
    read -p "Enter backup file name (or full path): " BACKUP_FILE
fi

# Resolve backup file path
if [[ "$BACKUP_FILE" != /* ]]; then
    BACKUP_FILE="$BACKUP_DIR/$BACKUP_FILE"
fi

# Check if backup file exists
if [ ! -f "$BACKUP_FILE" ]; then
    log_error "Backup file not found: $BACKUP_FILE"
    exit 1
fi

log_info "Restoring from backup: $BACKUP_FILE"
log_info "Target database: $DATABASE_URL"

# Confirm restore
read -p "⚠️  This will overwrite the current database. Continue? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    log_info "Restore cancelled"
    exit 0
fi

# Extract database name from DATABASE_URL
DB_NAME=$(echo "$DATABASE_URL" | sed -n 's/.*\/\([^?]*\).*/\1/p')

if [ -z "$DB_NAME" ]; then
    log_error "Could not extract database name from DATABASE_URL"
    exit 1
fi

log_info "Database name: $DB_NAME"

# Decompress if needed
if [[ "$BACKUP_FILE" == *.gz ]]; then
    log_info "Decompressing backup..."
    TEMP_FILE="${BACKUP_FILE%.gz}"
    gunzip -c "$BACKUP_FILE" > "$TEMP_FILE"
    RESTORE_FILE="$TEMP_FILE"
else
    RESTORE_FILE="$BACKUP_FILE"
fi

# Restore database
log_info "Restoring database..."
if psql "$DATABASE_URL" < "$RESTORE_FILE"; then
    log_info "✅ Database restored successfully"
else
    log_error "Failed to restore database"
    # Clean up temp file if created
    if [ -n "$TEMP_FILE" ] && [ -f "$TEMP_FILE" ]; then
        rm "$TEMP_FILE"
    fi
    exit 1
fi

# Clean up temp file if created
if [ -n "$TEMP_FILE" ] && [ -f "$TEMP_FILE" ]; then
    rm "$TEMP_FILE"
fi

log_info "✅ Restore completed successfully"


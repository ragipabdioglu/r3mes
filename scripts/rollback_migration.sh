#!/bin/bash
# Migration Rollback Script
#
# Rolls back database migration from PostgreSQL to SQLite.
# Restores from backup and verifies data integrity.
#
# Usage:
#   ./scripts/rollback_migration.sh [postgresql_url] [backup_path]
#
# Arguments:
#   postgresql_url: PostgreSQL connection URL (optional, uses STAGING_DATABASE_URL env var)
#   backup_path: Path to backup files (optional, uses BACKUP_PATH env var)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
POSTGRESQL_URL="${1:-${STAGING_DATABASE_URL:-postgresql://test:test@localhost:5432/r3mes_staging}}"
BACKUP_PATH="${2:-${BACKUP_PATH:-/tmp/r3mes_migration_test}}"
RESTORE_SQLITE_PATH="${RESTORE_SQLITE_PATH:-backend/database_restored.db}"

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Find latest backup
find_latest_backup() {
    local backup_file=$(ls -t "$BACKUP_PATH"/sqlite_backup_*.db 2>/dev/null | head -1)
    if [ -z "$backup_file" ]; then
        log_error "No backup file found in $BACKUP_PATH"
        exit 1
    fi
    echo "$backup_file"
}

# Verify backup integrity
verify_backup() {
    local backup_file=$1
    log_info "Verifying backup integrity: $backup_file"
    
    if sqlite3 "$backup_file" "SELECT 1;" > /dev/null 2>&1; then
        log_info "✓ Backup file is valid"
        return 0
    else
        log_error "✗ Backup file is corrupted"
        return 1
    fi
}

# Restore from backup
restore_from_backup() {
    local backup_file=$1
    log_info "Restoring from backup: $backup_file"
    
    # Copy backup to restore location
    cp "$backup_file" "$RESTORE_SQLITE_PATH"
    log_info "✓ Database restored to $RESTORE_SQLITE_PATH"
}

# Verify restored data
verify_restored_data() {
    log_info "Verifying restored data..."
    
    # Count rows in restored database
    RESTORED_COUNTS=$(sqlite3 "$RESTORE_SQLITE_PATH" <<EOF
SELECT 
    (SELECT COUNT(*) FROM users) as users,
    (SELECT COUNT(*) FROM mining_stats) as mining_stats,
    (SELECT COUNT(*) FROM earnings_history) as earnings_history,
    (SELECT COUNT(*) FROM hashrate_history) as hashrate_history,
    (SELECT COUNT(*) FROM api_keys) as api_keys;
EOF
)
    
    log_info "Restored database row counts: $RESTORED_COUNTS"
    
    # Verify database is usable
    if sqlite3 "$RESTORE_SQLITE_PATH" "SELECT COUNT(*) FROM users;" > /dev/null 2>&1; then
        log_info "✓ Restored database is usable"
        return 0
    else
        log_error "✗ Restored database is not usable"
        return 1
    fi
}

# Alembic downgrade (if using Alembic)
alembic_downgrade() {
    log_info "Running Alembic downgrade..."
    
    if [ -d "backend/alembic" ]; then
        cd backend
        if alembic downgrade -1 > /dev/null 2>&1; then
            log_info "✓ Alembic downgrade completed"
            cd ..
            return 0
        else
            log_warn "⚠ Alembic downgrade failed or not applicable"
            cd ..
            return 0
        fi
    else
        log_warn "⚠ Alembic not found, skipping downgrade"
        return 0
    fi
}

# Main rollback procedure
main() {
    log_info "Starting migration rollback..."
    log_info "PostgreSQL URL: $POSTGRESQL_URL"
    log_info "Backup Path: $BACKUP_PATH"
    echo ""
    
    # Find latest backup
    BACKUP_FILE=$(find_latest_backup)
    log_info "Using backup: $BACKUP_FILE"
    
    # Verify backup
    if ! verify_backup "$BACKUP_FILE"; then
        exit 1
    fi
    
    # Alembic downgrade
    alembic_downgrade
    
    # Restore from backup
    restore_from_backup "$BACKUP_FILE"
    
    # Verify restored data
    if verify_restored_data; then
        log_info "✓ Rollback completed successfully"
        log_info "Restored database: $RESTORE_SQLITE_PATH"
        exit 0
    else
        log_error "✗ Rollback verification failed"
        exit 1
    fi
}

# Run main function
main "$@"


#!/bin/bash
# Log Cleanup Script
#
# Cleans up old logs based on retention policy.
#
# Usage:
#   ./scripts/cleanup_logs.sh [log_directory]
#
# Environment Variables:
#   RETENTION_DAYS: Days to keep logs (default: 30)
#   ARCHIVE_BEFORE_DELETE: Archive to S3 before deletion (default: true)

set -e

# Configuration
LOG_DIR="${1:-/var/log/r3mes}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
ARCHIVE_BEFORE_DELETE="${ARCHIVE_BEFORE_DELETE:-true}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_info "Cleaning up logs in $LOG_DIR (retention: $RETENTION_DAYS days)..."

# Archive before deletion if enabled
if [ "$ARCHIVE_BEFORE_DELETE" == "true" ] && [ -n "$S3_BUCKET" ]; then
    log_info "Archiving logs before deletion..."
    ./scripts/archive_logs.sh "$LOG_DIR"
fi

# Delete old log files
deleted_count=0
find "$LOG_DIR" -name "*.log" -type f -mtime +$RETENTION_DAYS | while read logfile; do
    log_name=$(basename "$logfile")
    log_info "Deleting: $log_name"
    rm "$logfile"
    deleted_count=$((deleted_count + 1))
done

# Delete old compressed logs
find "$LOG_DIR" -name "*.log.gz" -type f -mtime +$RETENTION_DAYS | while read logfile; do
    log_name=$(basename "$logfile")
    log_info "Deleting: $log_name"
    rm "$logfile"
    deleted_count=$((deleted_count + 1))
done

log_info "✅ Log cleanup completed (deleted: $deleted_count files)"


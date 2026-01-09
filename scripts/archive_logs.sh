#!/bin/bash
# Log Archiving Script
#
# Archives logs to S3 for long-term storage.
#
# Usage:
#   ./scripts/archive_logs.sh [log_directory]
#
# Environment Variables:
#   S3_BUCKET: S3 bucket name (required)
#   AWS_REGION: AWS region (default: us-east-1)
#   RETENTION_DAYS: Days to keep logs locally (default: 30)

set -e

# Configuration
LOG_DIR="${1:-/var/log/r3mes}"
S3_BUCKET="${S3_BUCKET}"
AWS_REGION="${AWS_REGION:-us-east-1}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
ARCHIVE_DIR="/tmp/r3mes-log-archive"
DATE=$(date +%Y%m%d)

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

# Check if S3_BUCKET is set
if [ -z "$S3_BUCKET" ]; then
    log_error "S3_BUCKET environment variable is not set"
    exit 1
fi

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    log_error "AWS CLI is not installed"
    exit 1
fi

# Create archive directory
mkdir -p "$ARCHIVE_DIR"

log_info "Archiving logs from $LOG_DIR..."

# Archive logs older than retention period
find "$LOG_DIR" -name "*.log" -type f -mtime +$RETENTION_DAYS | while read logfile; do
    log_name=$(basename "$logfile")
    archive_name="${DATE}_${log_name}.gz"
    
    log_info "Archiving: $log_name"
    
    # Compress log file
    gzip -c "$logfile" > "$ARCHIVE_DIR/$archive_name"
    
    # Upload to S3
    s3_path="s3://${S3_BUCKET}/logs/${archive_name}"
    if aws s3 cp "$ARCHIVE_DIR/$archive_name" "$s3_path" --region "$AWS_REGION"; then
        log_info "✅ Uploaded to S3: $s3_path"
        
        # Verify S3 upload success
        if aws s3 ls "$s3_path" --region "$AWS_REGION" > /dev/null 2>&1; then
            log_info "✅ S3 upload verified"
            # Remove local log file after successful upload and verification
            rm "$logfile"
        else
            log_error "❌ S3 upload verification failed for $log_name"
            # Keep local file if verification fails
        fi
    else
        log_error "❌ Failed to upload $log_name to S3"
        # Keep local file if upload fails
    fi
done

# Clean up archive directory
rm -rf "$ARCHIVE_DIR"

log_info "✅ Log archiving completed"


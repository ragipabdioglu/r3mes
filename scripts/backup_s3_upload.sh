#!/bin/bash
# S3 Backup Upload Script
#
# Uploads database backups to AWS S3 with lifecycle management.
#
# Usage:
#   ./scripts/backup_s3_upload.sh [backup_file]
#
# Environment Variables:
#   S3_BUCKET: S3 bucket name (required)
#   AWS_REGION: AWS region (default: us-east-1)
#   AWS_ACCESS_KEY_ID: AWS access key (optional, if using IAM role)
#   AWS_SECRET_ACCESS_KEY: AWS secret key (optional, if using IAM role)

set -e

# Configuration
S3_BUCKET="${S3_BUCKET}"
AWS_REGION="${AWS_REGION:-us-east-1}"
BACKUP_DIR="${BACKUP_DIR:-/backups}"

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

# Get backup file
if [ -n "$1" ]; then
    BACKUP_FILE="$1"
else
    # Get latest backup
    BACKUP_FILE=$(ls -t "$BACKUP_DIR"/r3mes_*.sql.gz 2>/dev/null | head -1)
    
    if [ -z "$BACKUP_FILE" ]; then
        log_error "No backup files found in $BACKUP_DIR"
        exit 1
    fi
fi

# Check if backup file exists
if [ ! -f "$BACKUP_FILE" ]; then
    log_error "Backup file not found: $BACKUP_FILE"
    exit 1
fi

BACKUP_NAME=$(basename "$BACKUP_FILE")
S3_PATH="s3://${S3_BUCKET}/database-backups/${BACKUP_NAME}"

log_info "Uploading backup to S3..."
log_info "Backup file: $BACKUP_FILE"
log_info "S3 path: $S3_PATH"

# Upload to S3
if aws s3 cp "$BACKUP_FILE" "$S3_PATH" --region "$AWS_REGION"; then
    log_info "✅ Backup uploaded to S3: $S3_PATH"
    
    # Verify upload
    if aws s3 ls "$S3_PATH" --region "$AWS_REGION" > /dev/null 2>&1; then
        log_info "✅ Upload verified"
    else
        log_warn "⚠️  Upload verification failed"
    fi
else
    log_error "❌ Failed to upload backup to S3"
    exit 1
fi

log_info "✅ S3 upload completed successfully"


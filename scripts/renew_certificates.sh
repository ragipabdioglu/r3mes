#!/bin/bash
# Certificate Renewal Script
#
# Renews Let's Encrypt certificates with zero-downtime.
#
# Usage:
#   sudo ./scripts/renew_certificates.sh

set -e

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

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    log_error "This script must be run as root (use sudo)"
    exit 1
fi

log_info "Renewing Let's Encrypt certificates..."

# Pre-renewal validation
log_info "Pre-renewal checks..."

# Check if certbot is installed
if ! command -v certbot &> /dev/null; then
    log_error "Certbot is not installed"
    exit 1
fi

# Check if Nginx is running
if ! systemctl is-active --quiet nginx; then
    log_error "Nginx is not running"
    exit 1
fi

# Renew certificates
log_info "Renewing certificates..."
if certbot renew --quiet --no-self-upgrade; then
    log_info "✅ Certificates renewed successfully"
else
    log_error "❌ Certificate renewal failed"
    exit 1
fi

# Reload Nginx with new certificates (graceful reload, zero downtime)
log_info "Reloading Nginx with new certificates..."
if systemctl reload nginx; then
    log_info "✅ Nginx reloaded successfully"
else
    log_error "❌ Nginx reload failed"
    exit 1
fi

# Verify certificates
log_info "Verifying certificates..."
certbot certificates

log_info "✅ Certificate renewal completed successfully"

# Optional: Send notification on failure
# if [ $? -ne 0 ]; then
#     # Send alert (Slack, email, etc.)
#     echo "Certificate renewal failed" | mail -s "R3MES Certificate Renewal Failed" admin@r3mes.network
# fi


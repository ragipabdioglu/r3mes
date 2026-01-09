#!/bin/bash
# Let's Encrypt SSL Certificate Setup
#
# Obtains and configures SSL certificates for R3MES production domains.
#
# Usage:
#   sudo ./scripts/setup_letsencrypt.sh [domain1] [domain2] ...
#
# Example:
#   sudo ./scripts/setup_letsencrypt.sh r3mes.network api.r3mes.network

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

# Get domains from arguments or use defaults
if [ $# -eq 0 ]; then
    DOMAINS=("r3mes.network" "api.r3mes.network" "www.r3mes.network")
    log_warn "No domains specified, using defaults: ${DOMAINS[*]}"
else
    DOMAINS=("$@")
fi

log_info "Setting up Let's Encrypt SSL certificates for domains: ${DOMAINS[*]}"

# Check if certbot is installed
if ! command -v certbot &> /dev/null; then
    log_info "Installing Certbot..."
    if command -v apt-get &> /dev/null; then
        apt-get update
        apt-get install -y certbot python3-certbot-nginx
    elif command -v yum &> /dev/null; then
        yum install -y certbot python3-certbot-nginx
    else
        log_error "Package manager not found (apt-get or yum required)"
        exit 1
    fi
fi

# Create certbot webroot directory
mkdir -p /var/www/certbot

# Ensure Nginx is running (required for domain validation)
if ! systemctl is-active --quiet nginx; then
    log_info "Starting Nginx..."
    systemctl start nginx
fi

# Obtain certificates for each domain
for DOMAIN in "${DOMAINS[@]}"; do
    log_info "Obtaining certificate for $DOMAIN..."
    
    # Check if certificate already exists
    if [ -d "/etc/letsencrypt/live/$DOMAIN" ]; then
        log_warn "Certificate for $DOMAIN already exists, skipping..."
        continue
    fi
    
    # Obtain certificate using webroot method
    if certbot certonly \
        --webroot \
        --webroot-path=/var/www/certbot \
        --email admin@r3mes.network \
        --agree-tos \
        --no-eff-email \
        --non-interactive \
        -d "$DOMAIN"; then
        log_info "✅ Certificate obtained for $DOMAIN"
    else
        log_error "❌ Failed to obtain certificate for $DOMAIN"
        exit 1
    fi
done

# Reload Nginx to use new certificates
log_info "Reloading Nginx to use new certificates..."
systemctl reload nginx

log_info ""
log_info "✅ SSL certificates setup completed successfully!"
log_info ""
log_info "Certificates are located at:"
for DOMAIN in "${DOMAINS[@]}"; do
    log_info "  - /etc/letsencrypt/live/$DOMAIN/"
done
log_info ""
log_info "Next steps:"
log_info "1. Set up automatic renewal:"
log_info "   ./scripts/setup_certbot_renewal.sh"
log_info "2. Verify certificates:"
log_info "   certbot certificates"
log_info "3. Test HTTPS:"
log_info "   curl -I https://${DOMAINS[0]}"


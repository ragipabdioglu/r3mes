#!/bin/bash
# Setup Nginx Reverse Proxy for Production
#
# Installs and configures Nginx with HTTPS for R3MES production deployment.
#
# Usage:
#   sudo ./scripts/setup_nginx.sh [frontend|backend|all]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

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

# Determine which configs to install
INSTALL_TYPE="${1:-all}"

log_info "Setting up Nginx reverse proxy for R3MES production..."

# Check if Nginx is installed
if ! command -v nginx &> /dev/null; then
    log_info "Installing Nginx..."
    if command -v apt-get &> /dev/null; then
        apt-get update
        apt-get install -y nginx
    elif command -v yum &> /dev/null; then
        yum install -y nginx
    else
        log_error "Package manager not found (apt-get or yum required)"
        exit 1
    fi
fi

# Create certbot directory for ACME challenges
mkdir -p /var/www/certbot

# Install frontend config
if [ "$INSTALL_TYPE" == "frontend" ] || [ "$INSTALL_TYPE" == "all" ]; then
    log_info "Installing frontend Nginx configuration..."
    cp "$PROJECT_ROOT/nginx/nginx.prod.conf" /etc/nginx/sites-available/r3mes-frontend
    ln -sf /etc/nginx/sites-available/r3mes-frontend /etc/nginx/sites-enabled/r3mes-frontend
    log_info "✅ Frontend configuration installed"
fi

# Install backend config
if [ "$INSTALL_TYPE" == "backend" ] || [ "$INSTALL_TYPE" == "all" ]; then
    log_info "Installing backend Nginx configuration..."
    cp "$PROJECT_ROOT/nginx/nginx-backend.conf" /etc/nginx/sites-available/r3mes-backend
    ln -sf /etc/nginx/sites-available/r3mes-backend /etc/nginx/sites-enabled/r3mes-backend
    log_info "✅ Backend configuration installed"
fi

# Test Nginx configuration
log_info "Testing Nginx configuration..."
if nginx -t; then
    log_info "✅ Nginx configuration is valid"
else
    log_error "❌ Nginx configuration test failed"
    exit 1
fi

# Reload Nginx
log_info "Reloading Nginx..."
systemctl reload nginx || service nginx reload

log_info ""
log_info "✅ Nginx setup completed successfully!"
log_info ""
log_info "Next steps:"
log_info "1. Obtain SSL certificates using Let's Encrypt:"
log_info "   ./scripts/setup_letsencrypt.sh"
log_info "2. Verify HTTPS is working:"
log_info "   curl -I https://r3mes.network"
log_info "3. Check Nginx status:"
log_info "   systemctl status nginx"


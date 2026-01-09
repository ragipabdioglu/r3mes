#!/bin/bash
# R3MES Production Firewall Setup Script
# Configures UFW firewall for production deployment

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    print_error "Please run as root (use sudo)"
    exit 1
fi

print_header "R3MES Production Firewall Setup"

# Check if UFW is installed
if ! command -v ufw &> /dev/null; then
    print_warning "UFW is not installed. Installing..."
    apt-get update
    apt-get install -y ufw
    print_success "UFW installed"
fi

# Reset UFW to defaults (optional, comment out if you want to keep existing rules)
print_warning "This will reset UFW to defaults. Continue? (y/N)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    ufw --force reset
    print_success "UFW reset"
else
    print_warning "Skipping UFW reset"
fi

# Set default policies
print_header "Setting Default Policies"
ufw default deny incoming
ufw default allow outgoing
print_success "Default policies set: deny incoming, allow outgoing"

# Allow SSH (IMPORTANT: Do this first!)
print_header "Configuring SSH Access"
print_warning "Allowing SSH on port 22. Make sure you have SSH access before continuing!"
ufw allow 22/tcp comment 'SSH'
print_success "SSH (port 22) allowed"

# Allow HTTP and HTTPS
print_header "Configuring Web Access"
ufw allow 80/tcp comment 'HTTP - Let'\''s Encrypt'
ufw allow 443/tcp comment 'HTTPS'
print_success "HTTP (port 80) and HTTPS (port 443) allowed"

# Optional: Allow blockchain ports if needed for external access
# WARNING: Only enable if you need external access to blockchain node
print_header "Blockchain Ports Configuration"
print_warning "Do you want to allow external access to blockchain ports? (y/N)"
print_warning "Ports: 26656 (P2P), 26657 (RPC), 9090 (gRPC), 1317 (REST)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    ufw allow 26656/tcp comment 'Blockchain P2P'
    ufw allow 26657/tcp comment 'Blockchain RPC'
    ufw allow 9090/tcp comment 'Blockchain gRPC'
    ufw allow 1317/tcp comment 'Blockchain REST'
    print_success "Blockchain ports allowed"
    print_warning "⚠️  Blockchain ports are now accessible from internet!"
else
    print_success "Blockchain ports kept internal (recommended)"
fi

# Enable UFW
print_header "Enabling Firewall"
ufw --force enable
print_success "UFW enabled"

# Show status
print_header "Firewall Status"
ufw status verbose

print_header "Firewall Setup Complete"
print_success "Firewall configured successfully!"
print_warning "Make sure you can still SSH to the server before closing this session!"


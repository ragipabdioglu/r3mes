#!/bin/bash
# R3MES Full Testnet Installation Script
# Bu script Docker dahil HER ŞEYİ kurar
# Kullanım: curl -sSL https://raw.githubusercontent.com/YOUR_USERNAME/R3MES/main/scripts/full_install.sh | bash

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

print_header "R3MES Full Testnet Installation"
echo ""
echo "Bu script şunları kuracak:"
echo "  1. Docker & Docker Compose"
echo "  2. Git (gerekirse)"
echo "  3. R3MES Projesi"
echo "  4. Tüm servisleri başlatacak"
echo ""

# ============================================
# STEP 1: System Update
# ============================================
print_header "Step 1: System Update"
apt update && apt upgrade -y
print_success "System updated"

# ============================================
# STEP 2: Install Dependencies
# ============================================
print_header "Step 2: Installing Dependencies"

# Git
if ! command -v git &> /dev/null; then
    apt install -y git
    print_success "Git installed"
else
    print_success "Git already installed"
fi

# curl, wget, openssl
apt install -y curl wget openssl ca-certificates gnupg lsb-release
print_success "Dependencies installed"

# ============================================
# STEP 3: Install Docker
# ============================================
print_header "Step 3: Installing Docker"

if ! command -v docker &> /dev/null; then
    print_info "Installing Docker..."
    
    # Docker GPG key
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    
    # Docker repository
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Install Docker
    apt update
    apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    
    # Enable Docker
    systemctl enable docker
    systemctl start docker
    
    print_success "Docker installed and started"
else
    print_success "Docker already installed"
fi

# Verify Docker
docker --version
docker compose version
print_success "Docker verification complete"

# ============================================
# STEP 4: Configure Firewall
# ============================================
print_header "Step 4: Configuring Firewall"

if command -v ufw &> /dev/null; then
    ufw allow 22/tcp   # SSH
    ufw allow 80/tcp   # HTTP
    ufw allow 443/tcp  # HTTPS
    ufw allow 26656/tcp # P2P
    ufw allow 26657/tcp # RPC
    ufw allow 9090/tcp  # gRPC
    ufw allow 1317/tcp  # REST
    ufw --force enable
    print_success "Firewall configured"
else
    print_warning "UFW not found, skipping firewall configuration"
fi

# ============================================
# STEP 5: Clone R3MES Project
# ============================================
print_header "Step 5: Cloning R3MES Project"

cd /root

if [ -d "R3MES" ]; then
    print_warning "R3MES directory exists, pulling latest changes..."
    cd R3MES
    git pull origin main || git pull origin master
else
    print_info "Cloning R3MES repository..."
    # TODO: Bu URL'yi kendi GitHub URL'inizle değiştirin
    git clone https://github.com/YOUR_USERNAME/R3MES.git
    cd R3MES
fi

print_success "R3MES project ready"

# ============================================
# STEP 6: Create Docker Secrets
# ============================================
print_header "Step 6: Creating Docker Secrets"

cd /root/R3MES/docker

if [ ! -d "secrets" ]; then
    mkdir -p secrets
fi

if [ ! -f "secrets/postgres_password.txt" ]; then
    openssl rand -base64 32 | tr -d "=+/" | cut -c1-32 > secrets/postgres_password.txt
    openssl rand -base64 32 | tr -d "=+/" | cut -c1-32 > secrets/redis_password.txt
    openssl rand -base64 32 | tr -d "=+/" | cut -c1-32 > secrets/grafana_admin_password.txt
    chmod 600 secrets/*.txt
    print_success "Secrets created"
else
    print_success "Secrets already exist"
fi

# ============================================
# STEP 7: Setup Environment
# ============================================
print_header "Step 7: Setting Up Environment"

if [ ! -f ".env" ]; then
    cp .env.testnet .env
    print_success "Environment file created from .env.testnet"
else
    print_success "Environment file already exists"
fi

# ============================================
# STEP 8: Build and Start Services
# ============================================
print_header "Step 8: Building Docker Images"

docker compose -f docker-compose.prod.yml build

print_header "Step 9: Starting Services"

docker compose -f docker-compose.prod.yml up -d

# ============================================
# STEP 10: Wait for Services
# ============================================
print_header "Step 10: Waiting for Services to Start"

echo ""
print_info "Waiting for blockchain to initialize (1-3 minutes on first start)..."

MAX_WAIT=180
WAIT_COUNT=0
while ! curl -s http://localhost:26657/status > /dev/null 2>&1; do
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
    if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
        print_warning "Blockchain still starting. Check: docker logs r3mes-blockchain-prod"
        break
    fi
    echo -n "."
done
echo ""

# Check services
sleep 10

echo ""
print_header "Service Status"

if curl -s http://localhost:26657/status > /dev/null 2>&1; then
    print_success "Blockchain: Running ✅"
else
    print_warning "Blockchain: Starting..."
fi

if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    print_success "Backend API: Running ✅"
else
    print_warning "Backend API: Starting..."
fi

if curl -s http://localhost:3000 > /dev/null 2>&1; then
    print_success "Frontend: Running ✅"
else
    print_warning "Frontend: Starting..."
fi

# ============================================
# FINAL SUMMARY
# ============================================
print_header "🎉 Installation Complete!"

echo ""
echo "Services:"
echo "  - Blockchain RPC:    http://$(hostname -I | awk '{print $1}'):26657"
echo "  - Blockchain gRPC:   $(hostname -I | awk '{print $1}'):9090"
echo "  - Blockchain REST:   http://$(hostname -I | awk '{print $1}'):1317"
echo "  - Backend API:       http://$(hostname -I | awk '{print $1}'):8000"
echo "  - Frontend:          http://$(hostname -I | awk '{print $1}'):3000"
echo ""
echo "Commands:"
echo "  - View logs:    docker logs -f r3mes-blockchain-prod"
echo "  - All logs:     docker compose -f docker-compose.prod.yml logs -f"
echo "  - Stop:         docker compose -f docker-compose.prod.yml down"
echo "  - Restart:      docker compose -f docker-compose.prod.yml restart"
echo ""
echo "Grafana Password: $(cat /root/R3MES/docker/secrets/grafana_admin_password.txt)"
echo ""
print_success "Testnet is ready! 🚀"

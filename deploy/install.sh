#!/bin/bash

# ============================================================================
# R3MES One-Click Production Deployment Script
# Domain: remes.network
# VPS IP: 38.242.246.6
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOMAIN="remes.network"
API_DOMAIN="api.remes.network"
TESTNET_DOMAIN="testnet.remes.network"
RPC_DOMAIN="rpc.remes.network"
VPS_IP="38.242.246.6"
SSL_EMAIL="admin@remes.network"
INSTALL_DIR="/opt/r3mes"

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║     ██████╗ ██████╗ ███╗   ███╗███████╗███████╗             ║"
echo "║     ██╔══██╗╚════██╗████╗ ████║██╔════╝██╔════╝             ║"
echo "║     ██████╔╝ █████╔╝██╔████╔██║█████╗  ███████╗             ║"
echo "║     ██╔══██╗ ╚═══██╗██║╚██╔╝██║██╔══╝  ╚════██║             ║"
echo "║     ██║  ██║██████╔╝██║ ╚═╝ ██║███████╗███████║             ║"
echo "║     ╚═╝  ╚═╝╚═════╝ ╚═╝     ╚═╝╚══════╝╚══════╝             ║"
echo "║                                                              ║"
echo "║           Production Deployment Script v1.0                  ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${YELLOW}Configuration:${NC}"
echo "  Domain: $DOMAIN"
echo "  API: $API_DOMAIN"
echo "  Testnet: $TESTNET_DOMAIN"
echo "  VPS IP: $VPS_IP"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Error: Please run as root (sudo ./install.sh)${NC}"
    exit 1
fi

# ============================================================================
# Step 1: System Update & Dependencies
# ============================================================================
echo -e "\n${GREEN}[1/8] Updating system and installing dependencies...${NC}"

apt-get update -qq
apt-get upgrade -y -qq
apt-get install -y -qq \
    curl \
    wget \
    git \
    vim \
    htop \
    ufw \
    fail2ban \
    ca-certificates \
    gnupg \
    lsb-release

# ============================================================================
# Step 2: Install Docker
# ============================================================================
echo -e "\n${GREEN}[2/8] Installing Docker...${NC}"

if ! command -v docker &> /dev/null; then
    # Add Docker's official GPG key
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg

    # Add the repository
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      tee /etc/apt/sources.list.d/docker.list > /dev/null

    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    systemctl start docker
    systemctl enable docker
    echo -e "${GREEN}Docker installed successfully${NC}"
else
    echo -e "${YELLOW}Docker already installed${NC}"
fi

# ============================================================================
# Step 3: Configure Firewall
# ============================================================================
echo -e "\n${GREEN}[3/8] Configuring firewall...${NC}"

ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 8000/tcp
ufw allow 26656/tcp
ufw allow 26657/tcp
ufw --force enable

echo -e "${GREEN}Firewall configured${NC}"

# ============================================================================
# Step 4: Create Directory Structure
# ============================================================================
echo -e "\n${GREEN}[4/8] Creating directory structure...${NC}"

mkdir -p $INSTALL_DIR
cd $INSTALL_DIR

# Clone or update repository
if [ -d "$INSTALL_DIR/.git" ]; then
    echo "Updating existing repository..."
    git pull origin main
else
    echo "Cloning repository..."
    # If running from local, copy files instead
    if [ -d "/tmp/r3mes-deploy" ]; then
        cp -r /tmp/r3mes-deploy/* $INSTALL_DIR/
    else
        git clone https://github.com/r3mes/R3MES.git $INSTALL_DIR
    fi
fi

# Create required directories
mkdir -p $INSTALL_DIR/deploy/certbot/conf
mkdir -p $INSTALL_DIR/deploy/certbot/www
mkdir -p $INSTALL_DIR/deploy/nginx/conf.d
mkdir -p $INSTALL_DIR/backups

echo -e "${GREEN}Directory structure created${NC}"

# ============================================================================
# Step 5: Setup Environment
# ============================================================================
echo -e "\n${GREEN}[5/8] Setting up environment...${NC}"

cd $INSTALL_DIR/deploy

# Copy environment file if not exists
if [ ! -f ".env" ]; then
    cp .env.production .env
    echo -e "${GREEN}Environment file created${NC}"
else
    echo -e "${YELLOW}Environment file already exists${NC}"
fi

# ============================================================================
# Step 6: Get SSL Certificates
# ============================================================================
echo -e "\n${GREEN}[6/8] Obtaining SSL certificates...${NC}"

# Create temporary nginx config for SSL challenge
cat > $INSTALL_DIR/deploy/nginx/conf.d/temp-ssl.conf << 'TEMPSSL'
server {
    listen 80;
    server_name remes.network www.remes.network api.remes.network testnet.remes.network rpc.remes.network;

    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }

    location / {
        return 200 'SSL setup in progress';
        add_header Content-Type text/plain;
    }
}
TEMPSSL

# Start nginx temporarily for SSL challenge
docker compose -f docker-compose.production.yml up -d nginx 2>/dev/null || true
sleep 5

# Get SSL certificate
docker run --rm \
    -v "$INSTALL_DIR/deploy/certbot/conf:/etc/letsencrypt" \
    -v "$INSTALL_DIR/deploy/certbot/www:/var/www/certbot" \
    certbot/certbot certonly \
    --webroot \
    --webroot-path=/var/www/certbot \
    --email $SSL_EMAIL \
    --agree-tos \
    --no-eff-email \
    -d $DOMAIN \
    -d www.$DOMAIN \
    -d $API_DOMAIN \
    -d $TESTNET_DOMAIN \
    -d $RPC_DOMAIN \
    --force-renewal 2>/dev/null || {
        echo -e "${YELLOW}SSL certificate request failed. Using self-signed for now.${NC}"
        # Create self-signed certificate as fallback
        mkdir -p $INSTALL_DIR/deploy/certbot/conf/live/$DOMAIN
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout $INSTALL_DIR/deploy/certbot/conf/live/$DOMAIN/privkey.pem \
            -out $INSTALL_DIR/deploy/certbot/conf/live/$DOMAIN/fullchain.pem \
            -subj "/CN=$DOMAIN"
    }

# Stop temporary nginx
docker compose -f docker-compose.production.yml down 2>/dev/null || true

# Remove temporary config
rm -f $INSTALL_DIR/deploy/nginx/conf.d/temp-ssl.conf

echo -e "${GREEN}SSL certificates configured${NC}"

# ============================================================================
# Step 7: Build and Start Services
# ============================================================================
echo -e "\n${GREEN}[7/8] Building and starting services...${NC}"

cd $INSTALL_DIR/deploy

# Build backend image
echo "Building backend image..."
docker compose -f docker-compose.production.yml build --no-cache backend

# Start all services
echo "Starting all services..."
docker compose -f docker-compose.production.yml up -d

# Wait for services to be healthy
echo "Waiting for services to start..."
sleep 30

# Check service status
echo -e "\n${BLUE}Service Status:${NC}"
docker compose -f docker-compose.production.yml ps

# ============================================================================
# Step 8: Setup Cron Jobs
# ============================================================================
echo -e "\n${GREEN}[8/8] Setting up maintenance tasks...${NC}"

# SSL renewal cron
(crontab -l 2>/dev/null | grep -v "certbot renew"; echo "0 0 * * * docker compose -f $INSTALL_DIR/deploy/docker-compose.production.yml run --rm certbot renew && docker compose -f $INSTALL_DIR/deploy/docker-compose.production.yml exec nginx nginx -s reload") | crontab -

# Backup cron
cat > /usr/local/bin/r3mes-backup.sh << 'BACKUP'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR=/opt/r3mes/backups
docker exec r3mes-postgres pg_dump -U r3mes_admin r3mes > $BACKUP_DIR/db_$DATE.sql
gzip $BACKUP_DIR/db_$DATE.sql
find $BACKUP_DIR -name "*.gz" -mtime +7 -delete
BACKUP
chmod +x /usr/local/bin/r3mes-backup.sh
(crontab -l 2>/dev/null | grep -v "r3mes-backup"; echo "0 2 * * * /usr/local/bin/r3mes-backup.sh") | crontab -

echo -e "${GREEN}Maintenance tasks configured${NC}"

# ============================================================================
# Complete!
# ============================================================================
echo -e "\n${GREEN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║              🎉 DEPLOYMENT COMPLETE! 🎉                      ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${BLUE}Your R3MES backend is now running!${NC}"
echo ""
echo -e "${YELLOW}URLs:${NC}"
echo "  API:     https://api.remes.network"
echo "  Testnet: https://testnet.remes.network"
echo "  RPC:     https://rpc.remes.network"
echo ""
echo -e "${YELLOW}Health Check:${NC}"
curl -s https://api.remes.network/health 2>/dev/null || curl -s http://localhost:8000/health || echo "  Waiting for backend to start..."
echo ""
echo -e "${YELLOW}DNS Configuration Required:${NC}"
echo "  Add these DNS records to your domain registrar:"
echo ""
echo "  Type  | Name     | Value"
echo "  ------|----------|------------------"
echo "  A     | @        | $VPS_IP"
echo "  A     | api      | $VPS_IP"
echo "  A     | testnet  | $VPS_IP"
echo "  A     | rpc      | $VPS_IP"
echo "  A     | www      | $VPS_IP"
echo ""
echo -e "${YELLOW}Netlify Configuration:${NC}"
echo "  Set these environment variables in Netlify:"
echo ""
echo "  NEXT_PUBLIC_API_URL=https://api.remes.network"
echo "  NEXT_PUBLIC_BACKEND_URL=https://api.remes.network"
echo "  NEXT_PUBLIC_SITE_URL=https://remes.network"
echo "  NEXT_PUBLIC_RPC_URL=https://rpc.remes.network"
echo "  NEXT_PUBLIC_CHAIN_ID=remes-testnet-1"
echo ""
echo -e "${YELLOW}Useful Commands:${NC}"
echo "  View logs:    docker compose -f $INSTALL_DIR/deploy/docker-compose.production.yml logs -f"
echo "  Restart:      docker compose -f $INSTALL_DIR/deploy/docker-compose.production.yml restart"
echo "  Stop:         docker compose -f $INSTALL_DIR/deploy/docker-compose.production.yml down"
echo "  Backup:       /usr/local/bin/r3mes-backup.sh"
echo ""
echo -e "${GREEN}Installation directory: $INSTALL_DIR${NC}"
echo -e "${GREEN}Environment file: $INSTALL_DIR/deploy/.env${NC}"

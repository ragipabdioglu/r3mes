#!/bin/bash
# ===========================================
# R3MES SSL Certificate Setup Script
# Uses Let's Encrypt for free SSL certificates
# ===========================================

set -e

echo "🔐 R3MES SSL Setup Script"
echo "================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
DOMAIN="r3mes.network"
EMAIL="admin@r3mes.network"  # Change this to your email
DEPLOY_DIR="/opt/r3mes"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Please run as root${NC}"
    exit 1
fi

echo -e "${BLUE}📦 Step 1: Installing Certbot...${NC}"
apt-get update
apt-get install -y certbot

echo ""
echo -e "${BLUE}🔐 Step 2: Obtaining SSL certificates...${NC}"
echo ""
echo "Make sure your DNS records are configured:"
echo "  - api.r3mes.network    -> 38.242.246.6"
echo "  - rpc.r3mes.network    -> 38.242.246.6"
echo "  - rest.r3mes.network   -> 38.242.246.6"
echo "  - ipfs.r3mes.network   -> 38.242.246.6"
echo "  - grafana.r3mes.network -> 38.242.246.6"
echo ""
read -p "Are DNS records configured? (yes/no): " dns_confirm
if [ "$dns_confirm" != "yes" ]; then
    echo "Please configure DNS records first."
    exit 1
fi

# Stop nginx if running
docker stop r3mes-nginx 2>/dev/null || true

# Get certificates for all subdomains
certbot certonly --standalone \
    -d api.r3mes.network \
    -d rpc.r3mes.network \
    -d rest.r3mes.network \
    -d ipfs.r3mes.network \
    -d grafana.r3mes.network \
    --email "$EMAIL" \
    --agree-tos \
    --non-interactive

echo ""
echo -e "${BLUE}📁 Step 3: Copying certificates...${NC}"
mkdir -p "$DEPLOY_DIR/ssl/live/r3mes.network"
cp -L /etc/letsencrypt/live/api.r3mes.network/fullchain.pem "$DEPLOY_DIR/ssl/live/r3mes.network/"
cp -L /etc/letsencrypt/live/api.r3mes.network/privkey.pem "$DEPLOY_DIR/ssl/live/r3mes.network/"

echo ""
echo -e "${BLUE}🔄 Step 4: Setting up auto-renewal...${NC}"
cat > /etc/cron.d/certbot-renew << 'EOF'
0 0,12 * * * root certbot renew --quiet --post-hook "docker restart r3mes-nginx"
EOF

echo ""
echo -e "${GREEN}✅ SSL setup complete!${NC}"
echo ""
echo "Certificates location: $DEPLOY_DIR/ssl/live/r3mes.network/"
echo "Auto-renewal: Configured (runs twice daily)"

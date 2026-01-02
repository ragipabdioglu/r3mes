#!/bin/bash
# ===========================================
# R3MES Server Setup Script
# Run this AFTER cleanup to prepare the server
# ===========================================

set -e

echo "🚀 R3MES Server Setup Script"
echo "================================"
echo "Server: 38.242.246.6"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Please run as root${NC}"
    exit 1
fi

echo -e "${BLUE}📦 Step 1: Updating system...${NC}"
apt-get update && apt-get upgrade -y

echo ""
echo -e "${BLUE}📦 Step 2: Installing dependencies...${NC}"
apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    git \
    jq \
    htop \
    ufw \
    fail2ban

echo ""
echo -e "${BLUE}🐳 Step 3: Installing Docker...${NC}"
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
    systemctl enable docker
    systemctl start docker
else
    echo "Docker already installed"
fi

echo ""
echo -e "${BLUE}🐳 Step 4: Installing Docker Compose...${NC}"
if ! command -v docker-compose &> /dev/null; then
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
else
    echo "Docker Compose already installed"
fi

echo ""
echo -e "${BLUE}🔥 Step 5: Configuring firewall...${NC}"
ufw --force reset
ufw default deny incoming
ufw default allow outgoing

# SSH
ufw allow 22/tcp

# HTTP/HTTPS
ufw allow 80/tcp
ufw allow 443/tcp

# Blockchain P2P
ufw allow 26656/tcp

# Blockchain RPC (through nginx)
# ufw allow 26657/tcp  # Not needed, nginx handles this

# IPFS P2P
ufw allow 4001/tcp

ufw --force enable
ufw status

echo ""
echo -e "${BLUE}🔒 Step 6: Configuring fail2ban...${NC}"
cat > /etc/fail2ban/jail.local << 'EOF'
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
EOF

systemctl enable fail2ban
systemctl restart fail2ban

echo ""
echo -e "${BLUE}📁 Step 7: Creating R3MES directories...${NC}"
mkdir -p /opt/r3mes
mkdir -p /opt/r3mes/ssl
mkdir -p /opt/r3mes/data
mkdir -p /opt/r3mes/logs

echo ""
echo -e "${BLUE}👤 Step 8: Creating r3mes user...${NC}"
if ! id "r3mes" &>/dev/null; then
    useradd -m -s /bin/bash r3mes
    usermod -aG docker r3mes
fi

chown -R r3mes:r3mes /opt/r3mes

echo ""
echo -e "${GREEN}✅ Server setup complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Clone the R3MES repository to /opt/r3mes"
echo "2. Configure .env file"
echo "3. Set up SSL certificates"
echo "4. Run docker-compose"
echo ""
echo "Docker version: $(docker --version)"
echo "Docker Compose version: $(docker-compose --version)"

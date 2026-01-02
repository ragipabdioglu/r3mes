#!/bin/bash
# ===========================================
# R3MES Server Cleanup Script
# Run this FIRST on the server to remove old installation
# ===========================================

set -e

echo "🧹 R3MES Server Cleanup Script"
echo "================================"
echo "Server: 38.242.246.6"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Confirm before proceeding
echo -e "${YELLOW}⚠️  WARNING: This will remove all existing R3MES data!${NC}"
read -p "Are you sure you want to continue? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "📦 Step 1: Stopping all Docker containers..."
docker stop $(docker ps -aq) 2>/dev/null || true
docker rm $(docker ps -aq) 2>/dev/null || true

echo ""
echo "🗑️  Step 2: Removing Docker volumes..."
docker volume rm $(docker volume ls -q) 2>/dev/null || true

echo ""
echo "🧹 Step 3: Removing Docker networks..."
docker network rm $(docker network ls -q --filter type=custom) 2>/dev/null || true

echo ""
echo "🗑️  Step 4: Cleaning Docker system..."
docker system prune -af --volumes

echo ""
echo "📁 Step 5: Removing old R3MES directories..."
rm -rf /opt/r3mes 2>/dev/null || true
rm -rf /root/.remes 2>/dev/null || true
rm -rf ~/r3mes 2>/dev/null || true

echo ""
echo "🔧 Step 6: Removing old systemd services..."
systemctl stop remesd 2>/dev/null || true
systemctl disable remesd 2>/dev/null || true
rm -f /etc/systemd/system/remesd.service 2>/dev/null || true
systemctl daemon-reload

echo ""
echo -e "${GREEN}✅ Cleanup complete!${NC}"
echo ""
echo "Docker status:"
docker ps -a
echo ""
echo "Docker volumes:"
docker volume ls
echo ""
echo "Ready for fresh installation."

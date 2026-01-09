#!/bin/bash
# R3MES Quick Installation Script
# Installs all components for a complete R3MES setup

set -e

echo "=========================================="
echo "R3MES Complete Installation"
echo "=========================================="
echo ""
echo "This script will install:"
echo "  1. Blockchain Node (remesd)"
echo "  2. Miner Engine (PyPI package)"
echo "  3. IPFS Daemon"
echo "  4. Web Dashboard"
echo "  5. Desktop Launcher (optional)"
echo ""
read -p "Continue? (y/n): " CONTINUE
if [ "$CONTINUE" != "y" ]; then
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# 1. Install Blockchain Node
echo ""
echo "=========================================="
echo "1. Installing Blockchain Node..."
echo "=========================================="
if [ -f "scripts/install_founder.sh" ]; then
    echo "Run as founder/admin? (y/n): "
    read -p "  " IS_FOUNDER
    if [ "$IS_FOUNDER" = "y" ]; then
        bash scripts/install_founder.sh
    else
        bash scripts/install_validator.sh
    fi
else
    echo "⚠️  Install scripts not found. Skipping..."
fi

# 2. Install Miner Engine
echo ""
echo "=========================================="
echo "2. Installing Miner Engine (PyPI)..."
echo "=========================================="
if [ -f "scripts/install_miner_pypi.sh" ]; then
    bash scripts/install_miner_pypi.sh
else
    echo "⚠️  Install script not found. Skipping..."
fi

# 3. Install IPFS
echo ""
echo "=========================================="
echo "3. Installing IPFS..."
echo "=========================================="
if command -v ipfs &> /dev/null; then
    echo "✅ IPFS already installed"
else
    echo "Installing IPFS..."
    # IPFS installation (platform-specific)
    if [ "$(uname)" = "Linux" ]; then
        wget https://dist.ipfs.tech/kubo/v0.24.0/kubo_v0.24.0_linux-amd64.tar.gz
        tar -xzf kubo_v0.24.0_linux-amd64.tar.gz
        sudo mv kubo/ipfs /usr/local/bin/
        rm -rf kubo kubo_v0.24.0_linux-amd64.tar.gz
        echo "✅ IPFS installed"
    else
        echo "⚠️  Please install IPFS manually for your platform"
        echo "   https://docs.ipfs.tech/install/"
    fi
fi

# 4. Install Web Dashboard
echo ""
echo "=========================================="
echo "4. Installing Web Dashboard..."
echo "=========================================="
read -p "Install web dashboard? (y/n): " INSTALL_DASHBOARD
if [ "$INSTALL_DASHBOARD" = "y" ]; then
    if [ -f "scripts/install_web_dashboard.sh" ]; then
        bash scripts/install_web_dashboard.sh
    else
        echo "⚠️  Install script not found. Skipping..."
    fi
fi

# 5. Install Desktop Launcher
echo ""
echo "=========================================="
echo "5. Installing Desktop Launcher..."
echo "=========================================="
read -p "Install desktop launcher? (y/n): " INSTALL_LAUNCHER
if [ "$INSTALL_LAUNCHER" = "y" ]; then
    if [ -f "scripts/install_desktop_launcher.sh" ]; then
        bash scripts/install_desktop_launcher.sh
    else
        echo "⚠️  Install script not found. Skipping..."
    fi
fi

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Next Steps:"
echo ""
echo "1. Setup miner (if not done):"
echo "   r3mes-miner setup"
echo ""
echo "2. Start services:"
echo "   sudo systemctl start remesd"
echo "   sudo systemctl start ipfs"
echo "   r3mes-miner start"
echo ""
echo "3. Start web dashboard:"
echo "   cd web-dashboard && npm start"
echo ""
echo "4. See PRODUCTION_COMPLETE_GUIDE.md for detailed instructions"
echo ""
echo "=========================================="


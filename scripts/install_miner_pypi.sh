#!/bin/bash
# R3MES Miner Installation Script (PyPI Package)
# This script installs the miner from PyPI and sets up the system

set -e

echo "=========================================="
echo "R3MES Miner Installation (PyPI Package)"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.10+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python 3.10+ required. Found: $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"
echo ""

# Check pip
echo "Checking pip..."
if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
    echo "❌ pip not found. Installing pip..."
    python3 -m ensurepip --upgrade
fi

PIP_CMD="pip3"
if ! command -v pip3 &> /dev/null; then
    PIP_CMD="pip"
fi

echo "✅ pip found"
echo ""

# Install from PyPI
echo "Installing r3mes package from PyPI..."
echo "This may take a few minutes..."
$PIP_CMD install --upgrade pip
$PIP_CMD install r3mes

echo ""
echo "✅ R3MES miner installed successfully!"
echo ""

# Check installation
if command -v r3mes-miner &> /dev/null; then
    echo "✅ r3mes-miner command available"
    r3mes-miner --version
else
    echo "⚠️  r3mes-miner command not found in PATH"
    echo "   Try: python3 -m r3mes.cli.commands --help"
fi

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. Run setup wizard:"
echo "   r3mes-miner setup"
echo ""
echo "2. Start mining:"
echo "   r3mes-miner start"
echo ""
echo "3. Check status:"
echo "   r3mes-miner status"
echo ""
echo "4. For production, install systemd service:"
echo "   sudo cp scripts/systemd/r3mes-miner.service /etc/systemd/system/"
echo "   sudo systemctl enable r3mes-miner"
echo "   sudo systemctl start r3mes-miner"
echo ""
echo "=========================================="


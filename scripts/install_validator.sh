#!/bin/bash
# R3MES Validator Installation Script
# This script sets up a validator node

set -e

echo "=========================================="
echo "R3MES Validator Installation"
echo "=========================================="
echo ""

# Check Go
echo "Checking Go installation..."
if ! command -v go &> /dev/null; then
    echo "❌ Go not found. Please install Go 1.24.0+ first."
    echo "   Download from: https://go.dev/dl/"
    exit 1
fi

GO_VERSION=$(go version | awk '{print $3}' | sed 's/go//')
echo "✅ Go $GO_VERSION detected"
echo ""

# Build remesd
echo "Building remesd..."
cd "$(dirname "$0")/../remes"
make build

if [ ! -f "./build/remesd" ]; then
    echo "❌ Build failed. remesd binary not found."
    exit 1
fi

echo "✅ remesd built successfully"
echo ""

# Install binary
echo "Installing remesd to /usr/local/bin..."
sudo cp ./build/remesd /usr/local/bin/remesd
sudo chmod +x /usr/local/bin/remesd

echo "✅ remesd installed"
echo ""

# Initialize chain (if not already done)
if [ ! -d "$HOME/.remesd" ]; then
    echo "Initializing chain..."
    read -p "Chain ID (default: remes-1): " CHAIN_ID
    CHAIN_ID=${CHAIN_ID:-remes-1}
    
    remesd init validator-node --chain-id "$CHAIN_ID"
    echo "✅ Chain initialized"
    echo ""
fi

# Create validator key (if not exists)
if [ ! -f "$HOME/.remesd/config/priv_validator_key.json" ]; then
    echo "Creating validator key..."
    remesd keys add validator-key
    echo "✅ Validator key created"
    echo ""
fi

# Install systemd service
echo "Installing systemd service..."
sudo cp "$(dirname "$0")/systemd/remesd.service" /etc/systemd/system/
sudo systemctl daemon-reload

echo "✅ Systemd service installed"
echo ""

echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. Register as validator:"
echo "   remesd tx remes register-node \\"
echo "     --node-type validator \\"
echo "     --from validator-key \\"
echo "     --chain-id remes-1 \\"
echo "     --yes"
echo ""
echo "2. Create validator:"
echo "   remesd tx staking create-validator \\"
echo "     --amount 1000000uremes \\"
echo "     --pubkey \$(remesd tendermint show-validator) \\"
echo "     --moniker \"my-validator\" \\"
echo "     --commission-rate 0.1 \\"
echo "     --from validator-key \\"
echo "     --chain-id remes-1 \\"
echo "     --yes"
echo ""
echo "3. Start validator:"
echo "   sudo systemctl enable remesd"
echo "   sudo systemctl start remesd"
echo ""
echo "4. Check status:"
echo "   sudo systemctl status remesd"
echo "   remesd status"
echo ""
echo "=========================================="


#!/bin/bash
# R3MES Founder/Admin Installation Script
# This script sets up the initial blockchain network

set -e

echo "=========================================="
echo "R3MES Founder/Admin Setup"
echo "=========================================="
echo ""

# Check Go
echo "Checking Go installation..."
if ! command -v go &> /dev/null; then
    echo "❌ Go not found. Please install Go 1.24.0+ first."
    exit 1
fi

echo "✅ Go detected"
echo ""

# Build remesd
echo "Building remesd..."
cd "$(dirname "$0")/../remes"
make build

echo "✅ remesd built"
echo ""

# Install binary
echo "Installing remesd..."
sudo cp ./build/remesd /usr/local/bin/remesd
sudo chmod +x /usr/local/bin/remesd

echo "✅ remesd installed"
echo ""

# Initialize genesis
echo "Initializing genesis..."
read -p "Chain ID (default: remes-1): " CHAIN_ID
CHAIN_ID=${CHAIN_ID:-remes-1}

read -p "Moniker (default: genesis-node): " MONIKER
MONIKER=${MONIKER:-genesis-node}

if [ -f "$HOME/.remesd/config/genesis.json" ] || [ -f "$HOME/.remes/config/genesis.json" ]; then
    echo "⚠️  Genesis file already exists. Skipping init..."
else
    remesd init "$MONIKER" --chain-id "$CHAIN_ID"
    echo "✅ Genesis initialized"
fi
echo ""

# Create founder key
echo "Creating founder key..."
if remesd keys show founder-key &> /dev/null; then
    echo "⚠️  Key 'founder-key' already exists. Using existing key."
else
    remesd keys add founder-key
fi

FOUNDER_ADDRESS=$(remesd keys show founder-key -a)
echo "✅ Founder address: $FOUNDER_ADDRESS"
echo ""

# Add genesis account
# Add genesis account (only if genesis specific marker not found/simplified check)
# Ideally we check if address is already in genesis, but simple append might fail or duplicate.
# For now, we try-catch or warn.
echo "Adding genesis account..."
remesd add-genesis-account "$FOUNDER_ADDRESS" "$INITIAL_BALANCE" || echo "⚠️  Could not add genesis account (maybe already exists?)"

echo "✅ Genesis account added check done"
echo ""

# Create genesis validator
# Create genesis validator
echo "Creating genesis validator..."
if [ -f "$HOME/.remesd/config/gentx/gentx-*.json" ] || [ -f "$HOME/.remes/config/gentx/gentx-*.json" ]; then
     echo "⚠️  Gentx already exists. Skipping..."
else
    remesd gentx founder-key 1000000uremes \
        --chain-id "$CHAIN_ID" \
        --moniker "$MONIKER" \
        --commission-rate 0.1 \
        --commission-max-rate 0.2 \
        --commission-max-change-rate 0.01 || echo "⚠️  Failed to create gentx (check if already part of genesis)"

    remesd collect-gentxs || echo "⚠️  Failed to collect gentxs"
fi

echo "✅ Genesis validator step done"
echo ""

# Edit genesis (optional)
echo "Genesis file created at: ~/.remesd/config/genesis.json"
echo "You can edit it to:"
echo "  - Add more validators"
echo "  - Adjust network parameters"
echo "  - Set initial balances"
echo ""

read -p "Edit genesis.json now? (y/n): " EDIT_GENESIS
if [ "$EDIT_GENESIS" = "y" ]; then
    ${EDITOR:-nano} ~/.remesd/config/genesis.json
fi

# Install systemd service
echo "Installing systemd service..."
sudo cp "$(dirname "$0")/systemd/remesd.service" /etc/systemd/system/
sudo systemctl daemon-reload

echo "✅ Systemd service installed"
echo ""

echo "=========================================="
echo "Genesis Setup Complete!"
echo "=========================================="
echo ""
echo "Next Steps:"
echo ""
echo "1. Review genesis file:"
echo "   cat ~/.remesd/config/genesis.json"
echo ""
echo "2. Start the network:"
echo "   sudo systemctl enable remesd"
echo "   sudo systemctl start remesd"
echo ""
echo "3. Check status:"
echo "   sudo systemctl status remesd"
echo "   remesd status"
echo ""
echo "4. Share genesis file with other validators"
echo ""
echo "=========================================="


#!/bin/sh
# Blockchain Node Initialization Script
# Checks if chain is initialized, if not, initializes it

set -e

CHAIN_ID=${CHAIN_ID:-remes-mainnet}
HOME_DIR=${HOME:-/app/.remesd}

echo "=========================================="
echo "R3MES Blockchain Node Initialization"
echo "=========================================="
echo "Chain ID: $CHAIN_ID"
echo "Home Directory: $HOME_DIR"
echo "=========================================="

# Check if chain is already initialized
if [ -f "$HOME_DIR/config/config.toml" ]; then
    echo "✅ Chain already initialized"
    echo "Starting blockchain node..."
    exec remesd start --home "$HOME_DIR"
else
    echo "⚠️  Chain not initialized, initializing now..."
    
    # Initialize the chain
    remesd init "$CHAIN_ID" --chain-id "$CHAIN_ID" --home "$HOME_DIR"
    
    if [ $? -eq 0 ]; then
        echo "✅ Chain initialized successfully"
        echo "Starting blockchain node..."
        exec remesd start --home "$HOME_DIR"
    else
        echo "❌ Chain initialization failed"
        exit 1
    fi
fi


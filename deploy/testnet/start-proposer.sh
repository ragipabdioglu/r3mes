#!/bin/bash
set -e

echo "=========================================="
echo "R3MES Proposer Node Startup"
echo "=========================================="

CHAIN_ID="${CHAIN_ID:-r3mes-testnet-1}"
MONIKER="${MONIKER:-r3mes-proposer-1}"
VALIDATOR_RPC="${VALIDATOR_RPC:-http://r3mes-validator:26657}"

HOME_DIR="/root/.remes"

# Function to get validator node ID from RPC
get_validator_node_id() {
    local node_id=""
    for i in {1..30}; do
        node_id=$(curl -s "$VALIDATOR_RPC/status" | jq -r '.result.node_info.id' 2>/dev/null)
        if [ -n "$node_id" ] && [ "$node_id" != "null" ]; then
            echo "$node_id"
            return 0
        fi
        echo "Waiting for validator node ID... ($i/30)" >&2
        sleep 2
    done
    return 1
}

# Initialize if not already done
if [ ! -f "$HOME_DIR/config/genesis.json" ]; then
    echo "Initializing proposer node..."
    remesd init "$MONIKER" --chain-id "$CHAIN_ID" --home "$HOME_DIR"
    
    # Get genesis from validator
    echo "Fetching genesis from validator..."
    for i in {1..30}; do
        if curl -s "$VALIDATOR_RPC/genesis" | jq -r '.result.genesis' > "$HOME_DIR/config/genesis.json" 2>/dev/null; then
            if [ -s "$HOME_DIR/config/genesis.json" ] && [ "$(cat $HOME_DIR/config/genesis.json)" != "null" ]; then
                echo "Genesis fetched successfully"
                break
            fi
        fi
        echo "Waiting for validator genesis... ($i/30)"
        sleep 5
    done
    
    # Get validator node ID dynamically
    echo "Getting validator node ID..."
    VALIDATOR_NODE_ID=$(get_validator_node_id)
    if [ -n "$VALIDATOR_NODE_ID" ]; then
        echo "Validator node ID: $VALIDATOR_NODE_ID"
        sed -i "s/persistent_peers = \"\"/persistent_peers = \"${VALIDATOR_NODE_ID}@r3mes-validator:26656\"/" "$HOME_DIR/config/config.toml"
    else
        echo "WARNING: Could not get validator node ID, peer connection may fail"
    fi
    
    # Disable creating empty blocks
    sed -i 's/create_empty_blocks = true/create_empty_blocks = false/' "$HOME_DIR/config/config.toml"
    
    # Set minimum gas prices in app.toml
    sed -i 's/minimum-gas-prices = ""/minimum-gas-prices = "0.001ur3mes"/' "$HOME_DIR/config/app.toml"
fi

# Import proposer key if mnemonic provided
if [ -n "$PROPOSER_MNEMONIC" ]; then
    echo "Importing proposer key..."
    echo "$PROPOSER_MNEMONIC" | remesd keys add proposer --recover --keyring-backend test --home "$HOME_DIR" 2>/dev/null || true
    
    PROPOSER_ADDRESS=$(remesd keys show proposer -a --keyring-backend test --home "$HOME_DIR" 2>/dev/null || echo "unknown")
    echo "Proposer address: $PROPOSER_ADDRESS"
fi

echo "Starting proposer node..."
exec remesd start \
    --home "$HOME_DIR" \
    --rpc.laddr "tcp://0.0.0.0:26657" \
    --grpc.address "0.0.0.0:9090" \
    --p2p.laddr "tcp://0.0.0.0:26656" \
    --minimum-gas-prices "0.001ur3mes" \
    --log_level info

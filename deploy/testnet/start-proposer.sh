#!/bin/bash
set -e

echo "=========================================="
echo "R3MES Proposer Node Startup"
echo "=========================================="

CHAIN_ID="${CHAIN_ID:-r3mes-testnet-1}"
MONIKER="${MONIKER:-r3mes-proposer-1}"
VALIDATOR_RPC="${VALIDATOR_RPC:-http://r3mes-validator:26657}"

HOME_DIR="/home/proposer/.remes"

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
        echo "Waiting for validator... ($i/30)"
        sleep 5
    done
    
    # Configure for proposer role
    sed -i 's/persistent_peers = ""/persistent_peers = "'"$VALIDATOR_NODE_ID@r3mes-validator:26656"'"/' "$HOME_DIR/config/config.toml"
    
    # Enable proposer mode
    sed -i 's/mode = "full"/mode = "full"/' "$HOME_DIR/config/config.toml"
    
    # Configure for gradient aggregation
    cat >> "$HOME_DIR/config/app.toml" << EOF

# Proposer Configuration
[proposer]
enabled = true
aggregation_interval = "30s"
min_gradients_for_aggregation = 2
max_gradients_per_aggregation = 100
ipfs_upload_enabled = true
EOF
fi

# Import proposer key if mnemonic provided
if [ -n "$PROPOSER_MNEMONIC" ]; then
    echo "Importing proposer key..."
    echo "$PROPOSER_MNEMONIC" | remesd keys add proposer --recover --keyring-backend test --home "$HOME_DIR" 2>/dev/null || true
    
    PROPOSER_ADDRESS=$(remesd keys show proposer -a --keyring-backend test --home "$HOME_DIR")
    echo "Proposer address: $PROPOSER_ADDRESS"
fi

echo "Starting proposer node..."
exec remesd start \
    --home "$HOME_DIR" \
    --rpc.laddr "tcp://0.0.0.0:26657" \
    --grpc.address "0.0.0.0:9090" \
    --p2p.laddr "tcp://0.0.0.0:26656" \
    --log_level info

#!/bin/bash
# Initialize R3MES blockchain from scratch

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REmes_DIR="$PROJECT_ROOT/remes"

echo "=========================================="
echo "R3MES Blockchain Initialization"
echo "=========================================="

# Check if remesd binary exists
if [ ! -f "$REmes_DIR/build/remesd" ]; then
    echo "Building remesd binary..."
    cd "$REmes_DIR"
    make build
fi

# Initialize chain
echo "Initializing chain..."
cd "$REmes_DIR"
./build/remesd init r3mes-test --chain-id remes-test

# Create genesis accounts
echo "Creating genesis accounts..."
./build/remesd keys add validator --keyring-backend test || true
./build/remesd keys add miner1 --keyring-backend test || true

# Add genesis accounts
echo "Adding genesis accounts..."
VALIDATOR_ADDR=$(./build/remesd keys show validator -a --keyring-backend test)
MINER1_ADDR=$(./build/remesd keys show miner1 -a --keyring-backend test)

./build/remesd genesis add-genesis-account $VALIDATOR_ADDR 1000000000000stake --keyring-backend test
./build/remesd genesis add-genesis-account $MINER1_ADDR 1000000000000stake --keyring-backend test

# Create genesis transaction
echo "Creating genesis transaction..."
./build/remesd genesis gentx validator 100000000000stake --chain-id remes-test --keyring-backend test

# Collect genesis transactions
echo "Collecting genesis transactions..."
./build/remesd genesis collect-gentxs

echo "=========================================="
echo "✅ Chain initialized successfully!"
echo "=========================================="
echo "Validator address: $VALIDATOR_ADDR"
echo "Miner1 address: $MINER1_ADDR"
echo ""
echo "To start the chain, run:"
echo "  cd $REmes_DIR && ./build/remesd start"


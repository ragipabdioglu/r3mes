#!/bin/bash
# Start R3MES miner engine

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MINER_DIR="$PROJECT_ROOT/miner-engine"

echo "=========================================="
echo "Starting R3MES Miner Engine"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "$MINER_DIR/venv" ]; then
    echo "Creating Python virtual environment..."
    cd "$MINER_DIR"
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    cd "$MINER_DIR"
    source venv/bin/activate
fi

# Check if IPFS is running
if ! curl -s http://127.0.0.1:5001/api/v0/version > /dev/null 2>&1; then
    echo "⚠️  Warning: IPFS daemon is not running"
    echo "   Start IPFS: ipfs daemon"
    echo "   Or continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
fi

# Check if blockchain node is running
if ! curl -s http://127.0.0.1:26657/status > /dev/null 2>&1; then
    echo "⚠️  Warning: Blockchain node is not running"
    echo "   Start blockchain: cd $PROJECT_ROOT/remes && ./build/remesd start"
    echo "   Or continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
fi

# Get miner private key (or generate new one)
if [ -z "$MINER_PRIVATE_KEY" ]; then
    echo "Miner private key not set. Generating new keypair..."
    python3 -c "
from bridge.crypto import generate_keypair, private_key_to_hex
priv, pub = generate_keypair()
print(f'Private key: {private_key_to_hex(priv)}')
print('Save this private key securely!')
"
    echo "Enter private key:"
    read -r MINER_PRIVATE_KEY
fi

# Start miner
echo "Starting miner engine..."
export MINER_PRIVATE_KEY="$MINER_PRIVATE_KEY"
export BLOCKCHAIN_URL="${BLOCKCHAIN_URL:-localhost:9090}"
export CHAIN_ID="${CHAIN_ID:-remes-test}"

python3 miner_engine.py \
    --private-key "$MINER_PRIVATE_KEY" \
    --blockchain-url "$BLOCKCHAIN_URL" \
    --chain-id "$CHAIN_ID" \
    --max-iterations "${MAX_ITERATIONS:-5}" \
    "$@"


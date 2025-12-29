#!/bin/bash
# Check miner status and statistics

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REmes_DIR="$PROJECT_ROOT/remes"
MINER_DIR="$PROJECT_ROOT/miner-engine"

# Default values
BLOCKCHAIN_URL="${BLOCKCHAIN_URL:-localhost:9090}"
CHAIN_ID="${CHAIN_ID:-remes-test}"
MINER_ADDRESS="${MINER_ADDRESS:-}"

echo "=========================================="
echo "R3MES Miner Status"
echo "=========================================="

# Check if blockchain node is running
if ! curl -s http://127.0.0.1:26657/status > /dev/null 2>&1; then
    echo "❌ Blockchain node is not running"
    echo "   Start with: cd $REmes_DIR && ./build/remesd start"
    exit 1
fi

# Check if IPFS is running
if ! curl -s http://127.0.0.1:5001/api/v0/version > /dev/null 2>&1; then
    echo "⚠️  Warning: IPFS daemon is not running"
fi

# Get miner address if not provided
if [ -z "$MINER_ADDRESS" ]; then
    echo "Enter miner address (or press Enter to skip):"
    read -r MINER_ADDRESS
fi

# Query miner score if address provided
if [ -n "$MINER_ADDRESS" ]; then
    echo ""
    echo "Miner Statistics:"
    echo "-----------------"
    cd "$REmes_DIR"
    ./build/remesd query remes get-miner-score "$MINER_ADDRESS" --chain-id "$CHAIN_ID" 2>/dev/null || echo "  No statistics found for this miner"
fi

# Query recent gradients
echo ""
echo "Recent Gradients:"
echo "-----------------"
cd "$REmes_DIR"
REMESD_CMD="./build/remesd"
[ ! -f "$REMESD_CMD" ] && REMESD_CMD="remesd"
$REMESD_CMD query remes list-stored-gradient --chain-id "$CHAIN_ID" --limit 5 2>/dev/null || echo "  No gradients found"

# Query model parameters
echo ""
echo "Model Parameters:"
echo "-----------------"
cd "$REmes_DIR"
REMESD_CMD="./build/remesd"
[ ! -f "$REMESD_CMD" ] && REMESD_CMD="remesd"
$REMESD_CMD query remes get-model-params --chain-id "$CHAIN_ID" 2>/dev/null || echo "  Model parameters not available"

echo ""
echo "=========================================="


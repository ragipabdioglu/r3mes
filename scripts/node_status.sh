#!/bin/bash
# Check blockchain node status and network health

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REmes_DIR="$PROJECT_ROOT/remes"

# Default values
CHAIN_ID="${CHAIN_ID:-remes-test}"

echo "=========================================="
echo "R3MES Blockchain Node Status"
echo "=========================================="

# Check if node is running
if ! curl -s http://127.0.0.1:26657/status > /dev/null 2>&1; then
    echo "❌ Node is not running"
    echo "   Start with: cd $REmes_DIR && ./build/remesd start"
    exit 1
fi

# Get node status
echo "Node Status:"
echo "------------"
STATUS=$(curl -s http://127.0.0.1:26657/status)
NODE_INFO=$(echo "$STATUS" | grep -o '"node_info":{[^}]*}' || echo "")
SYNC_INFO=$(echo "$STATUS" | grep -o '"sync_info":{[^}]*}' || echo "")

if [ -n "$NODE_INFO" ]; then
    echo "  Node ID: $(echo "$NODE_INFO" | grep -o '"id":"[^"]*"' | cut -d'"' -f4 | head -c 20)..."
    echo "  Network: $(echo "$NODE_INFO" | grep -o '"network":"[^"]*"' | cut -d'"' -f4 || echo "unknown")"
fi

if [ -n "$SYNC_INFO" ]; then
    LATEST_HEIGHT=$(echo "$SYNC_INFO" | grep -o '"latest_block_height":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
    CATCHING_UP=$(echo "$SYNC_INFO" | grep -o '"catching_up":[^,}]*' | cut -d':' -f2 || echo "unknown")
    echo "  Latest Block Height: $LATEST_HEIGHT"
    echo "  Catching Up: $CATCHING_UP"
fi

# Query network statistics
echo ""
echo "Network Statistics:"
echo "-------------------"
cd "$REmes_DIR"

# Count active miners (approximate)
REMESD_CMD="./build/remesd"
[ ! -f "$REMESD_CMD" ] && REMESD_CMD="remesd"
GRADIENT_COUNT=$($REMESD_CMD query remes list-stored-gradient --chain-id "$CHAIN_ID" --limit 1000 2>/dev/null | grep -c "id:" || echo "0")
echo "  Total Gradients Submitted: $GRADIENT_COUNT"

# Get model state
echo ""
echo "Model State:"
echo "------------"
$REMESD_CMD query remes get-model-params --chain-id "$CHAIN_ID" 2>/dev/null | head -10 || echo "  Model state not available"

echo ""
echo "=========================================="


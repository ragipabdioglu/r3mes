#!/bin/bash
# Submit a governance proposal (dataset proposal, parameter update, etc.)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REmes_DIR="$PROJECT_ROOT/remes"

# Default values
CHAIN_ID="${CHAIN_ID:-remes-test}"
PROPOSER="${PROPOSER:-}"
PROPOSAL_TYPE="${PROPOSAL_TYPE:-parameter-change}"
DEPOSIT="${DEPOSIT:-1000000stake}"

echo "=========================================="
echo "R3MES Governance Proposal"
echo "=========================================="

# Check if node is running
if ! curl -s http://127.0.0.1:26657/status > /dev/null 2>&1; then
    echo "❌ Node is not running"
    exit 1
fi

# Get proposer address
if [ -z "$PROPOSER" ]; then
    echo "Enter proposer key name (from keyring):"
    read -r PROPOSER
fi

echo ""
echo "Proposal Type: $PROPOSAL_TYPE"
echo "Proposer: $PROPOSER"
echo "Deposit: $DEPOSIT"
echo ""
echo "Note: This is a template script. Full governance proposal"
echo "      submission requires implementing the governance module"
echo "      integration with the remes module parameters."
echo ""
echo "For now, parameter updates can be done via:"
echo "  remesd tx remes update-params --from $PROPOSER --chain-id $CHAIN_ID"

echo ""
echo "=========================================="


#!/bin/bash
# Genesis Configuration for Mainnet
# Tokenomics parameters for R3MES mainnet

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REmes_DIR="$PROJECT_ROOT/remes"

echo "=========================================="
echo "R3MES Mainnet Genesis Configuration"
echo "=========================================="

# Check if remesd binary exists
if [ ! -f "$REmes_DIR/build/remesd" ]; then
    echo "Building remesd binary..."
    cd "$REmes_DIR"
    make build
fi

GENESIS_FILE="$HOME/.remesd/config/genesis.json"

if [ ! -f "$GENESIS_FILE" ]; then
    echo "Error: genesis.json not found. Please run 'init_chain.sh' first."
    exit 1
fi

echo "Configuring tokenomics parameters..."

# Tokenomics Parameters
# These values can be overridden via environment variables
# Defaults are provided for convenience, but should be reviewed for production

# 1. Mint Module Parameters (Inflation)
MINT_INFLATION_RATE="${MINT_INFLATION_RATE:-0.10}"  # 10% annual inflation
MINT_INFLATION_MAX="${MINT_INFLATION_MAX:-0.20}"   # Maximum 20% inflation
MINT_INFLATION_MIN="${MINT_INFLATION_MIN:-0.05}"   # Minimum 5% inflation
MINT_GOAL_BONDED="${MINT_GOAL_BONDED:-0.67}"     # Target 67% of tokens staked
MINT_BLOCKS_PER_YEAR="${MINT_BLOCKS_PER_YEAR:-6311520}"  # ~6.3M blocks per year (6s block time)

# 2. Staking Parameters
STAKING_UNBONDING_TIME="${STAKING_UNBONDING_TIME:-1814400s}"  # 21 days (in seconds)
STAKING_MAX_VALIDATORS="${STAKING_MAX_VALIDATORS:-100}"       # Maximum 100 validators
STAKING_MAX_ENTRIES="${STAKING_MAX_ENTRIES:-7}"            # Max unbonding entries
STAKING_HISTORICAL_ENTRIES="${STAKING_HISTORICAL_ENTRIES:-10000}" # Historical entries

# 3. Slashing Parameters
SLASHING_SIGNED_BLOCKS_WINDOW="${SLASHING_SIGNED_BLOCKS_WINDOW:-10000}"  # 10,000 blocks (~17 hours)
SLASHING_MIN_SIGNED_PER_WINDOW="${SLASHING_MIN_SIGNED_PER_WINDOW:-0.05}"  # 5% minimum signed blocks
SLASHING_DOWNTIME_JAIL_DURATION="${SLASHING_DOWNTIME_JAIL_DURATION:-600s}" # 10 minutes
SLASHING_SLASH_FRACTION_DOUBLE_SIGN="${SLASHING_SLASH_FRACTION_DOUBLE_SIGN:-0.05}"  # 5% for double signing
SLASHING_SLASH_FRACTION_DOWNTIME="${SLASHING_SLASH_FRACTION_DOWNTIME:-0.0001}"   # 0.01% for downtime

# 4. Distribution Parameters
DISTRIBUTION_COMMUNITY_TAX="${DISTRIBUTION_COMMUNITY_TAX:-0.02}"  # 2% community tax
DISTRIBUTION_BASE_PROPOSER_REWARD="${DISTRIBUTION_BASE_PROPOSER_REWARD:-0.01}"  # 1% base proposer reward
DISTRIBUTION_BONUS_PROPOSER_REWARD="${DISTRIBUTION_BONUS_PROPOSER_REWARD:-0.04}" # 4% bonus proposer reward
DISTRIBUTION_WITHDRAW_ADDR_ENABLED="${DISTRIBUTION_WITHDRAW_ADDR_ENABLED:-true}"

# 5. Governance Parameters
GOVERNANCE_MIN_DEPOSIT="${GOVERNANCE_MIN_DEPOSIT:-1000000uremes}"  # 1M remes minimum deposit
GOVERNANCE_MAX_DEPOSIT_PERIOD="${GOVERNANCE_MAX_DEPOSIT_PERIOD:-172800s}"  # 2 days
GOVERNANCE_VOTING_PERIOD="${GOVERNANCE_VOTING_PERIOD:-1209600s}"      # 14 days
GOVENRANCE_QUORUM="${GOVENRANCE_QUORUM:-0.40}"                 # 40% quorum
GOVERNANCE_THRESHOLD="${GOVERNANCE_THRESHOLD:-0.50}"              # 50% threshold
GOVERNANCE_VETO_THRESHOLD="${GOVERNANCE_VETO_THRESHOLD:-0.334}"        # 33.4% veto threshold

# 6. Crisis Parameters
CRISIS_CONSTANT_FEE="${CRISIS_CONSTANT_FEE:-1000uremes}"  # Constant fee for crisis module

# Use jq to update genesis.json (if available)
if command -v jq &> /dev/null; then
    echo "Updating genesis.json with tokenomics parameters..."
    
    # Backup original
    cp "$GENESIS_FILE" "${GENESIS_FILE}.backup"
    
    # Update mint parameters
    jq ".app_state.mint.params.inflation = \"$MINT_INFLATION_RATE\"" "$GENESIS_FILE" > "${GENESIS_FILE}.tmp" && mv "${GENESIS_FILE}.tmp" "$GENESIS_FILE"
    jq ".app_state.mint.params.inflation_max = \"$MINT_INFLATION_MAX\"" "$GENESIS_FILE" > "${GENESIS_FILE}.tmp" && mv "${GENESIS_FILE}.tmp" "$GENESIS_FILE"
    jq ".app_state.mint.params.inflation_min = \"$MINT_INFLATION_MIN\"" "$GENESIS_FILE" > "${GENESIS_FILE}.tmp" && mv "${GENESIS_FILE}.tmp" "$GENESIS_FILE"
    jq ".app_state.mint.params.goal_bonded = \"$MINT_GOAL_BONDED\"" "$GENESIS_FILE" > "${GENESIS_FILE}.tmp" && mv "${GENESIS_FILE}.tmp" "$GENESIS_FILE"
    jq ".app_state.mint.params.blocks_per_year = \"$MINT_BLOCKS_PER_YEAR\"" "$GENESIS_FILE" > "${GENESIS_FILE}.tmp" && mv "${GENESIS_FILE}.tmp" "$GENESIS_FILE"
    
    # Update staking parameters
    jq ".app_state.staking.params.unbonding_time = \"$STAKING_UNBONDING_TIME\"" "$GENESIS_FILE" > "${GENESIS_FILE}.tmp" && mv "${GENESIS_FILE}.tmp" "$GENESIS_FILE"
    jq ".app_state.staking.params.max_validators = $STAKING_MAX_VALIDATORS" "$GENESIS_FILE" > "${GENESIS_FILE}.tmp" && mv "${GENESIS_FILE}.tmp" "$GENESIS_FILE"
    jq ".app_state.staking.params.max_entries = $STAKING_MAX_ENTRIES" "$GENESIS_FILE" > "${GENESIS_FILE}.tmp" && mv "${GENESIS_FILE}.tmp" "$GENESIS_FILE"
    jq ".app_state.staking.params.historical_entries = $STAKING_HISTORICAL_ENTRIES" "$GENESIS_FILE" > "${GENESIS_FILE}.tmp" && mv "${GENESIS_FILE}.tmp" "$GENESIS_FILE"
    
    # Update slashing parameters
    jq ".app_state.slashing.params.signed_blocks_window = \"$SLASHING_SIGNED_BLOCKS_WINDOW\"" "$GENESIS_FILE" > "${GENESIS_FILE}.tmp" && mv "${GENESIS_FILE}.tmp" "$GENESIS_FILE"
    jq ".app_state.slashing.params.min_signed_per_window = \"$SLASHING_MIN_SIGNED_PER_WINDOW\"" "$GENESIS_FILE" > "${GENESIS_FILE}.tmp" && mv "${GENESIS_FILE}.tmp" "$GENESIS_FILE"
    jq ".app_state.slashing.params.downtime_jail_duration = \"$SLASHING_DOWNTIME_JAIL_DURATION\"" "$GENESIS_FILE" > "${GENESIS_FILE}.tmp" && mv "${GENESIS_FILE}.tmp" "$GENESIS_FILE"
    jq ".app_state.slashing.params.slash_fraction_double_sign = \"$SLASHING_SLASH_FRACTION_DOUBLE_SIGN\"" "$GENESIS_FILE" > "${GENESIS_FILE}.tmp" && mv "${GENESIS_FILE}.tmp" "$GENESIS_FILE"
    jq ".app_state.slashing.params.slash_fraction_downtime = \"$SLASHING_SLASH_FRACTION_DOWNTIME\"" "$GENESIS_FILE" > "${GENESIS_FILE}.tmp" && mv "${GENESIS_FILE}.tmp" "$GENESIS_FILE"
    
    # Update distribution parameters
    jq ".app_state.distribution.params.community_tax = \"$DISTRIBUTION_COMMUNITY_TAX\"" "$GENESIS_FILE" > "${GENESIS_FILE}.tmp" && mv "${GENESIS_FILE}.tmp" "$GENESIS_FILE"
    jq ".app_state.distribution.params.base_proposer_reward = \"$DISTRIBUTION_BASE_PROPOSER_REWARD\"" "$GENESIS_FILE" > "${GENESIS_FILE}.tmp" && mv "${GENESIS_FILE}.tmp" "$GENESIS_FILE"
    jq ".app_state.distribution.params.bonus_proposer_reward = \"$DISTRIBUTION_BONUS_PROPOSER_REWARD\"" "$GENESIS_FILE" > "${GENESIS_FILE}.tmp" && mv "${GENESIS_FILE}.tmp" "$GENESIS_FILE"
    jq ".app_state.distribution.params.withdraw_addr_enabled = $DISTRIBUTION_WITHDRAW_ADDR_ENABLED" "$GENESIS_FILE" > "${GENESIS_FILE}.tmp" && mv "${GENESIS_FILE}.tmp" "$GENESIS_FILE"
    
    # Update governance parameters
    jq ".app_state.gov.params.min_deposit[0].amount = \"1000000\"" "$GENESIS_FILE" > "${GENESIS_FILE}.tmp" && mv "${GENESIS_FILE}.tmp" "$GENESIS_FILE"
    jq ".app_state.gov.params.min_deposit[0].denom = \"uremes\"" "$GENESIS_FILE" > "${GENESIS_FILE}.tmp" && mv "${GENESIS_FILE}.tmp" "$GENESIS_FILE"
    jq ".app_state.gov.params.max_deposit_period = \"$GOVERNANCE_MAX_DEPOSIT_PERIOD\"" "$GENESIS_FILE" > "${GENESIS_FILE}.tmp" && mv "${GENESIS_FILE}.tmp" "$GENESIS_FILE"
    jq ".app_state.gov.params.voting_period = \"$GOVERNANCE_VOTING_PERIOD\"" "$GENESIS_FILE" > "${GENESIS_FILE}.tmp" && mv "${GENESIS_FILE}.tmp" "$GENESIS_FILE"
    jq ".app_state.gov.params.quorum = \"$GOVENRANCE_QUORUM\"" "$GENESIS_FILE" > "${GENESIS_FILE}.tmp" && mv "${GENESIS_FILE}.tmp" "$GENESIS_FILE"
    jq ".app_state.gov.params.threshold = \"$GOVERNANCE_THRESHOLD\"" "$GENESIS_FILE" > "${GENESIS_FILE}.tmp" && mv "${GENESIS_FILE}.tmp" "$GENESIS_FILE"
    jq ".app_state.gov.params.veto_threshold = \"$GOVERNANCE_VETO_THRESHOLD\"" "$GENESIS_FILE" > "${GENESIS_FILE}.tmp" && mv "${GENESIS_FILE}.tmp" "$GENESIS_FILE"
    
    # Update crisis parameters
    jq ".app_state.crisis.params.constant_fee.amount = \"1000\"" "$GENESIS_FILE" > "${GENESIS_FILE}.tmp" && mv "${GENESIS_FILE}.tmp" "$GENESIS_FILE"
    jq ".app_state.crisis.params.constant_fee.denom = \"uremes\"" "$GENESIS_FILE" > "${GENESIS_FILE}.tmp" && mv "${GENESIS_FILE}.tmp" "$GENESIS_FILE"
    
    echo "✅ Genesis.json updated with tokenomics parameters"
else
    echo "⚠️  jq not found. Please install jq to update genesis.json automatically."
    echo "   Or manually update genesis.json with the following parameters:"
    echo ""
    echo "Mint Parameters:"
    echo "  - inflation: $MINT_INFLATION_RATE"
    echo "  - inflation_max: $MINT_INFLATION_MAX"
    echo "  - inflation_min: $MINT_INFLATION_MIN"
    echo "  - goal_bonded: $MINT_GOAL_BONDED"
    echo "  - blocks_per_year: $MINT_BLOCKS_PER_YEAR"
    echo ""
    echo "Staking Parameters:"
    echo "  - unbonding_time: $STAKING_UNBONDING_TIME"
    echo "  - max_validators: $STAKING_MAX_VALIDATORS"
    echo ""
    echo "Slashing Parameters:"
    echo "  - signed_blocks_window: $SLASHING_SIGNED_BLOCKS_WINDOW"
    echo "  - min_signed_per_window: $SLASHING_MIN_SIGNED_PER_WINDOW"
    echo "  - slash_fraction_double_sign: $SLASHING_SLASH_FRACTION_DOUBLE_SIGN"
    echo "  - slash_fraction_downtime: $SLASHING_SLASH_FRACTION_DOWNTIME"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Mainnet Genesis Configuration Complete"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review genesis.json parameters"
echo "2. Collect gentx from genesis validators"
echo "3. Run: ./build/remesd genesis collect-gentxs"
echo "4. Validate genesis.json: ./build/remesd genesis validate-genesis"
echo "5. Distribute genesis.json to all validators"
echo ""


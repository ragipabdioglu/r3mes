#!/bin/bash
# ===========================================
# R3MES Testnet Deployment Script
# Final deployment step
# ===========================================

set -e

echo "🚀 R3MES Testnet Deployment"
echo "================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

DEPLOY_DIR="/opt/r3mes"
cd "$DEPLOY_DIR"

# Check if .env exists
if [ ! -f "deploy/testnet/.env" ]; then
    echo -e "${RED}Error: .env file not found!${NC}"
    echo "Please copy .env.example to .env and configure it:"
    echo "  cp deploy/testnet/.env.example deploy/testnet/.env"
    echo "  nano deploy/testnet/.env"
    exit 1
fi

# Load environment variables
source deploy/testnet/.env

echo -e "${BLUE}📦 Step 1: Building Docker images...${NC}"
cd deploy/testnet
docker-compose -f docker-compose.testnet.yml build

echo ""
echo -e "${BLUE}🔑 Step 2: Generating wallet keys...${NC}"

# Generate faucet wallet if mnemonic not set
if [ -z "$FAUCET_MNEMONIC" ] || [ "$FAUCET_MNEMONIC" == "your twenty four word mnemonic phrase here for the faucet wallet account" ]; then
    echo "Generating new faucet wallet..."
    FAUCET_OUTPUT=$(docker run --rm r3mes-validator remesd keys add faucet --output json 2>&1)
    FAUCET_ADDRESS=$(echo "$FAUCET_OUTPUT" | jq -r '.address')
    FAUCET_MNEMONIC=$(echo "$FAUCET_OUTPUT" | jq -r '.mnemonic')
    
    echo ""
    echo -e "${YELLOW}⚠️  IMPORTANT: Save this mnemonic securely!${NC}"
    echo "Faucet Address: $FAUCET_ADDRESS"
    echo "Faucet Mnemonic: $FAUCET_MNEMONIC"
    echo ""
    echo "Update your .env file with:"
    echo "FAUCET_MNEMONIC=\"$FAUCET_MNEMONIC\""
    echo ""
    read -p "Press Enter after saving the mnemonic..."
fi

# Generate validator wallet
echo "Generating validator wallet..."
VALIDATOR_OUTPUT=$(docker run --rm r3mes-validator remesd keys add validator --output json 2>&1)
VALIDATOR_ADDRESS=$(echo "$VALIDATOR_OUTPUT" | jq -r '.address')
VALIDATOR_MNEMONIC=$(echo "$VALIDATOR_OUTPUT" | jq -r '.mnemonic')

echo ""
echo -e "${YELLOW}⚠️  IMPORTANT: Save this mnemonic securely!${NC}"
echo "Validator Address: $VALIDATOR_ADDRESS"
echo "Validator Mnemonic: $VALIDATOR_MNEMONIC"
echo ""

# Generate treasury wallet
echo "Generating treasury wallet..."
TREASURY_OUTPUT=$(docker run --rm r3mes-validator remesd keys add treasury --output json 2>&1)
TREASURY_ADDRESS=$(echo "$TREASURY_OUTPUT" | jq -r '.address')

echo "Treasury Address: $TREASURY_ADDRESS"
echo ""

echo -e "${BLUE}📜 Step 3: Updating genesis file...${NC}"
# Update genesis with actual addresses
sed -i "s/FAUCET_ADDRESS_PLACEHOLDER/$FAUCET_ADDRESS/g" genesis.json
sed -i "s/VALIDATOR_ADDRESS_PLACEHOLDER/$VALIDATOR_ADDRESS/g" genesis.json
sed -i "s/TREASURY_ADDRESS_PLACEHOLDER/$TREASURY_ADDRESS/g" genesis.json

echo ""
echo -e "${BLUE}🚀 Step 4: Starting services...${NC}"
docker-compose -f docker-compose.testnet.yml up -d

echo ""
echo -e "${BLUE}⏳ Step 5: Waiting for services to start...${NC}"
sleep 30

echo ""
echo -e "${BLUE}🔍 Step 6: Checking service status...${NC}"
docker-compose -f docker-compose.testnet.yml ps

echo ""
echo -e "${BLUE}📊 Step 7: Checking blockchain status...${NC}"
curl -s http://localhost:26657/status | jq '.result.sync_info'

echo ""
echo -e "${GREEN}✅ Deployment complete!${NC}"
echo ""
echo "================================"
echo "🌐 Service URLs:"
echo "================================"
echo "API:      https://api.r3mes.network"
echo "RPC:      https://rpc.r3mes.network"
echo "REST:     https://rest.r3mes.network"
echo "IPFS:     https://ipfs.r3mes.network"
echo "Grafana:  https://grafana.r3mes.network"
echo ""
echo "================================"
echo "📝 Important Addresses:"
echo "================================"
echo "Faucet:    $FAUCET_ADDRESS"
echo "Validator: $VALIDATOR_ADDRESS"
echo "Treasury:  $TREASURY_ADDRESS"
echo ""
echo "================================"
echo "🔧 Useful Commands:"
echo "================================"
echo "View logs:    docker-compose -f docker-compose.testnet.yml logs -f"
echo "Stop:         docker-compose -f docker-compose.testnet.yml down"
echo "Restart:      docker-compose -f docker-compose.testnet.yml restart"
echo "Status:       docker-compose -f docker-compose.testnet.yml ps"

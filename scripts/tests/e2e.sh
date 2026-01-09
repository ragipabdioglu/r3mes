#!/bin/bash
# R3MES End-to-End Test Script

set -e

echo "=========================================="
echo "R3MES End-to-End Test"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. IPFS kontrolü
echo "1. Checking IPFS daemon..."
if curl -s -X POST http://127.0.0.1:5001/api/v0/version > /dev/null 2>&1; then
    IPFS_VERSION=$(curl -s -X POST http://127.0.0.1:5001/api/v0/version | grep -o '"Version":"[^"]*"' | cut -d'"' -f4)
    echo -e "${GREEN}✅ IPFS daemon is running (${IPFS_VERSION})${NC}"
else
    echo -e "${RED}❌ IPFS daemon is not running${NC}"
    echo "   Start it with: ipfs daemon"
    echo "   Or install: https://docs.ipfs.tech/install/"
    exit 1
fi

# 2. Blockchain kontrolü
echo ""
echo "2. Checking blockchain node..."
if curl -s http://localhost:26657/status > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Blockchain node is running${NC}"
    NODE_INFO=$(curl -s http://localhost:26657/status | grep -o '"network":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
    echo "   Network: ${NODE_INFO}"
else
    echo -e "${RED}❌ Blockchain node is not running${NC}"
    echo "   Start it with: cd remes && remesd start"
    echo "   Or: cd remes && ignite chain serve"
    exit 1
fi

# 3. Python dependencies kontrolü
echo ""
echo "3. Checking Python dependencies..."
cd /home/rabdi/R3MES/miner-engine
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found, creating...${NC}"
    python3 -m venv venv
fi

source venv/bin/activate

if python3 -c "import torch; import ipfshttpclient" 2>/dev/null; then
    echo -e "${GREEN}✅ Python dependencies are installed${NC}"
else
    echo -e "${YELLOW}⚠️  Installing Python dependencies...${NC}"
    pip install -r requirements.txt
fi

# 4. Miner engine test
echo ""
echo "4. Testing miner engine..."
if python3 test_miner.py; then
    echo -e "${GREEN}✅ Miner engine test passed${NC}"
else
    echo -e "${RED}❌ Miner engine test failed${NC}"
    exit 1
fi

# 5. Özet
echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo -e "${GREEN}✅ IPFS daemon: Running${NC}"
echo -e "${GREEN}✅ Blockchain node: Running${NC}"
echo -e "${GREEN}✅ Python dependencies: Installed${NC}"
echo -e "${GREEN}✅ Miner engine: Tested${NC}"
echo ""
echo "Next steps:"
echo "1. Create wallet: remesd keys add miner1"
echo "2. Fund wallet (for testnet or local dev)"
echo "3. Run miner: cd miner-engine && source venv/bin/activate && python3 test_miner.py"
echo "4. Submit gradient to blockchain using the IPFS hash from test output"
echo ""


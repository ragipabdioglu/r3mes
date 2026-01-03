#!/bin/bash
# R3MES Validator Deployment Script
# Server: 38.242.246.6
# Run this on the server: bash /opt/r3mes/deploy/testnet/deploy-validator.sh

set -e

echo "=========================================="
echo "R3MES Validator Deployment"
echo "=========================================="

cd /opt/r3mes

# 1. Cleanup
echo "[1/6] Cleaning up old containers and volumes..."
docker stop r3mes-validator 2>/dev/null || true
docker rm r3mes-validator 2>/dev/null || true
docker volume rm r3mes_validator_data testnet_validator_data 2>/dev/null || true

# Remove any directory that was mistakenly created as start-validator.sh
if [ -d "/opt/r3mes/deploy/testnet/start-validator.sh" ]; then
    echo "Removing incorrectly created directory..."
    rm -rf /opt/r3mes/deploy/testnet/start-validator.sh
fi

# 2. Pull latest code
echo "[2/6] Pulling latest code..."
git pull origin main || echo "Git pull skipped (may have local changes)"

# 3. Verify files
echo "[3/6] Verifying files..."
ls -la deploy/testnet/Dockerfile.validator
ls -la remes/go.mod

# 4. Build validator image
echo "[4/6] Building validator image..."
docker build -t r3mes-validator:latest \
    -f deploy/testnet/Dockerfile.validator \
    ./remes

# 5. Create network if not exists
echo "[5/6] Creating network..."
docker network create testnet_r3mes-network 2>/dev/null || echo "Network already exists"

# 6. Start validator
echo "[6/6] Starting validator..."
docker run -d \
    --name r3mes-validator \
    --network testnet_r3mes-network \
    -p 26656:26656 \
    -p 26657:26657 \
    -p 1317:1317 \
    -p 9090:9090 \
    -v r3mes_validator_data:/root/.remes \
    -e MONIKER=r3mes-testnet-validator-1 \
    -e CHAIN_ID=r3mes-testnet-1 \
    --restart always \
    r3mes-validator:latest

echo ""
echo "=========================================="
echo "Deployment complete!"
echo "=========================================="
echo ""
echo "Checking logs (Ctrl+C to exit)..."
sleep 3
docker logs -f r3mes-validator

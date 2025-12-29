#!/bin/bash
# R3MES Testnet Quick Start Script
# This script starts all testnet services with Docker Compose

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$SCRIPT_DIR"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$DOCKER_DIR"

print_header "R3MES Testnet Quick Start"

echo ""
echo "Domain: r3mes.network"
echo "Chain ID: remes-testnet-1"
echo ""

# Step 1: Check Docker
print_info "Checking Docker..."
if ! command -v docker &> /dev/null; then
    print_error "Docker not found. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    print_error "Docker Compose not found. Please install Docker Compose."
    exit 1
fi
print_success "Docker is available"

# Step 2: Create secrets if not exist
print_info "Checking secrets..."
if [ ! -d "$DOCKER_DIR/secrets" ]; then
    mkdir -p "$DOCKER_DIR/secrets"
fi

if [ ! -f "$DOCKER_DIR/secrets/postgres_password.txt" ]; then
    print_warning "Creating Docker secrets..."
    openssl rand -base64 32 | tr -d "=+/" | cut -c1-32 > "$DOCKER_DIR/secrets/postgres_password.txt"
    openssl rand -base64 32 | tr -d "=+/" | cut -c1-32 > "$DOCKER_DIR/secrets/redis_password.txt"
    openssl rand -base64 32 | tr -d "=+/" | cut -c1-32 > "$DOCKER_DIR/secrets/grafana_admin_password.txt"
    chmod 600 "$DOCKER_DIR/secrets/"*.txt
    print_success "Secrets created"
else
    print_success "Secrets already exist"
fi

# Step 3: Copy env file if not exist
if [ ! -f "$DOCKER_DIR/.env" ]; then
    if [ -f "$DOCKER_DIR/.env.testnet" ]; then
        cp "$DOCKER_DIR/.env.testnet" "$DOCKER_DIR/.env"
        print_success "Environment file copied from .env.testnet"
    else
        print_error ".env.testnet file not found!"
        exit 1
    fi
else
    print_success "Environment file exists"
fi

# Step 4: Build and start services
print_header "Starting Services"

echo ""
print_info "Building Docker images (this may take a few minutes on first run)..."
echo ""

# Use docker compose (v2) or docker-compose (v1)
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

# Build images
$COMPOSE_CMD -f docker-compose.prod.yml build

# Start services (without miner profile)
print_info "Starting services..."
$COMPOSE_CMD -f docker-compose.prod.yml up -d

# Step 5: Wait for services
print_header "Waiting for Services"

echo ""
print_info "Waiting for blockchain to initialize (this may take 1-2 minutes on first start)..."

# Wait for blockchain
MAX_WAIT=180
WAIT_COUNT=0
while ! curl -s http://localhost:26657/status > /dev/null 2>&1; do
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
    if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
        print_warning "Blockchain is still starting. Check logs with: docker logs r3mes-blockchain-prod"
        break
    fi
    echo -n "."
done
echo ""

if curl -s http://localhost:26657/status > /dev/null 2>&1; then
    print_success "Blockchain is running!"
else
    print_warning "Blockchain may still be initializing. Check logs."
fi

# Check other services
sleep 10

if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    print_success "Backend is running!"
else
    print_warning "Backend is still starting..."
fi

if curl -s http://localhost:3000 > /dev/null 2>&1; then
    print_success "Frontend is running!"
else
    print_warning "Frontend is still starting..."
fi

# Step 6: Print summary
print_header "Testnet Started!"

echo ""
echo "Services:"
echo "  - Blockchain RPC:    http://localhost:26657"
echo "  - Blockchain gRPC:   localhost:9090"
echo "  - Blockchain REST:   http://localhost:1317"
echo "  - Backend API:       http://localhost:8000"
echo "  - Frontend:          http://localhost:3000"
echo "  - IPFS:              http://localhost:5001"
echo "  - Grafana:           http://localhost:3001"
echo ""
echo "Useful commands:"
echo "  - View logs:         docker logs -f r3mes-blockchain-prod"
echo "  - View all logs:     $COMPOSE_CMD -f docker-compose.prod.yml logs -f"
echo "  - Stop services:     $COMPOSE_CMD -f docker-compose.prod.yml down"
echo "  - Restart:           $COMPOSE_CMD -f docker-compose.prod.yml restart"
echo ""
print_warning "Note: First startup may take 2-3 minutes for all services to be ready."
echo ""
print_success "Testnet is ready! 🚀"

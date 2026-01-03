#!/bin/bash
# R3MES Testnet Full Deployment Script
# Server: 38.242.246.6
# Domain: r3mes.network
#
# Usage: bash /opt/r3mes/deploy/testnet/deploy-validator.sh [command]
# Commands:
#   deploy    - Full deployment (default)
#   rebuild   - Rebuild and restart validator only
#   restart   - Restart all services
#   logs      - Show logs
#   status    - Show status

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/opt/r3mes"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.testnet.yml"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info() { echo -e "${CYAN}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if .env exists
check_env() {
    if [ ! -f "$SCRIPT_DIR/.env" ]; then
        log_error ".env file not found!"
        log_info "Copy .env.example to .env and fill in the values:"
        log_info "  cp $SCRIPT_DIR/.env.example $SCRIPT_DIR/.env"
        exit 1
    fi
}

# Full deployment
deploy() {
    echo "=========================================="
    echo "R3MES Testnet Full Deployment"
    echo "=========================================="
    
    cd "$PROJECT_ROOT"
    check_env
    
    # 1. Pull latest code
    log_info "[1/7] Pulling latest code..."
    git pull origin main || log_warn "Git pull skipped (may have local changes)"
    
    # 2. Stop and remove existing services
    log_info "[2/7] Stopping and removing existing services..."
    docker compose -f "$COMPOSE_FILE" down --remove-orphans 2>/dev/null || true
    
    # Force remove any leftover containers with our names
    docker rm -f r3mes-validator r3mes-backend r3mes-nginx r3mes-postgres r3mes-redis r3mes-ipfs 2>/dev/null || true
    
    # 3. Clean up old validator data if needed
    log_info "[3/7] Checking validator data..."
    if [ "$CLEAN_DATA" = "true" ]; then
        log_warn "Cleaning validator data (CLEAN_DATA=true)..."
        docker volume rm testnet_validator_data 2>/dev/null || true
    fi
    
    # 4. Build images
    log_info "[4/7] Building Docker images..."
    docker compose -f "$COMPOSE_FILE" build --no-cache validator backend
    
    # 5. Start infrastructure services first
    log_info "[5/7] Starting infrastructure services..."
    docker compose -f "$COMPOSE_FILE" up -d postgres redis ipfs
    
    # Wait for postgres
    log_info "Waiting for PostgreSQL to be ready..."
    sleep 10
    
    # 6. Start validator
    log_info "[6/7] Starting validator..."
    docker compose -f "$COMPOSE_FILE" up -d validator
    
    # Wait for validator to initialize
    log_info "Waiting for validator to initialize (60s)..."
    sleep 60
    
    # 7. Start remaining services
    log_info "[7/7] Starting backend and nginx..."
    docker compose -f "$COMPOSE_FILE" up -d backend nginx
    
    echo ""
    log_success "=========================================="
    log_success "Deployment complete!"
    log_success "=========================================="
    echo ""
    
    status
}

# Rebuild validator only
rebuild_validator() {
    log_info "Rebuilding validator..."
    cd "$PROJECT_ROOT"
    
    docker compose -f "$COMPOSE_FILE" stop validator
    docker compose -f "$COMPOSE_FILE" build --no-cache validator
    docker compose -f "$COMPOSE_FILE" up -d validator
    
    log_success "Validator rebuilt and restarted"
    docker logs -f r3mes-validator
}

# Restart all services
restart() {
    log_info "Restarting all services..."
    cd "$PROJECT_ROOT"
    
    docker compose -f "$COMPOSE_FILE" restart
    
    log_success "All services restarted"
    status
}

# Show logs
logs() {
    local service="${1:-validator}"
    log_info "Showing logs for $service..."
    docker compose -f "$COMPOSE_FILE" logs -f "$service"
}

# Show status
status() {
    echo ""
    log_info "Service Status:"
    echo "----------------------------------------"
    docker compose -f "$COMPOSE_FILE" ps
    echo ""
    
    log_info "Endpoint Status:"
    echo "----------------------------------------"
    
    # Check RPC
    if curl -s --max-time 5 http://localhost:26657/status > /dev/null 2>&1; then
        local height=$(curl -s http://localhost:26657/status | jq -r '.result.sync_info.latest_block_height')
        log_success "RPC (26657): OK - Block height: $height"
    else
        log_error "RPC (26657): FAILED"
    fi
    
    # Check REST API
    if curl -s --max-time 5 http://localhost:1317/cosmos/base/tendermint/v1beta1/node_info > /dev/null 2>&1; then
        log_success "REST API (1317): OK"
    else
        log_error "REST API (1317): FAILED"
    fi
    
    # Check Backend
    if curl -s --max-time 5 http://localhost:8000/health > /dev/null 2>&1; then
        log_success "Backend API (8000): OK"
    else
        log_error "Backend API (8000): FAILED"
    fi
    
    # Check Faucet
    if curl -s --max-time 5 http://localhost:8000/faucet/status > /dev/null 2>&1; then
        local faucet_status=$(curl -s http://localhost:8000/faucet/status | jq -r '.enabled')
        log_success "Faucet: $faucet_status"
    else
        log_warn "Faucet: Not available"
    fi
    
    echo ""
    log_info "External URLs:"
    echo "----------------------------------------"
    echo "  RPC:      https://rpc.r3mes.network"
    echo "  REST:     https://rest.r3mes.network"
    echo "  API:      https://api.r3mes.network"
    echo "  IPFS:     https://ipfs.r3mes.network"
    echo ""
}

# Test faucet
test_faucet() {
    log_info "Testing faucet..."
    
    # Get faucet status
    log_info "Faucet status:"
    curl -s http://localhost:8000/faucet/status | jq .
    
    # Test claim (will fail due to rate limit after first try)
    log_info "Testing claim (use a test address):"
    echo "curl -X POST http://localhost:8000/faucet/claim -H 'Content-Type: application/json' -d '{\"address\": \"remes1...\"}'"
}

# Main
case "${1:-deploy}" in
    deploy)
        deploy
        ;;
    rebuild)
        rebuild_validator
        ;;
    restart)
        restart
        ;;
    logs)
        logs "${2:-validator}"
        ;;
    status)
        status
        ;;
    test-faucet)
        test_faucet
        ;;
    *)
        echo "Usage: $0 {deploy|rebuild|restart|logs|status|test-faucet}"
        exit 1
        ;;
esac

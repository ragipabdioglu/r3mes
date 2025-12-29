#!/bin/bash
# R3MES Quick Deploy Script
# One-command deployment for testnet/mainnet
#
# Usage:
#   bash scripts/quick_deploy.sh --domain testnet.r3mes.network --email admin@r3mes.network --chain-id remes-testnet-1
#   bash scripts/quick_deploy.sh --domain r3mes.network --email admin@r3mes.network --chain-id remes-mainnet-1 --mainnet

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKER_DIR="$PROJECT_ROOT/docker"
SECRETS_DIR="$DOCKER_DIR/secrets"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DOMAIN=""
EMAIL=""
CHAIN_ID=""
NETWORK_TYPE="testnet"
AUTO_SECRETS=true
SKIP_DOCKER_CHECK=false

# Functions
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

# Parse arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --domain)
                DOMAIN="$2"
                shift 2
                ;;
            --email)
                EMAIL="$2"
                shift 2
                ;;
            --chain-id)
                CHAIN_ID="$2"
                shift 2
                ;;
            --mainnet)
                NETWORK_TYPE="mainnet"
                shift
                ;;
            --testnet)
                NETWORK_TYPE="testnet"
                shift
                ;;
            --no-auto-secrets)
                AUTO_SECRETS=false
                shift
                ;;
            --skip-docker-check)
                SKIP_DOCKER_CHECK=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

show_help() {
    cat << EOF
R3MES Quick Deploy Script

Usage:
    bash scripts/quick_deploy.sh [OPTIONS]

Required Options:
    --domain DOMAIN          Domain name (e.g., testnet.r3mes.network)
    --email EMAIL            Email for Let's Encrypt certificates

Optional Options:
    --chain-id CHAIN_ID      Chain ID (default: auto-detect from network type)
    --mainnet                Deploy for mainnet (default: testnet)
    --testnet                Deploy for testnet (default)
    --no-auto-secrets        Don't auto-generate secrets (use existing or create manually)
    --skip-docker-check      Skip Docker installation check
    --help, -h               Show this help message

Examples:
    # Testnet deployment
    bash scripts/quick_deploy.sh \\
        --domain testnet.r3mes.network \\
        --email admin@r3mes.network

    # Mainnet deployment
    bash scripts/quick_deploy.sh \\
        --domain r3mes.network \\
        --email admin@r3mes.network \\
        --mainnet \\
        --chain-id remes-mainnet-1

    # Custom chain ID
    bash scripts/quick_deploy.sh \\
        --domain testnet.r3mes.network \\
        --email admin@r3mes.network \\
        --chain-id remes-testnet-1
EOF
}

# Check if Docker is installed
check_docker() {
    if [ "$SKIP_DOCKER_CHECK" = true ]; then
        return 0
    fi

    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed."
        print_info "Installing Docker..."
        
        # Try to install Docker (Ubuntu/Debian)
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y docker.io docker-compose
            sudo systemctl start docker
            sudo systemctl enable docker
            sudo usermod -aG docker $USER
            print_warning "Docker installed. You may need to logout/login for group changes."
        else
            print_error "Please install Docker manually: https://docs.docker.com/get-docker/"
            exit 1
        fi
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null 2>&1; then
        print_error "Docker Compose is not installed."
        print_info "Installing Docker Compose..."
        
        if command -v apt-get &> /dev/null; then
            sudo apt-get install -y docker-compose
        else
            print_error "Please install Docker Compose manually: https://docs.docker.com/compose/install/"
            exit 1
        fi
    fi
    
    print_success "Docker and Docker Compose are installed"
}

# Generate random password
generate_password() {
    openssl rand -base64 32 | tr -d "=+/" | cut -c1-32 2>/dev/null || \
    head -c 32 /dev/urandom | base64 | tr -d "=+/" | cut -c1-32
}

# Create secrets automatically
create_secrets_auto() {
    print_header "Creating Docker Secrets"
    
    mkdir -p "$SECRETS_DIR"
    
    # Generate passwords
    POSTGRES_PASSWORD=$(generate_password)
    REDIS_PASSWORD=$(generate_password)
    GRAFANA_PASSWORD=$(generate_password)
    
    # Save to files
    echo -n "$POSTGRES_PASSWORD" > "$SECRETS_DIR/postgres_password.txt"
    echo -n "$REDIS_PASSWORD" > "$SECRETS_DIR/redis_password.txt"
    echo -n "$GRAFANA_PASSWORD" > "$SECRETS_DIR/grafana_admin_password.txt"
    
    # Set permissions
    chmod 600 "$SECRETS_DIR"/*.txt
    
    print_success "Docker secrets created"
    print_info "PostgreSQL password: [saved to secrets]"
    print_info "Redis password: [saved to secrets]"
    print_info "Grafana password: [saved to secrets]"
}

# Check if secrets exist
check_secrets() {
    if [ ! -d "$SECRETS_DIR" ]; then
        mkdir -p "$SECRETS_DIR"
    fi
    
    if [ ! -f "$SECRETS_DIR/postgres_password.txt" ] || \
       [ ! -f "$SECRETS_DIR/redis_password.txt" ] || \
       [ ! -f "$SECRETS_DIR/grafana_admin_password.txt" ]; then
        if [ "$AUTO_SECRETS" = true ]; then
            create_secrets_auto
        else
            print_warning "Docker secrets not found"
            print_info "Run: bash scripts/create_secrets.sh"
            exit 1
        fi
    else
        print_success "Docker secrets already exist"
    fi
}

# Create .env.production file
create_env_file() {
    print_header "Creating Environment Configuration"
    
    ENV_FILE="$DOCKER_DIR/.env.production"
    
    # Determine which example file to use
    if [ "$NETWORK_TYPE" = "mainnet" ]; then
        EXAMPLE_FILE="$DOCKER_DIR/env.production.example"
    else
        EXAMPLE_FILE="$DOCKER_DIR/env.testnet.example"
    fi
    
    if [ ! -f "$EXAMPLE_FILE" ]; then
        print_error "Example file not found: $EXAMPLE_FILE"
        exit 1
    fi
    
    # Copy example file
    cp "$EXAMPLE_FILE" "$ENV_FILE"
    
    # Set default chain ID if not provided
    if [ -z "$CHAIN_ID" ]; then
        if [ "$NETWORK_TYPE" = "mainnet" ]; then
            CHAIN_ID="remes-mainnet-1"
        else
            CHAIN_ID="remes-testnet-1"
        fi
    fi
    
    # Determine RPC and REST URLs based on domain
    if [[ "$DOMAIN" == *"testnet"* ]]; then
        # Extract base domain (remove testnet. prefix if exists)
        BASE_DOMAIN="${DOMAIN#testnet.}"
        RPC_URL="https://testnet-rpc.${BASE_DOMAIN}"
        REST_URL="https://testnet-api.${BASE_DOMAIN}"
    else
        # Mainnet: use rpc.domain and api.domain
        RPC_URL="https://rpc.${DOMAIN}"
        REST_URL="https://api.${DOMAIN}"
    fi
    
    # Update critical variables using sed
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s|^CHAIN_ID=.*|CHAIN_ID=$CHAIN_ID|" "$ENV_FILE"
        sed -i '' "s|^DOMAIN=.*|DOMAIN=$DOMAIN|" "$ENV_FILE"
        sed -i '' "s|^EMAIL=.*|EMAIL=$EMAIL|" "$ENV_FILE"
               sed -i '' "s|^NEXT_PUBLIC_CHAIN_ID=.*|NEXT_PUBLIC_CHAIN_ID=$CHAIN_ID|" "$ENV_FILE"
               sed -i '' "s|^NEXT_PUBLIC_SITE_URL=.*|NEXT_PUBLIC_SITE_URL=https://$DOMAIN|" "$ENV_FILE"
               sed -i '' "s|^NEXT_PUBLIC_API_URL=.*|NEXT_PUBLIC_API_URL=https://$DOMAIN/api|" "$ENV_FILE"
               sed -i '' "s|^NEXT_PUBLIC_BACKEND_URL=.*|NEXT_PUBLIC_BACKEND_URL=https://$DOMAIN/api|" "$ENV_FILE"
               sed -i '' "s|^NEXT_PUBLIC_RPC_URL=.*|NEXT_PUBLIC_RPC_URL=$RPC_URL|" "$ENV_FILE"
               sed -i '' "s|^NEXT_PUBLIC_REST_URL=.*|NEXT_PUBLIC_REST_URL=$REST_URL|" "$ENV_FILE"
               sed -i '' "s|^CORS_ALLOWED_ORIGINS=.*|CORS_ALLOWED_ORIGINS=https://$DOMAIN,https://www.$DOMAIN|" "$ENV_FILE"
    else
        # Linux
        sed -i "s|^CHAIN_ID=.*|CHAIN_ID=$CHAIN_ID|" "$ENV_FILE"
        sed -i "s|^DOMAIN=.*|DOMAIN=$DOMAIN|" "$ENV_FILE"
        sed -i "s|^EMAIL=.*|EMAIL=$EMAIL|" "$ENV_FILE"
        sed -i "s|^NEXT_PUBLIC_CHAIN_ID=.*|NEXT_PUBLIC_CHAIN_ID=$CHAIN_ID|" "$ENV_FILE"
        sed -i "s|^NEXT_PUBLIC_SITE_URL=.*|NEXT_PUBLIC_SITE_URL=https://$DOMAIN|" "$ENV_FILE"
        sed -i "s|^NEXT_PUBLIC_API_URL=.*|NEXT_PUBLIC_API_URL=https://$DOMAIN/api|" "$ENV_FILE"
        sed -i "s|^NEXT_PUBLIC_BACKEND_URL=.*|NEXT_PUBLIC_BACKEND_URL=https://$DOMAIN/api|" "$ENV_FILE"
        sed -i "s|^NEXT_PUBLIC_RPC_URL=.*|NEXT_PUBLIC_RPC_URL=$RPC_URL|" "$ENV_FILE"
        sed -i "s|^NEXT_PUBLIC_REST_URL=.*|NEXT_PUBLIC_REST_URL=$REST_URL|" "$ENV_FILE"
        sed -i "s|^CORS_ALLOWED_ORIGINS=.*|CORS_ALLOWED_ORIGINS=https://$DOMAIN,https://www.$DOMAIN|" "$ENV_FILE"
    fi
    
    print_success "Environment file created: $ENV_FILE"
    print_info "Domain: $DOMAIN"
    print_info "Email: $EMAIL"
    print_info "Chain ID: $CHAIN_ID"
    print_info "Network: $NETWORK_TYPE"
    print_info "RPC URL: $RPC_URL"
    print_info "REST URL: $REST_URL"
}

# Validate required parameters
validate_params() {
    local errors=0
    
    if [ -z "$DOMAIN" ]; then
        print_error "Domain is required (--domain)"
        errors=$((errors + 1))
    fi
    
    if [ -z "$EMAIL" ]; then
        print_error "Email is required (--email)"
        errors=$((errors + 1))
    fi
    
    # Validate email format (basic)
    if [ -n "$EMAIL" ] && [[ ! "$EMAIL" =~ ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$ ]]; then
        print_warning "Email format may be invalid: $EMAIL"
    fi
    
    if [ $errors -gt 0 ]; then
        print_error "Missing required parameters"
        show_help
        exit 1
    fi
}

# Main deployment
main() {
    print_header "R3MES Quick Deploy"
    print_info "Network Type: $NETWORK_TYPE"
    print_info "Domain: $DOMAIN"
    print_info "Email: $EMAIL"
    print_info "Chain ID: ${CHAIN_ID:-auto-detect}"
    echo ""
    
    # Validate parameters
    validate_params
    
    # Check Docker
    check_docker
    
    # Check/create secrets
    check_secrets
    
    # Create environment file
    create_env_file
    
    # Deploy using existing script
    print_header "Starting Deployment"
    print_info "Using deploy_production_docker.sh..."
    echo ""
    
    bash "$PROJECT_ROOT/scripts/deploy_production_docker.sh"
    
    echo ""
    print_header "Deployment Complete!"
    print_success "R3MES has been deployed successfully"
    echo ""
    print_info "Next steps:"
    echo "  1. Wait for services to start (may take 2-5 minutes)"
    echo "  2. Check service status: cd docker && docker compose -f docker-compose.prod.yml ps"
    echo "  3. Check logs: cd docker && docker compose -f docker-compose.prod.yml logs -f"
    echo "  4. Access dashboard: https://$DOMAIN"
    echo "  5. Access Grafana: https://$DOMAIN:3001"
    echo ""
    print_info "Health checks:"
    echo "  - Frontend: https://$DOMAIN/health"
    echo "  - Backend: https://$DOMAIN/api/health"
    echo "  - Blockchain: https://$DOMAIN/api/blockchain/health"
    echo ""
}

# Parse arguments and run
parse_args "$@"
main


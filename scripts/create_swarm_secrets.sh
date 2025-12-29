#!/bin/bash
# Create Docker Swarm Secrets
# This script creates secrets for Docker Swarm deployment
# Usage: bash scripts/create_swarm_secrets.sh [--rotate]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_header() { echo -e "\n${CYAN}=== $1 ===${NC}\n"; }

# Check if Docker Swarm is initialized
check_swarm() {
    if ! docker info 2>/dev/null | grep -q "Swarm: active"; then
        log_error "Docker Swarm is not initialized"
        log_info "Initialize with: docker swarm init"
        exit 1
    fi
    log_success "Docker Swarm is active"
}

# Generate secure random password
generate_password() {
    local length=${1:-32}
    openssl rand -base64 48 | tr -d "=+/" | cut -c1-$length
}

# Generate secure random key
generate_key() {
    openssl rand -hex 32
}

# Create or update a secret
create_secret() {
    local name=$1
    local value=$2
    local rotate=${3:-false}
    
    # Check if secret exists
    if docker secret inspect "$name" &>/dev/null; then
        if [ "$rotate" = true ]; then
            log_warn "Rotating secret: $name"
            # Remove old secret (services must be updated to use new version)
            docker secret rm "$name" 2>/dev/null || true
            echo -n "$value" | docker secret create "$name" -
            log_success "Secret rotated: $name"
        else
            log_warn "Secret already exists: $name (use --rotate to update)"
        fi
    else
        echo -n "$value" | docker secret create "$name" -
        log_success "Secret created: $name"
    fi
}

# Create secret from file
create_secret_from_file() {
    local name=$1
    local file=$2
    local rotate=${3:-false}
    
    if [ ! -f "$file" ]; then
        log_error "File not found: $file"
        return 1
    fi
    
    if docker secret inspect "$name" &>/dev/null; then
        if [ "$rotate" = true ]; then
            log_warn "Rotating secret from file: $name"
            docker secret rm "$name" 2>/dev/null || true
            docker secret create "$name" "$file"
            log_success "Secret rotated: $name"
        else
            log_warn "Secret already exists: $name (use --rotate to update)"
        fi
    else
        docker secret create "$name" "$file"
        log_success "Secret created from file: $name"
    fi
}

# Parse arguments
ROTATE=false
INTERACTIVE=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --rotate)
            ROTATE=true
            shift
            ;;
        --non-interactive)
            INTERACTIVE=false
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

log_header "Docker Swarm Secrets Creation"

# Check swarm status
check_swarm

# Database Secrets
log_header "Database Secrets"

if [ "$INTERACTIVE" = true ]; then
    read -sp "Enter PostgreSQL password (press Enter to generate): " POSTGRES_PASSWORD
    echo
    [ -z "$POSTGRES_PASSWORD" ] && POSTGRES_PASSWORD=$(generate_password)
    
    read -p "Enter PostgreSQL user [r3mes]: " POSTGRES_USER
    [ -z "$POSTGRES_USER" ] && POSTGRES_USER="r3mes"
else
    POSTGRES_PASSWORD=$(generate_password)
    POSTGRES_USER="r3mes"
fi

create_secret "postgres_password" "$POSTGRES_PASSWORD" $ROTATE
create_secret "postgres_user" "$POSTGRES_USER" $ROTATE

# Redis Secret
log_header "Redis Secret"

if [ "$INTERACTIVE" = true ]; then
    read -sp "Enter Redis password (press Enter to generate): " REDIS_PASSWORD
    echo
    [ -z "$REDIS_PASSWORD" ] && REDIS_PASSWORD=$(generate_password)
else
    REDIS_PASSWORD=$(generate_password)
fi

create_secret "redis_password" "$REDIS_PASSWORD" $ROTATE

# Application Secrets
log_header "Application Secrets"

if [ "$INTERACTIVE" = true ]; then
    read -sp "Enter JWT secret (press Enter to generate): " JWT_SECRET
    echo
    [ -z "$JWT_SECRET" ] && JWT_SECRET=$(generate_key)
    
    read -sp "Enter API secret key (press Enter to generate): " API_SECRET_KEY
    echo
    [ -z "$API_SECRET_KEY" ] && API_SECRET_KEY=$(generate_key)
else
    JWT_SECRET=$(generate_key)
    API_SECRET_KEY=$(generate_key)
fi

create_secret "jwt_secret" "$JWT_SECRET" $ROTATE
create_secret "api_secret_key" "$API_SECRET_KEY" $ROTATE

# Grafana Secret
log_header "Grafana Secret"

if [ "$INTERACTIVE" = true ]; then
    read -sp "Enter Grafana admin password (press Enter to generate): " GRAFANA_PASSWORD
    echo
    [ -z "$GRAFANA_PASSWORD" ] && GRAFANA_PASSWORD=$(generate_password 24)
else
    GRAFANA_PASSWORD=$(generate_password 24)
fi

create_secret "grafana_admin_password" "$GRAFANA_PASSWORD" $ROTATE

# Blockchain Secrets (optional)
log_header "Blockchain Secrets (Optional)"

VALIDATOR_KEY_FILE="$PROJECT_ROOT/docker/secrets/validator_key.json"
NODE_KEY_FILE="$PROJECT_ROOT/docker/secrets/node_key.json"

if [ -f "$VALIDATOR_KEY_FILE" ]; then
    create_secret_from_file "validator_key" "$VALIDATOR_KEY_FILE" $ROTATE
else
    log_warn "Validator key file not found: $VALIDATOR_KEY_FILE"
    log_info "Create it manually or skip for non-validator nodes"
fi

if [ -f "$NODE_KEY_FILE" ]; then
    create_secret_from_file "node_key" "$NODE_KEY_FILE" $ROTATE
else
    log_warn "Node key file not found: $NODE_KEY_FILE"
    log_info "Will be auto-generated on first run"
fi

# SSL Certificates (optional)
log_header "SSL Certificates (Optional)"

SSL_CERT_FILE="$PROJECT_ROOT/docker/secrets/ssl_certificate.pem"
SSL_KEY_FILE="$PROJECT_ROOT/docker/secrets/ssl_private_key.pem"

if [ -f "$SSL_CERT_FILE" ]; then
    create_secret_from_file "ssl_certificate" "$SSL_CERT_FILE" $ROTATE
else
    log_warn "SSL certificate not found: $SSL_CERT_FILE"
    log_info "Use Let's Encrypt or provide your own certificate"
fi

if [ -f "$SSL_KEY_FILE" ]; then
    create_secret_from_file "ssl_private_key" "$SSL_KEY_FILE" $ROTATE
else
    log_warn "SSL private key not found: $SSL_KEY_FILE"
fi

# Summary
log_header "Secrets Summary"

echo "Created/Updated secrets:"
docker secret ls --format "table {{.Name}}\t{{.CreatedAt}}"

log_header "Next Steps"

echo "1. Create Docker configs (if needed):"
echo "   docker config create prometheus_config monitoring/prometheus/prometheus.prod.yml"
echo "   docker config create prometheus_alerts monitoring/prometheus/alerts.prod.yml"
echo ""
echo "2. Deploy the stack:"
echo "   docker stack deploy -c docker/docker-compose.swarm.yml r3mes"
echo ""
echo "3. Check services:"
echo "   docker service ls"
echo ""

if [ "$INTERACTIVE" = true ]; then
    log_warn "IMPORTANT: Save these credentials securely!"
    echo ""
    echo "PostgreSQL User: $POSTGRES_USER"
    echo "PostgreSQL Password: $POSTGRES_PASSWORD"
    echo "Redis Password: $REDIS_PASSWORD"
    echo "JWT Secret: $JWT_SECRET"
    echo "API Secret Key: $API_SECRET_KEY"
    echo "Grafana Admin Password: $GRAFANA_PASSWORD"
fi

log_success "Swarm secrets creation completed!"

#!/bin/bash
# Create Docker Swarm Secrets - Production Ready
# This script creates secrets for Docker Swarm deployment with enhanced security
# Usage: bash scripts/create_swarm_secrets.sh [--rotate] [--non-interactive]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Disable history for this session to prevent password leakage
set +o history

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1" >&2; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1" >&2; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1" >&2; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }
log_header() { echo -e "\n${CYAN}=== $1 ===${NC}\n" >&2; }

# Cleanup function
cleanup() {
    local exit_code=$?
    # Clear any sensitive variables
    unset POSTGRES_PASSWORD REDIS_PASSWORD JWT_SECRET API_SECRET_KEY GRAFANA_PASSWORD
    # Re-enable history
    set -o history
    exit $exit_code
}

trap cleanup EXIT ERR

# Secure password reading
read_secret() {
    local prompt="$1"
    local secret
    echo -n "$prompt" >&2
    read -s secret
    echo >&2  # New line after hidden input
    echo "$secret"
}

# Check if Docker Swarm is initialized
check_swarm() {
    if ! docker info 2>/dev/null | grep -q "Swarm: active"; then
        log_error "Docker Swarm is not initialized"
        log_info "Initialize with: docker swarm init"
        exit 1
    fi
    log_success "Docker Swarm is active"
}

# Generate cryptographically secure password
generate_password() {
    local length=${1:-32}
    # Use /dev/urandom for better entropy
    if command -v openssl >/dev/null 2>&1; then
        openssl rand -base64 $((length * 3 / 4)) | tr -d '\n' | cut -c1-$length
    else
        # Fallback to /dev/urandom
        head -c $length /dev/urandom | base64 | tr -d '\n' | cut -c1-$length
    fi
}

# Generate secure random key
generate_key() {
    if command -v openssl >/dev/null 2>&1; then
        openssl rand -hex 32
    else
        head -c 32 /dev/urandom | xxd -p -c 32
    fi
}

# Create or update a secret
create_secret() {
    local name=$1
    local value=$2
    local rotate=${3:-false}
    
    # Validate secret name
    if [[ ! "$name" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        log_error "Invalid secret name: $name"
        return 1
    fi
    
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

# Create secret from file with validation
create_secret_from_file() {
    local name=$1
    local file=$2
    local rotate=${3:-false}
    
    # Validate inputs
    if [[ ! "$name" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        log_error "Invalid secret name: $name"
        return 1
    fi
    
    if [ ! -f "$file" ]; then
        log_error "File not found: $file"
        return 1
    fi
    
    # Check file permissions (should not be world-readable)
    if [ "$(stat -c %a "$file" 2>/dev/null || stat -f %A "$file" 2>/dev/null)" -gt 600 ]; then
        log_warn "File $file has overly permissive permissions"
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

# Validate password strength
validate_password_strength() {
    local password="$1"
    local min_length=12
    
    if [ ${#password} -lt $min_length ]; then
        log_error "Password too short (minimum $min_length characters)"
        return 1
    fi
    
    # Check for common weak passwords
    local weak_passwords=("password" "123456" "admin" "root" "test")
    for weak in "${weak_passwords[@]}"; do
        if [[ "$password" == *"$weak"* ]]; then
            log_error "Password contains weak pattern: $weak"
            return 1
        fi
    done
    
    return 0
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
    POSTGRES_PASSWORD=$(read_secret "Enter PostgreSQL password (press Enter to generate): ")
    [ -z "$POSTGRES_PASSWORD" ] && POSTGRES_PASSWORD=$(generate_password)
    
    if ! validate_password_strength "$POSTGRES_PASSWORD"; then
        log_error "Password validation failed"
        exit 1
    fi
    
    read -p "Enter PostgreSQL user [r3mes]: " POSTGRES_USER >&2
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
    REDIS_PASSWORD=$(read_secret "Enter Redis password (press Enter to generate): ")
    [ -z "$REDIS_PASSWORD" ] && REDIS_PASSWORD=$(generate_password)
    
    if ! validate_password_strength "$REDIS_PASSWORD"; then
        log_error "Redis password validation failed"
        exit 1
    fi
else
    REDIS_PASSWORD=$(generate_password)
fi

create_secret "redis_password" "$REDIS_PASSWORD" $ROTATE

# Application Secrets
log_header "Application Secrets"

if [ "$INTERACTIVE" = true ]; then
    JWT_SECRET=$(read_secret "Enter JWT secret (press Enter to generate): ")
    [ -z "$JWT_SECRET" ] && JWT_SECRET=$(generate_key)
    
    API_SECRET_KEY=$(read_secret "Enter API secret key (press Enter to generate): ")
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
    GRAFANA_PASSWORD=$(read_secret "Enter Grafana admin password (press Enter to generate): ")
    [ -z "$GRAFANA_PASSWORD" ] && GRAFANA_PASSWORD=$(generate_password 24)
    
    if ! validate_password_strength "$GRAFANA_PASSWORD"; then
        log_error "Grafana password validation failed"
        exit 1
    fi
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

echo "Created/Updated secrets:" >&2
docker secret ls --format "table {{.Name}}\t{{.CreatedAt}}"

log_header "Next Steps"

echo "1. Create Docker configs (if needed):" >&2
echo "   docker config create prometheus_config monitoring/prometheus/prometheus.prod.yml" >&2
echo "   docker config create prometheus_alerts monitoring/prometheus/alerts.prod.yml" >&2
echo "" >&2
echo "2. Deploy the stack:" >&2
echo "   docker stack deploy -c docker/docker-compose.swarm.yml r3mes" >&2
echo "" >&2
echo "3. Check services:" >&2
echo "   docker service ls" >&2
echo "" >&2

# Create secure backup file with restricted permissions
if [ "$INTERACTIVE" = true ]; then
    log_warn "IMPORTANT: Saving credentials to secure backup file"
    
    BACKUP_FILE="/tmp/r3mes_secrets_backup_$(date +%Y%m%d_%H%M%S).txt"
    umask 0077  # Ensure file is created with 600 permissions
    
    cat > "$BACKUP_FILE" << EOF
# R3MES Secrets Backup - $(date)
# KEEP THIS FILE SECURE AND DELETE AFTER COPYING TO SECURE STORAGE

PostgreSQL User: $POSTGRES_USER
PostgreSQL Password: $POSTGRES_PASSWORD
Redis Password: $REDIS_PASSWORD
JWT Secret: $JWT_SECRET
API Secret Key: $API_SECRET_KEY
Grafana Admin Password: $GRAFANA_PASSWORD

# IMPORTANT: 
# 1. Copy these credentials to your secure password manager
# 2. Delete this file: rm "$BACKUP_FILE"
# 3. Verify deletion: shred -vfz -n 3 "$BACKUP_FILE" (if available)
EOF
    
    log_success "Credentials saved to: $BACKUP_FILE"
    log_warn "Remember to copy to secure storage and delete this file!"
    echo "" >&2
    echo "To securely delete the backup file:" >&2
    echo "  rm '$BACKUP_FILE'" >&2
    echo "  # Or for secure deletion:" >&2
    echo "  shred -vfz -n 3 '$BACKUP_FILE' 2>/dev/null || rm '$BACKUP_FILE'" >&2
fi

log_success "Swarm secrets creation completed!"

# Final security reminder
log_header "Security Reminders"
echo "1. ✅ Secrets are stored securely in Docker Swarm" >&2
echo "2. ⚠️  Backup file (if created) should be moved to secure storage" >&2
echo "3. 🔒 Consider rotating secrets periodically with --rotate flag" >&2
echo "4. 📝 Document secret rotation procedures for your team" >&2

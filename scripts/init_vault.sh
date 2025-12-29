#!/bin/bash
# Initialize HashiCorp Vault for R3MES
# Usage: bash scripts/init_vault.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VAULT_ADDR="${VAULT_ADDR:-http://localhost:8200}"

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

# Check if vault CLI is available
check_vault_cli() {
    if ! command -v vault &> /dev/null; then
        log_error "Vault CLI not found. Please install it first."
        log_info "Installation: https://developer.hashicorp.com/vault/downloads"
        exit 1
    fi
}

# Wait for Vault to be ready
wait_for_vault() {
    log_info "Waiting for Vault to be ready..."
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "$VAULT_ADDR/v1/sys/health" | grep -q '"initialized"'; then
            log_success "Vault is ready"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    
    log_error "Vault did not become ready in time"
    exit 1
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

log_header "HashiCorp Vault Initialization for R3MES"

check_vault_cli
wait_for_vault

export VAULT_ADDR

# Check if Vault is already initialized
INIT_STATUS=$(curl -s "$VAULT_ADDR/v1/sys/init" | grep -o '"initialized":[^,]*' | cut -d':' -f2)

if [ "$INIT_STATUS" = "true" ]; then
    log_warn "Vault is already initialized"
    
    # Check if sealed
    SEAL_STATUS=$(curl -s "$VAULT_ADDR/v1/sys/seal-status" | grep -o '"sealed":[^,]*' | cut -d':' -f2)
    
    if [ "$SEAL_STATUS" = "true" ]; then
        log_warn "Vault is sealed. Please unseal it first."
        log_info "Use: vault operator unseal"
        exit 1
    fi
    
    log_info "Vault is initialized and unsealed"
else
    log_header "Initializing Vault"
    
    # Initialize Vault
    INIT_OUTPUT=$(vault operator init -key-shares=5 -key-threshold=3 -format=json)
    
    # Save keys securely
    KEYS_FILE="$PROJECT_ROOT/docker/secrets/vault_keys.json"
    echo "$INIT_OUTPUT" > "$KEYS_FILE"
    chmod 600 "$KEYS_FILE"
    
    log_success "Vault initialized. Keys saved to: $KEYS_FILE"
    log_warn "IMPORTANT: Store these keys securely and delete the file!"
    
    # Extract root token and unseal keys
    ROOT_TOKEN=$(echo "$INIT_OUTPUT" | jq -r '.root_token')
    UNSEAL_KEY_1=$(echo "$INIT_OUTPUT" | jq -r '.unseal_keys_b64[0]')
    UNSEAL_KEY_2=$(echo "$INIT_OUTPUT" | jq -r '.unseal_keys_b64[1]')
    UNSEAL_KEY_3=$(echo "$INIT_OUTPUT" | jq -r '.unseal_keys_b64[2]')
    
    # Unseal Vault
    log_header "Unsealing Vault"
    
    vault operator unseal "$UNSEAL_KEY_1"
    vault operator unseal "$UNSEAL_KEY_2"
    vault operator unseal "$UNSEAL_KEY_3"
    
    log_success "Vault unsealed"
    
    # Login with root token
    export VAULT_TOKEN="$ROOT_TOKEN"
fi

# Check if we have a token
if [ -z "$VAULT_TOKEN" ]; then
    log_error "VAULT_TOKEN not set. Please set it or run initialization."
    exit 1
fi

log_header "Configuring Vault for R3MES"

# Enable KV secrets engine v2
log_info "Enabling KV secrets engine..."
vault secrets enable -path=secret -version=2 kv 2>/dev/null || log_warn "KV engine already enabled"

# Create R3MES secrets
log_header "Creating R3MES Secrets"

# Generate secrets
POSTGRES_PASSWORD=$(generate_password)
REDIS_PASSWORD=$(generate_password)
JWT_SECRET=$(generate_key)
API_SECRET_KEY=$(generate_key)
GRAFANA_PASSWORD=$(generate_password 24)

# Store PostgreSQL secrets
log_info "Storing PostgreSQL secrets..."
vault kv put secret/r3mes/postgres \
    user="r3mes" \
    password="$POSTGRES_PASSWORD" \
    host="postgres" \
    port="5432" \
    database="r3mes"

# Store Redis secrets
log_info "Storing Redis secrets..."
vault kv put secret/r3mes/redis \
    password="$REDIS_PASSWORD" \
    host="redis" \
    port="6379"

# Store application secrets
log_info "Storing application secrets..."
vault kv put secret/r3mes/app \
    jwt_secret="$JWT_SECRET" \
    api_secret_key="$API_SECRET_KEY"

# Store monitoring secrets
log_info "Storing monitoring secrets..."
vault kv put secret/r3mes/monitoring \
    grafana_admin_password="$GRAFANA_PASSWORD"

log_success "All secrets stored in Vault"

# Create policy for R3MES backend
log_header "Creating Vault Policy"

cat <<EOF | vault policy write r3mes-backend -
# R3MES Backend Policy
path "secret/data/r3mes/*" {
  capabilities = ["read", "list"]
}

path "secret/metadata/r3mes/*" {
  capabilities = ["read", "list"]
}
EOF

log_success "Policy 'r3mes-backend' created"

# Create AppRole for backend authentication
log_header "Creating AppRole Authentication"

# Enable AppRole auth method
vault auth enable approle 2>/dev/null || log_warn "AppRole already enabled"

# Create role for backend
vault write auth/approle/role/r3mes-backend \
    token_policies="r3mes-backend" \
    token_ttl=1h \
    token_max_ttl=4h \
    secret_id_ttl=24h \
    secret_id_num_uses=0

# Get role ID and secret ID
ROLE_ID=$(vault read -field=role_id auth/approle/role/r3mes-backend/role-id)
SECRET_ID=$(vault write -field=secret_id -f auth/approle/role/r3mes-backend/secret-id)

log_success "AppRole created for r3mes-backend"

# Summary
log_header "Vault Configuration Summary"

echo "Vault Address: $VAULT_ADDR"
echo ""
echo "Secrets stored at:"
echo "  - secret/r3mes/postgres"
echo "  - secret/r3mes/redis"
echo "  - secret/r3mes/app"
echo "  - secret/r3mes/monitoring"
echo ""
echo "AppRole credentials for backend:"
echo "  Role ID: $ROLE_ID"
echo "  Secret ID: $SECRET_ID"
echo ""

# Save AppRole credentials
APPROLE_FILE="$PROJECT_ROOT/docker/secrets/vault_approle.json"
cat > "$APPROLE_FILE" <<EOF
{
  "role_id": "$ROLE_ID",
  "secret_id": "$SECRET_ID"
}
EOF
chmod 600 "$APPROLE_FILE"

log_success "AppRole credentials saved to: $APPROLE_FILE"

log_header "Generated Credentials"

log_warn "IMPORTANT: Save these credentials securely!"
echo ""
echo "PostgreSQL Password: $POSTGRES_PASSWORD"
echo "Redis Password: $REDIS_PASSWORD"
echo "JWT Secret: $JWT_SECRET"
echo "API Secret Key: $API_SECRET_KEY"
echo "Grafana Admin Password: $GRAFANA_PASSWORD"
echo ""
echo "Root Token: $VAULT_TOKEN"

log_header "Next Steps"

echo "1. Set environment variables:"
echo "   export VAULT_ADDR=$VAULT_ADDR"
echo "   export VAULT_TOKEN=$VAULT_TOKEN"
echo ""
echo "2. Or use AppRole authentication in backend:"
echo "   export VAULT_ROLE_ID=$ROLE_ID"
echo "   export VAULT_SECRET_ID=$SECRET_ID"
echo ""
echo "3. Start services:"
echo "   docker-compose -f docker/docker-compose.vault.yml up -d"

log_success "Vault initialization completed!"

#!/bin/bash
# Create Kubernetes Secrets for R3MES
# Usage: bash scripts/create_k8s_secrets.sh [--namespace NAMESPACE] [--from-env]

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

# Default values
NAMESPACE="r3mes"
FROM_ENV=false
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --namespace|-n)
            NAMESPACE="$2"
            shift 2
            ;;
        --from-env)
            FROM_ENV=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --namespace, -n NAMESPACE  Kubernetes namespace (default: r3mes)"
            echo "  --from-env                 Read secrets from environment variables"
            echo "  --dry-run                  Print commands without executing"
            echo "  --help                     Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check kubectl
if ! command -v kubectl &> /dev/null; then
    log_error "kubectl not found. Please install kubectl first."
    exit 1
fi

# Generate secure random password
generate_password() {
    local length=${1:-32}
    openssl rand -base64 48 | tr -d "=+/" | cut -c1-$length
}

# Generate secure random key
generate_key() {
    openssl rand -hex 32
}

# Execute or print command
run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] $*"
    else
        "$@"
    fi
}

log_header "Kubernetes Secrets Creation for R3MES"

# Create namespace if not exists
log_info "Creating namespace: $NAMESPACE"
run_cmd kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | run_cmd kubectl apply -f -

# Read or generate secrets
if [ "$FROM_ENV" = true ]; then
    log_info "Reading secrets from environment variables..."
    POSTGRES_USER="${POSTGRES_USER:-r3mes}"
    POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-$(generate_password)}"
    REDIS_PASSWORD="${REDIS_PASSWORD:-$(generate_password)}"
    JWT_SECRET="${JWT_SECRET:-$(generate_key)}"
    API_SECRET_KEY="${API_SECRET_KEY:-$(generate_key)}"
    GRAFANA_ADMIN_PASSWORD="${GRAFANA_ADMIN_PASSWORD:-$(generate_password 24)}"
else
    log_info "Generating new secrets..."
    POSTGRES_USER="r3mes"
    POSTGRES_PASSWORD=$(generate_password)
    REDIS_PASSWORD=$(generate_password)
    JWT_SECRET=$(generate_key)
    API_SECRET_KEY=$(generate_key)
    GRAFANA_ADMIN_PASSWORD=$(generate_password 24)
fi

# Create database secrets
log_header "Creating Database Secrets"

run_cmd kubectl create secret generic r3mes-database-secrets \
    --namespace="$NAMESPACE" \
    --from-literal=postgres-user="$POSTGRES_USER" \
    --from-literal=postgres-password="$POSTGRES_PASSWORD" \
    --dry-run=client -o yaml | run_cmd kubectl apply -f -

log_success "Database secrets created"

# Create Redis secrets
log_header "Creating Redis Secrets"

run_cmd kubectl create secret generic r3mes-redis-secrets \
    --namespace="$NAMESPACE" \
    --from-literal=redis-password="$REDIS_PASSWORD" \
    --dry-run=client -o yaml | run_cmd kubectl apply -f -

log_success "Redis secrets created"

# Create application secrets
log_header "Creating Application Secrets"

run_cmd kubectl create secret generic r3mes-app-secrets \
    --namespace="$NAMESPACE" \
    --from-literal=jwt-secret="$JWT_SECRET" \
    --from-literal=api-secret-key="$API_SECRET_KEY" \
    --dry-run=client -o yaml | run_cmd kubectl apply -f -

log_success "Application secrets created"

# Create monitoring secrets
log_header "Creating Monitoring Secrets"

run_cmd kubectl create secret generic r3mes-monitoring-secrets \
    --namespace="$NAMESPACE" \
    --from-literal=grafana-admin-password="$GRAFANA_ADMIN_PASSWORD" \
    --dry-run=client -o yaml | run_cmd kubectl apply -f -

log_success "Monitoring secrets created"

# Create blockchain secrets (if key files exist)
log_header "Creating Blockchain Secrets"

VALIDATOR_KEY_FILE="$PROJECT_ROOT/docker/secrets/validator_key.json"
NODE_KEY_FILE="$PROJECT_ROOT/docker/secrets/node_key.json"

if [ -f "$VALIDATOR_KEY_FILE" ] && [ -f "$NODE_KEY_FILE" ]; then
    run_cmd kubectl create secret generic r3mes-blockchain-secrets \
        --namespace="$NAMESPACE" \
        --from-file=validator-key="$VALIDATOR_KEY_FILE" \
        --from-file=node-key="$NODE_KEY_FILE" \
        --dry-run=client -o yaml | run_cmd kubectl apply -f -
    log_success "Blockchain secrets created"
else
    log_warn "Blockchain key files not found, skipping..."
    log_info "Create them manually or generate during node initialization"
fi

# Create TLS secrets (if certificate files exist)
log_header "Creating TLS Secrets"

TLS_CERT_FILE="$PROJECT_ROOT/docker/secrets/ssl_certificate.pem"
TLS_KEY_FILE="$PROJECT_ROOT/docker/secrets/ssl_private_key.pem"

if [ -f "$TLS_CERT_FILE" ] && [ -f "$TLS_KEY_FILE" ]; then
    run_cmd kubectl create secret tls r3mes-tls-secrets \
        --namespace="$NAMESPACE" \
        --cert="$TLS_CERT_FILE" \
        --key="$TLS_KEY_FILE" \
        --dry-run=client -o yaml | run_cmd kubectl apply -f -
    log_success "TLS secrets created"
else
    log_warn "TLS certificate files not found, skipping..."
    log_info "Use cert-manager or provide your own certificates"
fi

# Summary
log_header "Secrets Summary"

echo "Namespace: $NAMESPACE"
echo ""
echo "Created secrets:"
kubectl get secrets -n "$NAMESPACE" --selector=app=r3mes 2>/dev/null || \
    kubectl get secrets -n "$NAMESPACE" 2>/dev/null | grep r3mes || \
    echo "  (run without --dry-run to see actual secrets)"

log_header "Generated Credentials"

if [ "$DRY_RUN" = false ]; then
    log_warn "IMPORTANT: Save these credentials securely!"
    echo ""
    echo "PostgreSQL User: $POSTGRES_USER"
    echo "PostgreSQL Password: $POSTGRES_PASSWORD"
    echo "Redis Password: $REDIS_PASSWORD"
    echo "JWT Secret: $JWT_SECRET"
    echo "API Secret Key: $API_SECRET_KEY"
    echo "Grafana Admin Password: $GRAFANA_ADMIN_PASSWORD"
fi

log_header "Next Steps"

echo "1. Verify secrets:"
echo "   kubectl get secrets -n $NAMESPACE"
echo ""
echo "2. Deploy R3MES:"
echo "   kubectl apply -f k8s/ -n $NAMESPACE"
echo ""
echo "3. Check pods:"
echo "   kubectl get pods -n $NAMESPACE"

log_success "Kubernetes secrets creation completed!"

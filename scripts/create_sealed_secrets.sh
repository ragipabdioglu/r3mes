#!/bin/bash
# Create Sealed Secrets for Kubernetes - Production Ready
# This script creates encrypted secrets that can be safely stored in Git
# Usage: bash scripts/create_sealed_secrets.sh [--namespace r3mes]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default values
NAMESPACE="r3mes"
OUTPUT_DIR="$PROJECT_ROOT/k8s/sealed-secrets"

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

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--namespace NAMESPACE] [--output-dir DIR]"
            echo "  --namespace    Kubernetes namespace (default: r3mes)"
            echo "  --output-dir   Output directory for sealed secrets (default: k8s/sealed-secrets)"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check prerequisites
check_prerequisites() {
    log_header "Checking Prerequisites"
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check kubeseal
    if ! command -v kubeseal &> /dev/null; then
        log_error "kubeseal is not installed"
        log_info "Install with: wget https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.24.0/kubeseal-0.24.0-linux-amd64.tar.gz"
        log_info "Or: brew install kubeseal"
        exit 1
    fi
    
    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_success "All prerequisites met"
}

# Install Sealed Secrets Controller
install_controller() {
    log_header "Installing Sealed Secrets Controller"
    
    # Check if controller is already installed
    if kubectl get deployment sealed-secrets-controller -n kube-system &> /dev/null; then
        log_warn "Sealed Secrets controller already installed"
        return 0
    fi
    
    # Install controller
    kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.24.0/controller.yaml
    
    # Wait for controller to be ready
    log_info "Waiting for controller to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/sealed-secrets-controller -n kube-system
    
    log_success "Sealed Secrets controller installed"
}

# Create namespace if it doesn't exist
create_namespace() {
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Creating namespace: $NAMESPACE"
        kubectl create namespace "$NAMESPACE"
    fi
}

# Generate secure password
generate_password() {
    local length=${1:-32}
    openssl rand -base64 $((length * 3 / 4)) | tr -d '\n' | cut -c1-$length
}

# Generate secure key
generate_key() {
    openssl rand -hex 32
}

# Create sealed secret from literal values
create_sealed_secret() {
    local name="$1"
    local namespace="$2"
    local output_file="$3"
    shift 3
    local -a data=("$@")
    
    log_info "Creating sealed secret: $name"
    
    # Build kubectl command
    local cmd="kubectl create secret generic $name --namespace=$namespace --dry-run=client -o yaml"
    
    # Add data arguments
    for item in "${data[@]}"; do
        cmd="$cmd --from-literal=$item"
    done
    
    # Create sealed secret
    eval "$cmd" | kubeseal --format=yaml --cert="$PROJECT_ROOT/k8s/sealed-secrets-cert.pem" > "$output_file"
    
    log_success "Sealed secret created: $output_file"
}

# Fetch sealed secrets certificate
fetch_certificate() {
    log_header "Fetching Sealed Secrets Certificate"
    
    local cert_file="$PROJECT_ROOT/k8s/sealed-secrets-cert.pem"
    
    # Fetch certificate from cluster
    kubeseal --fetch-cert > "$cert_file"
    
    log_success "Certificate saved to: $cert_file"
    log_info "This certificate can be safely committed to Git"
}

# Create output directory
mkdir -p "$OUTPUT_DIR"

log_header "R3MES Sealed Secrets Creation"

# Run checks and setup
check_prerequisites
install_controller
create_namespace
fetch_certificate

# Generate secrets
log_header "Generating Secrets"

# Database secrets
POSTGRES_USER="r3mes"
POSTGRES_PASSWORD=$(generate_password)
POSTGRES_DATABASE="r3mes"

create_sealed_secret "database-secrets" "$NAMESPACE" "$OUTPUT_DIR/database-secrets.yaml" \
    "postgres-user=$POSTGRES_USER" \
    "postgres-password=$POSTGRES_PASSWORD" \
    "postgres-database=$POSTGRES_DATABASE"

# Application secrets
JWT_SECRET=$(generate_key)
API_SECRET=$(generate_key)
REDIS_PASSWORD=$(generate_password)

create_sealed_secret "r3mes-secrets" "$NAMESPACE" "$OUTPUT_DIR/r3mes-secrets.yaml" \
    "jwt-secret=$JWT_SECRET" \
    "api-key-secret=$API_SECRET" \
    "redis-password=$REDIS_PASSWORD"

# Blockchain secrets
CHAIN_ID="remes-mainnet-1"
NODE_KEY=$(generate_key)

create_sealed_secret "blockchain-secrets" "$NAMESPACE" "$OUTPUT_DIR/blockchain-secrets.yaml" \
    "chain-id=$CHAIN_ID" \
    "node-key=$NODE_KEY"

# Monitoring secrets
GRAFANA_PASSWORD=$(generate_password 24)

create_sealed_secret "monitoring-secrets" "$NAMESPACE" "$OUTPUT_DIR/monitoring-secrets.yaml" \
    "grafana-admin-password=$GRAFANA_PASSWORD"

# Create kustomization file
log_header "Creating Kustomization"

cat > "$OUTPUT_DIR/kustomization.yaml" << EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: $NAMESPACE

resources:
- database-secrets.yaml
- r3mes-secrets.yaml
- blockchain-secrets.yaml
- monitoring-secrets.yaml

commonLabels:
  app.kubernetes.io/name: r3mes
  app.kubernetes.io/managed-by: sealed-secrets
EOF

log_success "Kustomization created: $OUTPUT_DIR/kustomization.yaml"

# Create deployment script
log_header "Creating Deployment Script"

cat > "$OUTPUT_DIR/deploy.sh" << 'EOF'
#!/bin/bash
# Deploy Sealed Secrets to Kubernetes

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Deploying sealed secrets..."
kubectl apply -k "$SCRIPT_DIR"

echo "Verifying secrets..."
kubectl get secrets -n r3mes

echo "✅ Sealed secrets deployed successfully!"
EOF

chmod +x "$OUTPUT_DIR/deploy.sh"

log_success "Deployment script created: $OUTPUT_DIR/deploy.sh"

# Summary
log_header "Summary"

echo "Generated sealed secrets:" >&2
ls -la "$OUTPUT_DIR"/*.yaml

log_header "Next Steps"

echo "1. Review generated sealed secrets:" >&2
echo "   ls $OUTPUT_DIR/" >&2
echo "" >&2
echo "2. Commit sealed secrets to Git (they are encrypted):" >&2
echo "   git add $OUTPUT_DIR/" >&2
echo "   git commit -m 'Add sealed secrets for production'" >&2
echo "" >&2
echo "3. Deploy to cluster:" >&2
echo "   $OUTPUT_DIR/deploy.sh" >&2
echo "" >&2
echo "4. Verify deployment:" >&2
echo "   kubectl get secrets -n $NAMESPACE" >&2
echo "" >&2

# Create secure backup
log_header "Creating Secure Backup"

BACKUP_FILE="/tmp/r3mes_sealed_secrets_backup_$(date +%Y%m%d_%H%M%S).txt"
umask 0077

cat > "$BACKUP_FILE" << EOF
# R3MES Sealed Secrets Backup - $(date)
# KEEP THIS FILE SECURE - Contains unencrypted secrets

PostgreSQL User: $POSTGRES_USER
PostgreSQL Password: $POSTGRES_PASSWORD
PostgreSQL Database: $POSTGRES_DATABASE
JWT Secret: $JWT_SECRET
API Secret: $API_SECRET
Redis Password: $REDIS_PASSWORD
Chain ID: $CHAIN_ID
Node Key: $NODE_KEY
Grafana Password: $GRAFANA_PASSWORD

# IMPORTANT:
# 1. Copy these to your secure password manager
# 2. Delete this file: rm "$BACKUP_FILE"
# 3. Sealed secrets are encrypted and safe to commit to Git
EOF

log_success "Backup created: $BACKUP_FILE"
log_warn "Remember to copy to secure storage and delete backup file!"

log_success "Sealed secrets creation completed!"

# Security reminders
log_header "Security Reminders"
echo "✅ Sealed secrets are encrypted and safe to commit to Git" >&2
echo "🔐 Only the cluster with the private key can decrypt them" >&2
echo "📝 Backup file contains unencrypted secrets - handle securely" >&2
echo "🔄 Rotate secrets periodically by regenerating sealed secrets" >&2
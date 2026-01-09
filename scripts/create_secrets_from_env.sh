#!/bin/bash
# Create Docker Secrets from .env.production file
# This script reads .env.production and creates secret files

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKER_DIR="$PROJECT_ROOT/docker"
SECRETS_DIR="$DOCKER_DIR/secrets"
ENV_FILE="$DOCKER_DIR/.env.production"

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

# Check if .env.production exists
if [ ! -f "$ENV_FILE" ]; then
    print_error ".env.production file not found at $ENV_FILE"
    print_warning "Please create it first: cp docker/env.production.example docker/.env.production"
    exit 1
fi

# Source the .env.production file
set -a
source "$ENV_FILE"
set +a

# Create secrets directory
mkdir -p "$SECRETS_DIR"

print_header "Creating Docker Secrets from .env.production"

# PostgreSQL Password
if [ -z "$POSTGRES_PASSWORD" ] || [ "$POSTGRES_PASSWORD" = "CHANGE_ME_STRONG_PASSWORD" ]; then
    print_error "POSTGRES_PASSWORD not set or still has default value in .env.production"
    exit 1
fi

echo -n "$POSTGRES_PASSWORD" > "$SECRETS_DIR/postgres_password.txt"
chmod 600 "$SECRETS_DIR/postgres_password.txt"
print_success "PostgreSQL password secret created"

# Redis Password
if [ -z "$REDIS_PASSWORD" ] || [ "$REDIS_PASSWORD" = "CHANGE_ME_REDIS_PASSWORD" ]; then
    print_error "REDIS_PASSWORD not set or still has default value in .env.production"
    exit 1
fi

echo -n "$REDIS_PASSWORD" > "$SECRETS_DIR/redis_password.txt"
chmod 600 "$SECRETS_DIR/redis_password.txt"
print_success "Redis password secret created"

# Grafana Admin Password
if [ -z "$GRAFANA_ADMIN_PASSWORD" ]; then
    print_warning "GRAFANA_ADMIN_PASSWORD not set, generating random password"
    GRAFANA_ADMIN_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)
fi

echo -n "$GRAFANA_ADMIN_PASSWORD" > "$SECRETS_DIR/grafana_admin_password.txt"
chmod 600 "$SECRETS_DIR/grafana_admin_password.txt"
print_success "Grafana admin password secret created"

print_header "Secrets Created Successfully"

echo "Secrets directory: $SECRETS_DIR"
echo ""
echo "Created secrets:"
ls -lh "$SECRETS_DIR" | grep -v "^total"
echo ""
print_warning "IMPORTANT: Keep these files secure and never commit them to git!"
print_success "You can now start the services with: docker compose -f docker-compose.prod.yml up -d"


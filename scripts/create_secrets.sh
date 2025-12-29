#!/bin/bash
# Create Docker Secrets from user input
# This script creates secret files for Docker Compose

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

# Create secrets directory
mkdir -p "$SECRETS_DIR"

print_header "Docker Secrets Creation"

# Function to generate random password
generate_password() {
    openssl rand -base64 32 | tr -d "=+/" | cut -c1-32
}

# Function to read password securely
read_password() {
    local prompt=$1
    local default=$2
    local password
    
    if [ -n "$default" ]; then
        read -sp "$prompt (press Enter to generate random): " password
        echo
        if [ -z "$password" ]; then
            password=$(generate_password)
            echo "Generated random password"
        fi
    else
        read -sp "$prompt: " password
        echo
        if [ -z "$password" ]; then
            print_error "Password cannot be empty"
            exit 1
        fi
    fi
    
    echo "$password"
}

# PostgreSQL Password
if [ -f "$SECRETS_DIR/postgres_password.txt" ]; then
    print_warning "postgres_password.txt already exists"
    read -p "Overwrite? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        POSTGRES_PASSWORD=$(read_password "Enter PostgreSQL password" "generate")
        echo -n "$POSTGRES_PASSWORD" > "$SECRETS_DIR/postgres_password.txt"
        chmod 600 "$SECRETS_DIR/postgres_password.txt"
        print_success "PostgreSQL password secret created"
    else
        print_warning "Skipping PostgreSQL password"
    fi
else
    POSTGRES_PASSWORD=$(read_password "Enter PostgreSQL password" "generate")
    echo -n "$POSTGRES_PASSWORD" > "$SECRETS_DIR/postgres_password.txt"
    chmod 600 "$SECRETS_DIR/postgres_password.txt"
    print_success "PostgreSQL password secret created"
fi

# Redis Password
if [ -f "$SECRETS_DIR/redis_password.txt" ]; then
    print_warning "redis_password.txt already exists"
    read -p "Overwrite? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        REDIS_PASSWORD=$(read_password "Enter Redis password" "generate")
        echo -n "$REDIS_PASSWORD" > "$SECRETS_DIR/redis_password.txt"
        chmod 600 "$SECRETS_DIR/redis_password.txt"
        print_success "Redis password secret created"
    else
        print_warning "Skipping Redis password"
    fi
else
    REDIS_PASSWORD=$(read_password "Enter Redis password" "generate")
    echo -n "$REDIS_PASSWORD" > "$SECRETS_DIR/redis_password.txt"
    chmod 600 "$SECRETS_DIR/redis_password.txt"
    print_success "Redis password secret created"
fi

# Grafana Admin Password
if [ -f "$SECRETS_DIR/grafana_admin_password.txt" ]; then
    print_warning "grafana_admin_password.txt already exists"
    read -p "Overwrite? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        GRAFANA_PASSWORD=$(read_password "Enter Grafana admin password" "generate")
        echo -n "$GRAFANA_PASSWORD" > "$SECRETS_DIR/grafana_admin_password.txt"
        chmod 600 "$SECRETS_DIR/grafana_admin_password.txt"
        print_success "Grafana admin password secret created"
    else
        print_warning "Skipping Grafana admin password"
    fi
else
    GRAFANA_PASSWORD=$(read_password "Enter Grafana admin password" "generate")
    echo -n "$GRAFANA_PASSWORD" > "$SECRETS_DIR/grafana_admin_password.txt"
    chmod 600 "$SECRETS_DIR/grafana_admin_password.txt"
    print_success "Grafana admin password secret created"
fi

print_header "Secrets Created Successfully"

echo "Secrets directory: $SECRETS_DIR"
echo ""
echo "Created secrets:"
ls -lh "$SECRETS_DIR" | grep -v "^total"
echo ""
print_warning "IMPORTANT: Keep these files secure and never commit them to git!"
print_success "You can now start the services with: docker compose -f docker-compose.prod.yml up -d"


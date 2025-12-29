#!/bin/bash
# R3MES Production Docker Deployment Script
# Deploys the entire R3MES stack using Docker Compose

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKER_DIR="$PROJECT_ROOT/docker"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Docker and Docker Compose are installed"
}

# Check if secrets exist
check_secrets() {
    SECRETS_DIR="$DOCKER_DIR/secrets"
    
    if [ ! -d "$SECRETS_DIR" ]; then
        print_warning "Secrets directory not found"
        mkdir -p "$SECRETS_DIR"
    fi
    
    # Check if secrets exist
    if [ ! -f "$SECRETS_DIR/postgres_password.txt" ] || \
       [ ! -f "$SECRETS_DIR/redis_password.txt" ] || \
       [ ! -f "$SECRETS_DIR/grafana_admin_password.txt" ]; then
        print_warning "Docker secrets not found"
        
        # Check if .env.production exists
        ENV_FILE="$DOCKER_DIR/.env.production"
        if [ -f "$ENV_FILE" ]; then
            print_warning "Found .env.production, creating secrets from it..."
            bash "$PROJECT_ROOT/scripts/create_secrets_from_env.sh"
        else
            print_warning "Creating secrets interactively..."
            bash "$PROJECT_ROOT/scripts/create_secrets.sh"
        fi
    fi
    
    print_success "Docker secrets validated"
}

# Check if .env.production exists (for non-secret variables)
check_env_file() {
    ENV_FILE="$DOCKER_DIR/.env.production"
    
    if [ ! -f "$ENV_FILE" ]; then
        print_warning ".env.production file not found"
        if [ -f "$DOCKER_DIR/env.production.example" ]; then
            print_warning "Creating .env.production from example..."
            cp "$DOCKER_DIR/env.production.example" "$ENV_FILE"
            print_warning "Please edit $ENV_FILE and set all required values before continuing"
            print_warning "Especially: DOMAIN, EMAIL, CORS_ALLOWED_ORIGINS"
            read -p "Press Enter after you've edited .env.production..."
        else
            print_error "env.production.example not found. Cannot create .env.production"
            exit 1
        fi
    fi
    
    # Validate critical variables (non-secret)
    source "$ENV_FILE"
    if [ -z "$DOMAIN" ] || [ "$DOMAIN" = "r3mes.network" ]; then
        print_warning "DOMAIN is set to default. Make sure this is correct for your deployment."
    fi
    
    print_success ".env.production file validated"
}

# Check for GPU support (optional)
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected"
        if docker info | grep -q "nvidia"; then
            print_success "NVIDIA Container Toolkit is configured"
            USE_GPU=true
        else
            print_warning "NVIDIA GPU detected but nvidia-container-toolkit not configured"
            print_warning "Miner service will not be available"
            USE_GPU=false
        fi
    else
        print_warning "No NVIDIA GPU detected. Miner service will not be available."
        USE_GPU=false
    fi
}

# Create Docker network
create_network() {
    if ! docker network ls | grep -q "r3mes-network"; then
        print_warning "Creating Docker network..."
        docker network create r3mes-network || true
        print_success "Docker network created"
    else
        print_success "Docker network already exists"
    fi
}

# Initialize SSL certificates
init_ssl() {
    source "$DOCKER_DIR/.env.production"
    
    if [ -z "$DOMAIN" ] || [ -z "$EMAIL" ]; then
        print_warning "DOMAIN or EMAIL not set. SSL certificates will need to be obtained manually."
        return
    fi
    
    # Check if certificates already exist
    if docker volume ls | grep -q "nginx_certs"; then
        print_success "SSL certificates volume exists"
        return
    fi
    
    print_warning "SSL certificates not found. They will be obtained on first nginx start."
    print_warning "Make sure port 80 is accessible for Let's Encrypt validation."
}

# Build images
build_images() {
    print_header "Building Docker Images"
    cd "$DOCKER_DIR"
    
    docker-compose -f docker-compose.prod.yml build --no-cache
    
    print_success "Docker images built"
}

# Start services
start_services() {
    print_header "Starting Services"
    cd "$DOCKER_DIR"
    
    # Determine if GPU mining should be enabled
    PROFILE_ARGS=""
    if [ "$USE_GPU" = true ]; then
        read -p "Enable GPU mining? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            PROFILE_ARGS="--profile miner"
            print_success "GPU mining will be enabled"
        fi
    fi
    
    # Start services
    docker-compose -f docker-compose.prod.yml $PROFILE_ARGS up -d
    
    print_success "Services started"
}

# Wait for services to be healthy
wait_for_health() {
    print_header "Waiting for Services to be Healthy"
    
    SERVICES=("postgres" "redis" "ipfs" "remesd" "backend" "frontend" "nginx")
    MAX_WAIT=300  # 5 minutes
    ELAPSED=0
    
    for service in "${SERVICES[@]}"; do
        print_warning "Waiting for $service to be healthy..."
        
        while [ $ELAPSED -lt $MAX_WAIT ]; do
            if docker-compose -f "$DOCKER_DIR/docker-compose.prod.yml" ps | grep -q "$service.*healthy"; then
                print_success "$service is healthy"
                break
            fi
            
            sleep 5
            ELAPSED=$((ELAPSED + 5))
        done
        
        if [ $ELAPSED -ge $MAX_WAIT ]; then
            print_error "$service did not become healthy within $MAX_WAIT seconds"
            print_warning "Check logs: docker-compose -f $DOCKER_DIR/docker-compose.prod.yml logs $service"
        fi
        
        ELAPSED=0
    done
}

# Obtain SSL certificate (if needed)
obtain_ssl() {
    source "$DOCKER_DIR/.env.production"
    
    if [ -z "$DOMAIN" ] || [ -z "$EMAIL" ]; then
        print_warning "Skipping SSL certificate setup (DOMAIN or EMAIL not set)"
        return
    fi
    
    print_header "Obtaining SSL Certificate"
    
    # Check if certificate already exists
    if docker exec r3mes-nginx-prod test -f "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" 2>/dev/null; then
        print_success "SSL certificate already exists"
        return
    fi
    
    print_warning "Obtaining SSL certificate from Let's Encrypt..."
    print_warning "Make sure port 80 is accessible and DNS points to this server"
    
    docker exec r3mes-nginx-prod certbot certonly \
        --webroot \
        --webroot-path=/var/www/certbot \
        --email "$EMAIL" \
        --agree-tos \
        --no-eff-email \
        -d "$DOMAIN" \
        -d "www.$DOMAIN" || {
        print_error "Failed to obtain SSL certificate"
        print_warning "You may need to manually configure SSL certificates"
    }
    
    # Reload nginx
    docker exec r3mes-nginx-prod nginx -s reload
    
    print_success "SSL certificate obtained"
}

# Show status
show_status() {
    print_header "Service Status"
    cd "$DOCKER_DIR"
    docker-compose -f docker-compose.prod.yml ps
}

# Show logs
show_logs() {
    print_header "Recent Logs"
    cd "$DOCKER_DIR"
    docker-compose -f docker-compose.prod.yml logs --tail=50
}

# Main execution
main() {
    print_header "R3MES Production Docker Deployment"
    
    check_docker
    check_secrets
    check_env_file
    check_gpu
    create_network
    init_ssl
    build_images
    start_services
    wait_for_health
    
    # Give services a moment to fully start
    sleep 10
    
    # Try to obtain SSL certificate
    obtain_ssl
    
    show_status
    
    print_header "Deployment Complete"
    print_success "R3MES production stack is running"
    print_warning "View logs: cd $DOCKER_DIR && docker-compose -f docker-compose.prod.yml logs -f"
    print_warning "Stop services: cd $DOCKER_DIR && docker-compose -f docker-compose.prod.yml down"
    print_warning "Restart services: cd $DOCKER_DIR && docker-compose -f docker-compose.prod.yml restart"
}

# Run main function
main "$@"


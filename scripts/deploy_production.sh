#!/bin/bash
# Production Deployment Script for R3MES
#
# This script automates the deployment of R3MES to production.
# It includes:
# - Pre-deployment checks
# - Database migration
# - Docker build and push
# - Blue-green deployment
# - Health checks
# - Rollback on failure

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
DEPLOYMENT_HOST="${DEPLOYMENT_HOST:-localhost}"
DEPLOYMENT_USER="${DEPLOYMENT_USER:-deploy}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-docker.io}"
DOCKER_IMAGE_PREFIX="${DOCKER_IMAGE_PREFIX:-r3mes}"
VERSION="${VERSION:-$(git describe --tags --always)}"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check environment variables
    if [ -z "$POSTGRES_URL" ]; then
        log_error "POSTGRES_URL is not set"
        exit 1
    fi
    
    if [ -z "$SENTRY_DSN" ]; then
        log_warn "SENTRY_DSN is not set (optional)"
    fi
    
    log_info "Requirements check passed"
}

build_images() {
    log_info "Building Docker images..."
    
    docker build -t ${DOCKER_REGISTRY}/${DOCKER_IMAGE_PREFIX}/backend:${VERSION} -f docker/backend.Dockerfile .
    docker build -t ${DOCKER_REGISTRY}/${DOCKER_IMAGE_PREFIX}/frontend:${VERSION} -f docker/frontend.Dockerfile .
    docker build -t ${DOCKER_REGISTRY}/${DOCKER_IMAGE_PREFIX}/blockchain:${VERSION} -f docker/blockchain.Dockerfile .
    docker build -t ${DOCKER_REGISTRY}/${DOCKER_IMAGE_PREFIX}/miner:${VERSION} -f docker/miner.Dockerfile .
    
    log_info "Docker images built successfully"
}

push_images() {
    log_info "Pushing Docker images to registry..."
    
    docker push ${DOCKER_REGISTRY}/${DOCKER_IMAGE_PREFIX}/backend:${VERSION}
    docker push ${DOCKER_REGISTRY}/${DOCKER_IMAGE_PREFIX}/frontend:${VERSION}
    docker push ${DOCKER_REGISTRY}/${DOCKER_IMAGE_PREFIX}/blockchain:${VERSION}
    docker push ${DOCKER_REGISTRY}/${DOCKER_IMAGE_PREFIX}/miner:${VERSION}
    
    log_info "Docker images pushed successfully"
}

run_migrations() {
    log_info "Running database migrations..."
    
    # Run PostgreSQL migration if needed
    if [ -f "scripts/migrate_sqlite_to_postgres.py" ]; then
        python3 scripts/migrate_sqlite_to_postgres.py
    fi
    
    log_info "Database migrations completed"
}

switch_load_balancer() {
    local new_color=$1
    local old_color=$2
    
    log_info "Switching load balancer from ${old_color} to ${new_color}..."
    
    # Nginx upstream configuration update
    if [ -f "/etc/nginx/sites-available/r3mes-backend" ]; then
        # Backup current config
        BACKUP_FILE="/etc/nginx/sites-available/r3mes-backend.backup.$(date +%s)"
        cp /etc/nginx/sites-available/r3mes-backend "$BACKUP_FILE"
        log_info "Backed up Nginx config to ${BACKUP_FILE}"
        
        # Update upstream to point to new color
        sed -i "s/backend-${old_color}/backend-${new_color}/g" /etc/nginx/sites-available/r3mes-backend
        
        # Test configuration
        if nginx -t > /dev/null 2>&1; then
            # Reload Nginx
            systemctl reload nginx || service nginx reload
            log_info "Nginx configuration updated and reloaded"
        else
            log_error "Nginx configuration test failed, restoring backup"
            cp "$BACKUP_FILE" /etc/nginx/sites-available/r3mes-backend
            return 1
        fi
    elif [ -f "nginx/nginx-backend.conf" ]; then
        # Update local config file
        sed -i "s/backend-${old_color}/backend-${new_color}/g" nginx/nginx-backend.conf
        log_info "Updated local Nginx config (manual reload required)"
    else
        log_warn "Nginx configuration not found, skipping load balancer update"
    fi
    
    # Docker Compose service name update (if using docker-compose for load balancing)
    # This is handled by docker-compose service names, no manual update needed
    
    return 0
}

deploy_blue_green() {
    log_info "Starting blue-green deployment..."
    
    # Determine current color (blue or green)
    CURRENT_COLOR=$(docker-compose -f docker/docker-compose.prod.yml ps 2>/dev/null | grep -o "blue\|green" | head -1 || echo "blue")
    
    if [ "$CURRENT_COLOR" == "blue" ]; then
        NEW_COLOR="green"
    else
        NEW_COLOR="blue"
    fi
    
    log_info "Current deployment: ${CURRENT_COLOR}, New deployment: ${NEW_COLOR}"
    
    # Store colors for rollback
    export CURRENT_COLOR
    export NEW_COLOR
    
    # Deploy new color
    export DEPLOYMENT_COLOR=${NEW_COLOR}
    docker-compose -f docker/docker-compose.prod.yml up -d --scale backend-${NEW_COLOR}=1 --scale frontend-${NEW_COLOR}=1
    
    # Wait for health check
    log_info "Waiting for health check..."
    sleep 30
    
    # Run smoke tests
    if [ -f "scripts/smoke-tests.sh" ]; then
        if ! bash scripts/smoke-tests.sh; then
            log_error "Smoke tests failed for new deployment"
            return 1
        fi
    fi
    
    # Switch traffic to new color
    if ! switch_load_balancer "$NEW_COLOR" "$CURRENT_COLOR"; then
        log_error "Failed to switch load balancer"
        return 1
    fi
    
    # Wait a bit for traffic to switch
    sleep 10
    
    # Stop old color
    log_info "Stopping old deployment (${CURRENT_COLOR})..."
    docker-compose -f docker/docker-compose.prod.yml stop backend-${CURRENT_COLOR} frontend-${CURRENT_COLOR} 2>/dev/null || true
    
    log_info "Blue-green deployment completed"
    return 0
}

health_check() {
    log_info "Running health checks..."
    
    # Check backend
    if ! curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_error "Backend health check failed"
        return 1
    fi
    
    # Check frontend
    if ! curl -f http://localhost:3000 > /dev/null 2>&1; then
        log_error "Frontend health check failed"
        return 1
    fi
    
    log_info "Health checks passed"
    return 0
}

rollback() {
    log_error "Deployment failed, rolling back..."
    
    # Switch back to previous color
    if [ -n "$CURRENT_COLOR" ] && [ -n "$NEW_COLOR" ]; then
        log_info "Rolling back from ${NEW_COLOR} to ${CURRENT_COLOR}..."
        
        # Switch load balancer back
        if switch_load_balancer "$CURRENT_COLOR" "$NEW_COLOR"; then
            log_info "Load balancer switched back to ${CURRENT_COLOR}"
        else
            log_error "Failed to switch load balancer back"
        fi
        
        # Stop failed deployment
        log_info "Stopping failed deployment (${NEW_COLOR})..."
        docker-compose -f docker/docker-compose.prod.yml stop backend-${NEW_COLOR} frontend-${NEW_COLOR} 2>/dev/null || true
        
        # Restore previous version
        log_info "Restoring previous deployment (${CURRENT_COLOR})..."
        docker-compose -f docker/docker-compose.prod.yml start backend-${CURRENT_COLOR} frontend-${CURRENT_COLOR} 2>/dev/null || true
        
        # Verify rollback
        sleep 10
        if health_check; then
            log_info "Rollback successful - previous version is running"
        else
            log_error "Rollback completed but health check failed - manual intervention required"
        fi
    else
        log_error "Cannot rollback - color information not available"
    fi
    
    log_warn "Rollback completed"
}

# Main deployment flow
main() {
    log_info "Starting production deployment (version: ${VERSION})"
    
    # Pre-deployment checks
    check_requirements
    
    # Build and push images
    build_images
    push_images
    
    # Run migrations
    run_migrations
    
    # Deploy
    if deploy_blue_green; then
        # Health check
        if health_check; then
            log_info "Deployment successful!"
        else
            rollback
            exit 1
        fi
    else
        rollback
        exit 1
    fi
}

# Run main function
main "$@"


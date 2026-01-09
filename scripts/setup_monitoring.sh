#!/bin/bash
# Setup Production Monitoring Stack
#
# Deploys Prometheus, Grafana, and Alertmanager for production monitoring.
#
# Usage:
#   ./scripts/setup_monitoring.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    log_error "Docker is not running"
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    log_error "docker-compose is not installed"
    exit 1
fi

log_info "Setting up R3MES production monitoring stack..."

# Check required environment variables
if [ -z "$GRAFANA_ADMIN_PASSWORD" ]; then
    log_error "GRAFANA_ADMIN_PASSWORD environment variable must be set"
    log_error "Example: export GRAFANA_ADMIN_PASSWORD='your-secure-password'"
    exit 1
fi

# Validate password strength (minimum 8 characters)
if [ ${#GRAFANA_ADMIN_PASSWORD} -lt 8 ]; then
    log_error "GRAFANA_ADMIN_PASSWORD must be at least 8 characters long"
    exit 1
fi

# Create network if it doesn't exist
log_info "Creating Docker network..."
docker network create r3mes-network 2>/dev/null || log_info "Network already exists"

# Deploy monitoring stack
log_info "Deploying monitoring stack..."
cd "$PROJECT_ROOT/docker"
docker-compose -f docker-compose.monitoring.yml -f docker-compose.monitoring.prod.yml up -d

# Wait for services to be ready
log_info "Waiting for services to be ready..."
sleep 10

# Check Prometheus
if curl -f http://localhost:9090/-/healthy > /dev/null 2>&1; then
    log_info "✅ Prometheus is running"
else
    log_error "❌ Prometheus health check failed"
    exit 1
fi

# Check Grafana
if curl -f http://localhost:3001/api/health > /dev/null 2>&1; then
    log_info "✅ Grafana is running"
    log_info "Grafana URL: http://localhost:3001"
    log_info "Default credentials: admin / ${GRAFANA_ADMIN_PASSWORD}"
else
    log_error "❌ Grafana health check failed"
    exit 1
fi

# Check Alertmanager
if curl -f http://localhost:9093/-/healthy > /dev/null 2>&1; then
    log_info "✅ Alertmanager is running"
else
    log_error "❌ Alertmanager health check failed"
    exit 1
fi

log_info ""
log_info "✅ Monitoring stack deployed successfully!"
log_info ""
log_info "Next steps:"
log_info "1. Access Grafana: http://localhost:3001"
log_info "2. Configure alert notification channels in Alertmanager"
log_info "3. Import dashboards from monitoring/grafana/dashboards/"
log_info "4. Verify Prometheus is scraping metrics from all targets"


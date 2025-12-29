#!/bin/bash
# Security Scanning Script
#
# Runs comprehensive security scans for R3MES project.
#
# Usage:
#   ./scripts/security_scan.sh [python|nodejs|docker|all]

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

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

SCAN_TYPE="${1:-all}"

log_info "Running security scans for R3MES..."

# Python security scan
if [ "$SCAN_TYPE" == "python" ] || [ "$SCAN_TYPE" == "all" ]; then
    log_info "Scanning Python code..."
    
    # Install tools if needed
    if ! command -v bandit &> /dev/null; then
        log_info "Installing Bandit..."
        pip install bandit
    fi
    
    if ! command -v safety &> /dev/null; then
        log_info "Installing Safety..."
        pip install safety
    fi
    
    # Run Bandit
    log_info "Running Bandit..."
    cd "$PROJECT_ROOT/backend"
    bandit -r app/ -f json -o bandit-report.json || true
    bandit -r app/ -f txt
    
    # Run Safety
    log_info "Running Safety..."
    safety check --json --output safety-report.json || true
    safety check
fi

# Node.js security scan
if [ "$SCAN_TYPE" == "nodejs" ] || [ "$SCAN_TYPE" == "all" ]; then
    log_info "Scanning Node.js dependencies..."
    
    cd "$PROJECT_ROOT/web-dashboard"
    
    if [ -f "package-lock.json" ]; then
        npm audit --audit-level=moderate || true
        npm audit --json > npm-audit-report.json || true
    else
        log_warn "package-lock.json not found, skipping npm audit"
    fi
fi

# Docker security scan
if [ "$SCAN_TYPE" == "docker" ] || [ "$SCAN_TYPE" == "all" ]; then
    log_info "Scanning Docker images..."
    
    if command -v trivy &> /dev/null; then
        # Scan backend image if exists
        if docker images | grep -q r3mes-backend; then
            trivy image r3mes-backend:latest
        else
            log_warn "r3mes-backend image not found, skipping scan"
        fi
        
        # Scan frontend image if exists
        if docker images | grep -q r3mes-frontend; then
            trivy image r3mes-frontend:latest
        else
            log_warn "r3mes-frontend image not found, skipping scan"
        fi
    else
        log_warn "Trivy not installed, skipping Docker scan"
        log_info "Install Trivy: https://github.com/aquasecurity/trivy"
    fi
fi

log_info ""
log_info "✅ Security scans completed"
log_info "Review reports in:"
log_info "  - backend/bandit-report.json"
log_info "  - backend/safety-report.json"
log_info "  - web-dashboard/npm-audit-report.json"


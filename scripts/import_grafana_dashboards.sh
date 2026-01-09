#!/bin/bash
# Grafana Dashboard Import Automation Script
#
# Imports Grafana dashboards from JSON files using Grafana API.
# Supports dashboard creation and updates.
#
# Usage:
#   ./scripts/import_grafana_dashboards.sh [grafana_url] [admin_user] [admin_password]
#
# Environment Variables:
#   GRAFANA_URL: Grafana URL (default: http://localhost:3000)
#   GRAFANA_USER: Grafana admin username (default: admin)
#   GRAFANA_PASSWORD: Grafana admin password (default: admin)
#   DASHBOARD_DIR: Directory containing dashboard JSON files (default: monitoring/grafana/dashboards)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
GRAFANA_URL="${1:-${GRAFANA_URL:-http://localhost:3000}}"
GRAFANA_USER="${2:-${GRAFANA_USER:-admin}}"
DASHBOARD_DIR="${DASHBOARD_DIR:-monitoring/grafana/dashboards}"

# Validate GRAFANA_PASSWORD is set (no default for security)
if [ -z "$3" ] && [ -z "$GRAFANA_PASSWORD" ]; then
    log_error "GRAFANA_PASSWORD must be provided as argument or environment variable"
    log_error "Usage: $0 [grafana_url] [admin_user] [admin_password]"
    log_error "Or set: export GRAFANA_PASSWORD='your-password'"
    exit 1
fi
GRAFANA_PASSWORD="${3:-${GRAFANA_PASSWORD}}"

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    log_error "jq is not installed. Please install jq to use this script."
    exit 1
fi

# Check if curl is installed
if ! command -v curl &> /dev/null; then
    log_error "curl is not installed"
    exit 1
fi

# Get Grafana API key or create session
get_grafana_auth() {
    # Try to authenticate and get session cookie
    AUTH_RESPONSE=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "{\"user\":\"$GRAFANA_USER\",\"password\":\"$GRAFANA_PASSWORD\"}" \
        "$GRAFANA_URL/api/login" 2>/dev/null)
    
    if echo "$AUTH_RESPONSE" | grep -q "Logged in"; then
        # Extract session cookie
        SESSION_COOKIE=$(curl -s -c - -X POST \
            -H "Content-Type: application/json" \
            -d "{\"user\":\"$GRAFANA_USER\",\"password\":\"$GRAFANA_PASSWORD\"}" \
            "$GRAFANA_URL/api/login" 2>/dev/null | grep "grafana_sess" | awk '{print $7}')
        
        if [ -n "$SESSION_COOKIE" ]; then
            echo "$SESSION_COOKIE"
            return 0
        fi
    fi
    
    # Fallback: Try API key from environment
    if [ -n "$GRAFANA_API_KEY" ]; then
        echo "Bearer $GRAFANA_API_KEY"
        return 0
    fi
    
    log_error "Failed to authenticate with Grafana"
    return 1
}

# Import dashboard from JSON file
import_dashboard() {
    local dashboard_file=$1
    local auth_header=$2
    
    log_info "Importing dashboard: $dashboard_file"
    
    # Read dashboard JSON
    if [ ! -f "$dashboard_file" ]; then
        log_error "Dashboard file not found: $dashboard_file"
        return 1
    fi
    
    # Prepare dashboard payload
    DASHBOARD_JSON=$(cat "$dashboard_file")
    
    # Check if dashboard already exists
    DASHBOARD_TITLE=$(echo "$DASHBOARD_JSON" | jq -r '.dashboard.title // .title // "Unknown"')
    DASHBOARD_UID=$(echo "$DASHBOARD_JSON" | jq -r '.dashboard.uid // .uid // empty')
    
    # Search for existing dashboard
    if [ -n "$DASHBOARD_UID" ]; then
        EXISTING_DASHBOARD=$(curl -s \
            -H "Authorization: $auth_header" \
            "$GRAFANA_URL/api/dashboards/uid/$DASHBOARD_UID" 2>/dev/null)
        
        if echo "$EXISTING_DASHBOARD" | jq -e '.dashboard' > /dev/null 2>&1; then
            log_info "Dashboard exists (UID: $DASHBOARD_UID), updating..."
            UPDATE_MODE=true
        else
            log_info "Creating new dashboard..."
            UPDATE_MODE=false
        fi
    else
        UPDATE_MODE=false
    fi
    
    # Prepare import payload
    if [ "$UPDATE_MODE" = true ]; then
        # Update existing dashboard
        IMPORT_PAYLOAD=$(echo "$DASHBOARD_JSON" | jq '{
            dashboard: .dashboard,
            overwrite: true,
            folderId: 0
        }')
    else
        # Create new dashboard
        IMPORT_PAYLOAD=$(echo "$DASHBOARD_JSON" | jq '{
            dashboard: .dashboard,
            overwrite: false,
            folderId: 0
        }')
    fi
    
    # Import dashboard
    IMPORT_RESPONSE=$(curl -s -X POST \
        -H "Authorization: $auth_header" \
        -H "Content-Type: application/json" \
        -d "$IMPORT_PAYLOAD" \
        "$GRAFANA_URL/api/dashboards/db" 2>/dev/null)
    
    # Check response
    if echo "$IMPORT_RESPONSE" | jq -e '.uid' > /dev/null 2>&1; then
        IMPORTED_UID=$(echo "$IMPORT_RESPONSE" | jq -r '.uid')
        log_info "✅ Dashboard imported successfully (UID: $IMPORTED_UID)"
        return 0
    else
        ERROR_MSG=$(echo "$IMPORT_RESPONSE" | jq -r '.message // .error // "Unknown error"')
        log_error "Failed to import dashboard: $ERROR_MSG"
        return 1
    fi
}

# Main import procedure
main() {
    log_info "Starting Grafana dashboard import..."
    log_info "Grafana URL: $GRAFANA_URL"
    log_info "Dashboard Directory: $DASHBOARD_DIR"
    echo ""
    
    # Authenticate
    log_info "Authenticating with Grafana..."
    AUTH_HEADER=$(get_grafana_auth)
    
    if [ -z "$AUTH_HEADER" ]; then
        log_error "Failed to authenticate with Grafana"
        exit 1
    fi
    
    log_info "✓ Authentication successful"
    echo ""
    
    # Check if dashboard directory exists
    if [ ! -d "$DASHBOARD_DIR" ]; then
        log_error "Dashboard directory not found: $DASHBOARD_DIR"
        exit 1
    fi
    
    # Import all JSON dashboards
    local imported=0
    local failed=0
    
    for dashboard_file in "$DASHBOARD_DIR"/*.json; do
        if [ -f "$dashboard_file" ]; then
            if import_dashboard "$dashboard_file" "$AUTH_HEADER"; then
                ((imported++))
            else
                ((failed++))
            fi
            echo ""
        fi
    done
    
    # Summary
    log_info "=== Import Summary ==="
    log_info "Imported: $imported"
    log_error "Failed: $failed"
    
    if [ $failed -eq 0 ]; then
        log_info "✅ All dashboards imported successfully!"
        exit 0
    else
        log_error "❌ Some dashboards failed to import"
        exit 1
    fi
}

# Run main function
main "$@"


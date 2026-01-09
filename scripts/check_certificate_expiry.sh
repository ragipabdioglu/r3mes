#!/bin/bash
# SSL Certificate Expiry Check Script
#
# Checks SSL certificate expiration dates and exports metrics for Prometheus.
# Sends alerts if certificates are expiring soon.
#
# Usage:
#   ./scripts/check_certificate_expiry.sh [domain] [port]
#
# Environment Variables:
#   DOMAINS: Comma-separated list of domains to check (default: api.r3mes.network,r3mes.network)
#   WARNING_DAYS: Days before expiry to warn (default: 30)
#   CRITICAL_DAYS: Days before expiry to alert (default: 7)
#   PROMETHEUS_METRICS_FILE: Path to write Prometheus metrics (optional)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DOMAINS="${DOMAINS:-api.r3mes.network,r3mes.network}"
WARNING_DAYS="${WARNING_DAYS:-30}"
CRITICAL_DAYS="${CRITICAL_DAYS:-7}"
PROMETHEUS_METRICS_FILE="${PROMETHEUS_METRICS_FILE:-/tmp/ssl_certificate_metrics.prom}"
PORT="${2:-443}"

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Check certificate expiry for a domain
check_certificate() {
    local domain=$1
    local port=$2
    
    log_info "Checking certificate for $domain:$port"
    
    # Get certificate expiration date
    if command -v openssl &> /dev/null; then
        # Use openssl to get certificate expiry
        EXPIRY_DATE=$(echo | openssl s_client -servername "$domain" -connect "$domain:$port" 2>/dev/null | \
            openssl x509 -noout -enddate 2>/dev/null | cut -d= -f2)
        
        if [ -z "$EXPIRY_DATE" ]; then
            log_error "Failed to get certificate expiry for $domain"
            return 1
        fi
        
        # Convert to epoch timestamp
        EXPIRY_EPOCH=$(date -d "$EXPIRY_DATE" +%s 2>/dev/null || date -j -f "%b %d %H:%M:%S %Y %Z" "$EXPIRY_DATE" +%s 2>/dev/null)
        
        if [ -z "$EXPIRY_EPOCH" ]; then
            log_error "Failed to parse expiry date: $EXPIRY_DATE"
            return 1
        fi
        
        # Get current epoch timestamp
        CURRENT_EPOCH=$(date +%s)
        
        # Calculate days until expiry
        DAYS_UNTIL_EXPIRY=$(( (EXPIRY_EPOCH - CURRENT_EPOCH) / 86400 ))
        
        # Export Prometheus metric
        if [ -n "$PROMETHEUS_METRICS_FILE" ]; then
            echo "# HELP ssl_certificate_expiry_days Days until SSL certificate expires" >> "$PROMETHEUS_METRICS_FILE"
            echo "# TYPE ssl_certificate_expiry_days gauge" >> "$PROMETHEUS_METRICS_FILE"
            echo "ssl_certificate_expiry_days{domain=\"$domain\",port=\"$port\"} $DAYS_UNTIL_EXPIRY" >> "$PROMETHEUS_METRICS_FILE"
        fi
        
        # Check expiry status
        if [ $DAYS_UNTIL_EXPIRY -lt 0 ]; then
            log_error "✗ Certificate for $domain has EXPIRED ($DAYS_UNTIL_EXPIRY days ago)"
            return 2
        elif [ $DAYS_UNTIL_EXPIRY -lt $CRITICAL_DAYS ]; then
            log_error "✗ Certificate for $domain expires in $DAYS_UNTIL_EXPIRY days (CRITICAL: < $CRITICAL_DAYS days)"
            return 2
        elif [ $DAYS_UNTIL_EXPIRY -lt $WARNING_DAYS ]; then
            log_warn "⚠ Certificate for $domain expires in $DAYS_UNTIL_EXPIRY days (WARNING: < $WARNING_DAYS days)"
            return 1
        else
            log_info "✓ Certificate for $domain expires in $DAYS_UNTIL_EXPIRY days"
            return 0
        fi
    else
        log_error "openssl not found, cannot check certificate"
        return 1
    fi
}

# Main check procedure
main() {
    local domain="${1:-}"
    
    # Initialize Prometheus metrics file
    if [ -n "$PROMETHEUS_METRICS_FILE" ]; then
        > "$PROMETHEUS_METRICS_FILE"  # Clear file
    fi
    
    local exit_code=0
    local checked_domains=0
    local failed_domains=0
    
    # Check single domain if provided, otherwise check all domains
    if [ -n "$domain" ]; then
        if check_certificate "$domain" "$PORT"; then
            ((checked_domains++))
        else
            ((failed_domains++))
            exit_code=1
        fi
    else
        # Check all domains
        IFS=',' read -ra DOMAIN_ARRAY <<< "$DOMAINS"
        for domain in "${DOMAIN_ARRAY[@]}"; do
            domain=$(echo "$domain" | xargs)  # Trim whitespace
            if [ -n "$domain" ]; then
                if check_certificate "$domain" "$PORT"; then
                    ((checked_domains++))
                else
                    ((failed_domains++))
                    exit_code=1
                fi
            fi
        done
    fi
    
    # Summary
    echo ""
    log_info "=== Certificate Check Summary ==="
    log_info "Checked: $checked_domains"
    log_error "Failed/Warning: $failed_domains"
    
    if [ -n "$PROMETHEUS_METRICS_FILE" ] && [ -f "$PROMETHEUS_METRICS_FILE" ]; then
        log_info "Prometheus metrics written to: $PROMETHEUS_METRICS_FILE"
    fi
    
    if [ $exit_code -eq 0 ]; then
        log_info "✅ All certificates are valid"
    else
        log_error "❌ Some certificates need attention"
    fi
    
    exit $exit_code
}

# Run main function
main "$@"


#!/bin/bash
# R3MES Secrets Rotation Script
# Safely rotates secrets with zero-downtime for Docker Compose and Swarm deployments
# Usage: bash scripts/rotate_secrets.sh [--secret NAME] [--all] [--dry-run]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKER_DIR="$PROJECT_ROOT/docker"
SECRETS_DIR="$DOCKER_DIR/secrets"
BACKUP_DIR="$SECRETS_DIR/backup"

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

# Generate secure random password
generate_password() {
    local length=${1:-32}
    openssl rand -base64 48 | tr -d "=+/" | cut -c1-$length
}

# Backup current secret
backup_secret() {
    local name=$1
    local file="$SECRETS_DIR/${name}.txt"
    
    if [ -f "$file" ]; then
        mkdir -p "$BACKUP_DIR"
        local timestamp=$(date +%Y%m%d_%H%M%S)
        cp "$file" "$BACKUP_DIR/${name}_${timestamp}.txt"
        log_info "Backed up: ${name} -> backup/${name}_${timestamp}.txt"
    fi
}

# Rotate a single secret (Docker Compose - file-based)
rotate_compose_secret() {
    local name=$1
    local new_value=$2
    local file="$SECRETS_DIR/${name}.txt"
    
    # Backup current
    backup_secret "$name"
    
    # Write new secret
    echo -n "$new_value" > "$file"
    chmod 600 "$file"
    
    log_success "Rotated secret: $name"
}

# Rotate a single secret (Docker Swarm)
rotate_swarm_secret() {
    local name=$1
    local new_value=$2
    
    # Check if swarm is active
    if ! docker info 2>/dev/null | grep -q "Swarm: active"; then
        log_warn "Docker Swarm not active, skipping swarm secret rotation"
        return 0
    fi
    
    # Create new versioned secret
    local timestamp=$(date +%Y%m%d%H%M%S)
    local new_name="${name}_${timestamp}"
    
    echo -n "$new_value" | docker secret create "$new_name" -
    log_success "Created new swarm secret: $new_name"
    
    # Note: Services need to be updated to use new secret
    log_warn "Update services to use new secret: $new_name"
    log_info "Then remove old secret: docker secret rm $name"
}

# Rotate PostgreSQL password
rotate_postgres() {
    local dry_run=$1
    local new_password=$(generate_password)
    
    log_header "Rotating PostgreSQL Password"
    
    if [ "$dry_run" = true ]; then
        log_info "[DRY RUN] Would rotate postgres_password"
        return 0
    fi
    
    # Rotate file-based secret
    rotate_compose_secret "postgres_password" "$new_password"
    
    # Update PostgreSQL user password
    log_info "Updating PostgreSQL user password..."
    
    # Get current password for authentication
    local current_password=""
    if [ -f "$BACKUP_DIR/postgres_password_"*.txt ]; then
        current_password=$(cat "$BACKUP_DIR/postgres_password_"*.txt 2>/dev/null | tail -1)
    fi
    
    # Try to update password in running container
    if docker ps --format '{{.Names}}' | grep -q "r3mes-postgres"; then
        docker exec r3mes-postgres-prod psql -U r3mes -c "ALTER USER r3mes PASSWORD '$new_password';" 2>/dev/null || {
            log_warn "Could not update PostgreSQL password automatically"
            log_info "Manual update required: ALTER USER r3mes PASSWORD 'new_password';"
        }
    fi
    
    # Restart dependent services
    log_info "Restarting backend service..."
    docker compose -f "$DOCKER_DIR/docker-compose.prod.yml" restart backend 2>/dev/null || true
    
    log_success "PostgreSQL password rotated"
    echo "New password: $new_password"
}

# Rotate Redis password
rotate_redis() {
    local dry_run=$1
    local new_password=$(generate_password)
    
    log_header "Rotating Redis Password"
    
    if [ "$dry_run" = true ]; then
        log_info "[DRY RUN] Would rotate redis_password"
        return 0
    fi
    
    # Rotate file-based secret
    rotate_compose_secret "redis_password" "$new_password"
    
    # Update Redis password (requires restart)
    log_info "Redis password rotation requires service restart"
    
    # Restart Redis and dependent services
    docker compose -f "$DOCKER_DIR/docker-compose.prod.yml" restart redis backend 2>/dev/null || true
    
    log_success "Redis password rotated"
    echo "New password: $new_password"
}

# Rotate Grafana admin password
rotate_grafana() {
    local dry_run=$1
    local new_password=$(generate_password 24)
    
    log_header "Rotating Grafana Admin Password"
    
    if [ "$dry_run" = true ]; then
        log_info "[DRY RUN] Would rotate grafana_admin_password"
        return 0
    fi
    
    # Rotate file-based secret
    rotate_compose_secret "grafana_admin_password" "$new_password"
    
    # Update Grafana admin password via API
    if docker ps --format '{{.Names}}' | grep -q "r3mes-grafana"; then
        local current_password=$(cat "$BACKUP_DIR/grafana_admin_password_"*.txt 2>/dev/null | tail -1)
        
        docker exec r3mes-grafana-prod grafana-cli admin reset-admin-password "$new_password" 2>/dev/null || {
            log_warn "Could not update Grafana password automatically"
            log_info "Manual update required via Grafana UI"
        }
    fi
    
    log_success "Grafana admin password rotated"
    echo "New password: $new_password"
}

# Rotate JWT secret
rotate_jwt() {
    local dry_run=$1
    local new_secret=$(openssl rand -hex 32)
    
    log_header "Rotating JWT Secret"
    
    if [ "$dry_run" = true ]; then
        log_info "[DRY RUN] Would rotate jwt_secret"
        return 0
    fi
    
    # Create JWT secret file if it doesn't exist
    rotate_compose_secret "jwt_secret" "$new_secret"
    
    # Restart backend (will invalidate existing tokens)
    log_warn "JWT rotation will invalidate all existing tokens!"
    docker compose -f "$DOCKER_DIR/docker-compose.prod.yml" restart backend 2>/dev/null || true
    
    log_success "JWT secret rotated"
    echo "New secret: $new_secret"
}

# Rotate API secret key
rotate_api_key() {
    local dry_run=$1
    local new_key=$(openssl rand -hex 32)
    
    log_header "Rotating API Secret Key"
    
    if [ "$dry_run" = true ]; then
        log_info "[DRY RUN] Would rotate api_secret_key"
        return 0
    fi
    
    rotate_compose_secret "api_secret_key" "$new_key"
    
    # Restart backend
    docker compose -f "$DOCKER_DIR/docker-compose.prod.yml" restart backend 2>/dev/null || true
    
    log_success "API secret key rotated"
    echo "New key: $new_key"
}

# Rotate all secrets
rotate_all() {
    local dry_run=$1
    
    log_header "Rotating All Secrets"
    
    if [ "$dry_run" = true ]; then
        log_info "[DRY RUN] Would rotate all secrets"
    fi
    
    rotate_postgres "$dry_run"
    rotate_redis "$dry_run"
    rotate_grafana "$dry_run"
    rotate_jwt "$dry_run"
    rotate_api_key "$dry_run"
    
    log_success "All secrets rotated"
}

# Cleanup old backups
cleanup_backups() {
    local days=${1:-30}
    
    log_header "Cleaning Up Old Backups"
    
    if [ -d "$BACKUP_DIR" ]; then
        find "$BACKUP_DIR" -name "*.txt" -mtime +$days -delete
        log_success "Removed backups older than $days days"
    fi
}

# Show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --secret NAME    Rotate specific secret (postgres, redis, grafana, jwt, api_key)"
    echo "  --all            Rotate all secrets"
    echo "  --dry-run        Show what would be done without making changes"
    echo "  --cleanup [DAYS] Remove backups older than DAYS (default: 30)"
    echo "  --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --secret postgres           # Rotate PostgreSQL password"
    echo "  $0 --all --dry-run             # Preview all rotations"
    echo "  $0 --all                       # Rotate all secrets"
    echo "  $0 --cleanup 7                 # Remove backups older than 7 days"
}

# Parse arguments
DRY_RUN=false
SECRET_NAME=""
ROTATE_ALL=false
CLEANUP_DAYS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --secret)
            SECRET_NAME="$2"
            shift 2
            ;;
        --all)
            ROTATE_ALL=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --cleanup)
            CLEANUP_DAYS="${2:-30}"
            shift
            [ -n "$2" ] && [[ "$2" =~ ^[0-9]+$ ]] && shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution
log_header "R3MES Secrets Rotation"

if [ "$DRY_RUN" = true ]; then
    log_warn "DRY RUN MODE - No changes will be made"
fi

# Create secrets directory if needed
mkdir -p "$SECRETS_DIR"

if [ -n "$CLEANUP_DAYS" ]; then
    cleanup_backups "$CLEANUP_DAYS"
elif [ "$ROTATE_ALL" = true ]; then
    rotate_all "$DRY_RUN"
elif [ -n "$SECRET_NAME" ]; then
    case $SECRET_NAME in
        postgres)
            rotate_postgres "$DRY_RUN"
            ;;
        redis)
            rotate_redis "$DRY_RUN"
            ;;
        grafana)
            rotate_grafana "$DRY_RUN"
            ;;
        jwt)
            rotate_jwt "$DRY_RUN"
            ;;
        api_key)
            rotate_api_key "$DRY_RUN"
            ;;
        *)
            log_error "Unknown secret: $SECRET_NAME"
            log_info "Valid secrets: postgres, redis, grafana, jwt, api_key"
            exit 1
            ;;
    esac
else
    usage
    exit 1
fi

log_header "Rotation Complete"

if [ "$DRY_RUN" = false ]; then
    log_warn "Remember to:"
    echo "  1. Update any external systems using these credentials"
    echo "  2. Test all services are working correctly"
    echo "  3. Store new credentials securely"
fi

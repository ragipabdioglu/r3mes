#!/bin/bash
# Restore Test Script
#
# Tests database restore procedure on a test database.
# Verifies data integrity after restore.
#
# Usage:
#   ./scripts/test_restore.sh [backup_file] [test_database_url]
#
# Environment Variables:
#   BACKUP_FILE: Path to backup file (required if not provided as argument)
#   TEST_DATABASE_URL: PostgreSQL connection URL for test database (required if not provided as argument)
#   CLEANUP: Set to "true" to drop test database after restore (default: true)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BACKUP_FILE="${1:-${BACKUP_FILE}}"
TEST_DATABASE_URL="${2:-${TEST_DATABASE_URL:-postgresql://test:test@localhost:5432/r3mes_test_restore}}"
CLEANUP="${CLEANUP:-true}"

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Check if backup file exists
if [ -z "$BACKUP_FILE" ]; then
    log_error "Backup file not specified. Provide as argument or set BACKUP_FILE environment variable."
    exit 1
fi

if [ ! -f "$BACKUP_FILE" ]; then
    log_error "Backup file not found: $BACKUP_FILE"
    exit 1
fi

# Extract database name from URL
extract_db_name() {
    local url=$1
    echo "$url" | sed 's|.*/||' | sed 's|?.*||'
}

DB_NAME=$(extract_db_name "$TEST_DATABASE_URL")

# Create test database
create_test_database() {
    log_info "Creating test database: $DB_NAME"
    
    # Extract connection URL without database name
    BASE_URL=$(echo "$TEST_DATABASE_URL" | sed "s|/$DB_NAME.*||")
    BASE_URL="${BASE_URL}/postgres"  # Connect to default postgres database
    
    # Drop existing test database if it exists
    psql "$BASE_URL" -c "DROP DATABASE IF EXISTS $DB_NAME;" > /dev/null 2>&1 || true
    
    # Create new test database
    if psql "$BASE_URL" -c "CREATE DATABASE $DB_NAME;" > /dev/null 2>&1; then
        log_info "✓ Test database created"
        return 0
    else
        log_error "Failed to create test database"
        return 1
    fi
}

# Restore backup to test database
restore_backup() {
    log_info "Restoring backup to test database..."
    
    # Decompress and restore
    if zcat "$BACKUP_FILE" | psql "$TEST_DATABASE_URL" > /dev/null 2>&1; then
        log_info "✓ Backup restored successfully"
        return 0
    else
        log_error "Failed to restore backup"
        return 1
    fi
}

# Verify restored data
verify_restored_data() {
    log_info "Verifying restored data..."
    
    # Count tables
    TABLE_COUNT=$(psql "$TEST_DATABASE_URL" -t -c "
        SELECT COUNT(*) FROM information_schema.tables 
        WHERE table_schema = 'public';
    " | tr -d ' ')
    
    log_info "Tables found: $TABLE_COUNT"
    
    if [ "$TABLE_COUNT" -gt 0 ]; then
        log_info "✓ Database contains tables"
    else
        log_error "✗ Database is empty"
        return 1
    fi
    
    # Check for key tables
    KEY_TABLES=("users" "mining_stats" "earnings_history")
    for table in "${KEY_TABLES[@]}"; do
        if psql "$TEST_DATABASE_URL" -t -c "SELECT COUNT(*) FROM $table;" > /dev/null 2>&1; then
            ROW_COUNT=$(psql "$TEST_DATABASE_URL" -t -c "SELECT COUNT(*) FROM $table;" | tr -d ' ')
            log_info "  $table: $ROW_COUNT rows"
        else
            log_warn "  Table $table not found or empty"
        fi
    done
    
    # Test query
    log_info "Testing query execution..."
    if psql "$TEST_DATABASE_URL" -c "SELECT 1;" > /dev/null 2>&1; then
        log_info "✓ Query execution successful"
        return 0
    else
        log_error "✗ Query execution failed"
        return 1
    fi
}

# Cleanup test database
cleanup_test_database() {
    if [ "$CLEANUP" == "true" ]; then
        log_info "Cleaning up test database..."
        
        BASE_URL=$(echo "$TEST_DATABASE_URL" | sed "s|/$DB_NAME.*||")
        BASE_URL="${BASE_URL}/postgres"
        
        if psql "$BASE_URL" -c "DROP DATABASE IF EXISTS $DB_NAME;" > /dev/null 2>&1; then
            log_info "✓ Test database dropped"
        else
            log_warn "Failed to drop test database (may need manual cleanup)"
        fi
    else
        log_info "Test database preserved (CLEANUP=false)"
        log_info "Test database URL: $TEST_DATABASE_URL"
    fi
}

# Main restore test procedure
main() {
    log_info "Starting restore test..."
    log_info "Backup file: $BACKUP_FILE"
    log_info "Test database: $TEST_DATABASE_URL"
    echo ""
    
    # Create test database
    if ! create_test_database; then
        exit 1
    fi
    
    # Restore backup
    if ! restore_backup; then
        cleanup_test_database
        exit 1
    fi
    
    # Verify restored data
    if ! verify_restored_data; then
        cleanup_test_database
        exit 1
    fi
    
    # Cleanup
    cleanup_test_database
    
    log_info "✅ Restore test completed successfully!"
    exit 0
}

# Run main function
main "$@"


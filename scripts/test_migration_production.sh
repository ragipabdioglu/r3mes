#!/bin/bash
# Production Migration Test Script
#
# Tests database migration from SQLite to PostgreSQL in a staging environment.
# Includes data integrity verification, performance testing, and rollback testing.
#
# Usage:
#   ./scripts/test_migration_production.sh [--staging-db-url] [--source-sqlite]
#
# Environment Variables:
#   STAGING_DATABASE_URL: PostgreSQL connection URL for staging (required)
#   SOURCE_SQLITE_PATH: Path to SQLite database file (default: backend/database.db)
#   BACKUP_PATH: Path to store backups (default: /tmp/r3mes_migration_test)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
STAGING_DATABASE_URL="${STAGING_DATABASE_URL:-postgresql://test:test@localhost:5432/r3mes_staging}"
SOURCE_SQLITE_PATH="${SOURCE_SQLITE_PATH:-backend/database.db}"
BACKUP_PATH="${BACKUP_PATH:-/tmp/r3mes_migration_test}"
TEST_RESULTS_DIR="${BACKUP_PATH}/test_results"

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

# Create test results directory
mkdir -p "$TEST_RESULTS_DIR"

# Test 1: Full Migration Test
test_full_migration() {
    log_test "Test 1: Full Migration (SQLite → PostgreSQL)"
    
    # Create backup
    BACKUP_FILE="${BACKUP_PATH}/sqlite_backup_$(date +%Y%m%d_%H%M%S).db"
    log_info "Creating SQLite backup: $BACKUP_FILE"
    cp "$SOURCE_SQLITE_PATH" "$BACKUP_FILE" || {
        log_error "Failed to create backup"
        ((TESTS_FAILED++))
        return 1
    }
    
    # Get row counts from SQLite
    log_info "Counting rows in SQLite database..."
    SQLITE_COUNTS=$(sqlite3 "$SOURCE_SQLITE_PATH" <<EOF
SELECT 
    (SELECT COUNT(*) FROM users) as users,
    (SELECT COUNT(*) FROM mining_stats) as mining_stats,
    (SELECT COUNT(*) FROM earnings_history) as earnings_history,
    (SELECT COUNT(*) FROM hashrate_history) as hashrate_history,
    (SELECT COUNT(*) FROM api_keys) as api_keys;
EOF
)
    
    # Run migration
    log_info "Running migration..."
    START_TIME=$(date +%s)
    
    if python3 scripts/migrate_to_postgresql_production.py \
        --source-sqlite "$SOURCE_SQLITE_PATH" \
        --target-postgresql "$STAGING_DATABASE_URL" \
        --backup-path "$BACKUP_PATH" \
        > "$TEST_RESULTS_DIR/migration.log" 2>&1; then
        END_TIME=$(date +%s)
        MIGRATION_TIME=$((END_TIME - START_TIME))
        log_info "Migration completed in ${MIGRATION_TIME} seconds"
    else
        log_error "Migration failed"
        ((TESTS_FAILED++))
        return 1
    fi
    
    # Verify row counts in PostgreSQL
    log_info "Verifying row counts in PostgreSQL..."
    PG_COUNTS=$(psql "$STAGING_DATABASE_URL" -t -c "
SELECT 
    (SELECT COUNT(*) FROM users)::text || ',' ||
    (SELECT COUNT(*) FROM mining_stats)::text || ',' ||
    (SELECT COUNT(*) FROM earnings_history)::text || ',' ||
    (SELECT COUNT(*) FROM hashrate_history)::text || ',' ||
    (SELECT COUNT(*) FROM api_keys)::text;
" | tr -d ' ')
    
    # Compare counts
    if [ "$SQLITE_COUNTS" == "$PG_COUNTS" ]; then
        log_info "✓ Row counts match"
        echo "$SQLITE_COUNTS" > "$TEST_RESULTS_DIR/row_counts_match.txt"
        ((TESTS_PASSED++))
    else
        log_error "✗ Row counts do not match"
        log_error "SQLite: $SQLITE_COUNTS"
        log_error "PostgreSQL: $PG_COUNTS"
        echo "SQLite: $SQLITE_COUNTS" > "$TEST_RESULTS_DIR/row_counts_mismatch.txt"
        echo "PostgreSQL: $PG_COUNTS" >> "$TEST_RESULTS_DIR/row_counts_mismatch.txt"
        ((TESTS_FAILED++))
        return 1
    fi
    
    # Sample data verification
    log_info "Verifying sample data..."
    SAMPLE_USER=$(sqlite3 "$SOURCE_SQLITE_PATH" "SELECT wallet_address FROM users LIMIT 1;")
    if [ -n "$SAMPLE_USER" ]; then
        PG_USER=$(psql "$STAGING_DATABASE_URL" -t -c "SELECT wallet_address FROM users WHERE wallet_address = '$SAMPLE_USER';" | tr -d ' ')
        if [ "$SAMPLE_USER" == "$PG_USER" ]; then
            log_info "✓ Sample data matches"
            ((TESTS_PASSED++))
        else
            log_error "✗ Sample data mismatch"
            ((TESTS_FAILED++))
        fi
    fi
    
    return 0
}

# Test 2: Performance Test
test_performance() {
    log_test "Test 2: Performance Benchmark"
    
    # Test query performance
    log_info "Testing query performance..."
    
    START_TIME=$(date +%s%N)
    psql "$STAGING_DATABASE_URL" -c "SELECT COUNT(*) FROM users;" > /dev/null
    END_TIME=$(date +%s%N)
    QUERY_TIME=$(( (END_TIME - START_TIME) / 1000000 ))
    
    log_info "Simple query time: ${QUERY_TIME}ms"
    echo "$QUERY_TIME" > "$TEST_RESULTS_DIR/query_time_ms.txt"
    
    if [ $QUERY_TIME -lt 1000 ]; then
        log_info "✓ Query performance acceptable"
        ((TESTS_PASSED++))
    else
        log_warn "⚠ Query performance may be slow"
        ((TESTS_FAILED++))
    fi
}

# Test 3: Connection Pool Test
test_connection_pool() {
    log_test "Test 3: Connection Pool Test"
    
    # Test pool creation
    log_info "Testing connection pool..."
    
    python3 <<EOF
import asyncio
import asyncpg
import sys

async def test_pool():
    try:
        pool = await asyncpg.create_pool(
            "$STAGING_DATABASE_URL",
            min_size=5,
            max_size=20
        )
        
        # Test multiple concurrent connections
        async with pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            if result == 1:
                print("OK")
                sys.exit(0)
            else:
                print("FAIL")
                sys.exit(1)
        
        await pool.close()
    except Exception as e:
        print(f"FAIL: {e}")
        sys.exit(1)

asyncio.run(test_pool())
EOF
    
    if [ $? -eq 0 ]; then
        log_info "✓ Connection pool test passed"
        ((TESTS_PASSED++))
    else
        log_error "✗ Connection pool test failed"
        ((TESTS_FAILED++))
    fi
}

# Test 4: Rollback Test
test_rollback() {
    log_test "Test 4: Rollback Test"
    
    log_info "Testing rollback procedure..."
    
    if [ -f "scripts/rollback_migration.sh" ]; then
        if bash scripts/rollback_migration.sh "$STAGING_DATABASE_URL" "$BACKUP_PATH" > "$TEST_RESULTS_DIR/rollback.log" 2>&1; then
            log_info "✓ Rollback test passed"
            ((TESTS_PASSED++))
        else
            log_error "✗ Rollback test failed"
            ((TESTS_FAILED++))
        fi
    else
        log_warn "⚠ Rollback script not found, skipping rollback test"
    fi
}

# Main test execution
main() {
    log_info "Starting production migration tests..."
    log_info "Staging Database: $STAGING_DATABASE_URL"
    log_info "Source SQLite: $SOURCE_SQLITE_PATH"
    log_info "Backup Path: $BACKUP_PATH"
    echo ""
    
    # Run tests
    test_full_migration || true
    test_performance || true
    test_connection_pool || true
    test_rollback || true
    
    # Summary
    echo ""
    log_info "=== Test Summary ==="
    log_info "Passed: $TESTS_PASSED"
    log_error "Failed: $TESTS_FAILED"
    
    TOTAL=$((TESTS_PASSED + TESTS_FAILED))
    if [ $TOTAL -gt 0 ]; then
        SUCCESS_RATE=$((TESTS_PASSED * 100 / TOTAL))
        log_info "Success Rate: ${SUCCESS_RATE}%"
    fi
    
    if [ $TESTS_FAILED -eq 0 ]; then
        log_info "All migration tests passed!"
        exit 0
    else
        log_error "Some migration tests failed. Check $TEST_RESULTS_DIR for details."
        exit 1
    fi
}

# Run main function
main "$@"


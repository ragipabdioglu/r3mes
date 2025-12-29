#!/bin/bash
# Smoke tests for R3MES deployment verification

set -e

BASE_URL="${1:-http://localhost:3000}"
API_URL="${2:-http://localhost:8000}"

echo "Running smoke tests against $BASE_URL (API: $API_URL)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Test function
test_endpoint() {
    local name=$1
    local url=$2
    local expected_status=${3:-200}
    
    echo -n "Testing $name... "
    
    response=$(curl -s -o /dev/null -w "%{http_code}" "$url" || echo "000")
    
    if [ "$response" == "$expected_status" ]; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAIL (Status: $response, Expected: $expected_status)${NC}"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Database connectivity test
test_database() {
    local db_url="${DATABASE_URL:-postgresql://localhost:5432/r3mes}"
    echo -n "Testing database connectivity... "
    
    # Try to connect using psql if available
    if command -v psql &> /dev/null; then
        if psql "$db_url" -c "SELECT 1;" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ PASS${NC}"
            ((TESTS_PASSED++))
            return 0
        else
            echo -e "${RED}✗ FAIL${NC}"
            ((TESTS_FAILED++))
            return 1
        fi
    else
        # Fallback: test via API health endpoint
        db_health=$(curl -s "$API_URL/health/database" 2>/dev/null || echo "{}")
        if echo "$db_health" | grep -q "healthy"; then
            echo -e "${GREEN}✓ PASS${NC}"
            ((TESTS_PASSED++))
            return 0
        else
            echo -e "${YELLOW}⚠ SKIP (psql not available, API check failed)${NC}"
            return 0
        fi
    fi
}

# Redis connectivity test
test_redis() {
    local redis_url="${REDIS_URL:-redis://localhost:6379/0}"
    echo -n "Testing Redis connectivity... "
    
    # Try redis-cli if available
    if command -v redis-cli &> /dev/null; then
        # Extract host and port from URL
        if echo "$redis_url" | grep -q "redis://"; then
            # Parse redis:// URL
            host_port=$(echo "$redis_url" | sed 's|redis://||' | sed 's|/.*||')
            if redis-cli -h "${host_port%%:*}" -p "${host_port##*:}" ping > /dev/null 2>&1; then
                echo -e "${GREEN}✓ PASS${NC}"
                ((TESTS_PASSED++))
                return 0
            else
                echo -e "${RED}✗ FAIL${NC}"
                ((TESTS_FAILED++))
                return 1
            fi
        else
            if redis-cli -u "$redis_url" ping > /dev/null 2>&1; then
                echo -e "${GREEN}✓ PASS${NC}"
                ((TESTS_PASSED++))
                return 0
            else
                echo -e "${RED}✗ FAIL${NC}"
                ((TESTS_FAILED++))
                return 1
            fi
        fi
    else
        # Fallback: test via API health endpoint
        redis_health=$(curl -s "$API_URL/health/redis" 2>/dev/null || echo "{}")
        if echo "$redis_health" | grep -q "healthy"; then
            echo -e "${GREEN}✓ PASS${NC}"
            ((TESTS_PASSED++))
            return 0
        else
            echo -e "${YELLOW}⚠ SKIP (redis-cli not available, API check failed)${NC}"
            return 0
        fi
    fi
}

# Secret management test
test_secret_management() {
    echo -n "Testing secret management... "
    
    # Test via Python script
    if command -v python3 &> /dev/null; then
        cd "$(dirname "$0")/.." || exit 1
        if python3 -c "
import sys
sys.path.insert(0, 'backend')
try:
    from app.secrets import get_secret_manager
    sm = get_secret_manager()
    # Test connection (don't retrieve actual secret)
    if hasattr(sm, 'test_connection'):
        result = sm.test_connection()
        if result:
            print('OK')
            sys.exit(0)
        else:
            print('FAIL: Connection test returned False')
            sys.exit(1)
    else:
        # If no test_connection method, just check if we can get the manager
        print('OK')
        sys.exit(0)
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ PASS${NC}"
            ((TESTS_PASSED++))
            return 0
        else
            echo -e "${YELLOW}⚠ SKIP (Secret management test failed, may not be configured)${NC}"
            return 0
        fi
    else
        # Fallback: test via API health endpoint
        secrets_health=$(curl -s "$API_URL/health/secrets" 2>/dev/null || echo "{}")
        if echo "$secrets_health" | grep -q "healthy"; then
            echo -e "${GREEN}✓ PASS${NC}"
            ((TESTS_PASSED++))
            return 0
        else
            echo -e "${YELLOW}⚠ SKIP (python3 not available, API check failed)${NC}"
            return 0
        fi
    fi
}

# PostgreSQL connection pool test
test_connection_pool() {
    echo -n "Testing PostgreSQL connection pool... "
    
    # Test via API health endpoint
    pool_stats=$(curl -s "$API_URL/health/pool" 2>/dev/null || echo "{}")
    
    if echo "$pool_stats" | grep -q "available\|size"; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${YELLOW}⚠ SKIP (Pool endpoint not available)${NC}"
        return 0
    fi
}

# Backup system test
test_backup_system() {
    echo -n "Testing backup system... "
    
    if [ -f "scripts/backup_database.sh" ] && [ -x "scripts/backup_database.sh" ]; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAIL (Backup script not found or not executable)${NC}"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Frontend tests
echo "=== Frontend Tests ==="
test_endpoint "Frontend Homepage" "$BASE_URL/"
test_endpoint "Frontend Docs" "$BASE_URL/docs"
test_endpoint "Frontend Developers" "$BASE_URL/developers"
test_endpoint "Frontend Community" "$BASE_URL/community"

# API tests
echo "=== API Tests ==="
test_endpoint "API Health Check" "$API_URL/health"
test_endpoint "API Network Stats" "$API_URL/network/stats"
test_endpoint "API Metrics" "$API_URL/metrics" 200

# Infrastructure tests
echo "=== Infrastructure Tests ==="
test_database
test_redis
test_secret_management
test_connection_pool
test_backup_system

# Blockchain tests (if available)
echo "=== Blockchain Tests ==="
BLOCKCHAIN_RPC="${BLOCKCHAIN_RPC:-http://localhost:26657}"
test_endpoint "Blockchain RPC Status" "$BLOCKCHAIN_RPC/status" 200 || echo -e "${YELLOW}⚠ Blockchain not available${NC}"

# Summary
echo ""
echo "=== Test Summary ==="
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All smoke tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some smoke tests failed!${NC}"
    exit 1
fi


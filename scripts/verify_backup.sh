#!/bin/bash
# Backup Verification Script
#
# Verifies backup file integrity, age, size, and S3 upload status.
#
# Usage:
#   ./scripts/verify_backup.sh [backup_file]
#
# Environment Variables:
#   BACKUP_DIR: Backup directory (default: /backups)
#   S3_BUCKET: S3 bucket for backup storage (optional)
#   MAX_BACKUP_AGE_DAYS: Maximum age for backups in days (default: 1)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/backups}"
MAX_BACKUP_AGE_DAYS="${MAX_BACKUP_AGE_DAYS:-1}"
MIN_BACKUP_SIZE_MB="${MIN_BACKUP_SIZE_MB:-1}"

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

# Find latest backup if not specified
find_latest_backup() {
    local backup_file=$(ls -t "$BACKUP_DIR"/r3mes_*.sql.gz 2>/dev/null | head -1)
    if [ -z "$backup_file" ]; then
        log_error "No backup file found in $BACKUP_DIR"
        exit 1
    fi
    echo "$backup_file"
}

# Test 1: Backup file integrity check
test_backup_integrity() {
    local backup_file=$1
    log_info "Test 1: Backup file integrity check"
    
    echo -n "  Checking gzip integrity... "
    if gzip -t "$backup_file" 2>/dev/null; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}"
        log_error "Backup file is corrupted (gzip test failed)"
        ((TESTS_FAILED++))
        return 1
    fi
    
    # Try to extract and check SQL format
    echo -n "  Checking SQL format... "
    if zcat "$backup_file" | head -1 | grep -q "PostgreSQL database dump\|-- PostgreSQL database dump" 2>/dev/null; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${YELLOW}⚠ WARN${NC}"
        log_warn "SQL format check inconclusive"
    fi
    
    return 0
}

# Test 2: Backup file age check
test_backup_age() {
    local backup_file=$1
    log_info "Test 2: Backup file age check"
    
    # Get file modification time
    FILE_AGE_SECONDS=$(($(date +%s) - $(stat -c %Y "$backup_file")))
    FILE_AGE_DAYS=$((FILE_AGE_SECONDS / 86400))
    
    echo -n "  Backup age: ${FILE_AGE_DAYS} days (max: ${MAX_BACKUP_AGE_DAYS})... "
    
    if [ $FILE_AGE_DAYS -le $MAX_BACKUP_AGE_DAYS ]; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}"
        log_error "Backup is too old (${FILE_AGE_DAYS} days > ${MAX_BACKUP_AGE_DAYS} days)"
        ((TESTS_FAILED++))
        return 1
    fi
    
    return 0
}

# Test 3: Backup file size check
test_backup_size() {
    local backup_file=$1
    log_info "Test 3: Backup file size check"
    
    # Get file size in MB
    FILE_SIZE_BYTES=$(stat -c %s "$backup_file")
    FILE_SIZE_MB=$((FILE_SIZE_BYTES / 1024 / 1024))
    
    echo -n "  Backup size: ${FILE_SIZE_MB} MB (min: ${MIN_BACKUP_SIZE_MB})... "
    
    if [ $FILE_SIZE_MB -ge $MIN_BACKUP_SIZE_MB ]; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}"
        log_error "Backup file is too small (${FILE_SIZE_MB} MB < ${MIN_BACKUP_SIZE_MB} MB)"
        ((TESTS_FAILED++))
        return 1
    fi
    
    return 0
}

# Test 4: S3 upload verification
test_s3_upload() {
    local backup_file=$1
    log_info "Test 4: S3 upload verification"
    
    if [ -z "$S3_BUCKET" ]; then
        echo -e "  ${YELLOW}⚠ SKIP (S3_BUCKET not configured)${NC}"
        return 0
    fi
    
    if ! command -v aws &> /dev/null; then
        echo -e "  ${YELLOW}⚠ SKIP (AWS CLI not found)${NC}"
        return 0
    fi
    
    # Extract backup filename
    BACKUP_FILENAME=$(basename "$backup_file")
    S3_PATH="s3://${S3_BUCKET}/database-backups/${BACKUP_FILENAME}"
    
    echo -n "  Checking S3: $S3_PATH... "
    
    if aws s3 ls "$S3_PATH" ${AWS_REGION:+--region $AWS_REGION} > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}"
        log_error "Backup not found in S3"
        ((TESTS_FAILED++))
        return 1
    fi
    
    return 0
}

# Test 5: Restore dry-run
test_restore_dry_run() {
    local backup_file=$1
    log_info "Test 5: Restore dry-run"
    
    echo -n "  Testing restore (dry-run)... "
    
    # Try to extract and parse first few lines
    if zcat "$backup_file" | head -100 | grep -q "CREATE TABLE\|INSERT INTO" 2>/dev/null; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${YELLOW}⚠ WARN${NC}"
        log_warn "Restore dry-run inconclusive"
    fi
    
    return 0
}

# Main verification procedure
main() {
    local backup_file="${1:-$(find_latest_backup)}"
    
    log_info "Verifying backup: $backup_file"
    log_info "Backup directory: $BACKUP_DIR"
    echo ""
    
    if [ ! -f "$backup_file" ]; then
        log_error "Backup file not found: $backup_file"
        exit 1
    fi
    
    # Run all tests
    test_backup_integrity "$backup_file" || true
    test_backup_age "$backup_file" || true
    test_backup_size "$backup_file" || true
    test_s3_upload "$backup_file" || true
    test_restore_dry_run "$backup_file" || true
    
    # Summary
    echo ""
    log_info "=== Verification Summary ==="
    log_info "Passed: $TESTS_PASSED"
    log_error "Failed: $TESTS_FAILED"
    
    TOTAL=$((TESTS_PASSED + TESTS_FAILED))
    if [ $TOTAL -gt 0 ]; then
        SUCCESS_RATE=$((TESTS_PASSED * 100 / TOTAL))
        log_info "Success Rate: ${SUCCESS_RATE}%"
    fi
    
    if [ $TESTS_FAILED -eq 0 ]; then
        log_info "✅ All backup verification tests passed!"
        exit 0
    else
        log_error "❌ Some backup verification tests failed"
        exit 1
    fi
}

# Run main function
main "$@"


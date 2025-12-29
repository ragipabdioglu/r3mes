#!/bin/bash
# R3MES Backup Restore Test Script
# Tests PostgreSQL backup and restore functionality

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKER_DIR="$PROJECT_ROOT/docker"

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi

print_header "R3MES Backup Restore Test"

# Check if postgres container is running
if ! docker ps | grep -q r3mes-postgres-prod; then
    print_error "PostgreSQL container is not running. Please start the stack first."
    exit 1
fi

# Test database name
TEST_DB="r3mes_test_restore"
BACKUP_FILE="/tmp/r3mes_backup_test_$(date +%Y%m%d_%H%M%S).dump"

print_header "Step 1: Create Test Data"

# Create a test table and insert data
docker exec r3mes-postgres-prod psql -U r3mes -d r3mes -c "
CREATE TABLE IF NOT EXISTS backup_test (
    id SERIAL PRIMARY KEY,
    test_data TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO backup_test (test_data) VALUES 
    ('Test data 1'),
    ('Test data 2'),
    ('Test data 3');

SELECT COUNT(*) as test_records FROM backup_test;
" || {
    print_error "Failed to create test data"
    exit 1
}

print_success "Test data created"

# Get record count before backup
RECORD_COUNT_BEFORE=$(docker exec r3mes-postgres-prod psql -U r3mes -d r3mes -t -c "SELECT COUNT(*) FROM backup_test;" | tr -d ' ')

print_header "Step 2: Create Backup"

# Create backup using pg_dump
docker exec r3mes-postgres-prod pg_dump -U r3mes -F c -f "$BACKUP_FILE" r3mes || {
    print_error "Failed to create backup"
    exit 1
}

# Copy backup file from container to host
docker cp r3mes-postgres-prod:"$BACKUP_FILE" "$BACKUP_FILE" || {
    print_error "Failed to copy backup file from container"
    exit 1
}

print_success "Backup created: $BACKUP_FILE"
BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
print_success "Backup size: $BACKUP_SIZE"

print_header "Step 3: Delete Test Data"

# Delete test data
docker exec r3mes-postgres-prod psql -U r3mes -d r3mes -c "DELETE FROM backup_test;" || {
    print_error "Failed to delete test data"
    exit 1
}

# Verify deletion
RECORD_COUNT_AFTER_DELETE=$(docker exec r3mes-postgres-prod psql -U r3mes -d r3mes -t -c "SELECT COUNT(*) FROM backup_test;" | tr -d ' ')

if [ "$RECORD_COUNT_AFTER_DELETE" != "0" ]; then
    print_error "Test data was not deleted properly"
    exit 1
fi

print_success "Test data deleted (records: $RECORD_COUNT_AFTER_DELETE)"

print_header "Step 4: Restore Backup"

# Copy backup file back to container
docker cp "$BACKUP_FILE" r3mes-postgres-prod:"$BACKUP_FILE" || {
    print_error "Failed to copy backup file to container"
    exit 1
}

# Restore backup
docker exec r3mes-postgres-prod pg_restore -U r3mes -d r3mes --clean --if-exists "$BACKUP_FILE" || {
    print_error "Failed to restore backup"
    exit 1
}

print_success "Backup restored"

print_header "Step 5: Verify Restore"

# Verify data was restored
RECORD_COUNT_AFTER_RESTORE=$(docker exec r3mes-postgres-prod psql -U r3mes -d r3mes -t -c "SELECT COUNT(*) FROM backup_test;" | tr -d ' ')

if [ "$RECORD_COUNT_AFTER_RESTORE" != "$RECORD_COUNT_BEFORE" ]; then
    print_error "Data mismatch after restore!"
    print_error "Before: $RECORD_COUNT_BEFORE, After: $RECORD_COUNT_AFTER_RESTORE"
    exit 1
fi

print_success "Data verified: $RECORD_COUNT_AFTER_RESTORE records restored"

print_header "Step 6: Cleanup"

# Clean up test data
docker exec r3mes-postgres-prod psql -U r3mes -d r3mes -c "DROP TABLE IF EXISTS backup_test;" || {
    print_warning "Failed to drop test table (non-critical)"
}

# Remove backup file from container
docker exec r3mes-postgres-prod rm -f "$BACKUP_FILE" || {
    print_warning "Failed to remove backup file from container (non-critical)"
}

# Remove backup file from host
rm -f "$BACKUP_FILE" || {
    print_warning "Failed to remove backup file from host (non-critical)"
}

print_success "Cleanup completed"

print_header "Backup Restore Test Results"
print_success "✅ Backup creation: PASSED"
print_success "✅ Backup restore: PASSED"
print_success "✅ Data integrity: PASSED"
print_success "✅ All tests passed!"

print_header "Next Steps"
echo "1. Test automated backup service:"
echo "   docker logs r3mes-postgres-backup-prod"
echo ""
echo "2. List available backups:"
echo "   docker exec r3mes-postgres-backup-prod ls -lh /backups"
echo ""
echo "3. Manual backup:"
echo "   docker exec r3mes-postgres-prod pg_dump -U r3mes -F c r3mes > backup.dump"
echo ""
echo "4. Manual restore:"
echo "   docker exec -i r3mes-postgres-prod pg_restore -U r3mes -d r3mes --clean < backup.dump"


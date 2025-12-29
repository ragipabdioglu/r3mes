#!/bin/bash
# PostgreSQL Point-in-Time Recovery (PITR) Setup
#
# Configures PostgreSQL for WAL archiving and PITR support.
#
# Usage:
#   sudo ./scripts/setup_pitr.sh

set -e

# Configuration
PG_DATA_DIR="${PGDATA:-/var/lib/postgresql/data}"
WAL_ARCHIVE_DIR="${WAL_ARCHIVE_DIR:-/backups/wal}"
BASE_BACKUP_DIR="${BASE_BACKUP_DIR:-/backups/base}"

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

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    log_error "This script must be run as root (use sudo)"
    exit 1
fi

log_info "Setting up PostgreSQL Point-in-Time Recovery (PITR)..."

# Create WAL archive directory
log_info "Creating WAL archive directory: $WAL_ARCHIVE_DIR"
mkdir -p "$WAL_ARCHIVE_DIR"
chown postgres:postgres "$WAL_ARCHIVE_DIR"
chmod 700 "$WAL_ARCHIVE_DIR"

# Create base backup directory
log_info "Creating base backup directory: $BASE_BACKUP_DIR"
mkdir -p "$BASE_BACKUP_DIR"
chown postgres:postgres "$BASE_BACKUP_DIR"
chmod 700 "$BASE_BACKUP_DIR"

# Configure PostgreSQL for WAL archiving
log_info "Configuring PostgreSQL for WAL archiving..."

PG_CONF="$PG_DATA_DIR/postgresql.conf"

if [ ! -f "$PG_CONF" ]; then
    log_error "PostgreSQL configuration file not found: $PG_CONF"
    exit 1
fi

# Backup original config
cp "$PG_CONF" "${PG_CONF}.backup.$(date +%Y%m%d_%H%M%S)"

# Update postgresql.conf
log_info "Updating postgresql.conf..."

# Enable WAL archiving
sed -i "s/#wal_level = replica/wal_level = replica/" "$PG_CONF" || \
    echo "wal_level = replica" >> "$PG_CONF"

sed -i "s/#archive_mode = on/archive_mode = on/" "$PG_CONF" || \
    echo "archive_mode = on" >> "$PG_CONF"

sed -i "s|#archive_command = ''|archive_command = 'cp %p $WAL_ARCHIVE_DIR/%f'|" "$PG_CONF" || \
    echo "archive_command = 'cp %p $WAL_ARCHIVE_DIR/%f'" >> "$PG_CONF"

# Set max_wal_senders for replication
sed -i "s/#max_wal_senders = 10/max_wal_senders = 10/" "$PG_CONF" || \
    echo "max_wal_senders = 10" >> "$PG_CONF"

log_info "✅ PostgreSQL configuration updated"

# Create archive script
log_info "Creating WAL archive script..."
cat > /usr/local/bin/pg_archive_wal.sh << 'EOF'
#!/bin/bash
# WAL Archive Script
# Called by PostgreSQL to archive WAL files

WAL_FILE=$1
WAL_ARCHIVE_DIR="/backups/wal"

# Copy WAL file to archive
cp "$WAL_FILE" "$WAL_ARCHIVE_DIR/$(basename $WAL_FILE)"

# Optional: Upload to S3
if [ -n "$S3_BUCKET" ]; then
    aws s3 cp "$WAL_ARCHIVE_DIR/$(basename $WAL_FILE)" \
        "s3://${S3_BUCKET}/wal/$(basename $WAL_FILE)" \
        --region "${AWS_REGION:-us-east-1}" || true
fi

exit 0
EOF

chmod +x /usr/local/bin/pg_archive_wal.sh
chown postgres:postgres /usr/local/bin/pg_archive_wal.sh

# Update archive_command to use script
sed -i "s|archive_command = 'cp %p $WAL_ARCHIVE_DIR/%f'|archive_command = '/usr/local/bin/pg_archive_wal.sh %p'|" "$PG_CONF"

# Restart PostgreSQL to apply changes
log_info "Restarting PostgreSQL to apply changes..."
systemctl restart postgresql || service postgresql restart

# Wait for PostgreSQL to be ready
sleep 5

# Verify WAL archiving
log_info "Verifying WAL archiving..."
sudo -u postgres psql -c "SELECT name, setting FROM pg_settings WHERE name IN ('wal_level', 'archive_mode', 'archive_command');"

log_info ""
log_info "✅ PITR setup completed successfully!"
log_info ""
log_info "Next steps:"
log_info "1. Create initial base backup: sudo -u postgres pg_basebackup -D $BASE_BACKUP_DIR/base_$(date +%Y%m%d)"
log_info "2. Schedule regular base backups (weekly recommended)"
log_info "3. Monitor WAL archive directory: $WAL_ARCHIVE_DIR"
log_info "4. Test PITR restore procedure"


#!/bin/bash
# Setup Backup Cron Job
#
# Installs systemd timer for automated database backups.
#
# Usage:
#   sudo ./scripts/setup_backup_cron.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SYSTEMD_DIR="/etc/systemd/system"

log_info() {
    echo "[INFO] $1"
}

log_error() {
    echo "[ERROR] $1" >&2
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    log_error "This script must be run as root (use sudo)"
    exit 1
fi

log_info "Setting up R3MES database backup timer..."

# Copy service file
log_info "Installing systemd service..."
cp "$PROJECT_ROOT/scripts/systemd/r3mes-backup.service" "$SYSTEMD_DIR/r3mes-backup.service"

# Copy timer file
log_info "Installing systemd timer..."
cp "$PROJECT_ROOT/scripts/systemd/r3mes-backup.timer" "$SYSTEMD_DIR/r3mes-backup.timer"

# Reload systemd
log_info "Reloading systemd daemon..."
systemctl daemon-reload

# Enable and start timer
log_info "Enabling backup timer..."
systemctl enable r3mes-backup.timer
systemctl start r3mes-backup.timer

# Show timer status
log_info "Backup timer status:"
systemctl status r3mes-backup.timer --no-pager

log_info "✅ Backup timer installed successfully"
log_info "Next backup scheduled: $(systemctl list-timers r3mes-backup.timer --no-pager | grep r3mes-backup | awk '{print $1, $2}')"


#!/bin/bash
# R3MES Auto-Start Setup Script
# Configures R3MES to start automatically on system boot

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v systemctl &> /dev/null; then
            echo "linux-systemd"
        elif command -v service &> /dev/null; then
            echo "linux-sysvinit"
        else
            echo "linux-unknown"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R3MES_HOME="${R3MES_HOME:-/opt/r3mes}"

setup_linux_systemd() {
    log_info "Setting up systemd services..."
    
    # Check if running as root
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root for systemd setup"
        exit 1
    fi
    
    # Create r3mes user if not exists
    if ! id "r3mes" &>/dev/null; then
        log_info "Creating r3mes user..."
        useradd -r -s /bin/false -d "$R3MES_HOME" r3mes
    fi
    
    # Create directories
    mkdir -p "$R3MES_HOME"/{bin,data,logs,config}
    mkdir -p /var/log/r3mes/{node,miner,backend,frontend,ipfs,training,audit,error}
    chown -R r3mes:r3mes "$R3MES_HOME" /var/log/r3mes
    
    # Copy service files
    log_info "Installing systemd service files..."
    cp "$SCRIPT_DIR/systemd/"*.service /etc/systemd/system/
    cp "$SCRIPT_DIR/systemd/"*.timer /etc/systemd/system/ 2>/dev/null || true
    
    # Reload systemd
    systemctl daemon-reload
    
    # Enable services
    log_info "Enabling R3MES services..."
    systemctl enable remesd.service
    systemctl enable r3mes-miner.service
    systemctl enable r3mes-backend.service
    systemctl enable ipfs.service
    
    # Enable timers
    systemctl enable r3mes-backup.timer 2>/dev/null || true
    systemctl enable r3mes-log-cleanup.timer 2>/dev/null || true
    
    # Setup logrotate
    log_info "Setting up log rotation..."
    cp "$SCRIPT_DIR/logrotate/r3mes" /etc/logrotate.d/r3mes
    chmod 644 /etc/logrotate.d/r3mes
    
    log_success "Systemd services installed and enabled!"
    log_info "Start services with: sudo systemctl start remesd r3mes-miner r3mes-backend"
    log_info "Check status with: sudo systemctl status remesd"
}

setup_macos() {
    log_info "Setting up macOS launchd services..."
    
    LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
    mkdir -p "$LAUNCH_AGENTS_DIR"
    
    # Create R3MES directories
    R3MES_HOME_MAC="/usr/local/r3mes"
    sudo mkdir -p "$R3MES_HOME_MAC"/{bin,data,logs,config}
    sudo chown -R "$USER" "$R3MES_HOME_MAC"
    
    # Copy launchd plist files
    log_info "Installing launchd plist files..."
    
    # User-level launch agent (runs as current user)
    cp "$SCRIPT_DIR/launchd/network.r3mes.all.plist" "$LAUNCH_AGENTS_DIR/"
    
    # Load the launch agent
    launchctl load "$LAUNCH_AGENTS_DIR/network.r3mes.all.plist" 2>/dev/null || true
    
    log_success "macOS launchd services installed!"
    log_info "Service will start automatically on login"
    log_info "Start now with: launchctl start network.r3mes.all"
    log_info "Check status with: launchctl list | grep r3mes"
}

setup_windows() {
    log_info "Windows detected. Please run the PowerShell script as Administrator:"
    log_info "  powershell -ExecutionPolicy Bypass -File scripts/windows/r3mes-service.ps1 -Action install"
    log_info ""
    log_info "Or manually:"
    log_info "  1. Open PowerShell as Administrator"
    log_info "  2. Navigate to R3MES directory"
    log_info "  3. Run: .\\scripts\\windows\\r3mes-service.ps1 -Action install"
}

show_status() {
    OS=$(detect_os)
    
    echo ""
    echo "=== R3MES Auto-Start Status ==="
    echo ""
    
    case $OS in
        linux-systemd)
            echo "Platform: Linux (systemd)"
            echo ""
            echo "Service Status:"
            systemctl status remesd --no-pager 2>/dev/null || echo "  remesd: not installed"
            systemctl status r3mes-miner --no-pager 2>/dev/null || echo "  r3mes-miner: not installed"
            systemctl status r3mes-backend --no-pager 2>/dev/null || echo "  r3mes-backend: not installed"
            ;;
        macos)
            echo "Platform: macOS (launchd)"
            echo ""
            echo "Service Status:"
            launchctl list | grep r3mes || echo "  No R3MES services found"
            ;;
        windows)
            echo "Platform: Windows"
            echo "Run: Get-Service R3MES in PowerShell to check status"
            ;;
        *)
            echo "Platform: Unknown"
            ;;
    esac
}

uninstall() {
    OS=$(detect_os)
    
    case $OS in
        linux-systemd)
            log_info "Removing systemd services..."
            systemctl stop remesd r3mes-miner r3mes-backend 2>/dev/null || true
            systemctl disable remesd r3mes-miner r3mes-backend 2>/dev/null || true
            rm -f /etc/systemd/system/r3mes*.service
            rm -f /etc/systemd/system/remesd.service
            rm -f /etc/logrotate.d/r3mes
            systemctl daemon-reload
            log_success "Systemd services removed"
            ;;
        macos)
            log_info "Removing launchd services..."
            launchctl unload "$HOME/Library/LaunchAgents/network.r3mes.all.plist" 2>/dev/null || true
            rm -f "$HOME/Library/LaunchAgents/network.r3mes."*.plist
            log_success "Launchd services removed"
            ;;
        windows)
            log_info "Run PowerShell as Administrator:"
            log_info "  .\\scripts\\windows\\r3mes-service.ps1 -Action uninstall"
            ;;
    esac
}

# Main
case "${1:-install}" in
    install)
        OS=$(detect_os)
        log_info "Detected OS: $OS"
        
        case $OS in
            linux-systemd)
                setup_linux_systemd
                ;;
            macos)
                setup_macos
                ;;
            windows)
                setup_windows
                ;;
            *)
                log_error "Unsupported operating system: $OS"
                exit 1
                ;;
        esac
        ;;
    uninstall)
        uninstall
        ;;
    status)
        show_status
        ;;
    *)
        echo "Usage: $0 {install|uninstall|status}"
        exit 1
        ;;
esac

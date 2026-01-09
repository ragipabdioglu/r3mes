#!/bin/bash

# HashiCorp Vault Setup Script for R3MES
# This script sets up a self-hosted Vault instance on Contabo VPS

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VAULT_VERSION="1.15.4"
VAULT_USER="vault"
VAULT_HOME="/opt/vault"
VAULT_DATA_DIR="/opt/vault/data"
VAULT_CONFIG_DIR="/etc/vault.d"
VAULT_LOG_DIR="/var/log/vault"
VAULT_PORT="8200"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        print_error "This script must be run as root"
        exit 1
    fi
}

# Function to detect OS
detect_os() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        OS=$ID
        VERSION=$VERSION_ID
    else
        print_error "Cannot detect operating system"
        exit 1
    fi
    
    print_status "Detected OS: $OS $VERSION"
}

# Function to install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    case $OS in
        ubuntu|debian)
            apt-get update
            apt-get install -y curl unzip jq systemd
            ;;
        centos|rhel|fedora)
            if command -v dnf &> /dev/null; then
                dnf install -y curl unzip jq systemd
            else
                yum install -y curl unzip jq systemd
            fi
            ;;
        *)
            print_error "Unsupported operating system: $OS"
            exit 1
            ;;
    esac
    
    print_success "Dependencies installed"
}

# Function to create vault user
create_vault_user() {
    print_status "Creating vault user..."
    
    if id "$VAULT_USER" &>/dev/null; then
        print_warning "User $VAULT_USER already exists"
    else
        useradd --system --home $VAULT_HOME --shell /bin/false $VAULT_USER
        print_success "User $VAULT_USER created"
    fi
}

# Function to download and install Vault
install_vault() {
    print_status "Downloading and installing Vault $VAULT_VERSION..."
    
    # Detect architecture
    ARCH=$(uname -m)
    case $ARCH in
        x86_64)
            VAULT_ARCH="amd64"
            ;;
        aarch64|arm64)
            VAULT_ARCH="arm64"
            ;;
        *)
            print_error "Unsupported architecture: $ARCH"
            exit 1
            ;;
    esac
    
    # Download Vault
    cd /tmp
    VAULT_ZIP="vault_${VAULT_VERSION}_linux_${VAULT_ARCH}.zip"
    VAULT_URL="https://releases.hashicorp.com/vault/${VAULT_VERSION}/${VAULT_ZIP}"
    
    print_status "Downloading from: $VAULT_URL"
    curl -fsSL -o "$VAULT_ZIP" "$VAULT_URL"
    
    # Verify download
    if [[ ! -f "$VAULT_ZIP" ]]; then
        print_error "Failed to download Vault"
        exit 1
    fi
    
    # Extract and install
    unzip -q "$VAULT_ZIP"
    chmod +x vault
    mv vault /usr/local/bin/
    
    # Verify installation
    if vault version; then
        print_success "Vault installed successfully"
    else
        print_error "Vault installation failed"
        exit 1
    fi
    
    # Cleanup
    rm -f "$VAULT_ZIP"
}

# Function to create directories
create_directories() {
    print_status "Creating directories..."
    
    mkdir -p "$VAULT_HOME"
    mkdir -p "$VAULT_DATA_DIR"
    mkdir -p "$VAULT_CONFIG_DIR"
    mkdir -p "$VAULT_LOG_DIR"
    
    # Set ownership
    chown -R $VAULT_USER:$VAULT_USER "$VAULT_HOME"
    chown -R $VAULT_USER:$VAULT_USER "$VAULT_DATA_DIR"
    chown -R $VAULT_USER:$VAULT_USER "$VAULT_LOG_DIR"
    chown -R root:$VAULT_USER "$VAULT_CONFIG_DIR"
    
    # Set permissions
    chmod 750 "$VAULT_HOME"
    chmod 750 "$VAULT_DATA_DIR"
    chmod 750 "$VAULT_CONFIG_DIR"
    chmod 750 "$VAULT_LOG_DIR"
    
    print_success "Directories created"
}

# Function to generate TLS certificates
generate_tls_certificates() {
    print_status "Generating TLS certificates..."
    
    CERT_DIR="$VAULT_CONFIG_DIR/tls"
    mkdir -p "$CERT_DIR"
    
    # Generate private key
    openssl genrsa -out "$CERT_DIR/vault-key.pem" 2048
    
    # Generate certificate signing request
    cat > "$CERT_DIR/vault.conf" <<EOF
[req]
default_bits = 2048
prompt = no
distinguished_name = req_distinguished_name
req_extensions = v3_req

[req_distinguished_name]
C = US
ST = State
L = City
O = R3MES
CN = vault.r3mes.local

[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = vault.r3mes.local
DNS.2 = localhost
IP.1 = 127.0.0.1
EOF
    
    # Generate self-signed certificate
    openssl req -new -x509 -key "$CERT_DIR/vault-key.pem" \
        -out "$CERT_DIR/vault-cert.pem" \
        -days 365 \
        -config "$CERT_DIR/vault.conf" \
        -extensions v3_req
    
    # Set permissions
    chown -R root:$VAULT_USER "$CERT_DIR"
    chmod 640 "$CERT_DIR/vault-key.pem"
    chmod 644 "$CERT_DIR/vault-cert.pem"
    
    print_success "TLS certificates generated"
}

# Function to create Vault configuration
create_vault_config() {
    print_status "Creating Vault configuration..."
    
    cat > "$VAULT_CONFIG_DIR/vault.hcl" <<EOF
# Vault Configuration for R3MES

# Storage backend
storage "file" {
  path = "$VAULT_DATA_DIR"
}

# Listener configuration
listener "tcp" {
  address     = "0.0.0.0:$VAULT_PORT"
  tls_cert_file = "$VAULT_CONFIG_DIR/tls/vault-cert.pem"
  tls_key_file  = "$VAULT_CONFIG_DIR/tls/vault-key.pem"
  tls_min_version = "tls12"
}

# API address
api_addr = "https://127.0.0.1:$VAULT_PORT"

# Cluster address (for HA setups)
cluster_addr = "https://127.0.0.1:8201"

# UI
ui = true

# Logging
log_level = "INFO"
log_file = "$VAULT_LOG_DIR/vault.log"
log_rotate_duration = "24h"
log_rotate_max_files = 30

# Disable mlock (for containers/VPS)
disable_mlock = true

# Performance settings
default_lease_ttl = "768h"
max_lease_ttl = "8760h"
EOF
    
    # Set permissions
    chown root:$VAULT_USER "$VAULT_CONFIG_DIR/vault.hcl"
    chmod 640 "$VAULT_CONFIG_DIR/vault.hcl"
    
    print_success "Vault configuration created"
}

# Function to create systemd service
create_systemd_service() {
    print_status "Creating systemd service..."
    
    cat > /etc/systemd/system/vault.service <<EOF
[Unit]
Description=HashiCorp Vault
Documentation=https://www.vaultproject.io/docs/
Requires=network-online.target
After=network-online.target
ConditionFileNotEmpty=$VAULT_CONFIG_DIR/vault.hcl
StartLimitIntervalSec=60
StartLimitBurst=3

[Service]
Type=notify
User=$VAULT_USER
Group=$VAULT_USER
ProtectSystem=full
ProtectHome=read-only
PrivateTmp=yes
PrivateDevices=yes
SecureBits=keep-caps
AmbientCapabilities=CAP_IPC_LOCK
CapabilityBoundingSet=CAP_SYSLOG CAP_IPC_LOCK
NoNewPrivileges=yes
ExecStart=/usr/local/bin/vault server -config=$VAULT_CONFIG_DIR/vault.hcl
ExecReload=/bin/kill -HUP \$MAINPID
KillMode=process
Restart=on-failure
RestartSec=5
TimeoutStopSec=30
StartLimitInterval=60
StartLimitBurst=3
LimitNOFILE=65536
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd and enable service
    systemctl daemon-reload
    systemctl enable vault
    
    print_success "Systemd service created and enabled"
}

# Function to start Vault
start_vault() {
    print_status "Starting Vault service..."
    
    systemctl start vault
    
    # Wait for Vault to start
    sleep 5
    
    if systemctl is-active --quiet vault; then
        print_success "Vault service started successfully"
    else
        print_error "Failed to start Vault service"
        systemctl status vault
        exit 1
    fi
}

# Function to initialize Vault
initialize_vault() {
    print_status "Initializing Vault..."
    
    # Set Vault address
    export VAULT_ADDR="https://127.0.0.1:$VAULT_PORT"
    export VAULT_SKIP_VERIFY=1  # For self-signed certificate
    
    # Wait for Vault to be ready
    local retries=30
    while [[ $retries -gt 0 ]]; do
        if vault status &>/dev/null; then
            break
        fi
        print_status "Waiting for Vault to be ready... ($retries retries left)"
        sleep 2
        ((retries--))
    done
    
    if [[ $retries -eq 0 ]]; then
        print_error "Vault failed to become ready"
        exit 1
    fi
    
    # Check if already initialized
    if vault status | grep -q "Initialized.*true"; then
        print_warning "Vault is already initialized"
        return
    fi
    
    # Initialize Vault
    print_status "Initializing Vault with 5 key shares and threshold of 3..."
    vault operator init -key-shares=5 -key-threshold=3 -format=json > /tmp/vault-init.json
    
    if [[ $? -eq 0 ]]; then
        print_success "Vault initialized successfully"
        
        # Extract keys and root token
        UNSEAL_KEYS=($(jq -r '.unseal_keys_b64[]' /tmp/vault-init.json))
        ROOT_TOKEN=$(jq -r '.root_token' /tmp/vault-init.json)
        
        # Save to secure location
        mkdir -p "$VAULT_HOME/keys"
        cp /tmp/vault-init.json "$VAULT_HOME/keys/"
        chown $VAULT_USER:$VAULT_USER "$VAULT_HOME/keys/vault-init.json"
        chmod 600 "$VAULT_HOME/keys/vault-init.json"
        
        # Unseal Vault
        print_status "Unsealing Vault..."
        for i in {0..2}; do
            vault operator unseal "${UNSEAL_KEYS[$i]}"
        done
        
        print_success "Vault unsealed successfully"
        
        # Display important information
        echo
        print_success "=== VAULT INITIALIZATION COMPLETE ==="
        echo -e "${GREEN}Root Token:${NC} $ROOT_TOKEN"
        echo -e "${GREEN}Unseal Keys:${NC}"
        for i in "${!UNSEAL_KEYS[@]}"; do
            echo -e "  Key $((i+1)): ${UNSEAL_KEYS[$i]}"
        done
        echo
        print_warning "IMPORTANT: Save these keys and token in a secure location!"
        print_warning "The keys are also saved in: $VAULT_HOME/keys/vault-init.json"
        echo
        
        # Cleanup temporary file
        rm -f /tmp/vault-init.json
    else
        print_error "Failed to initialize Vault"
        exit 1
    fi
}

# Function to setup KV secrets engine
setup_secrets_engine() {
    print_status "Setting up KV secrets engine..."
    
    # Authenticate with root token
    if [[ -z "$ROOT_TOKEN" ]]; then
        if [[ -f "$VAULT_HOME/keys/vault-init.json" ]]; then
            ROOT_TOKEN=$(jq -r '.root_token' "$VAULT_HOME/keys/vault-init.json")
        else
            print_error "Root token not found. Please provide it manually."
            read -s -p "Enter root token: " ROOT_TOKEN
            echo
        fi
    fi
    
    export VAULT_TOKEN="$ROOT_TOKEN"
    
    # Enable KV v2 secrets engine
    vault secrets enable -path=secret kv-v2
    
    print_success "KV secrets engine enabled at path: secret/"
}

# Function to create initial secrets
create_initial_secrets() {
    print_status "Creating initial secrets for R3MES..."
    
    # Generate secure random values
    JWT_SECRET=$(openssl rand -base64 32)
    DB_PASSWORD=$(openssl rand -base64 24)
    API_SECRET=$(openssl rand -base64 32)
    
    # Create database secrets
    vault kv put secret/r3mes/database \
        user="r3mes" \
        password="$DB_PASSWORD" \
        host="localhost" \
        port="5432" \
        database="r3mes"
    
    # Create JWT secrets
    vault kv put secret/r3mes/jwt \
        secret_key="$JWT_SECRET" \
        algorithm="HS256" \
        expiration_hours="24"
    
    # Create API secrets
    vault kv put secret/r3mes/api \
        secret_key="$API_SECRET"
    
    print_success "Initial secrets created"
    
    # Display database password for manual setup
    echo
    print_warning "=== DATABASE SETUP REQUIRED ==="
    echo -e "${YELLOW}Database Password:${NC} $DB_PASSWORD"
    echo -e "${YELLOW}Please use this password when setting up PostgreSQL${NC}"
    echo
}

# Function to create Vault policies
create_policies() {
    print_status "Creating Vault policies..."
    
    # R3MES application policy
    cat > /tmp/r3mes-policy.hcl <<EOF
# R3MES Application Policy
# Allows read/write access to R3MES secrets

path "secret/data/r3mes/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

path "secret/metadata/r3mes/*" {
  capabilities = ["list"]
}
EOF
    
    vault policy write r3mes-app /tmp/r3mes-policy.hcl
    
    # Cleanup
    rm -f /tmp/r3mes-policy.hcl
    
    print_success "Vault policies created"
}

# Function to create application token
create_app_token() {
    print_status "Creating application token..."
    
    # Create token for R3MES application
    APP_TOKEN=$(vault token create \
        -policy=r3mes-app \
        -ttl=8760h \
        -renewable=true \
        -display-name="r3mes-backend" \
        -format=json | jq -r '.auth.client_token')
    
    if [[ -n "$APP_TOKEN" ]]; then
        print_success "Application token created"
        
        # Save token to file
        echo "$APP_TOKEN" > "$VAULT_HOME/keys/app-token"
        chown $VAULT_USER:$VAULT_USER "$VAULT_HOME/keys/app-token"
        chmod 600 "$VAULT_HOME/keys/app-token"
        
        echo
        print_success "=== APPLICATION TOKEN ==="
        echo -e "${GREEN}Token:${NC} $APP_TOKEN"
        echo -e "${GREEN}Saved to:${NC} $VAULT_HOME/keys/app-token"
        echo
        print_warning "Use this token in your R3MES application configuration"
        echo -e "${YELLOW}Set environment variable:${NC} VAULT_TOKEN=$APP_TOKEN"
        echo
    else
        print_error "Failed to create application token"
        exit 1
    fi
}

# Function to setup firewall
setup_firewall() {
    print_status "Setting up firewall rules..."
    
    # Check if ufw is available
    if command -v ufw &> /dev/null; then
        # Allow Vault port
        ufw allow $VAULT_PORT/tcp comment "HashiCorp Vault"
        print_success "UFW firewall rule added for port $VAULT_PORT"
    elif command -v firewall-cmd &> /dev/null; then
        # CentOS/RHEL firewall
        firewall-cmd --permanent --add-port=$VAULT_PORT/tcp
        firewall-cmd --reload
        print_success "Firewalld rule added for port $VAULT_PORT"
    else
        print_warning "No firewall detected. Please manually allow port $VAULT_PORT"
    fi
}

# Function to create environment file
create_env_file() {
    print_status "Creating environment file..."
    
    cat > /opt/vault/r3mes-vault.env <<EOF
# R3MES Vault Configuration
# Source this file in your application environment

export VAULT_ADDR="https://127.0.0.1:$VAULT_PORT"
export VAULT_TOKEN="$APP_TOKEN"
export VAULT_SKIP_VERIFY=1
export VAULT_MOUNT_POINT="secret"
export VAULT_PATH_PREFIX="r3mes"
EOF
    
    chown $VAULT_USER:$VAULT_USER /opt/vault/r3mes-vault.env
    chmod 640 /opt/vault/r3mes-vault.env
    
    print_success "Environment file created: /opt/vault/r3mes-vault.env"
}

# Function to display final instructions
display_final_instructions() {
    echo
    print_success "=== VAULT SETUP COMPLETE ==="
    echo
    echo -e "${GREEN}Vault Address:${NC} https://127.0.0.1:$VAULT_PORT"
    echo -e "${GREEN}Vault UI:${NC} https://127.0.0.1:$VAULT_PORT/ui"
    echo -e "${GREEN}Configuration:${NC} $VAULT_CONFIG_DIR/vault.hcl"
    echo -e "${GREEN}Data Directory:${NC} $VAULT_DATA_DIR"
    echo -e "${GREEN}Log Directory:${NC} $VAULT_LOG_DIR"
    echo
    echo -e "${YELLOW}Important Files:${NC}"
    echo -e "  Root Token & Unseal Keys: $VAULT_HOME/keys/vault-init.json"
    echo -e "  Application Token: $VAULT_HOME/keys/app-token"
    echo -e "  Environment File: /opt/vault/r3mes-vault.env"
    echo
    echo -e "${YELLOW}Service Management:${NC}"
    echo -e "  Start:   systemctl start vault"
    echo -e "  Stop:    systemctl stop vault"
    echo -e "  Status:  systemctl status vault"
    echo -e "  Logs:    journalctl -u vault -f"
    echo
    echo -e "${YELLOW}Next Steps:${NC}"
    echo -e "  1. Source the environment file: source /opt/vault/r3mes-vault.env"
    echo -e "  2. Test Vault access: vault status"
    echo -e "  3. Configure your R3MES application with the environment variables"
    echo -e "  4. Set up PostgreSQL with the generated database password"
    echo
    print_warning "SECURITY REMINDERS:"
    echo -e "  - Keep the unseal keys and root token secure"
    echo -e "  - Regularly backup the Vault data directory"
    echo -e "  - Monitor Vault logs for security events"
    echo -e "  - Consider setting up Vault auto-unseal for production"
    echo
}

# Main execution
main() {
    print_status "Starting HashiCorp Vault setup for R3MES..."
    
    check_root
    detect_os
    install_dependencies
    create_vault_user
    install_vault
    create_directories
    generate_tls_certificates
    create_vault_config
    create_systemd_service
    start_vault
    initialize_vault
    setup_secrets_engine
    create_initial_secrets
    create_policies
    create_app_token
    setup_firewall
    create_env_file
    display_final_instructions
    
    print_success "Vault setup completed successfully!"
}

# Run main function
main "$@"
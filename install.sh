#!/bin/bash
# R3MES Unified Installation Script
# Installs all required dependencies and components for R3MES
# Supports: Linux (Ubuntu/Debian/Fedora) and macOS

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/debian_version ]; then
            OS="debian"
        elif [ -f /etc/redhat-release ]; then
            OS="fedora"
        else
            OS="linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    else
        echo -e "${RED}Unsupported OS: $OSTYPE${NC}"
        exit 1
    fi
    echo -e "${BLUE}Detected OS: $OS${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install package manager dependencies
install_package_manager() {
    echo -e "${BLUE}Installing package manager dependencies...${NC}"
    if [ "$OS" == "debian" ]; then
        sudo apt-get update
        sudo apt-get install -y curl wget git build-essential
    elif [ "$OS" == "fedora" ]; then
        sudo dnf install -y curl wget git gcc gcc-c++ make
    elif [ "$OS" == "macos" ]; then
        if ! command_exists brew; then
            echo -e "${YELLOW}Installing Homebrew...${NC}"
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
    fi
}

# Check and install Go
check_go() {
    echo -e "${BLUE}Checking Go installation...${NC}"
    if command_exists go; then
        GO_VERSION=$(go version | awk '{print $3}' | sed 's/go//')
        echo -e "${GREEN}Go is installed: $GO_VERSION${NC}"
        # Check if version is >= 1.21
        REQUIRED_VERSION="1.21"
        if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$GO_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
            echo -e "${YELLOW}Go version $GO_VERSION is too old. Required: >= $REQUIRED_VERSION${NC}"
            install_go
        fi
    else
        echo -e "${YELLOW}Go is not installed. Installing...${NC}"
        install_go
    fi
}

install_go() {
    GO_VERSION="1.21.5"
    echo -e "${BLUE}Installing Go $GO_VERSION...${NC}"
    
    if [ "$OS" == "macos" ]; then
        brew install go
    else
        cd /tmp
        wget "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz"
        sudo rm -rf /usr/local/go
        sudo tar -C /usr/local -xzf "go${GO_VERSION}.linux-amd64.tar.gz"
        export PATH=$PATH:/usr/local/go/bin
        echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
        rm "go${GO_VERSION}.linux-amd64.tar.gz"
    fi
    
    echo -e "${GREEN}Go installed successfully${NC}"
}

# Check and install Python
check_python() {
    echo -e "${BLUE}Checking Python installation...${NC}"
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version | awk '{print $2}')
        echo -e "${GREEN}Python is installed: $PYTHON_VERSION${NC}"
        # Check if version is >= 3.10
        REQUIRED_VERSION="3.10"
        if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
            echo -e "${YELLOW}Python version $PYTHON_VERSION is too old. Required: >= $REQUIRED_VERSION${NC}"
            install_python
        fi
    else
        echo -e "${YELLOW}Python is not installed. Installing...${NC}"
        install_python
    fi
}

install_python() {
    if [ "$OS" == "macos" ]; then
        brew install python@3.11
    else
        if [ "$OS" == "debian" ]; then
            sudo apt-get install -y python3.11 python3.11-venv python3-pip
        elif [ "$OS" == "fedora" ]; then
            sudo dnf install -y python3.11 python3-pip
        fi
    fi
    echo -e "${GREEN}Python installed successfully${NC}"
}

# Check and install Node.js
check_nodejs() {
    echo -e "${BLUE}Checking Node.js installation...${NC}"
    if command_exists node; then
        NODE_VERSION=$(node --version | sed 's/v//')
        echo -e "${GREEN}Node.js is installed: $NODE_VERSION${NC}"
        # Check if version is >= 18
        REQUIRED_VERSION="18.0.0"
        if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$NODE_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
            echo -e "${YELLOW}Node.js version $NODE_VERSION is too old. Required: >= $REQUIRED_VERSION${NC}"
            install_nodejs
        fi
    else
        echo -e "${YELLOW}Node.js is not installed. Installing...${NC}"
        install_nodejs
    fi
}

install_nodejs() {
    echo -e "${BLUE}Installing Node.js via nvm...${NC}"
    if [ ! -d "$HOME/.nvm" ]; then
        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
        export NVM_DIR="$HOME/.nvm"
        [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
    fi
    
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
    
    nvm install 20
    nvm use 20
    nvm alias default 20
    
    echo -e "${GREEN}Node.js installed successfully${NC}"
}

# Check GPU drivers
check_gpu() {
    echo -e "${BLUE}Checking GPU drivers...${NC}"
    
    # Check NVIDIA
    if command_exists nvidia-smi; then
        NVIDIA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
        echo -e "${GREEN}NVIDIA GPU detected: Driver $NVIDIA_VERSION${NC}"
        
        # Check CUDA
        if command_exists nvcc; then
            CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
            echo -e "${GREEN}CUDA installed: $CUDA_VERSION${NC}"
        else
            echo -e "${YELLOW}CUDA not found. Install from: https://developer.nvidia.com/cuda-downloads${NC}"
        fi
    fi
    
    # Check AMD
    if command_exists rocm-smi; then
        echo -e "${GREEN}AMD GPU detected (ROCm)${NC}"
    fi
    
    # Check Intel
    if command_exists intel_gpu_top; then
        echo -e "${GREEN}Intel GPU detected${NC}"
    fi
    
    if ! command_exists nvidia-smi && ! command_exists rocm-smi && ! command_exists intel_gpu_top; then
        echo -e "${YELLOW}No GPU detected. R3MES can run in CPU-only mode, but mining will be slower.${NC}"
    fi
}

# Check and install Docker (optional)
check_docker() {
    echo -e "${BLUE}Checking Docker installation...${NC}"
    if command_exists docker; then
        DOCKER_VERSION=$(docker --version | awk '{print $3}' | sed 's/,//')
        echo -e "${GREEN}Docker is installed: $DOCKER_VERSION${NC}"
    else
        echo -e "${YELLOW}Docker is not installed. (Optional - only needed for containerized deployments)${NC}"
        read -p "Install Docker? (y/n): " INSTALL_DOCKER
        if [ "$INSTALL_DOCKER" == "y" ]; then
            install_docker
        fi
    fi
}

install_docker() {
    if [ "$OS" == "macos" ]; then
        brew install --cask docker
    elif [ "$OS" == "debian" ]; then
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        rm get-docker.sh
    elif [ "$OS" == "fedora" ]; then
        sudo dnf install -y docker
        sudo systemctl enable docker
        sudo systemctl start docker
    fi
    echo -e "${GREEN}Docker installed successfully${NC}"
}

# Check and install IPFS
check_ipfs() {
    echo -e "${BLUE}Checking IPFS installation...${NC}"
    if command_exists ipfs; then
        IPFS_VERSION=$(ipfs version --number)
        echo -e "${GREEN}IPFS is installed: $IPFS_VERSION${NC}"
    else
        echo -e "${YELLOW}IPFS is not installed. Installing...${NC}"
        install_ipfs
    fi
}

install_ipfs() {
    IPFS_VERSION="0.20.0"
    echo -e "${BLUE}Installing IPFS $IPFS_VERSION...${NC}"
    
    cd /tmp
    if [ "$OS" == "macos" ]; then
        wget "https://dist.ipfs.tech/kubo/v${IPFS_VERSION}/kubo_v${IPFS_VERSION}_darwin-amd64.tar.gz"
        tar -xzf "kubo_v${IPFS_VERSION}_darwin-amd64.tar.gz"
        sudo cp kubo/ipfs /usr/local/bin/
        rm -rf kubo "kubo_v${IPFS_VERSION}_darwin-amd64.tar.gz"
    else
        wget "https://dist.ipfs.tech/kubo/v${IPFS_VERSION}/kubo_v${IPFS_VERSION}_linux-amd64.tar.gz"
        tar -xzf "kubo_v${IPFS_VERSION}_linux-amd64.tar.gz"
        sudo cp kubo/ipfs /usr/local/bin/
        rm -rf kubo "kubo_v${IPFS_VERSION}_linux-amd64.tar.gz"
    fi
    
    echo -e "${GREEN}IPFS installed successfully${NC}"
}

# Check disk space
check_disk_space() {
    echo -e "${BLUE}Checking disk space...${NC}"
    AVAILABLE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    REQUIRED_SPACE=50  # GB
    
    if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
        echo -e "${RED}Insufficient disk space. Required: ${REQUIRED_SPACE}GB, Available: ${AVAILABLE_SPACE}GB${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Disk space OK: ${AVAILABLE_SPACE}GB available${NC}"
}

# Check RAM
check_ram() {
    echo -e "${BLUE}Checking RAM...${NC}"
    if [ "$OS" == "macos" ]; then
        TOTAL_RAM=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
    else
        TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
    fi
    
    REQUIRED_RAM=8  # GB
    
    if [ "$TOTAL_RAM" -lt "$REQUIRED_RAM" ]; then
        echo -e "${YELLOW}Low RAM detected: ${TOTAL_RAM}GB. Recommended: >= ${REQUIRED_RAM}GB${NC}"
    else
        echo -e "${GREEN}RAM OK: ${TOTAL_RAM}GB${NC}"
    fi
}

# Main installation function
main() {
    echo -e "${GREEN}=========================================="
    echo "R3MES Unified Installation Script"
    echo "==========================================${NC}"
    echo ""
    
    detect_os
    install_package_manager
    
    echo ""
    echo -e "${BLUE}Checking system requirements...${NC}"
    check_disk_space
    check_ram
    check_gpu
    
    echo ""
    echo -e "${BLUE}Installing dependencies...${NC}"
    check_go
    check_python
    check_nodejs
    check_ipfs
    check_docker
    
    echo ""
    echo -e "${GREEN}=========================================="
    echo "Installation Complete!"
    echo "==========================================${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Run 'source ~/.bashrc' or restart your terminal"
    echo "  2. Navigate to R3MES directory"
    echo "  3. Run component-specific install scripts:"
    echo "     - scripts/install_founder.sh (for validators)"
    echo "     - scripts/install_miner_pypi.sh (for miners)"
    echo "     - scripts/install_web_dashboard.sh (for web dashboard)"
    echo ""
}

# Run main function
main


# R3MES Installation Guide

This guide provides instructions for installing R3MES and all its dependencies.

## Quick Start

### Linux / macOS

```bash
# Run the unified installation script
./install.sh
```

### Windows

```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\install.ps1
```

## What Gets Installed

The unified installation script checks and installs:

1. **Go** (>= 1.21) - For blockchain node
2. **Python** (>= 3.10) - For miner engine
3. **Node.js** (>= 18) - For web dashboard
4. **IPFS** - For decentralized storage
5. **Docker** (optional) - For containerized deployments
6. **GPU Drivers** - Detects and verifies NVIDIA/AMD/Intel GPUs

## System Requirements

### Minimum Requirements

- **OS**: Linux (Ubuntu/Debian/Fedora), macOS, or Windows 10/11
- **CPU**: 4+ cores
- **RAM**: 8GB (16GB recommended)
- **Disk**: 50GB free space
- **GPU**: Optional (NVIDIA/AMD/Intel supported)

### Recommended Requirements

- **OS**: Ubuntu 22.04 LTS or macOS 13+
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **Disk**: 100GB+ SSD
- **GPU**: NVIDIA GPU with 8GB+ VRAM (for mining)

## Manual Installation

If you prefer to install dependencies manually:

### Linux (Ubuntu/Debian)

```bash
# Update package list
sudo apt-get update

# Install Go
wget https://go.dev/dl/go1.21.5.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.21.5.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin

# Install Python
sudo apt-get install -y python3.11 python3.11-venv python3-pip

# Install Node.js (via nvm)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 20
nvm use 20

# Install IPFS
wget https://dist.ipfs.tech/kubo/v0.20.0/kubo_v0.20.0_linux-amd64.tar.gz
tar -xzf kubo_v0.20.0_linux-amd64.tar.gz
sudo cp kubo/ipfs /usr/local/bin/
```

### macOS

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install go python@3.11 node@20 ipfs
```

### Windows

```powershell
# Install Chocolatey (if not installed)
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install dependencies
choco install golang python311 nodejs-lts ipfs -y
```

## GPU Setup

### NVIDIA GPU

1. Install NVIDIA drivers from [NVIDIA website](https://www.nvidia.com/Download/index.aspx)
2. Install CUDA Toolkit from [NVIDIA CUDA downloads](https://developer.nvidia.com/cuda-downloads)
3. Verify installation:
   ```bash
   nvidia-smi
   nvcc --version
   ```

### AMD GPU

1. Install ROCm from [AMD ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html)
2. Verify installation:
   ```bash
   rocm-smi
   ```

### Intel GPU

Intel GPU support is automatic on Linux. For Windows, install Intel Graphics drivers from Intel's website.

## Component-Specific Installation

After running the unified installer, install specific components:

### Blockchain Node (Validator/Founder)

```bash
# For validators
./scripts/install_validator.sh

# For founders/admins
./scripts/install_founder.sh
```

### Miner Engine

```bash
./scripts/install_miner_pypi.sh
```

### Web Dashboard

```bash
./scripts/install_web_dashboard.sh
```

### Desktop Launcher

```bash
./scripts/install_desktop_launcher.sh
```

## Troubleshooting

### Go Not Found

After installing Go, add it to your PATH:

```bash
# Linux/macOS
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
source ~/.bashrc

# Windows
# Add C:\Program Files\Go\bin to PATH via System Properties
```

### Python Not Found

Ensure Python 3.10+ is in your PATH:

```bash
# Check version
python3 --version

# If not found, create symlink (Linux)
sudo ln -s /usr/bin/python3.11 /usr/bin/python3
```

### Node.js Not Found

If using nvm, ensure it's loaded:

```bash
# Add to ~/.bashrc or ~/.zshrc
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
```

### IPFS Not Starting

```bash
# Initialize IPFS (first time only)
ipfs init

# Start IPFS daemon
ipfs daemon
```

### GPU Not Detected

1. Verify GPU is recognized by OS
2. Install appropriate drivers
3. For NVIDIA, ensure CUDA is installed
4. Restart system after driver installation

## Verification

After installation, verify all components:

```bash
# Check Go
go version

# Check Python
python3 --version

# Check Node.js
node --version
npm --version

# Check IPFS
ipfs version

# Check GPU (NVIDIA)
nvidia-smi

# Check GPU (AMD)
rocm-smi
```

## Next Steps

1. **Initialize Blockchain**: Run `./scripts/init_chain.sh` (for founders)
2. **Start Node**: Run `./scripts/start_node.sh`
3. **Start Miner**: Run `./scripts/start_miner.sh`
4. **Start Web Dashboard**: Run `cd web-dashboard && npm run dev`

For detailed component documentation, see:
- [Blockchain Node Setup](../docs/01_blockchain_infrastructure.md)
- [Miner Engine Setup](../docs/02_ai_training_system.md)
- [Web Dashboard Setup](../docs/08_web_dashboard_command_center.md)
- [Desktop Launcher Setup](../docs/10_desktop_launcher.md)


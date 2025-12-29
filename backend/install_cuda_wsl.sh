#!/bin/bash
# CUDA Toolkit Installation Script for WSL
# This installs CUDA runtime libraries needed for bitsandbytes

set -e

echo "🔧 Installing CUDA Toolkit for WSL..."
echo ""

# Check if running on WSL
if [ ! -f /proc/version ] || ! grep -q Microsoft /proc/version; then
    echo "⚠️  This script is designed for WSL. Continuing anyway..."
fi

# Detect Ubuntu version
if [ -f /etc/os-release ]; then
    . /etc/os-release
    UBUNTU_VERSION=$(echo $VERSION_ID | cut -d. -f1)
    echo "✅ Detected Ubuntu version: $UBUNTU_VERSION"
else
    echo "❌ Cannot detect Ubuntu version"
    exit 1
fi

# Determine CUDA version from PyTorch
echo "🔍 Detecting CUDA version from PyTorch..."
CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "12.1")
echo "   PyTorch CUDA version: $CUDA_VERSION"

# Extract major.minor version
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
CUDA_VERSION_SHORT="${CUDA_MAJOR}.${CUDA_MINOR}"

echo "   Installing CUDA ${CUDA_VERSION_SHORT} runtime..."

# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA runtime libraries (minimal installation)
echo "📦 Installing CUDA runtime libraries..."
sudo apt-get install -y cuda-runtime-${CUDA_MAJOR}-${CUDA_MINOR} || \
sudo apt-get install -y cuda-toolkit-${CUDA_MAJOR}-${CUDA_MINOR} || \
sudo apt-get install -y cuda-libraries-${CUDA_MAJOR}-${CUDA_MINOR} || {
    echo "⚠️  Specific CUDA version not found, trying generic installation..."
    sudo apt-get install -y cuda-runtime
}

# Add CUDA to PATH and LD_LIBRARY_PATH
if [ -d "/usr/local/cuda-${CUDA_VERSION_SHORT}" ]; then
    CUDA_PATH="/usr/local/cuda-${CUDA_VERSION_SHORT}"
elif [ -d "/usr/local/cuda" ]; then
    CUDA_PATH="/usr/local/cuda"
else
    echo "⚠️  CUDA installation path not found"
    exit 1
fi

echo "✅ CUDA installed at: $CUDA_PATH"

# Add to .bashrc
if ! grep -q "CUDA_PATH" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# CUDA Path" >> ~/.bashrc
    echo "export CUDA_PATH=$CUDA_PATH" >> ~/.bashrc
    echo "export PATH=\$CUDA_PATH/bin:\$PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=\$CUDA_PATH/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
    echo "✅ Added CUDA to ~/.bashrc"
fi

# Verify installation
echo ""
echo "🔍 Verifying CUDA installation..."
if [ -f "$CUDA_PATH/lib64/libcudart.so" ]; then
    echo "✅ libcudart.so found at: $CUDA_PATH/lib64/libcudart.so"
else
    echo "⚠️  libcudart.so not found. Trying to locate..."
    find /usr -name "libcudart.so*" 2>/dev/null | head -3 || echo "   Not found"
fi

echo ""
echo "✅ CUDA installation complete!"
echo "   Please restart your terminal or run: source ~/.bashrc"
echo "   Then try starting the backend again."


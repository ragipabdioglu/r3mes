#!/bin/bash
# R3MES Backend Startup Script with CUDA Support for WSL

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# CUDA Library Path Setup for WSL
# Try to find CUDA runtime library
CUDA_LIB_PATHS=(
    "/usr/local/cuda/lib64"
    "/usr/lib/wsl/lib"
    "/usr/lib/x86_64-linux-gnu"
    "$(python3 -c 'import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))' 2>/dev/null)"
)

# Add PyTorch lib path to LD_LIBRARY_PATH
TORCH_LIB_PATH=$(python3 -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null || echo "")
if [ -n "$TORCH_LIB_PATH" ] && [ -d "$TORCH_LIB_PATH" ]; then
    export LD_LIBRARY_PATH="$TORCH_LIB_PATH:$LD_LIBRARY_PATH"
    echo "✅ Added PyTorch lib path to LD_LIBRARY_PATH: $TORCH_LIB_PATH"
fi

# Try to find libcudart.so
FOUND_CUDA_LIB=""
for path in "${CUDA_LIB_PATHS[@]}"; do
    if [ -d "$path" ] && [ -f "$path/libcudart.so"* ] 2>/dev/null; then
        FOUND_CUDA_LIB="$path"
        export LD_LIBRARY_PATH="$path:$LD_LIBRARY_PATH"
        echo "✅ Found CUDA library at: $path"
        break
    fi
done

# If CUDA runtime not found, try to use PyTorch's bundled CUDA
if [ -z "$FOUND_CUDA_LIB" ]; then
    echo "⚠️  libcudart.so not found in standard locations"
    echo "   Attempting to use PyTorch's CUDA libraries..."
    
    # Set environment variable to help bitsandbytes find CUDA
    export BITSANDBYTES_NOWELCOME=1
    
    # Try to find CUDA in PyTorch installation
    TORCH_CUDA_PATH=$(python3 -c "import torch; import os; lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib'); print(lib_path)" 2>/dev/null || echo "")
    if [ -n "$TORCH_CUDA_PATH" ] && [ -d "$TORCH_CUDA_PATH" ]; then
        export LD_LIBRARY_PATH="$TORCH_CUDA_PATH:$LD_LIBRARY_PATH"
        echo "✅ Using PyTorch CUDA libraries from: $TORCH_CUDA_PATH"
    fi
fi

# Check if CUDA is available
echo "🔍 Checking CUDA availability..."
python3 -c "import torch; print('PyTorch CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')" 2>&1 || true

# Set GPU device
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Start backend
echo "🚀 Starting R3MES Backend..."
echo "   Backend will be available at: http://localhost:8000"
echo "   API docs at: http://localhost:8000/docs"
echo ""

python3 -m app.main


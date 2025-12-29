#!/bin/bash
#
# Build Windows EXE Script for R3MES Desktop Launcher
#
# Builds the Tauri desktop launcher as a Windows executable (.exe) or installer (.msi).
# Validates the build output and prepares it for distribution.
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LAUNCHER_DIR="$PROJECT_ROOT/desktop-launcher-tauri"

# Default values
BUILD_TYPE="exe"  # exe or msi
VALIDATE=true
OUTPUT_DIR="$LAUNCHER_DIR/dist"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --no-validate)
            VALIDATE=false
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--type exe|msi] [--no-validate] [--output-dir <dir>]"
            echo ""
            echo "Options:"
            echo "  --type <type>      Build type: 'exe' or 'msi' (default: exe)"
            echo "  --no-validate      Skip build validation"
            echo "  --output-dir <dir> Output directory (default: desktop-launcher-tauri/dist)"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "R3MES Desktop Launcher - Windows Build"
echo "=========================================="
echo "Build type: $BUILD_TYPE"
echo "Output directory: $OUTPUT_DIR"
echo "Working directory: $LAUNCHER_DIR"
echo "=========================================="
echo ""

# Check if running on Windows or using cross-compilation
if [[ "$OSTYPE" != "msys" && "$OSTYPE" != "win32" && "$OSTYPE" != "cygwin" ]]; then
    echo -e "${YELLOW}Warning: Not running on Windows. Cross-compilation may be required.${NC}"
    echo "Make sure Rust target 'x86_64-pc-windows-msvc' is installed:"
    echo "  rustup target add x86_64-pc-windows-msvc"
    echo ""
fi

# Check prerequisites
echo "Checking prerequisites..."

# Check Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}Error: Node.js is not installed${NC}"
    exit 1
fi
NODE_VERSION=$(node --version)
echo "✅ Node.js: $NODE_VERSION"

# Check npm
if ! command -v npm &> /dev/null; then
    echo -e "${RED}Error: npm is not installed${NC}"
    exit 1
fi
NPM_VERSION=$(npm --version)
echo "✅ npm: $NPM_VERSION"

# Check Rust
if ! command -v rustc &> /dev/null; then
    echo -e "${RED}Error: Rust is not installed${NC}"
    exit 1
fi
RUST_VERSION=$(rustc --version)
echo "✅ Rust: $RUST_VERSION"

# Check Rust target for Windows
if [[ "$OSTYPE" != "msys" && "$OSTYPE" != "win32" && "$OSTYPE" != "cygwin" ]]; then
    if ! rustup target list --installed | grep -q "x86_64-pc-windows-msvc"; then
        echo -e "${YELLOW}Windows target not installed. Installing...${NC}"
        rustup target add x86_64-pc-windows-msvc
    fi
    echo "✅ Windows target installed"
fi

echo ""

# Navigate to launcher directory
cd "$LAUNCHER_DIR"

# Install dependencies
echo "Installing dependencies..."
if [ ! -d "node_modules" ]; then
    npm install
else
    echo "✅ Dependencies already installed"
fi
echo ""

# Build frontend
echo "Building frontend..."
npm run build
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Frontend build failed${NC}"
    exit 1
fi
echo "✅ Frontend build completed"
echo ""

# Build Tauri application
echo "Building Tauri application..."
echo "This may take several minutes..."

if [ "$BUILD_TYPE" = "msi" ]; then
    # Build MSI installer
    npm run tauri build -- --bundles msi
else
    # Build EXE (default)
    npm run tauri build
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Tauri build failed${NC}"
    exit 1
fi
echo "✅ Tauri build completed"
echo ""

# Find output files
BUILD_OUTPUT_DIR="$LAUNCHER_DIR/src-tauri/target/release/bundle"
EXE_FILE=""
MSI_FILE=""

if [ -d "$BUILD_OUTPUT_DIR" ]; then
    # Find EXE file
    EXE_FILE=$(find "$BUILD_OUTPUT_DIR" -name "*.exe" -type f | head -n 1)
    
    # Find MSI file
    if [ "$BUILD_TYPE" = "msi" ]; then
        MSI_FILE=$(find "$BUILD_OUTPUT_DIR/msi" -name "*.msi" -type f | head -n 1)
    fi
fi

# Copy output files
mkdir -p "$OUTPUT_DIR"

if [ -n "$EXE_FILE" ] && [ -f "$EXE_FILE" ]; then
    EXE_NAME=$(basename "$EXE_FILE")
    cp "$EXE_FILE" "$OUTPUT_DIR/"
    echo "✅ EXE copied to: $OUTPUT_DIR/$EXE_NAME"
    
    # Get file size
    EXE_SIZE=$(stat -f%z "$EXE_FILE" 2>/dev/null || stat -c%s "$EXE_FILE" 2>/dev/null)
    EXE_SIZE_MB=$(awk "BEGIN {printf \"%.2f\", $EXE_SIZE/1024/1024}")
    echo "   Size: ${EXE_SIZE_MB} MB"
fi

if [ -n "$MSI_FILE" ] && [ -f "$MSI_FILE" ]; then
    MSI_NAME=$(basename "$MSI_FILE")
    cp "$MSI_FILE" "$OUTPUT_DIR/"
    echo "✅ MSI copied to: $OUTPUT_DIR/$MSI_NAME"
    
    # Get file size
    MSI_SIZE=$(stat -f%z "$MSI_FILE" 2>/dev/null || stat -c%s "$MSI_FILE" 2>/dev/null)
    MSI_SIZE_MB=$(awk "BEGIN {printf \"%.2f\", $MSI_SIZE/1024/1024}")
    echo "   Size: ${MSI_SIZE_MB} MB"
fi

echo ""

# Validate build (if enabled)
if [ "$VALIDATE" = true ]; then
    echo "Validating build..."
    
    if [ -n "$EXE_FILE" ] && [ -f "$EXE_FILE" ]; then
        # Check if file exists and is not empty
        if [ ! -s "$EXE_FILE" ]; then
            echo -e "${RED}Error: EXE file is empty${NC}"
            exit 1
        fi
        
        # Try to get file info (Windows only)
        if command -v file &> /dev/null; then
            FILE_INFO=$(file "$EXE_FILE" 2>/dev/null || echo "")
            echo "File info: $FILE_INFO"
        fi
        
        echo "✅ EXE validation passed"
    fi
    
    if [ -n "$MSI_FILE" ] && [ -f "$MSI_FILE" ]; then
        if [ ! -s "$MSI_FILE" ]; then
            echo -e "${RED}Error: MSI file is empty${NC}"
            exit 1
        fi
        echo "✅ MSI validation passed"
    fi
else
    echo -e "${YELLOW}Skipping build validation${NC}"
fi

echo ""
echo "=========================================="
echo "✅ Build completed successfully!"
echo "=========================================="
if [ -n "$EXE_FILE" ]; then
    echo "EXE: $OUTPUT_DIR/$(basename "$EXE_FILE")"
fi
if [ -n "$MSI_FILE" ]; then
    echo "MSI: $OUTPUT_DIR/$(basename "$MSI_FILE")"
fi
echo ""
echo "Next steps:"
echo "1. Test the executable on a Windows system"
echo "2. Create a code signature (recommended for distribution)"
echo "3. Upload to release distribution platform"
echo "=========================================="


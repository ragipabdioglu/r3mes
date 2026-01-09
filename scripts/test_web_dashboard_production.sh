#!/bin/bash
#
# Test Web Dashboard Production Build
#
# Tests the production build of the web dashboard, including:
# - Build validation
# - URL configuration
# - API endpoint connectivity
# - CORS configuration
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
DASHBOARD_DIR="$PROJECT_ROOT/web-dashboard"

# Default values
BUILD_ONLY=false
START_SERVER=false
TEST_URLS=true
TEST_CORS=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --start-server)
            START_SERVER=true
            shift
            ;;
        --no-url-test)
            TEST_URLS=false
            shift
            ;;
        --no-cors-test)
            TEST_CORS=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--build-only] [--start-server] [--no-url-test] [--no-cors-test]"
            echo ""
            echo "Options:"
            echo "  --build-only      Only build, don't test"
            echo "  --start-server    Start production server after build"
            echo "  --no-url-test     Skip URL configuration tests"
            echo "  --no-cors-test    Skip CORS tests"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "R3MES Web Dashboard - Production Test"
echo "=========================================="
echo "Dashboard directory: $DASHBOARD_DIR"
echo "=========================================="
echo ""

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

echo ""

# Navigate to dashboard directory
cd "$DASHBOARD_DIR"

# Check for production environment file
if [ ! -f ".env.production" ]; then
    echo -e "${YELLOW}Warning: .env.production not found${NC}"
    echo "Copying from .env.production.example..."
    if [ -f ".env.production.example" ]; then
        cp .env.production.example .env.production
        echo "✅ Created .env.production from example"
        echo -e "${YELLOW}Please edit .env.production with actual production values${NC}"
    else
        echo -e "${RED}Error: .env.production.example not found${NC}"
        exit 1
    fi
else
    echo "✅ .env.production found"
fi

# Source production environment variables
if [ -f ".env.production" ]; then
    set -a
    source .env.production
    set +a
    echo "✅ Loaded .env.production"
fi

echo ""

# Check environment variables
echo "Checking environment variables..."

REQUIRED_VARS=(
    "NEXT_PUBLIC_BACKEND_URL"
    "NEXT_PUBLIC_API_URL"
)

MISSING_VARS=()
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var:-}" ]; then
        MISSING_VARS+=("$var")
    else
        echo "✅ $var=${!var}"
    fi
done

if [ ${#MISSING_VARS[@]} -gt 0 ]; then
    echo -e "${YELLOW}Warning: Missing environment variables: ${MISSING_VARS[*]}${NC}"
    echo "These should be set in .env.production"
fi

echo ""

# Install dependencies
echo "Installing dependencies..."
if [ ! -d "node_modules" ]; then
    npm install
else
    echo "✅ Dependencies already installed"
fi
echo ""

# Build production
echo "Building production bundle..."
echo "This may take several minutes..."

npm run build

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Production build failed${NC}"
    exit 1
fi

echo "✅ Production build completed"
echo ""

# Check build output
BUILD_DIR="$DASHBOARD_DIR/.next"
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}Error: Build output directory not found: $BUILD_DIR${NC}"
    exit 1
fi

echo "✅ Build output directory exists: $BUILD_DIR"

# Check for standalone output (if configured)
STANDALONE_DIR="$BUILD_DIR/standalone"
if [ -d "$STANDALONE_DIR" ]; then
    echo "✅ Standalone output found: $STANDALONE_DIR"
else
    echo -e "${YELLOW}Info: Standalone output not found (this is OK if not using standalone mode)${NC}"
fi

echo ""

# Test URL configuration (if enabled)
if [ "$TEST_URLS" = true ]; then
    echo "Testing URL configuration..."
    
    # Check if URLs are accessible (basic connectivity test)
    if [ -n "${NEXT_PUBLIC_BACKEND_URL:-}" ]; then
        BACKEND_HOST=$(echo "$NEXT_PUBLIC_BACKEND_URL" | sed -e 's|^[^/]*//||' -e 's|/.*$||' | cut -d: -f1)
        if [ "$BACKEND_HOST" != "localhost" ] && [ "$BACKEND_HOST" != "127.0.0.1" ]; then
            echo "Testing backend URL: $NEXT_PUBLIC_BACKEND_URL"
            if command -v curl &> /dev/null; then
                if curl -s --head --fail --max-time 5 "$NEXT_PUBLIC_BACKEND_URL/health" > /dev/null 2>&1; then
                    echo "✅ Backend URL is accessible"
                else
                    echo -e "${YELLOW}Warning: Backend URL is not accessible (may be expected if backend is not running)${NC}"
                fi
            else
                echo -e "${YELLOW}Warning: curl not available, skipping URL test${NC}"
            fi
        else
            echo "✅ Backend URL configured (localhost - will work when backend is running)"
        fi
    fi
    
    if [ -n "${NEXT_PUBLIC_API_URL:-}" ]; then
        API_HOST=$(echo "$NEXT_PUBLIC_API_URL" | sed -e 's|^[^/]*//||' -e 's|/.*$||' | cut -d: -f1)
        if [ "$API_HOST" != "localhost" ] && [ "$API_HOST" != "127.0.0.1" ]; then
            echo "Testing API URL: $NEXT_PUBLIC_API_URL"
            if command -v curl &> /dev/null; then
                if curl -s --head --fail --max-time 5 "$NEXT_PUBLIC_API_URL" > /dev/null 2>&1; then
                    echo "✅ API URL is accessible"
                else
                    echo -e "${YELLOW}Warning: API URL is not accessible (may be expected if API is not running)${NC}"
                fi
            else
                echo -e "${YELLOW}Warning: curl not available, skipping URL test${NC}"
            fi
        else
            echo "✅ API URL configured (localhost - will work when API is running)"
        fi
    fi
    
    echo ""
fi

# Test CORS configuration (if enabled)
if [ "$TEST_CORS" = true ]; then
    echo "Testing CORS configuration..."
    
    # Check next.config.js for CORS settings
    if [ -f "next.config.js" ]; then
        if grep -q "CORS\|cors\|Access-Control" next.config.js; then
            echo "✅ CORS configuration found in next.config.js"
        else
            echo -e "${YELLOW}Info: No explicit CORS configuration found (Next.js handles CORS via rewrites)${NC}"
        fi
    fi
    
    # Check for rewrites in next.config.js (API proxy)
    if grep -q "rewrites\|/api/" next.config.js; then
        echo "✅ API rewrites configuration found"
    fi
    
    echo ""
fi

# Summary
echo "=========================================="
echo "✅ Production test completed!"
echo "=========================================="
echo "Build output: $BUILD_DIR"
echo ""

if [ "$BUILD_ONLY" = true ]; then
    echo "Build-only mode: Skipping server start"
else
    echo "Next steps:"
    echo "1. Review build output in $BUILD_DIR"
    echo "2. Test production server: npm start"
    echo "3. Or deploy standalone output (if configured)"
    
    if [ "$START_SERVER" = true ]; then
        echo ""
        echo "Starting production server..."
        npm start
    fi
fi

echo "=========================================="


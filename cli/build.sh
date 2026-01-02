#!/bin/bash

# R3MES CLI Build Script
# Builds CLI for multiple platforms

set -e

echo "🔨 Building R3MES CLI..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Build information
VERSION=${VERSION:-"v0.1.0"}
BUILD_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
GIT_COMMIT=${GIT_COMMIT:-$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")}

# Build flags
LDFLAGS="-X main.Version=${VERSION} -X main.BuildTime=${BUILD_TIME} -X main.GitCommit=${GIT_COMMIT}"

# Create build directory
BUILD_DIR="build"
mkdir -p ${BUILD_DIR}

echo -e "${BLUE}Build Information:${NC}"
echo -e "  Version: ${GREEN}${VERSION}${NC}"
echo -e "  Build Time: ${GREEN}${BUILD_TIME}${NC}"
echo -e "  Git Commit: ${GREEN}${GIT_COMMIT}${NC}"
echo ""

# Change to CLI directory
cd cli/r3mes-cli

# Download dependencies
echo -e "${YELLOW}📦 Downloading dependencies...${NC}"
go mod download
go mod tidy

# Build for different platforms
platforms=(
    "linux/amd64"
    "linux/arm64"
    "darwin/amd64"
    "darwin/arm64"
    "windows/amd64"
)

for platform in "${platforms[@]}"; do
    IFS='/' read -r GOOS GOARCH <<< "$platform"
    
    output_name="r3mes-cli-${VERSION}-${GOOS}-${GOARCH}"
    if [ "$GOOS" = "windows" ]; then
        output_name="${output_name}.exe"
    fi
    
    echo -e "${YELLOW}🔨 Building for ${GOOS}/${GOARCH}...${NC}"
    
    env GOOS=$GOOS GOARCH=$GOARCH go build \
        -ldflags="${LDFLAGS}" \
        -o "../../${BUILD_DIR}/${output_name}" \
        .
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Successfully built ${output_name}${NC}"
    else
        echo -e "${RED}❌ Failed to build for ${GOOS}/${GOARCH}${NC}"
        exit 1
    fi
done

# Create checksums
cd ../../${BUILD_DIR}
echo -e "${YELLOW}📝 Creating checksums...${NC}"
sha256sum r3mes-cli-* > checksums.txt

echo ""
echo -e "${GREEN}🎉 Build completed successfully!${NC}"
echo -e "${BLUE}Built binaries:${NC}"
ls -la r3mes-cli-*

echo ""
echo -e "${BLUE}Usage:${NC}"
echo -e "  Linux/macOS: ${GREEN}./r3mes-cli-${VERSION}-linux-amd64 --help${NC}"
echo -e "  Windows: ${GREEN}r3mes-cli-${VERSION}-windows-amd64.exe --help${NC}"
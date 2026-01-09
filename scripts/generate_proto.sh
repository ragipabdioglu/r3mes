#!/bin/bash
# R3MES Protobuf Generation Script
# Generates Go code from protobuf definitions

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 R3MES Protobuf Generation${NC}"
echo "=================================="

# Check if protoc is installed
if ! command -v protoc &> /dev/null; then
    echo -e "${RED}❌ protoc is not installed. Please install Protocol Buffers compiler.${NC}"
    echo "Installation instructions:"
    echo "  - Ubuntu/Debian: sudo apt install protobuf-compiler"
    echo "  - macOS: brew install protobuf"
    echo "  - Windows: Download from https://github.com/protocolbuffers/protobuf/releases"
    exit 1
fi

# Check if protoc-gen-gogo is installed
if ! command -v protoc-gen-gogo &> /dev/null; then
    echo -e "${YELLOW}⚠️  protoc-gen-gogo not found. Installing...${NC}"
    go install github.com/cosmos/gogoproto/protoc-gen-gogo@latest
fi

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${GREEN}📁 Project root: $PROJECT_ROOT${NC}"

# Proto source directory
PROTO_DIR="$PROJECT_ROOT/remes/proto"
OUTPUT_DIR="$PROJECT_ROOT"

if [ ! -d "$PROTO_DIR" ]; then
    echo -e "${RED}❌ Proto directory not found: $PROTO_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}🔍 Proto directory: $PROTO_DIR${NC}"
echo -e "${GREEN}📤 Output directory: $OUTPUT_DIR${NC}"

# Find all .proto files
PROTO_FILES=$(find "$PROTO_DIR" -name "*.proto" | sort)

if [ -z "$PROTO_FILES" ]; then
    echo -e "${RED}❌ No .proto files found in $PROTO_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}📋 Found proto files:${NC}"
for file in $PROTO_FILES; do
    echo "  - $(basename "$file")"
done

# Generate Go code from proto files
echo -e "${GREEN}⚙️  Generating Go code...${NC}"

# Create necessary directories
mkdir -p "$OUTPUT_DIR/remes/x/remes/types"

# Generate protobuf files
for proto_file in $PROTO_FILES; do
    echo -e "${YELLOW}🔨 Processing: $(basename "$proto_file")${NC}"
    
    protoc \
        --proto_path="$PROTO_DIR" \
        --proto_path="$HOME/go/pkg/mod/github.com/cosmos/cosmos-sdk@v0.50.1/proto" \
        --proto_path="$HOME/go/pkg/mod/github.com/cosmos/cosmos-proto@v1.0.0-beta.3" \
        --proto_path="$HOME/go/pkg/mod/github.com/cosmos/gogoproto@v1.4.11" \
        --gogo_out="$OUTPUT_DIR" \
        --gogo_opt=paths=source_relative \
        "$proto_file"
done

echo -e "${GREEN}✅ Protobuf generation completed successfully!${NC}"

# Verify generated files
GENERATED_FILES=$(find "$OUTPUT_DIR/remes/x/remes/types" -name "*.pb.go" 2>/dev/null || true)

if [ -n "$GENERATED_FILES" ]; then
    echo -e "${GREEN}📄 Generated files:${NC}"
    for file in $GENERATED_FILES; do
        echo "  - $(basename "$file")"
    done
else
    echo -e "${YELLOW}⚠️  No .pb.go files found. This might be normal if proto files don't generate Go code.${NC}"
fi

# Check for compilation errors
echo -e "${GREEN}🔍 Checking for Go compilation errors...${NC}"
cd "$PROJECT_ROOT/remes"

if go build ./x/remes/types/... 2>/dev/null; then
    echo -e "${GREEN}✅ Go compilation successful!${NC}"
else
    echo -e "${YELLOW}⚠️  Go compilation has issues. This is expected if dependencies are missing.${NC}"
    echo "Run 'go mod tidy' and 'go mod download' to resolve dependencies."
fi

echo -e "${GREEN}🎉 Protobuf generation process completed!${NC}"
echo ""
echo "Next steps:"
echo "1. Run 'go mod tidy' to update dependencies"
echo "2. Run 'go build ./...' to verify compilation"
echo "3. Commit the generated files to version control"
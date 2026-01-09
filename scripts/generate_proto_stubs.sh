#!/bin/bash
# Generate Python gRPC stubs from proto files
# This script generates Python code from .proto files for miner-engine

set -e

echo "🔧 R3MES Proto Stub Generator"
echo "=============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if protoc is installed
if ! command -v protoc &> /dev/null; then
    echo -e "${RED}❌ protoc not found${NC}"
    echo "Install protobuf compiler:"
    echo "  Ubuntu/Debian: sudo apt-get install protobuf-compiler"
    echo "  macOS: brew install protobuf"
    echo "  Windows: choco install protoc"
    exit 1
fi

# Check if grpcio-tools is installed
if ! python3 -c "import grpc_tools" &> /dev/null; then
    echo -e "${YELLOW}⚠️  grpcio-tools not found, installing...${NC}"
    pip install grpcio-tools
fi

# Directories
PROTO_DIR="remes/proto"
OUTPUT_DIR="miner-engine/bridge/proto"
REMES_PROTO_DIR="$PROTO_DIR/remes/remes/v1"

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/remes/remes/v1"

echo -e "${GREEN}✓${NC} Directories created"

# Generate amino and gogoproto stubs (dependencies)
echo "📦 Generating dependency stubs..."

# Create minimal amino stub
cat > "$OUTPUT_DIR/amino/__init__.py" << 'EOF'
"""Amino proto stubs (minimal implementation)."""
EOF

mkdir -p "$OUTPUT_DIR/amino"
cat > "$OUTPUT_DIR/amino/amino_pb2.py" << 'EOF'
"""Generated protocol buffer code for amino."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool

DESCRIPTOR = _descriptor_pool.Default().FindFileByName('amino/amino.proto')
EOF

# Create minimal gogoproto stub
mkdir -p "$OUTPUT_DIR/gogoproto"
cat > "$OUTPUT_DIR/gogoproto/__init__.py" << 'EOF'
"""Gogoproto stubs (minimal implementation)."""
EOF

cat > "$OUTPUT_DIR/gogoproto/gogo_pb2.py" << 'EOF'
"""Generated protocol buffer code for gogoproto."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool

DESCRIPTOR = _descriptor_pool.Default().FindFileByName('gogoproto/gogo.proto')
EOF

echo -e "${GREEN}✓${NC} Dependency stubs created"

# Generate R3MES proto stubs
echo "🔨 Generating R3MES proto stubs..."

# Check if proto files exist
if [ ! -d "$REMES_PROTO_DIR" ]; then
    echo -e "${RED}❌ Proto directory not found: $REMES_PROTO_DIR${NC}"
    echo "Make sure you're running this script from the project root"
    exit 1
fi

# Generate stubs for each proto file
PROTO_FILES=(
    "tx.proto"
    "query.proto"
    "stored_gradient.proto"
    "task_pool.proto"
    "node.proto"
    "genesis.proto"
    "params.proto"
)

for proto_file in "${PROTO_FILES[@]}"; do
    if [ -f "$REMES_PROTO_DIR/$proto_file" ]; then
        echo "  Generating $proto_file..."
        
        python3 -m grpc_tools.protoc \
            -I"$PROTO_DIR" \
            -I"$PROTO_DIR/third_party/proto" \
            --python_out="$OUTPUT_DIR" \
            --grpc_python_out="$OUTPUT_DIR" \
            "$REMES_PROTO_DIR/$proto_file"
        
        echo -e "  ${GREEN}✓${NC} $proto_file generated"
    else
        echo -e "  ${YELLOW}⚠️  $proto_file not found, skipping${NC}"
    fi
done

# Create __init__.py files
echo "📝 Creating __init__.py files..."

cat > "$OUTPUT_DIR/__init__.py" << 'EOF'
"""R3MES gRPC proto stubs."""
EOF

cat > "$OUTPUT_DIR/remes/__init__.py" << 'EOF'
"""R3MES proto package."""
EOF

cat > "$OUTPUT_DIR/remes/remes/__init__.py" << 'EOF'
"""R3MES proto package."""
EOF

cat > "$OUTPUT_DIR/remes/remes/v1/__init__.py" << 'EOF'
"""R3MES v1 proto stubs."""

# Import generated modules
try:
    from . import tx_pb2, tx_pb2_grpc
    from . import query_pb2, query_pb2_grpc
    from . import stored_gradient_pb2, stored_gradient_pb2_grpc
    from . import task_pool_pb2, task_pool_pb2_grpc
    from . import node_pb2, node_pb2_grpc
    from . import genesis_pb2
    from . import params_pb2
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import some proto modules: {e}")

__all__ = [
    'tx_pb2', 'tx_pb2_grpc',
    'query_pb2', 'query_pb2_grpc',
    'stored_gradient_pb2', 'stored_gradient_pb2_grpc',
    'task_pool_pb2', 'task_pool_pb2_grpc',
    'node_pb2', 'node_pb2_grpc',
    'genesis_pb2',
    'params_pb2',
]
EOF

echo -e "${GREEN}✓${NC} __init__.py files created"

# Fix import paths in generated files
echo "🔧 Fixing import paths..."

find "$OUTPUT_DIR" -name "*_pb2*.py" -type f | while read -r file; do
    # Fix amino imports
    sed -i.bak 's/from amino import amino_pb2/from amino import amino_pb2 as amino_dot_amino__pb2/g' "$file"
    # Fix gogoproto imports
    sed -i.bak 's/from gogoproto import gogo_pb2/from gogoproto import gogo_pb2 as gogoproto_dot_gogo__pb2/g' "$file"
    # Remove backup files
    rm -f "${file}.bak"
done

echo -e "${GREEN}✓${NC} Import paths fixed"

# Verify generated files
echo "🔍 Verifying generated files..."

REQUIRED_FILES=(
    "$OUTPUT_DIR/remes/remes/v1/tx_pb2.py"
    "$OUTPUT_DIR/remes/remes/v1/tx_pb2_grpc.py"
    "$OUTPUT_DIR/remes/remes/v1/query_pb2.py"
    "$OUTPUT_DIR/remes/remes/v1/query_pb2_grpc.py"
)

ALL_FOUND=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}✓${NC} $(basename $file)"
    else
        echo -e "  ${RED}✗${NC} $(basename $file) not found"
        ALL_FOUND=false
    fi
done

if [ "$ALL_FOUND" = true ]; then
    echo -e "\n${GREEN}✅ Proto stub generation completed successfully!${NC}"
    echo "Generated files are in: $OUTPUT_DIR"
else
    echo -e "\n${YELLOW}⚠️  Some files were not generated${NC}"
    echo "Check the proto files and try again"
    exit 1
fi

# Test import
echo "🧪 Testing imports..."
if python3 -c "from miner-engine.bridge.proto.remes.remes.v1 import tx_pb2, query_pb2" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Import test passed"
else
    echo -e "${YELLOW}⚠️  Import test failed (this is normal if miner-engine is not in PYTHONPATH)${NC}"
fi

echo -e "\n${GREEN}🎉 Done!${NC}"

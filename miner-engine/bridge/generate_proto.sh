#!/bin/bash
# Generate Python gRPC client from protocol buffers

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REmes_DIR="$PROJECT_ROOT/remes"
OUTPUT_DIR="$SCRIPT_DIR/proto"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Generating Python gRPC client from proto files..."

# Use grpc_tools.protoc (comes with grpcio-tools)
python3 -m grpc_tools.protoc \
    --python_out="$OUTPUT_DIR" \
    --grpc_python_out="$OUTPUT_DIR" \
    --proto_path="$REmes_DIR/proto" \
    "$REmes_DIR/proto/remes/remes/v1/tx.proto" \
    "$REmes_DIR/proto/remes/remes/v1/query.proto" \
    "$REmes_DIR/proto/remes/remes/v1/genesis.proto" \
    "$REmes_DIR/proto/remes/remes/v1/stored_gradient.proto" \
    "$REmes_DIR/proto/remes/remes/v1/params.proto" \
    2>&1 | grep -v "warning:" || true

# Create __init__.py files
touch "$OUTPUT_DIR/__init__.py"

echo "Python proto files generated in $OUTPUT_DIR"
echo "Note: You may need to install additional dependencies:"
echo "  pip install grpcio grpcio-tools protobuf"

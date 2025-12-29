# Python gRPC Client for R3MES Blockchain

This directory contains the Python gRPC client for communicating with the R3MES blockchain node.

## Setup

### Prerequisites

1. Install Python dependencies:
```bash
pip install grpcio grpcio-tools protobuf
```

2. Install Cosmos SDK proto dependencies:
```bash
cd ../../remes
buf dep update  # or buf config migrate if using newer buf
```

### Generating Python Proto Files

Run the generation script:
```bash
cd miner-engine/bridge
bash generate_proto.sh
```

**Note:** The script requires Cosmos SDK proto dependencies to be available. If you encounter import errors, you may need to:

1. Download Cosmos proto dependencies manually
2. Or use a simplified proto generation approach (see below)

### Manual Proto Generation

If automatic generation fails due to missing Cosmos dependencies, you can generate proto files manually:

```bash
# Find where buf stores dependencies
BUF_CACHE="$HOME/.cache/buf"  # or check buf.yaml for dependency locations

# Generate with all include paths
python3 -m grpc_tools.protoc \
    --python_out=proto \
    --grpc_python_out=proto \
    --proto_path=../../remes/proto \
    --proto_path="$BUF_CACHE/cosmos/cosmos-proto" \
    --proto_path="$BUF_CACHE/cosmos/cosmos-sdk" \
    --proto_path="$BUF_CACHE/cosmos/gogo-proto" \
    --proto_path="$BUF_CACHE/googleapis" \
    ../../remes/proto/remes/remes/v1/*.proto
```

## Usage

After generating proto files, import them in your Python code:

```python
from bridge.proto import remes_pb2
from bridge.proto import remes_pb2_grpc
```

## Status

**Current Status:** Proto generation requires Cosmos SDK dependencies. This will be completed when:
- Cosmos proto dependencies are properly configured
- Or simplified proto files are created for Python client

**Next Steps:** Complete Task 2 (Core PoUW Blockchain Module) first, then return to Python client integration.


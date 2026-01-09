# R3MES Docker Containerization

This directory contains Docker configurations for the R3MES PoUW protocol.

## Dockerfiles

### `Dockerfile.nvidia`
- **Base**: `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
- **PyTorch**: 2.1.0 (CUDA 12.1)
- **Purpose**: NVIDIA GPU support for miner engine
- **Deterministic Features**:
  - `CUBLAS_WORKSPACE_CONFIG=:4096:8` for deterministic CUDA operations
  - `TORCH_USE_CUDA_DSA=1` for deterministic algorithms
  - Fixed Python hash seed

### `Dockerfile.amd`
- **Base**: `rocm/pytorch:rocm5.7_ubuntu22.04_py3.10_pytorch_2.1.0`
- **PyTorch**: 2.1.0 (ROCm 5.7)
- **Purpose**: AMD GPU support for miner engine
- **Deterministic Features**:
  - `HIP_FORCE_DEV_KERNEL=1` for deterministic execution
  - Fixed Python hash seed

### `Dockerfile.go`
- **Base**: `golang:1.22-alpine`
- **Go**: 1.22
- **Purpose**: Blockchain node (remesd)
- **Build**: Static binary with all dependencies

## Docker Compose

The `docker-compose.yml` file orchestrates:
1. **IPFS daemon**: Decentralized storage
2. **Blockchain node**: Go-based Cosmos SDK node
3. **Miner engine (NVIDIA)**: GPU mining with CUDA
4. **Miner engine (AMD)**: GPU mining with ROCm

## Usage

### Build Images

```bash
# Build all images
docker-compose build

# Build specific service
docker-compose build miner-nvidia
docker-compose build miner-amd
docker-compose build blockchain
```

### Run Services

```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d ipfs
docker-compose up -d blockchain
docker-compose up -d miner-nvidia

# View logs
docker-compose logs -f miner-nvidia
docker-compose logs -f blockchain
```

### Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Deterministic Execution

The Docker containers are configured for deterministic execution:

1. **Fixed Dependencies**: All Python packages are pinned to specific versions
2. **Deterministic CUDA**: `CUBLAS_WORKSPACE_CONFIG` and `TORCH_USE_CUDA_DSA` enabled
3. **Fixed Seeds**: `PYTHONHASHSEED=0` for reproducible hash operations
4. **Locked Versions**: PyTorch 2.1.0, CUDA 12.1, Go 1.22

## GPU Requirements

### NVIDIA
- Requires NVIDIA Docker runtime (`nvidia-docker2`)
- GPU with CUDA 12.1 support
- Compute Capability 7.0+ (Volta, Turing, Ampere, Ada, Blackwell)

### AMD
- Requires ROCm 5.7+
- AMD GPU with ROCm support
- Access to `/dev/kfd` and `/dev/dri`

## Environment Variables

### Miner Engine
- `BLOCKCHAIN_URL`: gRPC endpoint (default: `blockchain:9090`)
- `IPFS_API_URL`: IPFS API endpoint (default: `http://ipfs:5001`)
- `CHAIN_ID`: Chain ID (default: `remes-test`)
- `NVIDIA_VISIBLE_DEVICES`: GPU device selection (NVIDIA only)

### Blockchain Node
- `CHAIN_ID`: Chain ID (default: `remes-test`)

## Volume Mounts

- `ipfs-data`: IPFS repository data
- `blockchain-data`: Blockchain node data (`~/.remesd`)
- `miner-data`: Miner engine data and logs

## Troubleshooting

### GPU Not Detected
```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Check AMD devices
ls -la /dev/kfd /dev/dri
```

### Build Failures
```bash
# Clean build cache
docker-compose build --no-cache

# Check logs
docker-compose logs build
```

### Network Issues
```bash
# Check service connectivity
docker-compose exec miner-nvidia ping blockchain
docker-compose exec miner-nvidia curl http://ipfs:5001/api/v0/version
```


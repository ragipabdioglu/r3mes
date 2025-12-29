# Serving Node Onboarding Guide

## Overview

Serving nodes provide AI model inference services to the R3MES network. They load models from IPFS, process inference requests from users, and earn fees for each successful inference.

## Requirements

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU/CPU** | CPU (4 cores) or GPU (4GB VRAM) | GPU (8GB+ VRAM) for faster inference |
| **RAM** | 8 GB | 16 GB+ |
| **Disk** | 20 GB (for model storage) | 50 GB+ SSD |
| **Network** | 10 Mbps upload | 100 Mbps+ upload |

### Software Requirements

- Python 3.10+
- IPFS daemon (optional, embedded IPFS available)
- Access to blockchain RPC endpoint

## Quick Start

### 1. Install R3MES Serving Package

```bash
pip install r3mes
```

### 2. Run Setup Wizard

```bash
r3mes-serving setup
```

The setup wizard will guide you through:
1. Wallet creation or import
2. Blockchain configuration (gRPC URL, Chain ID)
3. Model configuration (IPFS hash, version)

### 3. Start Serving Node

```bash
r3mes-serving start
```

## Configuration

### Configuration File

After running `r3mes-serving setup`, a configuration file is created at `~/.r3mes/serving_config.json`:

```json
{
  "private_key": "your_private_key",
  "blockchain_url": "localhost:9090",
  "chain_id": "remes-1",
  "model_ipfs_hash": "Qm...",
  "model_version": "v1.0.0",
  "node_type": "serving"
}
```

### Environment Variables

You can override configuration using environment variables:

```bash
export R3MES_BLOCKCHAIN_URL="grpc://localhost:9090"
export R3MES_CHAIN_ID="remes-1"
export R3MES_MODEL_IPFS_HASH="Qm..."
export R3MES_MODEL_VERSION="v1.0.0"
```

## Model Loading

### From IPFS Hash

The serving node automatically downloads and loads models from IPFS:

```bash
r3mes-serving start --model-ipfs-hash QmYourModelHash
```

### Model Version Management

Update model version on blockchain:

```bash
r3mes-serving register --model-version v2.0.0 --model-ipfs-hash QmNewHash
```

## Inference Request Workflow

1. **Request Received**: User submits inference request via blockchain (`MsgRequestInference`)
2. **Input Download**: Serving node downloads input data from IPFS
3. **Inference Execution**: Model processes input and generates output
4. **Result Upload**: Output is uploaded to IPFS
5. **Result Submission**: Serving node submits result via `MsgSubmitInferenceResult`
6. **Fee Collection**: Fees are automatically collected

## Status Management

### Update Node Status

Mark node as available/unavailable:

```bash
# Via CLI (updates blockchain)
r3mes-serving status --available true
```

### Check Node Status

```bash
r3mes-serving status
```

Output:
```
Serving Node Status
===================
Address: remes1abc...
Model Version: v1.0.0
Model IPFS Hash: Qm...
Status: Available
Total Requests: 150
Successful Requests: 148
Success Rate: 98.7%
Average Latency: 245ms
```

## Registration on Blockchain

Register your node as a serving node:

```bash
r3mes-serving register
```

This sends a `MsgRegisterNode` transaction with:
- Node address
- Role: `NODE_TYPE_SERVING` (2)
- Resource quotas
- Stake amount

## Desktop Launcher Integration

### Using Desktop Launcher

1. Open R3MES Desktop Launcher
2. In Setup Wizard, select "Serving" role
3. Configure model settings
4. Start serving node from launcher UI

### Process Management

The Desktop Launcher provides:
- Start/Stop serving node
- View logs
- Monitor status
- Resource usage tracking

## Monitoring & Statistics

### Web Dashboard

Access serving node statistics via web dashboard:
- Navigate to `/serving` page
- View active serving nodes
- Monitor request queue
- Check success rates and latency

### API Endpoints

Query serving node statistics via backend API:

```bash
# List all serving nodes
curl http://localhost:8000/api/serving/nodes

# Get specific node stats
curl http://localhost:8000/api/serving/nodes/{address}/stats

# Get inference requests
curl http://localhost:8000/api/serving/nodes/{address}/requests
```

## Troubleshooting

### Model Not Loading

**Problem**: Model fails to load from IPFS

**Solutions**:
1. Check IPFS connectivity: `ipfs swarm peers`
2. Verify IPFS hash is correct
3. Ensure sufficient disk space
4. Check model format compatibility

### Low Success Rate

**Problem**: High failure rate on inference requests

**Solutions**:
1. Check GPU/CPU resources
2. Monitor memory usage
3. Verify model version matches requests
4. Check input data format

### Blockchain Connection Issues

**Problem**: Cannot connect to blockchain

**Solutions**:
1. Verify `blockchain_url` in config
2. Check network connectivity
3. Ensure blockchain node is running
4. Verify chain ID matches network

## Best Practices

1. **Model Selection**: Use efficient models for faster inference
2. **Resource Allocation**: Reserve sufficient GPU/CPU for inference
3. **Network Stability**: Maintain stable connection for blockchain communication
4. **Monitoring**: Regularly check success rates and latency
5. **Updates**: Keep model versions updated for compatibility

## Fee Structure

Serving nodes earn fees for each successful inference:
- Fee amount is set by requester
- Fees are collected automatically
- Failed inferences do not earn fees
- Fees are paid in REMES tokens

## Advanced Configuration

### Custom Model Loading

For custom model formats, modify `r3mes/serving/engine.py`:

```python
def load_model(self, model_ipfs_hash: str):
    # Custom model loading logic
    model_path = self.ipfs_client.get(model_ipfs_hash)
    # Load your custom model format
    self.model = load_custom_model(model_path)
```

### Resource Limits

Set resource limits in configuration:

```json
{
  "resources": {
    "cpu_cores": 4,
    "memory_gb": 8,
    "gpu_count": 1,
    "gpu_memory_gb": 12
  }
}
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/r3mes/r3mes/issues
- Discord: https://discord.gg/r3mes
- Documentation: https://docs.r3mes.network


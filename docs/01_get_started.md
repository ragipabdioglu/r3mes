# Quick Start Guide

Get R3MES running and start earning rewards in under 10 minutes.

## Prerequisites

Before you begin, ensure your system meets these requirements:

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA 6GB VRAM | NVIDIA 12GB+ VRAM |
| CUDA | 11.8+ | 12.1+ |
| RAM | 8GB | 16GB |
| Storage | 50GB SSD | 100GB NVMe |
| OS | Windows 10, Ubuntu 20.04+, macOS 12+ | Ubuntu 22.04 |
| Python | 3.10+ | 3.11 |

## Step 1: Install R3MES

**Option A: PyPI (Recommended)**

```bash
pip install r3mes
```

**Option B: From Source**

```bash
git clone https://github.com/r3mes-network/r3mes.git
cd r3mes
pip install -e .
```

Verify installation:

```bash
r3mes --version
```

## Step 2: Run Setup Wizard

The setup wizard handles wallet creation, configuration, and initial setup:

```bash
r3mes setup
```

The wizard will:

1. Check system requirements and GPU compatibility
2. Create a new wallet or import existing one
3. Configure blockchain connection
4. Download the base model (~8GB)
5. Request testnet tokens from faucet

**Important**: Save your mnemonic phrase securely. This is the only way to recover your wallet.

## Step 3: Start Mining

```bash
r3mes-miner start
```

You should see output similar to:

```
R3MES Miner v1.0.0
Status: Mining
GPU: NVIDIA RTX 3080 (10GB)
Hashrate: 1,234 gradients/hour
Temperature: 65°C
```

## Step 4: Verify Setup

Check your wallet balance:

```bash
r3mes wallet balance
```

Check mining status:

```bash
r3mes-miner status
```

Check network connection:

```bash
r3mes network status
```

## Configuration

Configuration file location:

| OS | Path |
|----|------|
| Linux/macOS | `~/.r3mes/config.yaml` |
| Windows | `%USERPROFILE%\.r3mes\config.yaml` |

Example configuration:

```yaml
network:
  chain_id: remes-testnet-1
  rpc_url: https://rpc.testnet.r3mes.network:26657
  rest_url: https://api.testnet.r3mes.network:1317

mining:
  gpu_ids: [0]
  batch_size: auto
  auto_start: true

wallet:
  keyring_backend: file
```

## Common Issues

| Issue | Solution |
|-------|----------|
| CUDA not found | Install NVIDIA CUDA Toolkit 11.8+ |
| Out of memory | Reduce batch size: `r3mes-miner start --batch-size 1` |
| Connection refused | Check firewall settings for ports 26657, 1317, 9090 |
| Wallet not found | Re-run setup: `r3mes setup --force` |

## Next Steps

| Goal | Guide |
|------|-------|
| Optimize mining | [Mining Guide](/docs/mining-guide) |
| Stake tokens | [Staking Guide](/docs/staking-guide) |
| Run a validator | [Validator Guide](/docs/validators) |
| Build with R3MES | [API Reference](/docs/api-reference) |

## Get Help

| Resource | Link |
|----------|------|
| Discord | [discord.gg/r3mes](https://discord.gg/r3mes) |
| GitHub Issues | [github.com/r3mes-network/r3mes/issues](https://github.com/r3mes-network/r3mes/issues) |
| Troubleshooting | [Troubleshooting Guide](/docs/troubleshooting) |

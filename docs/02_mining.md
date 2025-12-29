# Mining Guide

Complete guide to mining REMES tokens with your GPU.

## How Mining Works

R3MES uses **Proof of Useful Work (PoUW)** — your GPU trains AI models instead of solving arbitrary puzzles. Each valid training contribution earns REMES tokens.

| Step | Description |
|------|-------------|
| 1. Receive Task | Get a training data chunk from the network |
| 2. Train Model | Run forward/backward pass on your GPU |
| 3. Submit Gradient | Upload results to IPFS, submit hash to blockchain |
| 4. Earn Reward | Receive REMES tokens after verification |

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA 6GB VRAM | NVIDIA 12GB+ VRAM |
| CUDA | 11.8+ | 12.1+ |
| RAM | 8GB | 16GB |
| Storage | 50GB SSD | 100GB NVMe |
| Internet | 10 Mbps | 50+ Mbps |

## GPU Performance Tiers

| Tier | GPUs | Expected Earnings | Power Efficiency |
|------|------|-------------------|------------------|
| S | RTX 4090, RTX 4080 | 50-80 REMES/day | Excellent |
| A | RTX 3090, RTX 3080 Ti | 35-55 REMES/day | Very Good |
| B | RTX 3080, RTX 3070 Ti | 25-40 REMES/day | Good |
| C | RTX 3070, RTX 3060 Ti | 18-30 REMES/day | Moderate |
| D | RTX 3060, GTX 1080 Ti | 12-20 REMES/day | Basic |
| E | GTX 1660, GTX 1070 | 5-12 REMES/day | Entry |

*Earnings estimates based on testnet data. Actual earnings vary with network conditions.*

## Installation

```bash
# Install R3MES
pip install r3mes

# Verify GPU detection
r3mes-miner check-gpu
```

Expected output:

```
GPU 0: NVIDIA GeForce RTX 3080
  VRAM: 10GB
  CUDA: 8.6
  Driver: 535.104.05
  Status: Compatible
```

## Start Mining

**Basic start:**

```bash
r3mes-miner start
```

**With options:**

```bash
# Specify GPU
r3mes-miner start --gpu 0

# Multiple GPUs
r3mes-miner start --gpu 0,1,2

# Custom batch size
r3mes-miner start --batch-size 8

# Background mode
r3mes-miner start --daemon
```

## Multi-GPU Mining

Configure multiple GPUs in `~/.r3mes/config.yaml`:

```yaml
mining:
  gpu_ids: [0, 1, 2, 3]
  batch_size: auto
  distributed: true
```

Start with all GPUs:

```bash
r3mes-miner start --gpu all
```

## Monitoring

**CLI Dashboard:**

```bash
r3mes-miner dashboard
```

**HTTP Stats Server:**

The miner runs a stats server on port 8080:

```bash
curl http://localhost:8080/stats
```

**Web Dashboard:**

Access mining stats at [localhost:3000/mine](http://localhost:3000/mine)

## Optimization

### VRAM Profiles

The miner automatically adjusts settings based on available VRAM:

| VRAM | Batch Size | Gradient Accumulation |
|------|------------|----------------------|
| 6GB | 1 | 32 |
| 8GB | 2 | 16 |
| 10GB | 4 | 8 |
| 12GB | 6 | 4 |
| 16GB+ | 8 | 2 |
| 24GB | 16 | 1 |

### Power Management

```bash
# Set power limit (requires root/admin)
sudo nvidia-smi -pl 250

# Or use miner setting
r3mes-miner start --power-limit 250
```

### Temperature Control

Configure in `~/.r3mes/config.yaml`:

```yaml
mining:
  max_temperature: 80
  resume_temperature: 70
```

## Reward Structure

| Component | Amount | Description |
|-----------|--------|-------------|
| Base Reward | 10 REMES | Per valid gradient submission |
| Quality Bonus | 0-50% | Based on gradient quality |
| Reputation Bonus | 0-15% | Based on trust score |
| Consistency Bonus | 0-20% | For continuous mining |

## Earnings Calculator

```bash
r3mes-miner estimate --gpu "RTX 3080"
```

Example output:

```
Estimated Earnings for RTX 3080 (10GB)
Hashrate: ~1,400 gradients/hour
Daily: ~28-35 REMES
Weekly: ~196-245 REMES
Monthly: ~840-1,050 REMES

Power Cost (at $0.10/kWh):
Daily: ~$0.53
Monthly: ~$16
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | `r3mes-miner start --batch-size 1` |
| Low hashrate | Check GPU utilization with `nvidia-smi` |
| Submissions failing | Check network: `r3mes network status` |
| GPU not detected | Reinstall PyTorch with CUDA support |

## Best Practices

| Practice | Reason |
|----------|--------|
| Keep drivers updated | Latest NVIDIA drivers improve performance |
| Monitor temperatures | Keep GPU below 80°C for longevity |
| Use SSD storage | Faster model loading and IPFS operations |
| Stable internet | Consistent connection prevents submission failures |
| Regular updates | Keep R3MES miner updated for optimizations |
| Backup wallet | Store mnemonic phrase securely |

## FAQ

**Can I mine with AMD GPUs?**
Currently only NVIDIA GPUs are supported. AMD support is planned.

**Can I mine and game simultaneously?**
Not recommended. Mining uses 100% GPU resources.

**How do I withdraw earnings?**
Use `r3mes wallet send` or the web dashboard.

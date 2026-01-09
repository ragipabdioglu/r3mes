# Proposer Onboarding Guide

## Overview

Proposers aggregate gradients from miners, creating aggregated model updates that are submitted to the blockchain. They play a crucial role in the federated learning process by combining multiple gradient contributions.

## Requirements

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 8 GB | 16 GB+ |
| **Disk** | 50 GB (for gradient storage) | 100 GB+ SSD |
| **Network** | 50 Mbps upload | 100 Mbps+ upload |

### Software Requirements

- Python 3.10+
- IPFS daemon (required for gradient download)
- Access to blockchain RPC endpoint

## Quick Start

### 1. Install R3MES Package

```bash
pip install r3mes
```

### 2. Run Setup Wizard

```bash
r3mes-proposer setup
```

The setup wizard will guide you through:
1. Wallet creation or import
2. Blockchain configuration
3. Resource allocation

### 3. Start Proposer Service

```bash
r3mes-proposer start
```

## Aggregation Workflow

### Commit-Reveal-Submit Pattern

Proposers use a commit-reveal-submit pattern to prevent front-running:

1. **Commit**: Hash of aggregation result is committed to blockchain
2. **Reveal**: After commit period, actual aggregation is revealed
3. **Submit**: Aggregation is submitted and validated

### Manual Aggregation

Trigger aggregation manually:

```bash
r3mes-proposer aggregate --gradient-ids 1,2,3,4,5 --training-round-id 10
```

### View Pending Gradients

Check available gradients for aggregation:

```bash
r3mes-proposer pool
```

Output:
```
Pending Gradients Pool
=====================
Total: 15 gradients

Gradient ID | Miner | Training Round | Status
1          | remes1... | 10          | pending
2          | remes1... | 10          | pending
3          | remes1... | 10          | pending
...
```

## Registration on Blockchain

Register as proposer node:

```bash
r3mes-proposer register
```

This sends a `MsgRegisterNode` transaction with:
- Node address
- Role: `NODE_TYPE_PROPOSER` (4)
- Resource quotas
- Stake amount

## Gradient Aggregation

### Automatic Aggregation

The proposer service automatically:
1. Queries pending gradients from blockchain
2. Downloads gradients from IPFS
3. Aggregates using weighted average
4. Commits, reveals, and submits aggregation

### Aggregation Algorithm

Gradients are aggregated using weighted average:
- Each gradient is weighted by miner's reputation
- Aggregated result is uploaded to IPFS
- Merkle root of included gradients is computed

### Aggregation Parameters

Configure aggregation behavior:

```json
{
  "aggregation": {
    "min_gradients": 5,
    "max_gradients": 50,
    "weight_by_reputation": true,
    "training_round_id": 10
  }
}
```

## Rewards

Proposers earn rewards for successful aggregations:
- Reward based on number of gradients aggregated
- Bonus for including high-reputation miners
- Penalties for invalid aggregations

### Check Rewards

View accumulated rewards:

```bash
# Via web dashboard
# Navigate to /proposer page

# Via API
curl http://localhost:8000/api/proposer/nodes/{address}
```

## Desktop Launcher Integration

### Using Desktop Launcher

1. Open R3MES Desktop Launcher
2. In Setup Wizard, select "Proposer" role
3. Configure aggregation settings
4. Start proposer service from launcher UI

## Monitoring

### Web Dashboard

Access proposer statistics:
- Navigate to `/proposer` page
- View pending gradients pool
- Monitor aggregation history
- Check rewards

### API Endpoints

Query proposer data:

```bash
# List proposer nodes
curl http://localhost:8000/api/proposer/nodes

# Get aggregations
curl http://localhost:8000/api/proposer/aggregations

# Get gradient pool
curl http://localhost:8000/api/proposer/pool
```

## Best Practices

1. **Gradient Selection**: Choose gradients from reputable miners
2. **Timing**: Aggregate when sufficient gradients are available
3. **Resource Management**: Monitor CPU and memory usage
4. **Network Stability**: Maintain stable connection for IPFS and blockchain
5. **Stake Management**: Maintain sufficient stake for registration

## Troubleshooting

### IPFS Download Failures

**Problem**: Cannot download gradients from IPFS

**Solutions**:
1. Check IPFS daemon is running: `ipfs daemon`
2. Verify IPFS connectivity: `ipfs swarm peers`
3. Check disk space for gradient storage
4. Verify gradient IPFS hashes are correct

### Aggregation Failures

**Problem**: Aggregation submission fails

**Solutions**:
1. Verify sufficient stake for registration
2. Check gradient format compatibility
3. Ensure training round ID matches
4. Verify Merkle root computation

### Low Rewards

**Problem**: Earning fewer rewards than expected

**Solutions**:
1. Aggregate more gradients per round
2. Include high-reputation miners
3. Submit aggregations more frequently
4. Verify aggregation quality

## Advanced Configuration

### Custom Aggregation Algorithm

Modify aggregation logic in `r3mes/proposer/aggregator.py`:

```python
def aggregate_gradients(self, gradients: List[bytes]) -> bytes:
    # Custom aggregation algorithm
    # e.g., federated averaging, secure aggregation
    aggregated = custom_aggregate(gradients)
    return aggregated
```

### Resource Allocation

Configure resource limits:

```json
{
  "resources": {
    "cpu_cores": 8,
    "memory_gb": 16,
    "storage_gb": 100,
    "network_bandwidth_mbps": 1000
  }
}
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/r3mes/r3mes/issues
- Discord: https://discord.gg/r3mes
- Documentation: https://docs.r3mes.network


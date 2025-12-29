# Multi-Role Node Setup Guide

## Overview

R3MES supports multi-role nodes, allowing a single node to perform multiple roles simultaneously (e.g., Miner + Serving, Validator + Proposer). This guide explains how to configure and manage multi-role nodes.

## Architecture

### Role Combinations

Common role combinations:

1. **Miner + Serving**: Train models and serve inference
2. **Validator + Proposer**: Validate blocks and aggregate gradients
3. **Miner + Proposer**: Train models and aggregate gradients
4. **All Roles**: Full-featured node (requires significant resources)

### Resource Allocation

Each role consumes resources:

| Role | CPU | RAM | GPU | Disk | Network |
|------|-----|-----|-----|------|---------|
| Miner | 2 cores | 4 GB | Required | 20 GB | 10 Mbps |
| Serving | 2 cores | 4 GB | Optional | 20 GB | 10 Mbps |
| Validator | 1 core | 2 GB | - | 10 GB | 5 Mbps |
| Proposer | 2 cores | 4 GB | - | 50 GB | 50 Mbps |

### Multi-Role Considerations

- **Resource Conflicts**: Ensure sufficient resources for all roles
- **Priority Management**: Set priorities for resource allocation
- **Process Isolation**: Each role runs in separate process
- **Monitoring**: Track resource usage per role

## Setup Methods

### Method 1: Desktop Launcher (Recommended)

1. **Run Setup Wizard**:
   - Launch R3MES Desktop Launcher
   - In Role Selection step, select multiple roles
   - Configure resources for each role

2. **Start Roles**:
   - Use launcher UI to start/stop individual roles
   - Monitor each role's status separately

### Method 2: CLI Registration

Register node with multiple roles:

```bash
# Register with Miner + Serving roles
r3mes-miner register --roles 1,2

# Or use role registration API
curl -X POST http://localhost:8000/api/roles/register \
  -H "Content-Type: application/json" \
  -d '{
    "node_address": "remes1...",
    "roles": [1, 2],
    "stake": "1000000uremes",
    "resources": {
      "cpu_cores": 8,
      "memory_gb": 16,
      "gpu_count": 1,
      "gpu_memory_gb": 12,
      "storage_gb": 100,
      "network_bandwidth_mbps": 1000
    }
  }'
```

### Method 3: Web Dashboard

1. Navigate to `/roles` page
2. Select multiple roles
3. Configure resource allocation
4. Submit registration transaction

## Resource Allocation Strategies

### Equal Allocation

Divide resources equally among roles:

```json
{
  "roles": [1, 2],
  "resources": {
    "cpu_cores": 8,
    "memory_gb": 16,
    "gpu_count": 1
  },
  "allocation": {
    "miner": {
      "cpu_cores": 4,
      "memory_gb": 8,
      "gpu_usage": 0.5
    },
    "serving": {
      "cpu_cores": 4,
      "memory_gb": 8,
      "gpu_usage": 0.5
    }
  }
}
```

### Priority-Based Allocation

Allocate more resources to primary role:

```json
{
  "primary_role": "miner",
  "allocation": {
    "miner": {
      "cpu_cores": 6,
      "memory_gb": 12,
      "gpu_usage": 0.8
    },
    "serving": {
      "cpu_cores": 2,
      "memory_gb": 4,
      "gpu_usage": 0.2
    }
  }
}
```

## Example Configurations

### Configuration 1: Miner + Serving

**Use Case**: Train models and serve inference simultaneously

**Hardware**:
- GPU: RTX 3090 (24GB VRAM)
- CPU: 8 cores
- RAM: 32 GB
- Disk: 100 GB SSD

**Configuration**:
```json
{
  "roles": [1, 2],
  "resources": {
    "cpu_cores": 8,
    "memory_gb": 32,
    "gpu_count": 1,
    "gpu_memory_gb": 24,
    "storage_gb": 100
  },
  "allocation": {
    "miner": {
      "cpu_cores": 4,
      "memory_gb": 16,
      "gpu_usage": 0.7
    },
    "serving": {
      "cpu_cores": 4,
      "memory_gb": 16,
      "gpu_usage": 0.3
    }
  }
}
```

### Configuration 2: Validator + Proposer

**Use Case**: Validate blocks and aggregate gradients

**Hardware**:
- CPU: 16 cores
- RAM: 32 GB
- Disk: 200 GB SSD
- Network: 100 Mbps upload

**Configuration**:
```json
{
  "roles": [3, 4],
  "resources": {
    "cpu_cores": 16,
    "memory_gb": 32,
    "storage_gb": 200,
    "network_bandwidth_mbps": 100
  },
  "allocation": {
    "validator": {
      "cpu_cores": 4,
      "memory_gb": 8,
      "network_usage": 0.2
    },
    "proposer": {
      "cpu_cores": 12,
      "memory_gb": 24,
      "network_usage": 0.8
    }
  }
}
```

### Configuration 3: All Roles

**Use Case**: Full-featured node with all capabilities

**Hardware**:
- GPU: RTX 4090 (24GB VRAM)
- CPU: 32 cores
- RAM: 64 GB
- Disk: 500 GB NVMe SSD
- Network: 1 Gbps

**Configuration**:
```json
{
  "roles": [1, 2, 3, 4],
  "resources": {
    "cpu_cores": 32,
    "memory_gb": 64,
    "gpu_count": 1,
    "gpu_memory_gb": 24,
    "storage_gb": 500,
    "network_bandwidth_mbps": 1000
  },
  "allocation": {
    "miner": {
      "cpu_cores": 8,
      "memory_gb": 16,
      "gpu_usage": 0.5
    },
    "serving": {
      "cpu_cores": 8,
      "memory_gb": 16,
      "gpu_usage": 0.3
    },
    "validator": {
      "cpu_cores": 4,
      "memory_gb": 8
    },
    "proposer": {
      "cpu_cores": 12,
      "memory_gb": 24
    }
  }
}
```

## Process Management

### Desktop Launcher

Manage each role independently:
- Start/Stop individual roles
- View logs per role
- Monitor resource usage per role

### CLI Management

Start roles individually:

```bash
# Start miner
r3mes-miner start

# Start serving (in separate terminal)
r3mes-serving start

# Start proposer (in separate terminal)
r3mes-proposer start
```

### Systemd Services

Create systemd services for each role:

```ini
# /etc/systemd/system/r3mes-miner.service
[Unit]
Description=R3MES Miner
After=network.target

[Service]
Type=simple
User=r3mes
WorkingDirectory=/home/r3mes
ExecStart=/usr/local/bin/r3mes-miner start
Restart=always

[Install]
WantedBy=multi-user.target
```

## Performance Considerations

### Resource Contention

- **GPU**: Miner and Serving compete for GPU
- **CPU**: All roles compete for CPU
- **Memory**: Shared memory pool
- **Network**: Bandwidth shared across roles

### Optimization Tips

1. **GPU Scheduling**: Use CUDA MPS for GPU sharing
2. **CPU Affinity**: Pin roles to specific CPU cores
3. **Memory Limits**: Set memory limits per role
4. **Network QoS**: Prioritize critical role traffic

## Monitoring

### Per-Role Statistics

Monitor each role separately:

```bash
# Miner stats
r3mes-miner status

# Serving stats
r3mes-serving status

# Proposer stats
r3mes-proposer pool
```

### Web Dashboard

View multi-role node status:
- Navigate to `/roles` page
- See all active roles
- Monitor resource usage per role

## Troubleshooting

### Resource Exhaustion

**Problem**: One role consumes all resources

**Solutions**:
1. Set resource limits per role
2. Adjust allocation ratios
3. Stop non-critical roles
4. Upgrade hardware

### Role Conflicts

**Problem**: Roles interfere with each other

**Solutions**:
1. Isolate roles in separate processes
2. Use resource limits
3. Schedule roles at different times
4. Use containerization (Docker)

## Best Practices

1. **Start Small**: Begin with single role, add roles gradually
2. **Monitor Resources**: Track usage before adding roles
3. **Test Combinations**: Verify role compatibility
4. **Plan Allocation**: Reserve resources for each role
5. **Backup Config**: Save configuration before changes

## Support

For issues and questions:
- GitHub Issues: https://github.com/r3mes/r3mes/issues
- Discord: https://discord.gg/r3mes
- Documentation: https://docs.r3mes.network


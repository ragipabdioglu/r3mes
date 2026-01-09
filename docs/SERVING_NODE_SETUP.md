# Serving Node Setup Guide

This guide explains how to set up a miner as a serving node for distributed LoRA inference.

## Overview

A serving node is a miner that:
1. Downloads and loads LoRA adapters from IPFS
2. Registers itself with the backend as available for inference
3. Receives chat requests from the backend
4. Performs inference using the selected LoRA adapter
5. Streams responses back to the backend

## Prerequisites

- Miner engine installed and configured
- Backend API accessible
- IPFS daemon running
- At least one LoRA adapter registered in the backend

## Setup Steps

### 1. Enable Serving Node Mode

Set the environment variable to enable serving node functionality:

```bash
export R3MES_ENABLE_SERVING_NODE=true
```

### 2. Configure Backend URL

Set the backend API URL:

```bash
export R3MES_BACKEND_URL=https://api.r3mes.network
# Or for local development:
export R3MES_BACKEND_URL=http://localhost:${BACKEND_PORT:-8000}
```

### 3. Configure Serving Node Endpoint

Set the serving node endpoint URL (optional, defaults to localhost:8081):

```bash
export R3MES_SERVING_NODE_HOST=0.0.0.0  # Listen on all interfaces
export R3MES_SERVING_NODE_PORT=8081     # Port for inference server
```

Or set the full URL:

```bash
export R3MES_SERVING_NODE_URL=http://your-public-ip:8081
```

### 4. Start Miner

Start the miner as usual. The serving node functionality will be automatically initialized:

```bash
r3mes-miner start
```

Or using continuous mining:

```bash
r3mes-miner continuous --batch-size 10
```

## Verification

### Check Serving Node Status

1. **Check miner logs** for serving node initialization:
   ```
   ✅ Serving node initialized with LoRAs: legal, medical
   ✅ Inference server started on 0.0.0.0:8081
   ✅ Successfully registered as serving node at http://your-ip:8081
   ```

2. **Check backend API** for registered serving nodes:
   ```bash
   curl http://localhost:8000/api/serving-node/list?lora=legal
   ```

3. **Test inference endpoint** directly:
   ```bash
   curl -X POST http://localhost:8081/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "What is a contract?", "lora_name": "legal"}'
   ```

4. **Check health endpoint**:
   ```bash
   curl http://localhost:8081/health
   ```

## How It Works

### Initialization Flow

1. Miner starts and loads base model
2. If `R3MES_ENABLE_SERVING_NODE=true`:
   - Fetches available LoRAs from backend
   - Selects at least one LoRA (currently selects first available)
   - Downloads selected LoRAs from IPFS
   - Loads LoRAs into the model using PEFT
   - Registers with backend as serving node
   - Starts inference HTTP server (port 8081)
   - Starts heartbeat thread (sends heartbeat every 30 seconds)

### Request Flow

1. User sends chat request to backend with LoRA selection
2. Backend queries database for serving nodes with that LoRA
3. Backend routes request to selected serving node
4. Serving node performs inference with selected LoRA
5. Serving node streams response back to backend
6. Backend streams response to user

### Fallback

If no serving nodes are available for the requested LoRA, the backend falls back to local inference (if available).

## LoRA Management

### Registering LoRAs

LoRAs must be registered in the backend before miners can download them:

```bash
curl -X POST http://localhost:${BACKEND_PORT:-8000}/api/lora/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "legal",
    "ipfs_hash": "Qm...",
    "description": "Legal domain LoRA",
    "category": "legal",
    "version": "1.0"
  }'
```

### Listing Available LoRAs

```bash
curl http://localhost:${BACKEND_PORT:-8000}/api/lora/list
```

## Troubleshooting

### Serving Node Not Registering

- Check that `R3MES_ENABLE_SERVING_NODE=true` is set
- Verify backend URL is correct and accessible
- Check miner logs for registration errors
- Ensure at least one LoRA is available in backend

### LoRA Download Fails

- Verify IPFS daemon is running: `ipfs daemon`
- Check IPFS connection: `ipfs swarm peers`
- Verify LoRA IPFS hash is correct in backend
- Check miner logs for IPFS errors

### Inference Not Working

- Verify model is loaded correctly
- Check that LoRA adapters are loaded: check logs for "Loaded LoRA adapter"
- Test inference endpoint directly (see Verification section)
- Check that tokenizer is available (for PyTorch models)

### Heartbeat Failing

- Check network connectivity to backend
- Verify backend endpoint is correct
- Check backend logs for heartbeat errors
- Ensure serving node is registered before heartbeat starts

## Environment Variables Summary

| Variable | Description | Default |
|----------|-------------|---------|
| `R3MES_ENABLE_SERVING_NODE` | Enable serving node mode | `false` |
| `R3MES_BACKEND_URL` | Backend API URL | `http://localhost:${BACKEND_PORT:-8000}` (development) or production URL |
| `R3MES_SERVING_NODE_URL` | Full serving node endpoint URL | Auto-detected |
| `R3MES_SERVING_NODE_HOST` | Serving node host | `0.0.0.0` |
| `R3MES_SERVING_NODE_PORT` | Serving node port | `8081` |

## Advanced Configuration

### Custom LoRA Selection

Currently, the miner selects the first available LoRA. To customize selection, modify `LoRAManager.select_loras()` method.

### Load Balancing

The backend uses random selection for load balancing. Multiple serving nodes with the same LoRA will share the load.

### Heartbeat Interval

The heartbeat interval is hardcoded to 30 seconds. To change it, modify `MinerEngine.heartbeat_interval` in `engine.py`.

## Security Considerations

- Serving nodes should be behind a firewall
- Consider using TLS for serving node endpoints
- Validate requests in inference server
- Monitor serving node load and capacity
- Implement rate limiting if needed


# Serving in R3MES

Learn how to run a serving node on R3MES and provide AI inference services to users while earning rewards.

---

## What is Serving?

Serving nodes provide AI model inference services to users of the R3MES network. They load trained models (BitNet 1.58-bit with LoRA adapters) and respond to inference requests from users.

As a serving node, you:
- Load AI models and LoRA adapters
- Process inference requests from users
- Earn tokens for serving requests
- Compete with other serving nodes on quality and availability

---

## Why Become a Serving Node?

### ✅ Earn Revenue

Serving nodes earn tokens for each inference request they process. Active serving nodes with good uptime can generate steady income.

### ✅ Network Utility

By providing inference services, you make the trained AI models accessible to users, creating real utility for the network.

### ✅ Low Barrier to Entry

Serving requires less intensive hardware than mining. You can serve with a single GPU and stable internet connection.

### ✅ Flexible Operation

Serving nodes can run alongside other operations (mining, validating) if you have sufficient resources.

---

## Requirements

### Hardware

- **GPU**: NVIDIA GPU with CUDA support (RTX 3060 or better)
- **VRAM**: 8GB+ (12GB+ recommended for optimal performance)
- **RAM**: 16GB+ system RAM
- **Disk**: 50GB+ for model storage
- **Network**: Stable internet connection with low latency

### Software

- **Python**: 3.10+
- **CUDA**: NVIDIA CUDA toolkit
- **PyTorch**: CUDA-enabled PyTorch
- **Backend API**: R3MES backend inference service

### Network

- **Blockchain Node**: Connection to R3MES blockchain (can be remote)
- **API Endpoint**: Public API endpoint for serving requests
- **Registration**: Register as serving node on blockchain

---

## Quick Start

### Option 1: Desktop Launcher

1. **Download** Desktop Launcher
2. **Launch** and run setup wizard
3. **Select** "Serving" role during setup
4. **Configure** serving parameters:
   - API endpoint URL
   - Model configuration
   - LoRA adapters to load
5. **Register** on blockchain via Web Dashboard
6. **Start** serving node from launcher

[Desktop Launcher Guide →](10_desktop_launcher.md)

### Option 2: CLI

1. **Install** the Python package:
   ```bash
   pip install r3mes
   ```

2. **Configure** serving node:
   ```bash
   r3mes-serving configure
   ```
   This will prompt for:
   - API endpoint configuration
   - Model path and LoRA adapters
   - Blockchain RPC endpoint
   - Serving parameters (batch size, max requests, etc.)

3. **Register** on blockchain:
   ```bash
   r3mes-serving register --from my-key
   ```

4. **Start** serving:
   ```bash
   r3mes-serving start
   ```

### Option 3: Backend API

The serving functionality is part of the R3MES backend API. You can run the backend API with serving enabled:

```bash
# Configure backend
export ENABLE_SERVING=true
export MODEL_PATH=/path/to/models
export LORA_ADAPTERS_PATH=/path/to/adapters

# Start backend
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## How Serving Works

### Request Flow

1. **User Request**: User sends inference request via Web Dashboard or API
2. **Router Selection**: Semantic router selects appropriate LoRA adapter based on request
3. **Model Loading**: Serving node loads base model + selected LoRA adapter
4. **Inference**: Process request and generate response
5. **Response**: Return response to user
6. **Reward**: Earn tokens for serving the request

### Model Management

Serving nodes manage:
- **Base Model**: BitNet 1.58-bit base model (28GB, one-time download)
- **LoRA Adapters**: Domain-specific adapters (10-100MB each)
- **Adapter Selection**: Intelligent routing based on request content

### Quality Metrics

Serving nodes are evaluated on:
- **Response Time**: Latency from request to response
- **Availability**: Uptime percentage
- **Quality**: Response quality (user ratings, if implemented)
- **Throughput**: Requests processed per second

---

## Optimizing Your Serving Node

### GPU Selection

- **Higher VRAM**: Load multiple adapters simultaneously
- **CUDA Cores**: Faster inference processing
- **Memory Bandwidth**: Faster model loading

### Model Loading Strategy

- **Pre-load**: Pre-load frequently used adapters
- **Lazy Loading**: Load adapters on-demand to save memory
- **Adapter Pool**: Keep pool of loaded adapters for common requests

### Network Optimization

- **CDN**: Use CDN for static assets if serving web interface
- **Load Balancing**: Multiple serving nodes behind load balancer
- **Geographic Distribution**: Serve from multiple regions for lower latency

### Caching

- **Response Caching**: Cache responses for identical requests
- **Model Caching**: Cache loaded models/adapters in memory
- **Request Batching**: Batch similar requests for efficiency

---

## Monitoring Your Serving Node

### Key Metrics

Monitor these metrics:

- **Requests Served**: Total requests processed
- **Average Latency**: Response time
- **Error Rate**: Failed requests percentage
- **GPU Utilization**: GPU usage during inference
- **Memory Usage**: VRAM and system RAM usage
- **Earnings**: Tokens earned from serving

### Desktop Launcher

The Desktop Launcher shows:
- Serving node status
- Request statistics
- Earnings and rewards
- Resource usage

### Web Dashboard

View serving statistics at [dashboard.r3mes.network](https://dashboard.r3mes.network):
- Serving node details
- Request history
- Earnings over time
- Network comparison

### API Endpoints

```bash
# Health check
curl http://your-serving-node:8000/health

# Serving stats
curl http://your-serving-node:8000/api/serving/stats
```

---

## Earnings and Economics

### Earning Model

Serving nodes earn tokens based on:
- **Request Volume**: More requests = more earnings
- **Quality**: Higher quality responses may earn more
- **Availability**: Consistent uptime increases request volume
- **Competition**: Compete with other serving nodes

### Payment Flow

1. User pays credits/tokens for inference request
2. Credit is deducted from user balance
3. Serving node receives portion of payment
4. Rewards accumulate and can be claimed

### Claiming Rewards

```bash
# Claim serving rewards
r3mes-serving claim-rewards --from my-key
```

Or use the Web Dashboard to claim rewards.

---

## Troubleshooting

### Model Loading Fails

- Check disk space (50GB+ required)
- Verify model file integrity
- Check CUDA/PyTorch installation
- Review logs for specific errors

### High Latency

- Optimize model loading strategy
- Use GPU acceleration
- Reduce batch size if needed
- Check network latency

### Low Earnings

- Improve uptime (24/7 recommended)
- Optimize response times
- Ensure good network connectivity
- Check if serving requests (monitor logs)

### GPU Out of Memory

- Reduce number of loaded adapters
- Use smaller batch sizes
- Implement lazy loading
- Use model quantization if available

**More Help:** [Troubleshooting Guide →](TROUBLESHOOTING.md)

---

## Next Steps

- [Mining Guide →](02_mining.md) - Train AI models
- [Validating Guide →](03_validating.md) - Run a validator
- [Staking Guide →](07_staking.md) - Stake tokens
- [API Reference →](api/) - Complete API documentation

---

**Need Help?** [Join Discord](https://discord.gg/r3mes) | [View Documentation](00_home.md)


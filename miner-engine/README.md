# R3MES Miner Engine

🚀 **Decentralized AI Training Engine** with BitNet 1.58-bit quantization, LoRA adapters, and blockchain coordination.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](#testing)

## 🌟 Features

### Core Training Engine
- **🔢 BitNet 1.58-bit Training**: Ultra-efficient quantized neural network training
- **🎯 LoRA Adapters**: Low-rank adaptation for parameter-efficient fine-tuning
- **⚡ GPU Optimization**: Automatic VRAM profiling and adaptive batch sizing
- **🔄 Deterministic Training**: Reproducible results across different hardware

### Decentralized Infrastructure
- **⛓️ Blockchain Integration**: Seamless coordination with R3MES blockchain
- **📦 IPFS Storage**: Distributed gradient storage and retrieval
- **🌐 P2P Communication**: Arrow Flight for high-performance data transfer
- **🔐 Privacy Protection**: Intel SGX integration for secure computation

### Production Ready
- **📊 Real-time Monitoring**: Web-based dashboard with performance metrics
- **🛠️ Advanced Configuration**: Hot-reloadable YAML configuration system
- **🔧 CLI Management**: Comprehensive command-line tools
- **🐳 Docker Support**: Complete containerization with Docker Compose

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/r3mes/miner-engine.git
cd miner-engine

# Run automated setup
python scripts/setup.py --environment development

# Or manual installation
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (set your private key and blockchain URL)
nano .env
```

### 3. Start Mining

```bash
# Start with management tool
python scripts/r3mes-manager.py start

# Or start directly
python miner_engine.py

# Start with monitoring dashboard
python scripts/r3mes-manager.py start --daemon
python scripts/r3mes-manager.py monitor
```

### 4. Monitor Progress

- **Dashboard**: http://localhost:8080
- **Stats API**: http://localhost:8080/api/metrics
- **Logs**: `tail -f logs/miner-engine.log`

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        R3MES MINER ENGINE ARCHITECTURE                          │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI Tools     │    │  Web Dashboard  │    │   Management    │
│   (r3mes-cli)   │    │   (FastAPI)     │    │   (Manager)     │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    MINER ENGINE CORE    │
                    │   (Async Processing)    │
                    └────────────┬────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                       │                        │
        ▼                       ▼                        ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ MINER NODE   │    │   SERVING NODE   │    │  PROPOSER NODE   │
│              │    │                  │    │                  │
│• BitNet      │    │• Inference       │    │• Gradient        │
│  Training    │    │  Server          │    │  Aggregation     │
│• LoRA        │    │• Model Serving   │    │• IPFS Hash       │
│  Adapters    │    │• Load Balancing  │    │  Lookup          │
│• Gradient    │    │• Arrow Flight    │    │• Blockchain      │
│  Compression │    │• Stats HTTP      │    │  Query           │
│• IPFS Upload │    │                  │    │                  │
└──────┬───────┘    └─────────┬────────┘    └─────────┬────────┘
       │                      │                       │
       └──────────────────────┼───────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   CORE MODULES    │
                    ├───────────────────┤
                    │• BitLinear Layer  │
                    │• LoRA Trainer     │
                    │• Verification     │
                    │• Serialization    │
                    │• Compression      │
                    │• Coordinator      │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │  BRIDGE LAYER     │
                    ├───────────────────┤
                    │• Blockchain RPC   │
                    │• Crypto Signing   │
                    │• Arrow Flight     │
                    │• Proof of Work    │
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  BLOCKCHAIN  │    │     IPFS     │    │  MONITORING  │
│              │    │              │    │              │
│• Go Node     │    │• Gradient    │    │• Dashboard   │
│  gRPC        │    │  Storage     │    │• Metrics     │
│• Tendermint  │    │• Hash        │    │• Alerts      │
│• Cosmos SDK  │    │  Retrieval   │    │• Analytics   │
│• Seed Sync   │    │• Pinning     │    │• Logs        │
└──────────────┘    └──────────────┘    └──────────────┘
```

## 📋 Components

### Core Training (`core/`)
- **BitLinear Layer**: 1.58-bit quantized layers with LoRA adapters
- **LoRA Trainer**: Parameter-efficient training with frozen backbones
- **Verification**: Deterministic hash verification across architectures
- **Serialization**: Efficient gradient compression and storage
- **Coordination**: Atomic transaction management and rollback

### Blockchain Bridge (`bridge/`)
- **gRPC Client**: Communication with R3MES blockchain node
- **Crypto Operations**: Secp256k1 signing and verification
- **Arrow Flight**: High-performance tensor transfer
- **Transaction Builder**: Cosmos SDK compatible transactions

### Node Types (`r3mes/`)
- **Miner Node**: BitNet training with LoRA adapters
- **Serving Node**: Model inference and serving
- **Proposer Node**: Gradient aggregation and consensus

### Utilities (`utils/`)
- **Performance Monitor**: Advanced metrics collection and analysis
- **Monitoring Dashboard**: Real-time web-based monitoring
- **Advanced Config**: Hot-reloadable configuration management
- **API Documentation**: Automatic API docs generation

## ⚙️ Configuration

### Environment Variables

```bash
# Core Configuration
R3MES_ENV=development                    # Environment mode
R3MES_BLOCKCHAIN_URL=localhost:9090      # Blockchain endpoint
R3MES_PRIVATE_KEY=your_private_key       # Signing key

# Performance Tuning
R3MES_BATCH_SIZE=4                       # Training batch size
R3MES_LORA_RANK=8                        # LoRA adapter rank
R3MES_MAX_MEMORY_MB=8192                 # Memory limit
R3MES_GPU_MEMORY_FRACTION=0.8            # GPU memory usage

# Node Roles
R3MES_ENABLE_MINER=true                  # Enable miner functionality
R3MES_ENABLE_SERVING_NODE=false          # Enable serving node
R3MES_ENABLE_PROPOSER_NODE=false         # Enable proposer node
```

### Configuration Files

- `config/default.yaml` - Default settings for all environments
- `config/local.yaml` - Local development overrides
- `config/production.yaml` - Production optimized settings
- `.env` - Environment-specific variables

## 🛠️ Management Tools

### R3MES Manager CLI

```bash
# Start/stop miner
python scripts/r3mes-manager.py start --daemon
python scripts/r3mes-manager.py stop
python scripts/r3mes-manager.py restart

# Monitor status
python scripts/r3mes-manager.py status
python scripts/r3mes-manager.py logs --follow

# System diagnostics
python scripts/r3mes-manager.py diagnostics
python scripts/r3mes-manager.py cleanup

# Monitoring dashboard
python scripts/r3mes-manager.py monitor
```

### Setup Script

```bash
# Automated setup for different environments
python scripts/setup.py --environment development
python scripts/setup.py --environment production --docker --systemd
```

## 🐳 Docker Deployment

### Quick Start with Docker Compose

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f r3mes-miner

# Scale miners
docker-compose up -d --scale r3mes-miner=3
```

### Production Deployment

```bash
# Create production environment
python scripts/setup.py --environment production --docker

# Deploy with Docker Swarm
docker stack deploy -c docker-compose.yml r3mes
```

## 📊 Monitoring & Analytics

### Real-time Dashboard
- **Performance Metrics**: CPU, memory, GPU utilization
- **Training Progress**: Loss curves, gradient statistics
- **Network Status**: Blockchain connectivity, IPFS health
- **System Health**: Error rates, uptime monitoring

### API Endpoints
- `GET /api/metrics` - Current performance metrics
- `GET /api/status` - System status and health
- `GET /api/profiles` - Operation profiling data
- `GET /api/recommendations` - Optimization suggestions

### Performance Profiling

```python
from utils.performance_monitor import get_global_monitor

# Profile code blocks
monitor = get_global_monitor()
with monitor.profile("training_step"):
    # Your training code here
    pass

# Get performance summary
summary = monitor.get_metrics_summary()
recommendations = monitor.get_optimization_recommendations()
```

## 🧪 Testing

### Run Test Suite

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_trainer.py -v
python -m pytest tests/test_integration_full.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### Integration Tests

```bash
# Full end-to-end integration test
python -m pytest tests/test_integration_full.py::TestFullIntegration::test_end_to_end_mining_workflow -v

# Performance benchmarks
python -m pytest tests/test_integration_full.py::TestFullIntegration::test_performance_benchmarks -v
```

## 🔐 Security & Privacy

### Intel SGX Integration

```bash
# Install SGX SDK (see privacy/sgx_integration_guide.md)
bash privacy/build_sgx.sh

# Enable SGX in configuration
R3MES_ENABLE_SGX=true
R3MES_SGX_ENCLAVE_PATH=privacy/enclave/r3mes_enclave.signed.so
```

### Security Best Practices

- **Private Key Management**: Store keys securely, never commit to version control
- **Network Security**: Use TLS for production deployments
- **Access Control**: Implement proper authentication and authorization
- **Audit Logging**: Enable comprehensive audit trails

## 📚 Documentation

### API Documentation

```bash
# Generate API documentation
python -m utils.api_doc_generator --project-root . --output-dir docs/api

# View generated docs
open docs/api/README.md
```

### Comprehensive Guides

- **[Installation Guide](INSTALLATION.md)** - Detailed installation instructions
- **[Configuration Guide](config/README.md)** - Complete configuration reference
- **[SGX Integration Guide](privacy/sgx_integration_guide.md)** - Intel SGX setup
- **[API Documentation](docs/api/)** - Auto-generated API docs
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Common issues and solutions

## 🤝 Contributing

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/r3mes/miner-engine.git
cd miner-engine
python scripts/setup.py --environment development --install-optional

# Install pre-commit hooks
pre-commit install

# Run development server
python miner_engine.py --test-mode
```

### Code Quality

- **Linting**: `flake8`, `black`, `isort`
- **Type Checking**: `mypy`
- **Testing**: `pytest` with coverage
- **Documentation**: Comprehensive docstrings and examples

## 📈 Performance

### Benchmarks

| Operation | Time (ms) | Memory (MB) | Notes |
|-----------|-----------|-------------|-------|
| BitLinear Forward | 2.3 | 45 | 768→768, rank=8 |
| LoRA Training Step | 15.7 | 120 | Batch size 4 |
| Gradient Serialization | 0.8 | 12 | Gzip compression |
| IPFS Upload | 45.2 | 8 | 1MB gradient file |

### Optimization Tips

- **Batch Size**: Increase for better GPU utilization
- **LoRA Rank**: Balance between quality and efficiency
- **Memory Management**: Use gradient checkpointing for large models
- **Compression**: Enable gradient compression for bandwidth savings

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size and LoRA rank
R3MES_BATCH_SIZE=2
R3MES_LORA_RANK=4
```

**Blockchain Connection Failed**
```bash
# Check blockchain node status
python scripts/r3mes-manager.py diagnostics
```

**IPFS Upload Timeout**
```bash
# Increase timeout and check IPFS node
R3MES_TIMEOUT_SECONDS=60
```

### Debug Mode

```bash
# Enable debug logging
R3MES_LOG_LEVEL=DEBUG
R3MES_ENABLE_DEBUG_ENDPOINTS=true

# Run diagnostics
python scripts/r3mes-manager.py diagnostics
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **BitNet**: Inspired by Microsoft's BitNet architecture
- **LoRA**: Based on Microsoft's Low-Rank Adaptation technique
- **Cosmos SDK**: Blockchain integration framework
- **IPFS**: Distributed storage protocol
- **Intel SGX**: Trusted execution environment

## 📞 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/r3mes/miner-engine/issues)
- **Discussions**: [GitHub Discussions](https://github.com/r3mes/miner-engine/discussions)
- **Discord**: [R3MES Community](https://discord.gg/r3mes)

---

**Built with ❤️ by the R3MES Team**
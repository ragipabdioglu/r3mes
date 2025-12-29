# R3MES - Revolutionary Resource-Efficient Machine Learning Ecosystem

[![Production Ready](https://img.shields.io/badge/status-production--ready-brightgreen)](https://github.com/r3mes-network/r3mes)
[![Docker](https://img.shields.io/badge/docker-compose-blue)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Go Version](https://img.shields.io/badge/go-1.24+-blue)](https://golang.org/)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue)](https://python.org/)
[![Node Version](https://img.shields.io/badge/node-18+-green)](https://nodejs.org/)

**R3MES** is a revolutionary blockchain protocol that combines Proof of Useful Work (PoUW) consensus mechanism with AI model training. Unlike traditional blockchain mining that wastes energy on meaningless computations, R3MES enables miners to earn tokens by training AI models, creating a self-sustaining ecosystem for decentralized machine learning.

## 🚀 Quick Start

### Prerequisites

- **Docker 20.10+** and **Docker Compose 2.0+** (script otomatik kurar)
- **Minimum**: 4 vCPU, 8GB RAM, 75GB disk
- **Recommended**: 6 vCPU, 12GB RAM, 200GB disk
- (Optional) NVIDIA GPU with nvidia-container-toolkit for mining

### ⚡ One-Command Deployment (Recommended)

**Testnet:**
```bash
git clone https://github.com/r3mes-network/r3mes.git R3MES && cd R3MES
bash scripts/quick_deploy.sh --domain testnet.r3mes.network --email admin@r3mes.network
```

**Mainnet:**
```bash
git clone https://github.com/r3mes-network/r3mes.git R3MES && cd R3MES
bash scripts/quick_deploy.sh --domain r3mes.network --email admin@r3mes.network --mainnet
```

See **[Quick Deploy Guide](QUICK_DEPLOY.md)** for more options and details.

### Production Deployment (Docker - Manual)

#### Quick Deploy (Recommended - One Command)

**Testnet:**
```bash
git clone https://github.com/r3mes-network/r3mes.git R3MES
cd R3MES
bash scripts/quick_deploy.sh --domain testnet.r3mes.network --email admin@r3mes.network
```

**Mainnet:**
```bash
git clone https://github.com/r3mes-network/r3mes.git R3MES
cd R3MES
bash scripts/quick_deploy.sh --domain r3mes.network --email admin@r3mes.network --mainnet
```

The quick deploy script automatically:
- ✅ Checks/installs Docker
- ✅ Creates Docker secrets
- ✅ Configures environment variables
- ✅ Deploys all services

#### Manual Deploy (Step by Step)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/r3mes-network/r3mes.git R3MES
   cd R3MES
   ```

2. **Create Docker secrets:**
   ```bash
   bash scripts/create_secrets.sh
   ```
   This will create secure password files for PostgreSQL, Redis, and Grafana.

3. **Configure environment:**
   ```bash
   cd docker
   cp env.production.example .env.production
   nano .env.production  # Set DOMAIN, EMAIL, and other non-sensitive values
   ```

4. **Deploy:**
   ```bash
   bash scripts/deploy_production_docker.sh
   ```

For detailed deployment instructions, see:
- **[Docker Production Guide](docker/README_PRODUCTION.md)** - Complete Docker deployment guide
- **[Testnet Deployment Guide](docs/TESTNET_DEPLOYMENT.md)** - Testnet-specific deployment instructions
- **[Contabo VPS Deployment](docker/CONTOBO_DEPLOYMENT_GUIDE.md)** - Step-by-step VPS setup
- **[Docker Secrets Guide](docker/DOCKER_SECRETS_GUIDE.md)** - Secure secret management

## 📋 What's Included

### Core Services

- **Blockchain Node** (`remesd`) - Cosmos SDK-based blockchain with PoUW consensus
- **Backend API** (FastAPI) - AI inference service with semantic routing
- **Miner Engine** (Python) - GPU-based AI model training
- **Web Dashboard** (Next.js) - Unified user interface
- **IPFS** - Distributed storage for training data
- **PostgreSQL** - Relational database
- **Redis** - Caching layer

### Production Features

- ✅ **Docker Secrets** - Secure password management
- ✅ **SSL/HTTPS** - Automatic Let's Encrypt certificates via Certbot
- ✅ **Monitoring Stack** - Prometheus, Grafana, Alertmanager
- ✅ **Automated Backups** - Daily PostgreSQL backups with 7-day retention
- ✅ **Health Checks** - Comprehensive service health monitoring
- ✅ **GPU Support** - NVIDIA GPU integration for mining
- ✅ **Auto-Migrations** - Automatic database schema updates

## 🏗️ Architecture

```
┌─────────────────┐
│   Web Dashboard │
│   (Next.js)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Backend API    │
│   (FastAPI)     │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────────┐
│Postgres│ │  Blockchain  │
│  Redis │ │ (Cosmos SDK) │
└────────┘ └──────┬───────┘
                  │
         ┌────────┴────────┐
         ▼                 ▼
   ┌──────────┐      ┌──────────┐
   │  Miner   │      │ Validator│
   │  Engine  │      │   Node   │
   │ (Python) │      │   (Go)   │
   └──────────┘      └──────────┘
```

## 📚 Documentation

### Getting Started
- **[Project Overview](docs/PROJECT_OVERVIEW.md)** - High-level project overview
- **[Technical Analysis Report](TECHNICAL_ANALYSIS_REPORT.md)** - Comprehensive technical analysis (2025-01-14)
- **[Architecture Overview](docs/ARCHITECTURE_OVERVIEW.md)** - System architecture

### Deployment
- **[Docker Production Guide](docker/README_PRODUCTION.md)** - Complete Docker deployment
- **[Contabo VPS Guide](docker/CONTOBO_DEPLOYMENT_GUIDE.md)** - VPS deployment instructions
- **[Docker Secrets](docker/DOCKER_SECRETS_GUIDE.md)** - Secret management guide

### Development
- **[Architecture](docs/ARCHITECTURE_OVERVIEW.md)** - System architecture
- **[API Reference](docs/13_api_reference.md)** - API documentation
- **[Testing Guide](TEST_GUIDE.md)** - Testing instructions

### Full Documentation Index
See **[docs/README.md](docs/README.md)** for complete documentation index.

## 🔧 Key Features

### Proof of Useful Work (PoUW)
Miners earn tokens by training AI models instead of wasting energy on meaningless computations.

### Model-Agnostic Architecture
R3MES supports any AI model architecture. Genesis period uses BitNet b1.58.

### Distributed Training
Multiple miners contribute to training the same model, with gradients aggregated on-chain.

### Economic Incentives
- **Miners**: Earn tokens by training models
- **Validators**: Earn fees by verifying gradients
- **Users**: Pay credits to use inference services

## 🛠️ Development

### Local Development Setup

1. **Backend:**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   python -m app.main
   ```

2. **Frontend:**
   ```bash
   cd web-dashboard
   npm install
   npm run dev
   ```

3. **Blockchain:**
   ```bash
   cd remes
   make install
   remesd init validator --chain-id remes-mainnet
   remesd start
   ```

### Testing

```bash
# Run all tests
make test

# Run specific test suite
cd backend && pytest
cd web-dashboard && npm test
```

## 📦 Services Overview

| Service | Port | Description |
|---------|------|-------------|
| Frontend | 3000 (internal) | Next.js web dashboard |
| Backend | 8000 (internal) | FastAPI inference service |
| Nginx | 80, 443 | Reverse proxy + SSL |
| PostgreSQL | 5432 (internal) | Database |
| Redis | 6379 (internal) | Cache |
| IPFS | 5001, 4001 (internal) | Distributed storage |
| Blockchain | 26657, 9090 (internal) | Cosmos SDK node |
| Prometheus | 9090 (internal) | Metrics collection |
| Grafana | 3001 (internal) | Monitoring dashboards |

## 🔒 Security

- **Docker Secrets** - Secure password storage
- **SSL/HTTPS** - Automatic certificate management
- **CORS Protection** - Strict CORS configuration
- **Rate Limiting** - API protection
- **Input Validation** - Comprehensive validation
- **Non-root Containers** - Security best practices

## 📊 Monitoring

The production stack includes a complete monitoring solution:

- **Prometheus** - Metrics collection and storage
- **Grafana** - Visualization dashboards
- **Alertmanager** - Alert routing and notification
- **Node Exporter** - System metrics
- **Redis Exporter** - Redis metrics
- **PostgreSQL Exporter** - Database metrics

Access Grafana at `http://your-domain:3001` (configured via Nginx).

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Website**: [https://r3mes.network](https://r3mes.network)
- **Documentation**: [docs/README.md](docs/README.md)
- **API Reference**: [https://api.r3mes.network/docs](https://api.r3mes.network/docs)
- **Block Explorer**: [https://explorer.r3mes.network](https://explorer.r3mes.network)
- **GitHub**: [https://github.com/r3mes-network/r3mes](https://github.com/r3mes-network/r3mes)
- **Discord**: [https://discord.gg/r3mes](https://discord.gg/r3mes)
- **Twitter**: [https://twitter.com/r3mes_network](https://twitter.com/r3mes_network)

## 📞 Support

For support and questions:
- Check the [documentation](docs/README.md)
- Review [troubleshooting guides](docs/TROUBLESHOOTING.md)
- Join our [Discord community](https://discord.gg/r3mes)
- Open an [issue](https://github.com/r3mes-network/r3mes/issues)

---

**Last Updated**: 2025-01-01  
**Version**: 1.0.0  
**Status**: Production Ready  
**Maintained by**: R3MES Foundation


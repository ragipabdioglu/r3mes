# Project Structure

Technical overview of the R3MES codebase organization.

---

## Repository Layout

```
R3MES/
├── remes/                    # Blockchain (Cosmos SDK)
├── miner-engine/             # Python Miner Engine
├── web-dashboard/            # Next.js Web Dashboard
├── desktop-launcher-tauri/   # Tauri Desktop Launcher
├── backend/                  # FastAPI Backend Service
├── sdk/                      # SDKs (Python, Go, JavaScript)
├── scripts/                  # Utility Scripts
├── docs/                     # Documentation
├── docker/                   # Docker Configurations
├── k8s/                      # Kubernetes Manifests
└── monitoring/               # Prometheus & Grafana
```

---

## Core Components

### Blockchain (`remes/`)

Cosmos SDK-based blockchain for consensus and coordination.

| Directory | Purpose |
|-----------|---------|
| `app/` | Application configuration |
| `cmd/remesd/` | CLI binary entry point |
| `x/remes/` | Custom PoUW module |
| `x/remes/keeper/` | Business logic |
| `x/remes/types/` | Type definitions |
| `proto/` | Protocol Buffer definitions |

**Key Files:**
- `app/app.go` - Main app configuration
- `x/remes/keeper/msg_server.go` - Transaction handlers
- `x/remes/keeper/query_server.go` - Query handlers

### Miner Engine (`miner-engine/`)

Python-based AI training engine.

| Directory | Purpose |
|-----------|---------|
| `r3mes/cli/` | CLI commands and wizard |
| `r3mes/miner/` | Core miner engine |
| `core/` | Training logic (BitLinear, gradients) |
| `bridge/` | Blockchain communication |
| `utils/` | Utilities (GPU, IPFS, etc.) |

**Key Files:**
- `r3mes/miner/engine.py` - Main miner engine
- `core/trainer.py` - Training logic
- `bridge/blockchain_client.py` - gRPC client

### Web Dashboard (`web-dashboard/`)

Next.js 14 web application.

| Directory | Purpose |
|-----------|---------|
| `app/` | Next.js App Router pages |
| `components/` | React components |
| `lib/` | Utilities and helpers |
| `hooks/` | Custom React hooks |
| `providers/` | Context providers |

**Key Files:**
- `app/page.tsx` - Home page
- `app/dashboard/page.tsx` - Dashboard
- `components/NetworkExplorer.tsx` - 3D globe visualization

### Desktop Launcher (`desktop-launcher-tauri/`)

Cross-platform desktop application.

| Directory | Purpose |
|-----------|---------|
| `src/` | React frontend |
| `src-tauri/src/` | Rust backend |

**Key Files:**
- `src/App.tsx` - Main React component
- `src-tauri/src/main.rs` - Rust entry point
- `src-tauri/src/process_manager.rs` - Process management

### Backend Service (`backend/`)

FastAPI inference and API service.

| Directory | Purpose |
|-----------|---------|
| `app/` | FastAPI application |
| `routers/` | API route handlers |
| `services/` | Business logic |

**Key Files:**
- `app/main.py` - FastAPI entry point
- `app/inference.py` - AI inference logic

---

## SDKs (`sdk/`)

Client libraries for R3MES integration.

| SDK | Language | Package |
|-----|----------|---------|
| `sdk/python/` | Python | `r3mes-sdk` |
| `sdk/javascript/` | TypeScript | `@r3mes/sdk` |
| `sdk/go/` | Go | `github.com/r3mes/sdk-go` |

---

## Scripts (`scripts/`)

Utility and deployment scripts.

| Directory | Purpose |
|-----------|---------|
| `tests/` | Test scripts (e2e, integration) |
| `setup/` | Setup and installation |
| `systemd/` | Systemd service files |

**Key Scripts:**
- `node_control.sh` - Node start/stop/status
- `init_chain.sh` - Initialize blockchain
- `quick_install_all.sh` - Full installation

---

## Configuration Files

| File | Purpose |
|------|---------|
| `Makefile` | Build and test commands |
| `.env.example` | Environment variables template |
| `docker-compose.yml` | Docker development setup |
| `k8s/` | Kubernetes manifests |

---

## Port Mappings

| Port | Service |
|------|---------|
| 26657 | Tendermint RPC |
| 26656 | P2P Network |
| 9090 | gRPC |
| 1317 | REST API |
| 5001 | IPFS API |
| 3000 | Web Dashboard |
| 8000 | Backend Service |
| 8080 | Miner Stats Server |

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Blockchain | Go, Cosmos SDK v0.50.9, CometBFT v0.38.19 |
| Miner | Python 3.10+, PyTorch 2.1.0 |
| Dashboard | Next.js 14, TypeScript, React 18 |
| Launcher | Tauri 1.5, Rust, React |
| Backend | Python, FastAPI |

---

## Build Outputs

| Component | Output Location |
|-----------|-----------------|
| Blockchain | `remes/build/remesd` |
| Dashboard | `web-dashboard/.next/` |
| Launcher | `desktop-launcher-tauri/src-tauri/target/` |

---

## Next Steps

- [API Reference →](api-reference) - API documentation
- [Quick Start →](quick-start) - Get started
- [How It Works →](how-it-works) - Architecture overview

---

**Need Help?** [Join Discord](https://discord.gg/r3mes) | [GitHub](https://github.com/r3mes-network/r3mes)

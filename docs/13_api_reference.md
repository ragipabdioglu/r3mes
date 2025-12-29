# API Reference

R3MES provides three API protocols for integration: REST, gRPC, and WebSocket.

---

## Overview

| Protocol | Use Case | Port |
|----------|----------|------|
| REST API | HTTP/JSON requests | 8000 (Backend), 1317 (Blockchain) |
| gRPC API | Efficient binary protocol | 9090 |
| WebSocket | Real-time streaming | 8000 |

---

## REST API

### Backend API (Port 8000)

The FastAPI backend provides application-level endpoints.

**Base URL:**
- Development: `http://localhost:8000`
- Production: `https://api.r3mes.network`

#### Network & Statistics

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/network/stats` | GET | Network-wide statistics |
| `/blocks` | GET | Recent blocks (limit, offset params) |
| `/health` | GET | Health check |
| `/queue/stats` | GET | Task queue statistics |
| `/metrics` | GET | Prometheus metrics |

#### Miner Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/miner/stats/{address}` | GET | Miner statistics |
| `/miner/earnings/{address}` | GET | Earnings history |
| `/miner/hashrate/{address}` | GET | Hashrate history |

#### User & API Keys

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/user/info/{address}` | GET | User information |
| `/api-keys/create` | POST | Create API key |
| `/api-keys/list/{address}` | GET | List API keys |
| `/api-keys/revoke` | POST | Revoke API key |
| `/api-keys/delete` | DELETE | Delete API key |

#### Inference

| Endpoint | Method | Description | Rate Limit |
|----------|--------|-------------|------------|
| `/chat` | POST | AI inference (streaming) | 10/minute |

#### LoRA & Serving Nodes

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/lora/list` | GET | List LoRA adapters |
| `/api/lora/register` | POST | Register LoRA adapter |
| `/api/serving-node/register` | POST | Register serving node |
| `/api/serving-node/list` | GET | List serving nodes |

#### Leaderboards

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/leaderboard/miners` | GET | Top miners by reputation |
| `/leaderboard/validators` | GET | Top validators by trust score |

### Blockchain API (Port 1317)

Cosmos SDK REST endpoints for blockchain queries.

**Base URL:**
- Development: `http://localhost:1317`
- Production: `https://rpc.r3mes.network`

---

## gRPC API

**Base URL:** `localhost:9090` (dev) | `node.r3mes.network:9090` (prod)

### Query Service

| RPC | Description |
|-----|-------------|
| `QueryMiners` | List miners with pagination |
| `QueryStatistics` | Network statistics |
| `QueryBlocks` | Recent blocks |
| `QueryActivePool` | Active task pool |

### Transaction Service

| RPC | Description |
|-----|-------------|
| `SubmitGradient` | Submit training gradient |

---

## WebSocket API

**Base URL:** `ws://localhost:8000/ws` (dev) | `wss://api.r3mes.network/ws` (prod)

### Topics

| Topic | Update Interval | Description |
|-------|-----------------|-------------|
| `miner_stats` | 2 seconds | GPU temp, hashrate, power |
| `training_metrics` | Per step | Loss, epoch, gradient norm |
| `network_status` | 5 seconds | Active miners, block height |
| `blocks` | Per block | New block notifications |

---

## Authentication

### API Key

```bash
X-API-Key: r3mes_...
# or
Authorization: Bearer r3mes_...
```

### Rate Limiting

| Endpoint Type | Limit |
|---------------|-------|
| Chat/Inference | 10/minute |
| General API | 30/minute |
| Health/Metrics | 100/minute |

---

## SDKs

| Language | Package |
|----------|---------|
| Python | `r3mes-sdk` |
| JavaScript | `@r3mes/sdk` |
| Go | `github.com/r3mes/sdk-go` |

---

**Need Help?** [Join Discord](https://discord.gg/r3mes) | [GitHub](https://github.com/r3mes-network/r3mes)

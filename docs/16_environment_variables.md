# R3MES Environment Variables Documentation

## Genel Bakış

R3MES projesi, farklı bileşenler için environment variable'lar kullanır. Bu doküman, tüm environment variable'ları ve kullanımlarını açıklar.

---

## Backend (FastAPI)

### Database Configuration

#### Production (Docker Secrets) - RECOMMENDED

For production deployments, use Docker secrets to securely manage passwords:

```bash
# Docker secrets (production)
POSTGRES_PASSWORD_FILE=/run/secrets/postgres_password
POSTGRES_USER=r3mes
POSTGRES_DB=r3mes
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# DATABASE_URL is automatically constructed from the above components
# when POSTGRES_PASSWORD_FILE is set
```

**How it works**:
- The backend reads the password from the file specified in `POSTGRES_PASSWORD_FILE`
- It automatically constructs `DATABASE_URL` from `POSTGRES_USER`, password, `POSTGRES_HOST`, `POSTGRES_PORT`, and `POSTGRES_DB`
- This is the **secure and recommended** method for production

#### Development (Direct Environment Variables)

For local development, you can use direct environment variables:

```bash
# Database path (SQLite) - for local development
DATABASE_PATH=backend/data/r3mes.db

# PostgreSQL (development)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=r3mes
POSTGRES_PASSWORD=your_password
POSTGRES_DB=r3mes

# Or use DATABASE_URL directly
DATABASE_URL=postgresql://r3mes:your_password@localhost:5432/r3mes

# Chain JSON path
CHAIN_JSON_PATH=backend/data/chain.json
```

**Note**: In production, `POSTGRES_PASSWORD` should NOT be set directly. Use `POSTGRES_PASSWORD_FILE` instead.

### API Configuration

```bash
# API base URL
BACKEND_URL=http://localhost:8000
API_BASE_URL=http://localhost:1317

# Rate limiting
RATE_LIMIT_CHAT=10/minute
RATE_LIMIT_GET=60/minute
RATE_LIMIT_POST=30/minute

# CORS
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
R3MES_ENV=development  # development | production
R3MES_TEST_MODE=false
```

### Model Configuration

```bash
# Model loading
R3MES_USE_LLAMA3=true
R3MES_MODEL_NAME=meta-llama/Meta-Llama-3-8B

# Inference
MAX_WORKERS=1
```

### Notification Configuration

```bash
# Email (SMTP)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
SMTP_FROM=noreply@r3mes.network

# Slack
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
SLACK_CHANNEL=#r3mes-alerts

# Notification channels (comma-separated)
NOTIFICATION_CHANNELS=email,slack

# Enable/disable notifications
ENABLE_NOTIFICATIONS=true
```

### Monitoring & Logging

```bash
# Sentry
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id
R3MES_VERSION=1.0.0
HOSTNAME=r3mes-backend

# Logging
LOG_LEVEL=INFO  # DEBUG | INFO | WARNING | ERROR
ENABLE_FILE_LOGGING=true
```

### Cache Configuration

#### Production (Docker Secrets) - RECOMMENDED

For production deployments, use Docker secrets to securely manage Redis password:

```bash
# Docker secrets (production)
REDIS_PASSWORD_FILE=/run/secrets/redis_password
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0

# REDIS_URL is automatically constructed from the above components
# when REDIS_PASSWORD_FILE is set
```

**How it works**:
- The backend reads the password from the file specified in `REDIS_PASSWORD_FILE`
- It automatically constructs `REDIS_URL` from `REDIS_HOST`, password, `REDIS_PORT`, and `REDIS_DB`
- This is the **secure and recommended** method for production

#### Development (Direct Environment Variables)

For local development, you can use direct environment variables:

```bash
# Redis (development)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=  # Optional for local development
REDIS_DB=0

# Or use REDIS_URL directly
REDIS_URL=redis://localhost:6379/0
```

**Note**: In production, `REDIS_PASSWORD` should NOT be set directly. Use `REDIS_PASSWORD_FILE` instead.

---

## Miner Engine

### Blockchain Configuration

```bash
# Blockchain connection
BLOCKCHAIN_URL=localhost:9090
CHAIN_ID=remes-test

# Private key (required)
PRIVATE_KEY=your_private_key_here
```

### Model Configuration

```bash
# Model parameters
MODEL_HIDDEN_SIZE=768
LORA_RANK=8
LEARNING_RATE=0.0001

# Training
DETERMINISTIC=true
GRADIENT_ACCUMULATION_STEPS=4
TOP_K_COMPRESSION=0.1

# Multi-GPU
USE_MULTI_GPU=false
MULTI_GPU_DEVICE_IDS=0,1  # comma-separated GPU IDs
USE_DDP=false  # Use DistributedDataParallel
```

### Logging

```bash
LOG_LEVEL=INFO
USE_JSON_LOGS=false
```

---

## Frontend (Next.js)

### API Configuration

```bash
# Backend API
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
NEXT_PUBLIC_API_URL=http://localhost:1317

# Blockchain
NEXT_PUBLIC_CHAIN_ID=remes-test
NEXT_PUBLIC_RPC_URL=http://localhost:26657
```

### Analytics

```bash
# Google Analytics
NEXT_PUBLIC_GA_ID=G-XXXXXXXXXX

# Sentry
NEXT_PUBLIC_SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id
```

### Environment

```bash
NODE_ENV=development  # development | production
```

---

## Blockchain (Cosmos SDK)

### Node Configuration

```bash
# Chain ID
CHAIN_ID=remes-test

# RPC
RPC_URL=http://localhost:26657
RPC_PORT=26657

# API
API_URL=http://localhost:1317
API_PORT=1317

# gRPC
GRPC_URL=localhost:9090
GRPC_PORT=9090
```

### Network Configuration

```bash
# P2P
P2P_PORT=26656
P2P_SEEDS=seed1.example.com:26656,seed2.example.com:26656
P2P_PERSISTENT_PEERS=peer1.example.com:26656,peer2.example.com:26656
```

---

## IPFS

### IPFS Configuration

```bash
# IPFS node
IPFS_HOST=localhost
IPFS_PORT=5001
IPFS_GATEWAY_URL=http://localhost:8080/ipfs/

# IPFS API
IPFS_API_URL=http://localhost:5001/api/v0
```

---

## Desktop Launcher (Tauri)

### System Requirements

```bash
# Docker
DOCKER_REQUIRED=true
DOCKER_MIN_VERSION=20.10

# Python
PYTHON_REQUIRED=true
PYTHON_MIN_VERSION=3.9

# Node.js
NODEJS_REQUIRED=true
NODEJS_MIN_VERSION=18.0

# CUDA (optional)
CUDA_REQUIRED=false
CUDA_MIN_VERSION=11.8
```

---

## Debug Mode

### Global Debug Configuration

```bash
# Global Debug Mode
R3MES_DEBUG_MODE=true  # Enable debug mode globally
R3MES_DEBUG_LEVEL=verbose  # verbose|standard|minimal
R3MES_DEBUG_COMPONENTS=blockchain,backend,miner,launcher,frontend  # Comma-separated component list

# Debug Features
R3MES_DEBUG_LOGGING=true  # Enable enhanced logging
R3MES_DEBUG_PROFILING=true  # Enable performance profiling
R3MES_DEBUG_STATE_INSPECTION=true  # Enable state inspection
R3MES_DEBUG_TRACE=true  # Enable distributed tracing

# Logging Configuration
R3MES_DEBUG_LOG_LEVEL=TRACE  # TRACE|DEBUG|INFO|WARN|ERROR
R3MES_DEBUG_LOG_FORMAT=json  # json|text
R3MES_DEBUG_LOG_FILE=~/.r3mes/debug.log  # Path to debug log file

# Performance Profiling
R3MES_DEBUG_PROFILE_OUTPUT=~/.r3mes/profiles  # Directory for profile outputs
R3MES_DEBUG_PROFILE_INTERVAL=60  # Interval in seconds for periodic profiling

# Trace Configuration
R3MES_DEBUG_TRACE_ENABLED=true  # Enable trace collection
R3MES_DEBUG_TRACE_BUFFER_SIZE=10000  # Maximum number of traces to keep in memory
R3MES_DEBUG_TRACE_EXPORT_PATH=~/.r3mes/traces  # Path to export traces
```

**Security Note**: Debug mode should only be enabled in development/testing environments. In production, `R3MES_DEBUG_MODE=true` will trigger a security validation error.

---

## Monitoring (Prometheus/Grafana)

### Prometheus

```bash
# Prometheus
PROMETHEUS_PORT=9090
PROMETHEUS_RETENTION=30d

# Alertmanager
ALERTMANAGER_PORT=9093
```

### Grafana

```bash
# Grafana
GRAFANA_PORT=3000
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin
```

---

## CI/CD

### GitHub Actions

```bash
# Secrets required in GitHub
DOCKER_USERNAME=your_docker_username
DOCKER_PASSWORD=your_docker_password

# Deployment
DEPLOYMENT_ENV=staging  # staging | production
DEPLOYMENT_HOST=your-deployment-host.com
DEPLOYMENT_USER=deploy
DEPLOYMENT_KEY=your_ssh_private_key
```

---

## Production Checklist

### Required Variables

- [ ] `POSTGRES_URL` - PostgreSQL connection string
- [ ] `SENTRY_DSN` - Sentry error tracking
- [ ] `SMTP_*` - Email notification configuration
- [ ] `SLACK_WEBHOOK_URL` - Slack notifications
- [ ] `CORS_ALLOWED_ORIGINS` - CORS configuration
- [ ] `R3MES_ENV=production` - Environment mode

### Security Variables

- [ ] `PRIVATE_KEY` - Miner private key (keep secure!)
- [ ] `SMTP_PASSWORD` - Email password
- [ ] `POSTGRES_PASSWORD` - Database password
- [ ] `DOCKER_PASSWORD` - Docker registry password

### Optional Variables

- [ ] `REDIS_*` - Redis cache (optional)
- [ ] `NEXT_PUBLIC_GA_ID` - Google Analytics (optional)
- [ ] `USE_MULTI_GPU` - Multi-GPU training (optional)

---

## Environment File Examples

### Development (.env.development)

```bash
R3MES_ENV=development
DATABASE_PATH=backend/data/r3mes.db
LOG_LEVEL=DEBUG
ENABLE_NOTIFICATIONS=false
CORS_ALLOWED_ORIGINS=http://localhost:3000
```

### Production (.env.production)

```bash
R3MES_ENV=production
POSTGRES_URL=postgresql://user:pass@db:5432/r3mes
SENTRY_DSN=https://...
SMTP_HOST=smtp.gmail.com
SMTP_USER=alerts@r3mes.network
SLACK_WEBHOOK_URL=https://...
CORS_ALLOWED_ORIGINS=https://app.r3mes.network
LOG_LEVEL=INFO
ENABLE_NOTIFICATIONS=true
```

---

## Notes

- **Never commit `.env` files** to version control
- Use `.env.example` as a template
- Use secrets management in production (AWS Secrets Manager, HashiCorp Vault, etc.)
- Rotate sensitive credentials regularly
- Use different credentials for development, staging, and production


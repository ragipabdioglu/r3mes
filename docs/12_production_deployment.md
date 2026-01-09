# R3MES Production Deployment

## Genel Bakış

R3MES production deployment, systemd services, Docker, ve CI/CD pipeline ile production-ready ortam sağlar.

---

## 🏗️ Production Mimarisi

### Bileşenler

1. **Blockchain Node** (remesd): Cosmos SDK blockchain
2. **Miner Engine** (r3mes-miner): Python miner
3. **IPFS Daemon**: Embedded veya standalone
4. **Backend Inference Service** (r3mes-backend): FastAPI inference service
   - Async database (aiosqlite)
   - Semantic router
   - Config manager
   - Structured logging
   - API key management
5. **Web Dashboard**: Next.js 14 unified application
   - 8 pages: `/`, `/chat`, `/mine`, `/wallet`, `/network`, `/settings`, `/help`, `/onboarding`
   - React Query for data fetching
   - WalletContext & ThemeContext
   - Security headers
6. **Desktop Launcher**: Tauri application (Rust + React)

---

## 📦 Systemd Services

### remesd.service

```bash
# Service dosyası: scripts/systemd/remesd.service
[Unit]
Description=R3MES Blockchain Node
After=network.target

[Service]
Type=simple
User=remes
WorkingDirectory=/home/remes/R3MES/remes
ExecStart=/home/remes/R3MES/remes/build/remesd start --home /home/remes/.remes --chain-id remes-1
Restart=always
RestartSec=3
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
```

**Kurulum**:
```bash
sudo cp scripts/systemd/remesd.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable remesd
sudo systemctl start remesd
```

### r3mes-miner.service

```bash
# Service dosyası: scripts/systemd/r3mes-miner.service
[Unit]
Description=R3MES Miner Engine
After=network.target remesd.service ipfs.service

[Service]
Type=simple
User=remes
WorkingDirectory=/home/remes/R3MES/miner-engine
ExecStart=/bin/bash -c 'source venv/bin/activate && r3mes-miner start'
Restart=always
RestartSec=5
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
```

**Kurulum**:
```bash
sudo cp scripts/systemd/r3mes-miner.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable r3mes-miner
sudo systemctl start r3mes-miner
```

### ipfs.service

```bash
# Service dosyası: scripts/systemd/ipfs.service
[Unit]
Description=IPFS Daemon
After=network.target

[Service]
Type=simple
User=remes
ExecStart=/usr/local/bin/ipfs daemon
Restart=always
RestartSec=3
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
```

**Kurulum**:
```bash
sudo cp scripts/systemd/ipfs.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ipfs
sudo systemctl start ipfs
```

### r3mes-backend.service

```bash
# Service dosyası: scripts/systemd/r3mes-backend.service
[Unit]
Description=R3MES Backend Inference Service
After=network.target remesd.service
Requires=network.target

[Service]
Type=simple
User=remes
WorkingDirectory=/home/remes/R3MES
Environment="PYTHONPATH=/home/remes/R3MES/backend"
Environment="BASE_MODEL_PATH=checkpoints/base_model"
Environment="DATABASE_PATH=backend/database.db"
Environment="CHAIN_JSON_PATH=chain.json"
Environment="USE_SEMANTIC_ROUTER=true"
Environment="LOG_LEVEL=INFO"
Environment="ENABLE_FILE_LOGGING=true"
Environment="R3MES_ENV=production"
Environment="CORS_ALLOWED_ORIGINS=https://dashboard.r3mes.network"
ExecStart=/bin/bash -c 'source backend/venv/bin/activate && python run_backend.py'
Restart=always
RestartSec=5
LimitNOFILE=65535
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

**Kurulum**:
```bash
sudo cp scripts/systemd/r3mes-backend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable r3mes-backend
sudo systemctl start r3mes-backend
```

**Status Kontrolü**:
```bash
sudo systemctl status r3mes-backend
sudo journalctl -u r3mes-backend -f
```

---

## 🐳 Docker Deployment (Önerilen Production Yöntemi)

R3MES production deployment için **Docker Compose** kullanımı önerilir. Bu yöntem tüm servisleri tek bir komutla yönetmenizi sağlar ve production-grade özellikler içerir.

### Özellikler

- ✅ **Docker Secrets** - Güvenli şifre yönetimi
- ✅ **SSL/HTTPS** - Otomatik Let's Encrypt sertifikaları
- ✅ **Monitoring Stack** - Prometheus, Grafana, Alertmanager
- ✅ **Automated Backups** - Günlük PostgreSQL yedeklemeleri
- ✅ **Health Checks** - Kapsamlı servis sağlık kontrolü
- ✅ **GPU Support** - NVIDIA GPU entegrasyonu

### Hızlı Başlangıç

1. **Docker Secrets Oluştur:**
   ```bash
   bash scripts/create_secrets.sh
   ```

2. **Environment Dosyası Hazırla:**
   ```bash
   cd docker
   cp env.production.example .env.production
   nano .env.production  # DOMAIN, EMAIL vs. doldurun
   ```

3. **Deploy:**
   ```bash
   bash scripts/deploy_production_docker.sh
   ```

### Detaylı Dokümantasyon

- **[Docker Production Guide](../docker/README_PRODUCTION.md)** - Tam Docker deployment rehberi
- **[Contabo VPS Guide](../docker/CONTOBO_DEPLOYMENT_GUIDE.md)** - VPS kurulum adımları
- **[Docker Secrets Guide](../docker/DOCKER_SECRETS_GUIDE.md)** - Secret yönetimi

### Docker Compose Servisleri

Production stack şunları içerir:

1. **PostgreSQL** - Veritabanı
2. **Redis** - Cache katmanı
3. **IPFS** - Dağıtık depolama
4. **remesd** - Blockchain node
5. **Backend** - FastAPI inference service
6. **Frontend** - Next.js dashboard
7. **Nginx** - Reverse proxy + SSL
8. **Certbot** - Let's Encrypt otomasyonu
9. **Prometheus** - Metrics toplama
10. **Grafana** - Monitoring dashboards
11. **Alertmanager** - Alert yönetimi
12. **Exporters** - Node, Redis, PostgreSQL metrics
13. **Postgres Backup** - Otomatik yedekleme

### Monitoring

Production stack tam bir monitoring çözümü içerir:

- **Prometheus** - Metrics toplama ve depolama
- **Grafana** - Görselleştirme dashboards
- **Alertmanager** - Alert routing ve bildirimler
- **Exporters** - Sistem, Redis, PostgreSQL metrikleri

Grafana'ya `http://your-domain:3001` üzerinden erişilebilir (Nginx üzerinden yapılandırılabilir).

### Notification Service

The notification service provides automated alerts for system events:

- **Database Connection Failures**: Critical alerts when database connections fail
- **Blockchain Connection Failures**: Critical alerts when blockchain RPC/REST connections fail
- **High Error Rates**: High priority alerts when API error rate exceeds threshold (default: 10%)
- **Health Check Failures**: Critical alerts when health checks fail

**Configuration:**
- `NOTIFICATION_CHANNELS`: Comma-separated list (default: `email,slack`)
- `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`: Email configuration
- `SLACK_WEBHOOK_URL`: Slack webhook URL
- `ERROR_RATE_THRESHOLD`: Error rate threshold (default: `0.1` = 10%)
- `ERROR_RATE_CHECK_INTERVAL`: Check interval in seconds (default: `60`)

See `docker/env.production.example` for full configuration options.

### Semantic Router

The semantic router is enabled by default and uses embedding-based similarity to intelligently select LoRA adapters.

**Configuration:**
- `USE_SEMANTIC_ROUTER`: Enable/disable (default: `true`)
- `SEMANTIC_ROUTER_THRESHOLD`: Similarity threshold (default: `0.7`)

**Note**: Semantic router works on CPU and does not require CUDA. It uses the `all-MiniLM-L6-v2` model which is lightweight and fast.

### Blockchain Indexer

The blockchain indexer automatically indexes blockchain events for historical analytics:

- **Transaction-level Parsing**: Parses Cosmos SDK events from transactions
- **Batch Processing**: Processes blocks in batches for better performance
- **Indexing Lag Monitoring**: Tracks how many blocks behind the indexer is
- **Health Check**: `/health/indexer` endpoint for indexer status

**Configuration:**
- `INDEXER_BATCH_SIZE`: Number of blocks to process in batch (default: `10`)

**Health Check:**
```bash
curl http://localhost:8000/health/indexer
```

Response includes:
- `running`: Whether indexer is running
- `last_indexed_height`: Last indexed block height
- `indexing_lag`: Number of blocks behind
- `lag_percentage`: Percentage of blocks behind

### Eski Docker Compose (Legacy)

Aşağıdaki yapılandırma eski versiyonlar için referans amaçlıdır:

```yaml
# miner-engine/docker/docker-compose.yml
version: '3.8'

services:
  blockchain-node:
    build:
      context: ../../remes
      dockerfile: docker/Dockerfile.go
    ports:
      - "26657:26657"
      - "1317:1317"
      - "9090:9090"
    
  miner:
    build:
      context: ..
      dockerfile: docker/Dockerfile.nvidia
    depends_on:
      - blockchain-node
    environment:
      - R3MES_NODE_GRPC_URL=blockchain-node:9090
      - R3MES_IPFS_URL=http://ipfs:5001
    volumes:
      - ~/.r3mes:/root/.r3mes
    
  ipfs:
    image: ipfs/kubo:latest
    ports:
      - "5001:5001"
      - "4001:4001"
  
  backend:
    build:
      context: ../../backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    depends_on:
      - blockchain-node
    environment:
      - BASE_MODEL_PATH=checkpoints/base_model
      - DATABASE_PATH=backend/database.db
      - CHAIN_JSON_PATH=chain.json
    volumes:
      - ~/.r3mes/checkpoints:/app/checkpoints
      - ~/.r3mes/backend:/app/backend
```

**Kullanım**:
```bash
cd miner-engine/docker
docker-compose up -d
```

---

## 🔄 CI/CD Pipeline

### GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-go@v4
        with:
          go-version: '1.24'
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: cd remes && go test ./...
      - run: cd miner-engine && pytest tests/
```

---

## 🔒 Security Considerations

### Firewall Rules

```bash
# Allow only necessary ports
sudo ufw allow 26656/tcp  # P2P
sudo ufw allow 26657/tcp  # RPC
sudo ufw allow 9090/tcp   # gRPC
sudo ufw allow 1317/tcp   # REST API
sudo ufw allow 8000/tcp   # Backend Inference Service
sudo ufw allow 80/tcp     # HTTP
sudo ufw allow 443/tcp    # HTTPS
```

### HTTPS Enforcement

- Let's Encrypt SSL certificates
- Nginx reverse proxy with SSL
- Next.js middleware HTTPS enforcement

### Authentication

- API key authentication
- JWT tokens for dashboard access
- Rate limiting

---

## 🚀 CI/CD Pipeline

### GitHub Actions Workflows

**CI Pipeline** (`.github/workflows/ci.yml`):
- Lint checks (Python, TypeScript, Go)
- Unit tests (Backend, Blockchain, Frontend)
- Build verification
- E2E tests (Playwright)

**CD Pipeline** (`.github/workflows/cd.yml`):
- Security scanning (Bandit, Safety, npm audit, Trivy)
- Docker image building and pushing
- Staging deployment (automatic on main branch)
- Production deployment (on version tags)
- Blue-green deployment strategy
- Rollback mechanism

**Rollback Workflow** (`.github/workflows/rollback.yml`):
- Manual rollback to previous version
- Environment selection (staging/production)
- Health check verification

### Deployment Process

1. **Staging Deployment**:
   - Automatic on push to `main` branch
   - Runs smoke tests after deployment
   - E2E tests against staging environment
   - Slack notifications on failure

2. **Production Deployment**:
   - Triggered by version tags (`v*`)
   - Requires manual approval
   - Creates database backup before deployment
   - Blue-green deployment (zero downtime)
   - Smoke tests after deployment
   - Automatic rollback on failure

### Smoke Tests

**Script**: `scripts/smoke-tests.sh`

**Tests**:
- Frontend pages (Homepage, Docs, Developers, Community)
- API health check
- Network stats endpoint
- Blockchain RPC status

**Usage**:
```bash
./scripts/smoke-tests.sh https://staging.r3mes.network http://staging-api.r3mes.network
```

## 💾 Database Migration (PostgreSQL)

### Setup

**Script**: `scripts/setup_postgres.sh`

**Features**:
- Docker Compose PostgreSQL setup
- Automatic database initialization
- Connection pooling configuration
- Health checks

**Usage**:
```bash
cd docker
./scripts/setup_postgres.sh
```

### Migration from SQLite

**Script**: `scripts/migrate_sqlite_to_postgres.py`

**Process**:
1. Export data from SQLite
2. Import to PostgreSQL
3. Verify data integrity
4. Switch application to PostgreSQL

**Usage**:
```bash
# Development
export DATABASE_URL=postgresql://user:password@localhost:5432/r3mes
python scripts/migrate_sqlite_to_postgres.py

# Production: Use environment variable (no localhost)
export DATABASE_URL=postgresql://user:password@prod-db.r3mes.network:5432/r3mes
python scripts/migrate_sqlite_to_postgres.py
```

### Connection Pooling

**Configuration**:
- Min pool size: 5 (development), 10 (production)
- Max pool size: 20 (development), 50 (production)
- Connection timeout: 60 seconds

**Environment Variables**:
- `DATABASE_TYPE=postgresql`
- `DATABASE_URL=postgresql://user:password@host:port/dbname`
- `DATABASE_POOL_MIN_SIZE=10`
- `DATABASE_POOL_MAX_SIZE=50`

### Performance Optimization

**PostgreSQL Tuning** (Production):
- Shared buffers: 256MB
- Effective cache size: 1GB
- Maintenance work mem: 64MB
- Max connections: 200
- WAL settings optimized for write-heavy workloads

## 📊 Monitoring & Alerting

### Prometheus

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'remesd'
    static_configs:
      - targets: ['${PROMETHEUS_REMESD_TARGET:-remesd.r3mes.network:26660}']
  
  - job_name: 'r3mes-miner'
    static_configs:
      - targets: ['${PROMETHEUS_MINER_TARGET:-miner.r3mes.network:8080}']
  
  - job_name: 'r3mes-backend'
    static_configs:
      - targets: ['${PROMETHEUS_BACKEND_TARGET:-backend.r3mes.network:8000}']
  
  # Note: Replace localhost with actual production hostnames
  # Development: Use localhost
  # Production: Use environment variables or actual hostnames

### Grafana Dashboards

- Node metrics (block height, transaction count)
- Miner metrics (hash rate, GPU stats)
- Network metrics (active miners, total gradients)
- Backend metrics (API latency, error rates, cache hit rates)

### Alertmanager

**Configuration**: `docker/alertmanager/alertmanager.yml`

**Alert Rules**: `docker/prometheus/alert_rules.yml`

**Alert Categories**:
- **Critical Alerts**: Backend down, blockchain node down, no new blocks
- **Warning Alerts**: High error rate, high latency, low cache hit rate, sync lag
- **System Alerts**: High memory/CPU usage, low disk space
- **Network Alerts**: Low miner count, high network latency

**Notification Channels**:
- Email (SMTP)
- Slack webhooks
- PagerDuty (optional)

**Alert Routing**:
- Critical alerts → Immediate notification (email + Slack)
- Backend alerts → Backend team channel
- Blockchain alerts → Blockchain team channel
- Miner alerts → Miner team channel

### Sentry Error Tracking

**Frontend (Next.js)**:
- Client-side error tracking (`@sentry/nextjs`)
- Server-side error tracking
- Edge runtime error tracking
- Session replay (optional)
- Performance monitoring

**Backend (FastAPI)**:
- Exception tracking
- Performance monitoring
- Request/response logging
- SQL query tracking

**Blockchain (Go)**:
- Panic recovery
- Error context tracking
- Performance monitoring

**Configuration**:
- Environment variables: `SENTRY_DSN`, `SENTRY_DEBUG`
- Filtering: Sensitive data (passwords, API keys) automatically filtered
- Sample rates: 10% in production, 100% in development

---

## 💾 Backup & Recovery

### Backup Chain State

```bash
# Backup data directory
tar -czf remes-backup-$(date +%Y%m%d).tar.gz ~/.remesd/data/

# Backup keys (encrypted)
./build/remesd keys export mykey --output-file mykey-backup.json
```

### Recovery

```bash
# Restore data directory
tar -xzf remes-backup-YYYYMMDD.tar.gz -C ~/.remesd/

# Restore keys
./build/remesd keys import mykey mykey-backup.json
```

---

## 🚀 Deployment Scripts

### install_founder.sh

```bash
# Kurucu/Admin kurulum script'i
bash scripts/install_founder.sh
```

**Yapılanlar**:
- Go binary build
- Genesis oluşturma
- Validator key oluşturma
- Systemd service kurulumu

### install_miner_pypi.sh

```bash
# Miner PyPI kurulum script'i
bash scripts/install_miner_pypi.sh
```

**Yapılanlar**:
- Virtual environment oluşturma
- PyPI'den r3mes paketi kurulumu
- Setup wizard çalıştırma
- Systemd service kurulumu

### install_validator.sh

```bash
# Validator kurulum script'i
bash scripts/install_validator.sh
```

**Yapılanlar**:
- Go binary build
- Node initialization
- Validator key oluşturma
- Systemd service kurulumu

---

## 🌐 Environment Variables

### Important Security Note

**CRITICAL**: Environment files containing production secrets (`.env.production`) must NEVER be committed to version control. These files should be:

1. **Managed via Secrets Manager**: Use AWS Secrets Manager, HashiCorp Vault, or similar service
2. **Injected at Runtime**: Environment variables should be injected during deployment (Docker, Kubernetes, systemd)
3. **Example Files Only**: Only `.env.production.example` files (with placeholder values) should be in the repository

### Environment File Management

- **Backend**: Copy `backend/.env.production.example` to `backend/.env.production` and fill in actual values
- **Miner Engine**: Copy `miner-engine/env.production.example` to `miner-engine/.env.production` and fill in actual values
- **Never commit**: `.env.production` files are in `.gitignore` and should never be committed

### Blockchain Node

```bash
# Production: MUST use production URLs (no localhost)
export R3MES_GRPC_ADDR=node.r3mes.network:9090
export R3MES_TENDERMINT_RPC_ADDR=http://node.r3mes.network:26657
export CORS_ALLOWED_ORIGINS=https://dashboard.r3mes.network
export R3MES_DASHBOARD_CACHE_TTL_SECONDS=30

# Development: Can use localhost
# export R3MES_GRPC_ADDR=localhost:9090
# export R3MES_TENDERMINT_RPC_ADDR=http://localhost:26657
```

### Miner Engine

```bash
# Production: MUST use production URLs (no localhost)
export R3MES_NODE_GRPC_URL=node.r3mes.network:9090
export R3MES_IPFS_URL=http://ipfs.r3mes.network:5001
export R3MES_TLS_ENABLED=true

# Development: Can use localhost
# export R3MES_NODE_GRPC_URL=localhost:9090
# export R3MES_IPFS_URL=http://localhost:5001
# export R3MES_TLS_ENABLED=false
```

### Web Dashboard

```bash
# Production: MUST use production URLs (no localhost)
export NEXT_PUBLIC_API_URL=https://api.r3mes.network
export NEXT_PUBLIC_WS_URL=wss://api.r3mes.network/ws
export RATE_LIMIT_WINDOW_MS=60000
export RATE_LIMIT_MAX_REQUESTS=100

# Development: Can use localhost
# export NEXT_PUBLIC_API_URL=http://localhost:1317
# export NEXT_PUBLIC_WS_URL=ws://localhost:1317
```

---

## 📋 Production Checklist

### Pre-Deployment

- [ ] Tüm testler geçiyor
- [ ] Security audit tamamlandı
- [ ] Monitoring kuruldu
- [ ] Backup stratejisi hazır
- [ ] SSL certificates kuruldu
- [ ] Firewall rules ayarlandı

### Deployment

- [ ] Systemd services kuruldu
- [ ] Environment variables ayarlandı
- [ ] Log rotation yapılandırıldı
- [ ] Monitoring aktif
- [ ] Backup otomatikleştirildi

### Post-Deployment

- [ ] Health checks çalışıyor
- [ ] Metrics toplanıyor
- [ ] Alerting aktif
- [ ] Documentation güncel

---

Bu production deployment rehberi, R3MES sistemini production ortamında güvenli ve güvenilir şekilde çalıştırmak için gerekli tüm adımları sağlar.


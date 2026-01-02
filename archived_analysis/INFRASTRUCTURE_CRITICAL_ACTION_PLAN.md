# R3MES Infrastructure - Critical Action Plan

**Priority**: URGENT - Must Complete Before Production  
**Timeline**: 2-3 weeks  
**Effort**: ~150 hours

---

## PHASE 1: CRITICAL SECURITY FIXES (Week 1)

### 1.1 Container Security - Backend Dockerfile

**File**: `backend/Dockerfile`

```dockerfile
# BEFORE (INSECURE):
FROM python:3.10-slim
WORKDIR /app
COPY . .
EXPOSE 8000
CMD ["/app/scripts/init_db.sh"]

# AFTER (SECURE):
FROM python:3.10-slim@sha256:SPECIFIC_HASH

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    bash \
    curl \
    postgresql-client \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1001 -s /sbin/nologin appuser

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories with correct permissions
RUN mkdir -p checkpoints logs && \
    chown -R appuser:appuser checkpoints logs

# Switch to non-root user
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/app/scripts/init_db.sh"]
```

**Changes**:
- Pin base image with SHA256
- Create non-root user (appuser:1001)
- Set proper file ownership
- Use --chown in COPY commands
- Switch to non-root user before CMD

**Testing**:
```bash
# Verify non-root user
docker run --rm r3mes/backend:latest id
# Should output: uid=1001(appuser) gid=1001(appuser) groups=1001(appuser)

# Verify no root access
docker run --rm r3mes/backend:latest whoami
# Should output: appuser
```

---

### 1.2 Kubernetes Security Contexts

**File**: `k8s/production-deployment.yaml`

Add to all pod specs:

```yaml
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1001
    runAsGroup: 1001
    fsGroup: 1001
    seccompProfile:
      type: RuntimeDefault
  
  containers:
  - name: backend
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
          - ALL
    volumeMounts:
    - name: tmp
      mountPath: /tmp
    - name: cache
      mountPath: /app/cache
  
  volumes:
  - name: tmp
    emptyDir: {}
  - name: cache
    emptyDir: {}
```

**Apply to**: All deployments (backend, frontend, blockchain, miner)

---

### 1.3 Pod Security Policy

**File**: `k8s/pod-security-policy.yaml` (NEW)

```yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: r3mes-restricted
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  hostNetwork: false
  hostIPC: false
  hostPID: false
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'MustRunAs'
    seLinuxOptions:
      level: "s0:c123,c456"
  fsGroup:
    rule: 'MustRunAs'
    ranges:
      - min: 1000
        max: 65535
  readOnlyRootFilesystem: false
```

**Apply**:
```bash
kubectl apply -f k8s/pod-security-policy.yaml
```

---

### 1.4 Database Encryption

**File**: `docker/docker-compose.prod.yml`

Add PostgreSQL encryption:

```yaml
postgres:
  environment:
    # Enable encryption
    POSTGRES_INITDB_ARGS: "-c ssl=on -c ssl_cert_file=/etc/ssl/certs/server.crt -c ssl_key_file=/etc/ssl/private/server.key"
  volumes:
    - postgres_data:/var/lib/postgresql/data
    - ./certs/server.crt:/etc/ssl/certs/server.crt:ro
    - ./certs/server.key:/etc/ssl/private/server.key:ro
```

**Generate certificates**:
```bash
# Generate self-signed certificate
openssl req -new -x509 -days 365 -nodes \
  -out docker/certs/server.crt \
  -keyout docker/certs/server.key \
  -subj "/CN=postgres"

chmod 600 docker/certs/server.key
```

---

### 1.5 Backup Encryption

**File**: `scripts/backup_database.sh` (UPDATE)

```bash
#!/bin/bash
# Encrypted database backup

BACKUP_DIR="/backups"
ENCRYPTION_KEY="${BACKUP_ENCRYPTION_KEY:-}"

if [ -z "$ENCRYPTION_KEY" ]; then
    echo "Error: BACKUP_ENCRYPTION_KEY not set"
    exit 1
fi

# Create backup
BACKUP_FILE="$BACKUP_DIR/backup_$(date +%Y%m%d_%H%M%S).dump"

docker exec r3mes-postgres-prod pg_dump \
    -U r3mes \
    -d r3mes \
    -F c \
    | openssl enc -aes-256-cbc -salt -pass pass:"$ENCRYPTION_KEY" \
    > "$BACKUP_FILE.enc"

echo "Encrypted backup created: $BACKUP_FILE.enc"
```

---

### 1.6 Centralized Logging with Loki

**File**: `docker/docker-compose.logging.yml` (UPDATE)

```yaml
version: '3.8'

services:
  loki:
    image: grafana/loki:latest
    container_name: r3mes-loki-prod
    ports:
      - "3100:3100"
    volumes:
      - loki_data:/loki
      - ./loki/loki-config.yml:/etc/loki/local-config.yml:ro
    command: -config.file=/etc/loki/local-config.yml
    networks:
      - r3mes-network
    restart: always

  promtail:
    image: grafana/promtail:latest
    container_name: r3mes-promtail-prod
    volumes:
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
      - /var/run/docker.sock:/var/run/docker.sock
      - ./promtail/promtail-config.yml:/etc/promtail/config.yml:ro
    command: -config.file=/etc/promtail/config.yml
    networks:
      - r3mes-network
    depends_on:
      - loki
    restart: always

volumes:
  loki_data:

networks:
  r3mes-network:
    external: true
```

---

### 1.7 Distributed Tracing with Jaeger

**File**: `docker/docker-compose.tracing.yml` (UPDATE)

```yaml
version: '3.8'

services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    container_name: r3mes-jaeger-prod
    ports:
      - "6831:6831/udp"
      - "16686:16686"
    environment:
      - COLLECTOR_ZIPKIN_HOST_PORT=:9411
    volumes:
      - jaeger_data:/badger/data
    networks:
      - r3mes-network
    restart: always

volumes:
  jaeger_data:

networks:
  r3mes-network:
    external: true
```

**Backend Integration**:
```python
# backend/app/main.py
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)
```

---

### 1.8 Missing Alert Rules

**File**: `monitoring/prometheus/alerts.prod.yml` (ADD)

```yaml
groups:
  - name: r3mes_critical_alerts
    interval: 30s
    rules:
      # Database connection pool exhaustion
      - alert: DatabaseConnectionPoolExhausted
        expr: database_connections_active >= database_connections_max * 0.9
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Database connection pool nearly exhausted"
          description: "{{ $value }} of {{ $labels.max }} connections in use"

      # Redis memory exhaustion
      - alert: RedisMemoryExhausted
        expr: redis_memory_used_bytes >= redis_memory_max_bytes * 0.9
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Redis memory nearly exhausted"
          description: "{{ $value | humanize }}B of {{ $labels.max | humanize }}B used"

      # Disk space critical
      - alert: DiskSpaceCritical
        expr: (node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) < 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Disk space critical"
          description: "Only {{ $value | humanizePercentage }} disk space remaining"

      # Pod restart loop
      - alert: PodRestartLoop
        expr: rate(kube_pod_container_status_restarts_total[15m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Pod in restart loop"
          description: "Pod {{ $labels.pod }} restarting {{ $value }} times/min"

      # Deployment replica mismatch
      - alert: DeploymentReplicasMismatch
        expr: kube_deployment_spec_replicas != kube_deployment_status_replicas_available
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Deployment replicas mismatch"
          description: "Deployment {{ $labels.deployment }} has {{ $value }} unavailable replicas"

      # StatefulSet replica mismatch
      - alert: StatefulSetReplicasMismatch
        expr: kube_statefulset_status_replicas_ready != kube_statefulset_status_replicas
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "StatefulSet replicas mismatch"
          description: "StatefulSet {{ $labels.statefulset }} has {{ $value }} unavailable replicas"
```

---

## PHASE 2: HIGH PRIORITY IMPROVEMENTS (Week 2-3)

### 2.1 Pin Base Image Versions

**Update all Dockerfiles**:

```bash
# Get SHA256 for specific image version
docker pull python:3.10-slim
docker inspect python:3.10-slim | grep -i digest

# Update Dockerfile
FROM python:3.10-slim@sha256:abc123def456...
```

**Files to update**:
- backend/Dockerfile
- web-dashboard/Dockerfile
- miner-engine/docker/Dockerfile.go
- docker/nginx/Dockerfile

---

### 2.2 Secret Rotation

**File**: `scripts/rotate_secrets.sh` (NEW)

```bash
#!/bin/bash
# Rotate Docker secrets

set -e

SECRETS_DIR="docker/secrets"

echo "Rotating secrets..."

# Generate new passwords
NEW_POSTGRES_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)
NEW_REDIS_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)
NEW_GRAFANA_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)

# Update secret files
echo -n "$NEW_POSTGRES_PASSWORD" > "$SECRETS_DIR/postgres_password.txt"
echo -n "$NEW_REDIS_PASSWORD" > "$SECRETS_DIR/redis_password.txt"
echo -n "$NEW_GRAFANA_PASSWORD" > "$SECRETS_DIR/grafana_admin_password.txt"

chmod 600 "$SECRETS_DIR"/*.txt

# Restart services
docker-compose -f docker/docker-compose.prod.yml restart postgres redis grafana

echo "Secrets rotated successfully"
```

---

### 2.3 Database Replication

**Set up PostgreSQL streaming replication**:

```yaml
# docker-compose.prod.yml - Add replica
postgres-replica:
  image: postgres:16-alpine
  environment:
    POSTGRES_REPLICATION_MODE: slave
    POSTGRES_MASTER_SERVICE: postgres
  depends_on:
    - postgres
  volumes:
    - postgres_replica_data:/var/lib/postgresql/data
```

---

### 2.4 Redis Replication

**Set up Redis Sentinel**:

```yaml
redis-sentinel:
  image: redis:7-alpine
  command: redis-sentinel /etc/redis/sentinel.conf
  volumes:
    - ./redis/sentinel.conf:/etc/redis/sentinel.conf:ro
  depends_on:
    - redis
```

---

### 2.5 CI/CD Pipeline

**File**: `.github/workflows/deploy.yml` (NEW)

```yaml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          format: 'sarif'
          output: 'trivy-results.sarif'

  build:
    needs: security-scan
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker images
        run: |
          docker build -t r3mes/backend:${{ github.sha }} backend/
          docker build -t r3mes/frontend:${{ github.sha }} web-dashboard/

  test:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: |
          docker-compose -f docker/docker-compose.test.yml up --abort-on-container-exit

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to production
        run: |
          bash scripts/deploy_production_docker.sh
```

---

## PHASE 3: MEDIUM PRIORITY ENHANCEMENTS (Week 4+)

### 3.1 WAF Integration

Use ModSecurity with Nginx:

```dockerfile
# docker/nginx/Dockerfile
FROM nginx:alpine

RUN apk add --no-cache modsecurity modsecurity-nginx

COPY modsecurity.conf /etc/nginx/modsecurity/modsecurity.conf
COPY rules/ /etc/nginx/modsecurity/rules/
```

---

### 3.2 VPN for Admin Access

Use WireGuard or OpenVPN:

```bash
# Install WireGuard
sudo apt-get install -y wireguard wireguard-tools

# Generate keys
wg genkey | tee privatekey | wg pubkey > publickey

# Configure VPN access for SSH
```

---

### 3.3 MFA Implementation

Use TOTP for SSH:

```bash
# Install google-authenticator
sudo apt-get install -y libpam-google-authenticator

# Configure PAM
echo "auth required pam_google_authenticator.so" >> /etc/pam.d/sshd
```

---

## VALIDATION CHECKLIST

### Security Validation
- [ ] Backend container runs as non-root user
- [ ] All K8s pods have security contexts
- [ ] Pod Security Policy enforced
- [ ] Database encryption enabled
- [ ] Backups encrypted
- [ ] Centralized logging operational
- [ ] Distributed tracing operational
- [ ] All critical alerts configured

### Operational Validation
- [ ] Base images pinned with SHA256
- [ ] Secret rotation tested
- [ ] Database replication tested
- [ ] Redis replication tested
- [ ] CI/CD pipeline operational
- [ ] Automated security scanning working
- [ ] Backup encryption tested
- [ ] Disaster recovery drill completed

### Performance Validation
- [ ] Load testing completed
- [ ] Performance targets met
- [ ] No memory leaks detected
- [ ] Database performance acceptable
- [ ] Cache hit rate > 85%

---

## DEPLOYMENT STEPS

1. **Week 1**: Implement all critical security fixes
2. **Week 2**: Implement high priority improvements
3. **Week 3**: Complete testing and validation
4. **Week 4+**: Implement medium priority enhancements

---

**Timeline**: 2-3 weeks  
**Effort**: ~150 hours  
**Team**: DevOps + Security + Backend Engineers

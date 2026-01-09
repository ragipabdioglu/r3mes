# R3MES Docker Secrets Implementation Guide

## Overview

R3MES supports multiple secrets management backends for secure credential storage:

1. **Docker Compose Secrets** (file-based) - Development/Simple Production
2. **Docker Swarm Secrets** - Production with Docker Swarm
3. **Kubernetes Secrets** - Production with Kubernetes
4. **HashiCorp Vault** - Enterprise-grade secrets management
5. **AWS Secrets Manager** - Cloud-native on AWS

## Quick Start

### Docker Compose (Development)

```bash
# Create secrets
bash scripts/create_secrets.sh

# Start services
docker-compose -f docker-compose.prod.yml up -d
```

### Docker Swarm (Production)

```bash
# Initialize swarm
docker swarm init

# Create secrets
bash scripts/create_swarm_secrets.sh

# Deploy stack
docker stack deploy -c docker-compose.swarm.yml r3mes
```

### Kubernetes

```bash
# Create secrets
bash scripts/create_k8s_secrets.sh --namespace r3mes

# Deploy
kubectl apply -f k8s/ -n r3mes
```

### HashiCorp Vault

```bash
# Start Vault
docker-compose -f docker-compose.vault.yml up -d vault

# Initialize and configure
bash scripts/init_vault.sh

# Start all services
docker-compose -f docker-compose.vault.yml up -d
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     R3MES Backend                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  SecretsManager                          │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐   │   │
│  │  │ Docker  │ │   K8s   │ │  Vault  │ │     AWS     │   │   │
│  │  │ Secrets │ │ Secrets │ │         │ │   Secrets   │   │   │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └──────┬──────┘   │   │
│  │       │           │           │              │          │   │
│  │       └───────────┴───────────┴──────────────┘          │   │
│  │                         │                                │   │
│  │                    get_secret()                          │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Secrets Reference

| Secret Name | Description | Used By |
|-------------|-------------|---------|
| `postgres_password` | PostgreSQL database password | Backend, PostgreSQL |
| `postgres_user` | PostgreSQL username | Backend, PostgreSQL |
| `redis_password` | Redis cache password | Backend, Redis |
| `jwt_secret` | JWT signing key | Backend |
| `api_secret_key` | API authentication key | Backend |
| `grafana_admin_password` | Grafana admin password | Grafana |
| `validator_key` | Blockchain validator key | remesd |
| `node_key` | Blockchain node key | remesd |
| `ssl_certificate` | TLS certificate | Nginx |
| `ssl_private_key` | TLS private key | Nginx |

## Backend Integration

### Using SecretsManager

```python
from app.secrets_manager import get_secrets_manager, SecretKey

# Get secrets manager instance
secrets = get_secrets_manager()

# Get a specific secret
db_password = secrets.get(SecretKey.POSTGRES_PASSWORD)

# Get database URL (built from secrets)
db_url = secrets.get_database_url()

# Get Redis URL
redis_url = secrets.get_redis_url()
```

### Environment Variables

The backend automatically reads secrets from:

1. `/run/secrets/` (Docker Secrets)
2. `*_FILE` environment variables (file paths)
3. Direct environment variables (fallback)

Example:
```bash
# Docker Secrets (automatic)
/run/secrets/postgres_password

# File-based
POSTGRES_PASSWORD_FILE=/path/to/secret

# Direct (least secure)
POSTGRES_PASSWORD=mypassword
```

## Security Best Practices

### 1. Never Commit Secrets

```gitignore
# .gitignore
docker/secrets/
*.txt
vault_keys.json
```

### 2. Use Strong Passwords

```bash
# Generate strong password
openssl rand -base64 32 | tr -d "=+/" | cut -c1-32
```

### 3. Rotate Secrets Regularly

```bash
# Rotate all secrets
bash scripts/rotate_secrets.sh --all

# Rotate specific secret
bash scripts/rotate_secrets.sh --secret postgres
```

### 4. Limit Access

```bash
# Set proper file permissions
chmod 600 docker/secrets/*.txt
```

### 5. Use Vault for Production

For production environments, use HashiCorp Vault:
- Automatic secret rotation
- Audit logging
- Fine-grained access control
- Dynamic secrets

## Troubleshooting

### Secret Not Found

```bash
# Check if secret file exists
ls -la docker/secrets/

# Check Docker secret
docker secret ls

# Check Kubernetes secret
kubectl get secrets -n r3mes
```

### Permission Denied

```bash
# Fix file permissions
chmod 600 docker/secrets/*.txt
chown $(whoami) docker/secrets/*.txt
```

### Container Can't Read Secret

```bash
# Check secret mount in container
docker exec r3mes-backend-prod ls -la /run/secrets/

# Check secret content (careful!)
docker exec r3mes-backend-prod cat /run/secrets/postgres_password
```

### Vault Connection Failed

```bash
# Check Vault status
vault status

# Check Vault logs
docker logs r3mes-vault

# Verify token
vault token lookup
```

## Migration Guide

### From Environment Variables to Docker Secrets

1. Create secrets files:
```bash
bash scripts/create_secrets_from_env.sh
```

2. Update docker-compose.yml to use secrets
3. Remove passwords from .env file
4. Restart services

### From Docker Compose to Swarm

1. Initialize swarm:
```bash
docker swarm init
```

2. Create swarm secrets:
```bash
bash scripts/create_swarm_secrets.sh
```

3. Deploy with swarm compose:
```bash
docker stack deploy -c docker-compose.swarm.yml r3mes
```

### From Docker to Kubernetes

1. Create Kubernetes secrets:
```bash
bash scripts/create_k8s_secrets.sh
```

2. Update deployments to use secrets
3. Apply Kubernetes manifests

## Monitoring Secrets

### Audit Secret Access

With Vault:
```bash
vault audit enable file file_path=/var/log/vault/audit.log
```

### Alert on Secret Changes

Configure Prometheus alerts for:
- Secret rotation events
- Failed authentication attempts
- Unusual secret access patterns

## Backup and Recovery

### Backup Secrets

```bash
# Backup Docker secrets
tar -czf secrets_backup.tar.gz docker/secrets/

# Backup Vault
vault operator raft snapshot save backup.snap
```

### Restore Secrets

```bash
# Restore Docker secrets
tar -xzf secrets_backup.tar.gz

# Restore Vault
vault operator raft snapshot restore backup.snap
```

## References

- [Docker Secrets Documentation](https://docs.docker.com/engine/swarm/secrets/)
- [Kubernetes Secrets](https://kubernetes.io/docs/concepts/configuration/secret/)
- [HashiCorp Vault](https://www.vaultproject.io/docs)
- [AWS Secrets Manager](https://aws.amazon.com/secrets-manager/)

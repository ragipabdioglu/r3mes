# R3MES Secret Management Strategy

## Overview

This document outlines the secret management strategy for the R3MES project, including how secrets are stored, accessed, and rotated in different environments.

## Secret Categories

### 1. Application Secrets

- **Database Credentials**: PostgreSQL username, password, connection strings
- **Redis Credentials**: Redis password, connection URLs
- **API Keys**: External service API keys (Sentry, SMTP, etc.)
- **JWT Secrets**: Token signing keys (if applicable)

### 2. Blockchain Secrets

- **Private Keys**: Miner private keys (hex format)
- **Validator Keys**: Validator operator keys (if applicable)
- **Node Keys**: Blockchain node keys

### 3. Infrastructure Secrets

- **SSH Keys**: Deployment SSH keys (production, staging)
- **Docker Registry**: Docker registry credentials
- **Cloud Provider**: AWS/GCP/Azure credentials (if applicable)

## Secret Storage

### Development Environment

**Location**: `.env` files (git-ignored)

```bash
# .env.development
DATABASE_URL=postgresql://user:pass@localhost:5432/r3mes
REDIS_URL=redis://localhost:6379/0
PRIVATE_KEY=your_dev_private_key_here
```

**Security Notes**:
- Never commit `.env` files to version control
- Use `.env.example` as a template
- Rotate secrets regularly
- Use different secrets for each developer

### Staging Environment

**Location**: Environment variables or secret management service

**Recommended**: Use a secret management service:
- **AWS Secrets Manager**
- **HashiCorp Vault**
- **Google Secret Manager**
- **Azure Key Vault**

**Example (AWS Secrets Manager)**:
```python
import boto3
import json

def get_secret(secret_name: str) -> dict:
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

# Usage
secrets = get_secret('r3mes/staging')
DATABASE_URL = secrets['database_url']
```

### Production Environment

**Location**: Secret management service (MANDATORY)

**Requirements**:
1. **Centralized Secret Management**: Use AWS Secrets Manager, HashiCorp Vault, or similar
2. **Encryption at Rest**: All secrets encrypted
3. **Encryption in Transit**: TLS for all secret access
4. **Access Control**: IAM roles, RBAC
5. **Audit Logging**: All secret access logged
6. **Rotation**: Automatic secret rotation where possible

## Secret Access Patterns

### Backend Service

**Implementation**: `backend/app/secrets.py`

The backend uses a unified secret management interface that supports multiple providers:

```python
from app.secrets import get_secret, get_secret_manager

# Get a single secret
database_url = get_secret("DATABASE_URL")

# Get secret manager instance
secret_manager = get_secret_manager()

# Test connection
if secret_manager.test_connection():
    print("Secret management service connected")
```

**Supported Providers**:
1. **AWS Secrets Manager** (priority in production if `AWS_SECRETS_MANAGER_REGION` or `AWS_DEFAULT_REGION` is set)
2. **HashiCorp Vault** (if `VAULT_ADDR` is set)
3. **Environment Variables** (fallback for development)

**Usage Example**:
```python
from app.secrets import get_secret

# In production, this will use AWS Secrets Manager or Vault
# In development, this will use environment variables
database_url = get_secret("DATABASE_URL")

# With default value (development only)
redis_url = get_secret("REDIS_URL", default="redis://localhost:6379/0")
```

**Caching**: All secrets are cached for 5 minutes to reduce API calls.

### Miner Engine

```python
# miner-engine/r3mes/utils/secrets.py
import os
from typing import Optional

def get_private_key() -> str:
    """Get miner private key from secure storage."""
    env_mode = os.getenv("R3MES_ENV", "development").lower()
    
    if env_mode == "production":
        # Production: use secret management service
        return get_secret_from_vault("miner/private_key")
    else:
        # Development: use environment variable
        private_key = os.getenv("PRIVATE_KEY")
        if not private_key:
            raise ValueError("PRIVATE_KEY must be set")
        return private_key
```

### Blockchain Node

**Go Implementation**:
```go
package keeper

import (
    "os"
    "github.com/aws/aws-sdk-go/service/secretsmanager"
)

func getSecret(secretName string) (string, error) {
    envMode := os.Getenv("R3MES_ENV")
    
    if envMode == "production" {
        // Use AWS Secrets Manager or Vault
        return getSecretFromAWS(secretName)
    } else {
        // Development: use environment variable
        value := os.Getenv(secretName)
        if value == "" {
            return "", fmt.Errorf("secret %s not found", secretName)
        }
        return value, nil
    }
}
```

## Secret Rotation

### Automatic Rotation

**Recommended Services**:
- **AWS Secrets Manager**: Automatic rotation with Lambda functions
- **HashiCorp Vault**: Dynamic secrets with TTL
- **Kubernetes Secrets**: External secret operator

### Manual Rotation

**Process**:
1. Generate new secret
2. Update secret in secret management service
3. Restart services (or use zero-downtime deployment)
4. Verify new secret works
5. Revoke old secret
6. Update documentation

**Rotation Schedule**:
- **Database Passwords**: Every 90 days
- **API Keys**: Every 180 days
- **Private Keys**: Never (unless compromised)
- **SSH Keys**: Every 90 days

## Secret Injection

### Docker Compose

```yaml
services:
  backend:
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
    secrets:
      - private_key
    secrets:
      private_key:
        external: true  # Use Docker secrets
```

### Kubernetes

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: r3mes-secrets
type: Opaque
stringData:
  database_url: "postgresql://..."
  redis_url: "redis://..."
---
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: backend
        envFrom:
        - secretRef:
            name: r3mes-secrets
```

### Environment Variables

```bash
# Production deployment script
export DATABASE_URL=$(aws secretsmanager get-secret-value --secret-id r3mes/database_url --query SecretString --output text)
export REDIS_URL=$(aws secretsmanager get-secret-value --secret-id r3mes/redis_url --query SecretString --output text)
./start_backend.sh
```

## Security Best Practices

### 1. Never Log Secrets

```python
# BAD
logger.info(f"Database URL: {database_url}")

# GOOD
logger.info("Database connection established")
```

### 2. Use Secret Filtering

All loggers automatically filter sensitive data (see `LOGGING_POLICY.md`).

### 3. Principle of Least Privilege

- Each service should only have access to secrets it needs
- Use IAM roles with minimal permissions
- Rotate credentials regularly

### 4. Encrypt Secrets at Rest

- Use encrypted storage (AWS S3 with encryption, encrypted volumes)
- Use key management services (AWS KMS, HashiCorp Vault)

### 5. Encrypt Secrets in Transit

- Use TLS for all secret access
- Use secure channels (SSH tunnels, VPN) for secret retrieval

### 6. Audit Secret Access

- Log all secret access attempts
- Monitor for unauthorized access
- Alert on suspicious patterns

## Secret Management Services

### Google Cloud Secret Manager

**Setup**:
```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Set up Application Default Credentials
gcloud auth application-default login

# Or use service account (recommended for production)
gcloud iam service-accounts create r3mes-secret-manager \
    --display-name="R3MES Secret Manager"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:r3mes-secret-manager@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/secretmanager.admin"

gcloud iam service-accounts keys create ~/r3mes-gcp-key.json \
    --iam-account=r3mes-secret-manager@YOUR_PROJECT_ID.iam.gserviceaccount.com

export GOOGLE_APPLICATION_CREDENTIALS=~/r3mes-gcp-key.json
```

**Create Secrets**:
```bash
# Set project ID
export GOOGLE_CLOUD_PROJECT="your-project-id"

# Create secrets
echo -n "postgresql://user:pass@host:5432/db" | \
    gcloud secrets create r3mes-production-database-url \
    --data-file=- \
    --replication-policy="automatic"

echo -n "redis://password@host:6379/0" | \
    gcloud secrets create r3mes-production-redis-url \
    --data-file=- \
    --replication-policy="automatic"

echo -n "your-api-key-secret-min-32-chars" | \
    gcloud secrets create r3mes-production-api-key-secret \
    --data-file=- \
    --replication-policy="automatic"

echo -n "your-jwt-secret-min-32-chars" | \
    gcloud secrets create r3mes-production-jwt-secret \
    --data-file=- \
    --replication-policy="automatic"

echo -n "https://rpc.r3mes.network:26657" | \
    gcloud secrets create r3mes-production-blockchain-rpc-url \
    --data-file=- \
    --replication-policy="automatic"

echo -n "rpc.r3mes.network:9090" | \
    gcloud secrets create r3mes-production-blockchain-grpc-url \
    --data-file=- \
    --replication-policy="automatic"

echo -n "https://rpc.r3mes.network:1317" | \
    gcloud secrets create r3mes-production-blockchain-rest-url \
    --data-file=- \
    --replication-policy="automatic"
```

**Python Client**:
```python
from app.secrets import GoogleCloudSecretManager

# Initialize
secret_manager = GoogleCloudSecretManager(project_id="your-project-id")

# Get secret
database_url = secret_manager.get_secret("r3mes-production-database-url")

# Get multiple secrets
secrets = secret_manager.get_secrets("r3mes-production")
```

**Configuration**:
- Set `GOOGLE_CLOUD_PROJECT` environment variable
- Set `GOOGLE_APPLICATION_CREDENTIALS` for service account authentication (optional, uses ADC if not set)
- Secrets are cached for 5 minutes
- Secret names use format: `r3mes-production-secret-name`

**Secret Naming Convention**:
- Use kebab-case: `r3mes-production-database-url`
- Prefix with environment: `r3mes-production-*` or `r3mes-staging-*`
- Descriptive names: `r3mes-production-blockchain-rpc-url`

**Access Control**:
```bash
# Grant service account access to secrets
for secret in r3mes-production-database-url r3mes-production-redis-url; do
    gcloud secrets add-iam-policy-binding $secret \
        --member="serviceAccount:r3mes-secret-manager@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
        --role="roles/secretmanager.secretAccessor"
done
```

### AWS Secrets Manager

**Setup**:
```bash
# Create secret
aws secretsmanager create-secret \
  --name r3mes/database_url \
  --secret-string "postgresql://user:pass@host:5432/db"

# Retrieve secret
aws secretsmanager get-secret-value \
  --secret-id r3mes/database_url \
  --query SecretString --output text
```

**Python Client**:
```python
from app.secrets import AWSSecretsManager

# Initialize
secret_manager = AWSSecretsManager(region="us-east-1")

# Get secret
database_url = secret_manager.get_secret("r3mes/production/database_url")

# Get multiple secrets
secrets = secret_manager.get_secrets("r3mes/production")
```

**Configuration**:
- Set `AWS_SECRETS_MANAGER_REGION` or `AWS_DEFAULT_REGION` environment variable
- AWS credentials via IAM role, `~/.aws/credentials`, or environment variables
- Secrets are cached for 5 minutes

**Secret Naming Convention**:
- Use hierarchical naming: `r3mes/production/database_url`
- Store JSON objects for multiple related secrets

### HashiCorp Vault

**Setup**:
```bash
# Enable KV secrets engine
vault secrets enable -path=r3mes kv-v2

# Store secret
vault kv put r3mes/database url="postgresql://..."

# Retrieve secret
vault kv get -field=url r3mes/database
```

**Python Client**:
```python
from app.secrets import HashiCorpVault

# Initialize
secret_manager = HashiCorpVault(
    vault_url="https://vault.example.com:8200",
    vault_token=os.getenv("VAULT_TOKEN")
)

# Get secret (path format: secret/data/r3mes/production/database_url)
database_url = secret_manager.get_secret("secret/data/r3mes/production/database_url")

# Get multiple secrets
secrets = secret_manager.get_secrets("secret/data/r3mes/production")
```

**Configuration**:
- Set `VAULT_ADDR` environment variable
- Set `VAULT_TOKEN` environment variable
- Secrets are cached for 5 minutes

**Secret Path Format**:
- KV v2 engine: `secret/data/path/to/secret`
- The `data` part is required for KV v2

## Environment-Specific Configuration

### Development

```bash
# .env.development
R3MES_ENV=development
DATABASE_URL=postgresql://localhost:5432/r3mes_dev
REDIS_URL=redis://localhost:6379/0
PRIVATE_KEY=dev_private_key_here
```

### Staging

```bash
# Use secret management service
R3MES_ENV=staging
# Secrets loaded from AWS Secrets Manager or Vault
```

### Production

```bash
# MANDATORY: Use secret management service
R3MES_ENV=production
# All secrets MUST come from secret management service
# Environment variables are NOT allowed for sensitive data
```

## Secret Validation

### Pre-Deployment Validation

Use the validation script before production deployment:

```bash
python scripts/validate_production_env.py
```

This script validates:
- All required environment variables
- Secret management service connection
- Production configuration requirements

### Startup Checks

The backend automatically validates secrets on startup:

```python
from app.env_validator import validate_environment
from app.secrets import get_secret_manager

# Validate environment variables (includes secret management check)
validate_environment()

# Test secret management connection
secret_manager = get_secret_manager()
if not secret_manager.test_connection():
    raise ValueError("Secret management service connection failed")
```

### Manual Validation

```python
from app.secrets import get_secret

required_secrets = [
    "DATABASE_URL",
    "REDIS_URL",
    "PRIVATE_KEY",  # For miners
]

missing = []
for secret in required_secrets:
    try:
        get_secret(secret)
    except Exception as e:
        missing.append(f"{secret}: {e}")

if missing:
    raise ValueError(f"Missing required secrets:\n" + "\n".join(missing))
```

## Emergency Procedures

### Secret Compromise

1. **Immediate Actions**:
   - Revoke compromised secret
   - Generate new secret
   - Update all services
   - Audit access logs

2. **Investigation**:
   - Review access logs
   - Identify breach vector
   - Assess impact
   - Document incident

3. **Prevention**:
   - Update security policies
   - Rotate all related secrets
   - Enhance monitoring
   - Review access controls

## Compliance

### GDPR

- Encrypt personal data at rest and in transit
- Implement access controls
- Log all data access
- Support data deletion requests

### SOC 2

- Encrypt secrets at rest
- Implement access controls
- Regular security audits
- Incident response procedures

## References

- [AWS Secrets Manager Best Practices](https://docs.aws.amazon.com/secretsmanager/latest/userguide/best-practices.html)
- [HashiCorp Vault Documentation](https://www.vaultproject.io/docs)
- [OWASP Secret Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)


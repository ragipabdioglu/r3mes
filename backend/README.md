# R3MES Backend API

Production-ready FastAPI backend with JWT authentication, input sanitization, and blockchain integration.

## 🚀 Features

### Security
- **JWT Authentication**: RS256 asymmetric signing with token refresh
- **Input Sanitization**: Multi-layer protection against XSS, SQL injection, NoSQL injection, command injection
- **Secrets Management**: Support for AWS Secrets Manager, HashiCorp Vault, Azure Key Vault
- **Rate Limiting**: Configurable rate limits per endpoint
- **CORS Protection**: Configurable allowed origins

### API Endpoints

#### Public Endpoints
- `GET /` - Health check
- `GET /health` - Detailed health status
- `GET /chain/status` - Blockchain status
- `POST /auth/login` - User login (JWT token generation)
- `POST /auth/refresh` - Token refresh
- `POST /generate` - AI text generation (optional auth)

#### Protected Endpoints (Require JWT)
- `POST /auth/logout` - User logout
- `POST /chat` - AI chat with conversation history
- `GET /user/profile` - User profile

## 📦 Installation

### Prerequisites
- Python 3.10+
- Redis (for caching and JWT blacklist)
- PostgreSQL (optional, for production)

### Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Generate RSA keys for JWT** (production):
```bash
# Generate private key
openssl genrsa -out private_key.pem 2048

# Generate public key
openssl rsa -in private_key.pem -pubout -out public_key.pem

# Update .env
JWT_PRIVATE_KEY_PATH=/path/to/private_key.pem
JWT_PUBLIC_KEY_PATH=/path/to/public_key.pem
```

4. **Start Redis**:
```bash
redis-server
```

5. **Run the backend**:
```bash
# Development
python main.py

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 🔐 Authentication Flow

### 1. Login
```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "wallet_address": "remes1...",
    "signature": "..."
  }'
```

Response:
```json
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 900
}
```

### 2. Use Access Token
```bash
curl -X GET http://localhost:8000/user/profile \
  -H "Authorization: Bearer eyJ..."
```

### 3. Refresh Token
```bash
curl -X POST http://localhost:8000/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "eyJ..."
  }'
```

### 4. Logout
```bash
curl -X POST http://localhost:8000/auth/logout \
  -H "Authorization: Bearer eyJ..."
```

## 🛡️ Input Sanitization

All user inputs are automatically sanitized to prevent:
- XSS (Cross-Site Scripting)
- SQL Injection
- NoSQL Injection
- Command Injection
- Path Traversal

Example:
```python
from backend.app.input_sanitizer import InputSanitizer

# Sanitize string
safe_input = InputSanitizer.sanitize_string(
    user_input,
    max_length=1000,
    strict=True  # Reject suspicious patterns
)

# Sanitize dictionary
safe_data = InputSanitizer.sanitize_dict(request_data)
```

## 🔑 Secrets Management

### Development (Environment Variables)
```bash
SECRETS_PROVIDER=env
API_SECRET_KEY=your-secret-key
```

### Production (AWS Secrets Manager)
```bash
SECRETS_PROVIDER=aws
AWS_REGION=us-east-1
AWS_SECRET_NAME=r3mes/production
```

### Production (HashiCorp Vault)
```bash
SECRETS_PROVIDER=vault
VAULT_ADDR=https://vault.example.com
VAULT_TOKEN=your-vault-token
VAULT_SECRET_PATH=secret/r3mes
```

## 📊 Monitoring

### Health Check
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "redis": "connected",
  "timestamp": 1704672000.0
}
```

### Logs
```bash
# View logs
tail -f logs/backend.log

# Log levels: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL=INFO
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=backend tests/
```

## 🚀 Deployment

### Docker
```bash
# Build image
docker build -t r3mes-backend .

# Run container
docker run -d \
  -p 8000:8000 \
  -e R3MES_ENV=production \
  -e REDIS_URL=redis://redis:6379/0 \
  --name r3mes-backend \
  r3mes-backend
```

### Kubernetes
```bash
# Apply manifests
kubectl apply -f k8s/backend/
```

## 📝 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `R3MES_ENV` | Environment (development/production) | `development` |
| `RPC_URL` | Blockchain RPC endpoint | `http://localhost:26657` |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` |
| `DATABASE_URL` | PostgreSQL connection URL | - |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | Access token expiration | `15` |
| `JWT_REFRESH_TOKEN_EXPIRE_DAYS` | Refresh token expiration | `30` |
| `CORS_ORIGINS` | Allowed CORS origins | `*` |
| `SECRETS_PROVIDER` | Secrets provider (env/aws/vault/azure) | `env` |

## 🔧 Development

### Project Structure
```
backend/
├── app/
│   ├── __init__.py
│   ├── jwt_auth.py          # JWT authentication
│   ├── input_sanitizer.py   # Input sanitization
│   ├── cache.py             # Redis cache manager
│   ├── exceptions.py        # Custom exceptions
│   └── secrets_provider.py  # Secrets management
├── tests/
│   ├── test_jwt_auth.py
│   ├── test_sanitizer.py
│   └── test_cache.py
└── README.md
```

### Adding New Endpoints

1. **Public endpoint**:
```python
@app.get("/public/endpoint")
async def public_endpoint():
    return {"message": "Public data"}
```

2. **Protected endpoint**:
```python
@app.get("/protected/endpoint")
async def protected_endpoint(
    current_user: str = Depends(get_current_user)
):
    return {"user": current_user, "data": "Protected data"}
```

3. **Optional auth endpoint**:
```python
@app.get("/optional/endpoint")
async def optional_endpoint(
    current_user: Optional[str] = Depends(get_current_user_optional)
):
    if current_user:
        return {"user": current_user, "premium": True}
    return {"premium": False}
```

## 📚 API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🤝 Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines.

## 📄 License

See [LICENSE](../LICENSE) for license information.

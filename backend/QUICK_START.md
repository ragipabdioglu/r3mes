# R3MES Backend - Hızlı Başlangıç

## 🚀 5 Dakikada Başlat

### 1. Bağımlılıkları Yükle

```bash
# Python sanal ortamı oluştur
python -m venv venv

# Sanal ortamı aktifleştir
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Bağımlılıkları yükle
pip install -r requirements.txt
```

### 2. Redis'i Başlat

```bash
# Docker ile (önerilen)
docker run -d -p 6379:6379 redis:alpine

# Veya yerel Redis
redis-server
```

### 3. Ortam Değişkenlerini Ayarla

```bash
# .env dosyası oluştur
cp .env.example .env

# Minimum gerekli ayarlar (development için)
R3MES_ENV=development
REDIS_URL=redis://localhost:6379/0
RPC_URL=http://localhost:26657
CORS_ORIGINS=http://localhost:3000,http://localhost:8000
```

### 4. Backend'i Başlat

```bash
# Development mode
python main.py

# Veya uvicorn ile
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Test Et

```bash
# Başka bir terminalde
python backend/test_integration.py
```

## 📝 Temel Kullanım

### Health Check

```bash
curl http://localhost:8000/health
```

### Login (JWT Token Al)

```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "wallet_address": "remes1abcdefghijklmnopqrstuvwxyz1234567890",
    "signature": "test_signature"
  }'
```

Yanıt:
```json
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 900
}
```

### AI Text Generation (Anonymous)

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Merhaba R3MES!",
    "max_length": 100
  }'
```

### AI Text Generation (Authenticated)

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -d '{
    "prompt": "Merhaba R3MES!",
    "max_length": 100
  }'
```

### Chat (Requires Auth)

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -d '{
    "message": "Merhaba, nasılsın?"
  }'
```

### User Profile (Requires Auth)

```bash
curl http://localhost:8000/user/profile \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## 🔧 Production Deployment

### 1. RSA Key Oluştur

```bash
# Private key
openssl genrsa -out private_key.pem 2048

# Public key
openssl rsa -in private_key.pem -pubout -out public_key.pem

# Güvenli bir yere taşı
mkdir -p /etc/r3mes/keys
mv private_key.pem public_key.pem /etc/r3mes/keys/
chmod 600 /etc/r3mes/keys/private_key.pem
```

### 2. Production .env

```bash
R3MES_ENV=production
REDIS_URL=redis://redis-server:6379/0
DATABASE_URL=postgresql://user:pass@db-server:5432/r3mes

# JWT Keys
JWT_PRIVATE_KEY_PATH=/etc/r3mes/keys/private_key.pem
JWT_PUBLIC_KEY_PATH=/etc/r3mes/keys/public_key.pem

# Secrets Management
SECRETS_PROVIDER=aws
AWS_REGION=us-east-1
AWS_SECRET_NAME=r3mes/production

# CORS
CORS_ORIGINS=https://app.r3mes.io,https://dashboard.r3mes.io
```

### 3. Docker ile Çalıştır

```bash
# Build
docker build -t r3mes-backend .

# Run
docker run -d \
  -p 8000:8000 \
  -v /etc/r3mes/keys:/keys:ro \
  -e R3MES_ENV=production \
  -e JWT_PRIVATE_KEY_PATH=/keys/private_key.pem \
  -e JWT_PUBLIC_KEY_PATH=/keys/public_key.pem \
  --name r3mes-backend \
  r3mes-backend
```

### 4. Kubernetes ile Deploy

```bash
# Secrets oluştur
kubectl create secret generic r3mes-jwt-keys \
  --from-file=private_key.pem \
  --from-file=public_key.pem

# Deploy
kubectl apply -f k8s/backend/
```

## 🐛 Troubleshooting

### Redis Bağlantı Hatası

```bash
# Redis çalışıyor mu kontrol et
redis-cli ping

# Docker ile Redis başlat
docker run -d -p 6379:6379 redis:alpine
```

### JWT Key Hatası

```bash
# Development'ta otomatik key üretilir
# Production'da mutlaka RSA key dosyaları gerekli

# Key'leri kontrol et
ls -la /etc/r3mes/keys/
```

### Import Hatası

```bash
# Backend modüllerinin yüklü olduğundan emin ol
pip install -e .

# Veya PYTHONPATH'i ayarla
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## 📚 Daha Fazla Bilgi

- [Backend README](README.md) - Detaylı dokümantasyon
- [API Docs](http://localhost:8000/docs) - Swagger UI
- [ReDoc](http://localhost:8000/redoc) - Alternative API docs

## 🤝 Yardım

Sorun mu yaşıyorsunuz? 

1. [GitHub Issues](https://github.com/r3mes/r3mes/issues)
2. [Discord Community](https://discord.gg/r3mes)
3. [Documentation](https://docs.r3mes.io)

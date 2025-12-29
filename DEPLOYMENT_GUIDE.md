# R3MES Deployment Guide

Bu rehber, R3MES projesini VPS sunucuda (backend) ve Netlify'da (frontend) yayınlamak için adım adım talimatlar içerir.

## 📋 İçindekiler

1. [Gereksinimler](#gereksinimler)
2. [Paket Sürüm Kontrolü](#paket-sürüm-kontrolü)
3. [VPS Sunucu Kurulumu](#vps-sunucu-kurulumu)
4. [SSL Sertifikası Alma](#ssl-sertifikası-alma)
5. [Backend Deployment](#backend-deployment)
6. [Netlify Frontend Deployment](#netlify-frontend-deployment)
7. [DNS Yapılandırması](#dns-yapılandırması)
8. [Monitoring ve Bakım](#monitoring-ve-bakım)

---

## Gereksinimler

### VPS Sunucu Gereksinimleri
- **OS:** Ubuntu 22.04 LTS (önerilen)
- **RAM:** Minimum 4GB (8GB önerilen)
- **CPU:** 2+ vCPU
- **Disk:** 50GB+ SSD
- **Portlar:** 80, 443, 8000, 26656, 26657

### Yazılım Gereksinimleri
- Docker 24.0+
- Docker Compose 2.20+
- Node.js 20+ (frontend build için)
- Python 3.10+
- Nginx
- Certbot (SSL için)

---

## Paket Sürüm Kontrolü

### Frontend (web-dashboard/package.json)

✅ **Uyumlu Paketler:**
- Next.js 14.2.35 - Stabil
- React 18.3.0 - Stabil
- Three.js 0.168.0 - Stabil
- Framer Motion 10.16.0 - Stabil

⚠️ **Dikkat Edilmesi Gerekenler:**
```json
// Jest 30.x yeni çıktı, 29.x'e düşürmeyi düşünün
"jest": "^29.7.0",
"jest-environment-jsdom": "^29.7.0",
"@types/jest": "^29.5.0",

// ESLint 9.x breaking changes içeriyor
"eslint": "^8.57.0",
"eslint-config-next": "^14.2.35"
```

### Backend (backend/requirements.txt)

✅ **Uyumlu Paketler:**
- FastAPI 0.104.1 - Stabil
- Uvicorn 0.24.0 - Stabil
- Pydantic 2.5.0 - Stabil
- PyTorch 2.1.0+ - Stabil

⚠️ **GPU Desteği için:**
```bash
# CUDA 11.8 için
pip install torch==2.1.0+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html
```

---

## VPS Sunucu Kurulumu

### Adım 1: Sunucuya Bağlanma

```bash
ssh root@YOUR_VPS_IP
```

### Adım 2: Sistem Güncelleme

```bash
# Sistem güncelleme
apt update && apt upgrade -y

# Gerekli paketleri yükle
apt install -y curl wget git vim htop ufw fail2ban
```

### Adım 3: Firewall Yapılandırması

```bash
# UFW'yi etkinleştir
ufw default deny incoming
ufw default allow outgoing

# Gerekli portları aç
ufw allow ssh
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 8000/tcp    # Backend API
ufw allow 26656/tcp   # Blockchain P2P
ufw allow 26657/tcp   # Blockchain RPC

# Firewall'u etkinleştir
ufw enable
ufw status
```

### Adım 4: Docker Kurulumu

```bash
# Docker GPG key ekle
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Docker repository ekle
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

# Docker yükle
apt update
apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Docker'ı başlat
systemctl start docker
systemctl enable docker

# Docker versiyonunu kontrol et
docker --version
docker compose version
```

### Adım 5: Kullanıcı Oluşturma

```bash
# r3mes kullanıcısı oluştur
adduser r3mes
usermod -aG docker r3mes
usermod -aG sudo r3mes

# Kullanıcıya geç
su - r3mes
```

---

## SSL Sertifikası Alma

### Yöntem 1: Let's Encrypt (Ücretsiz - Önerilen)

```bash
# Certbot yükle
apt install -y certbot python3-certbot-nginx

# SSL sertifikası al (domain'inizi değiştirin)
certbot --nginx -d api.r3mes.network -d r3mes.network

# Otomatik yenileme test et
certbot renew --dry-run

# Cron job ekle (otomatik yenileme)
echo "0 0,12 * * * root certbot renew --quiet" >> /etc/crontab
```

### Yöntem 2: Cloudflare (Ücretsiz SSL + CDN)

1. Cloudflare'e kaydolun: https://cloudflare.com
2. Domain'inizi ekleyin
3. Nameserver'ları Cloudflare'e yönlendirin
4. SSL/TLS ayarlarından "Full (strict)" seçin
5. Origin Certificate oluşturun:

```bash
# Cloudflare Origin Certificate'ı kaydet
mkdir -p /etc/ssl/cloudflare
nano /etc/ssl/cloudflare/cert.pem    # Certificate yapıştır
nano /etc/ssl/cloudflare/key.pem     # Private key yapıştır
chmod 600 /etc/ssl/cloudflare/key.pem
```

---

## Backend Deployment

### Adım 1: Proje Klonlama

```bash
cd /home/r3mes
git clone https://github.com/YOUR_USERNAME/R3MES.git
cd R3MES
```

### Adım 2: Environment Dosyası Oluşturma

```bash
# Backend .env dosyası
cat > backend/.env.production << 'EOF'
# Environment
R3MES_ENV=production
LOG_LEVEL=INFO

# Database
DATABASE_TYPE=postgresql
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=r3mes
POSTGRES_USER=r3mes_user
POSTGRES_PASSWORD=YOUR_SECURE_PASSWORD_HERE

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=YOUR_REDIS_PASSWORD_HERE

# Blockchain
BLOCKCHAIN_RPC_URL=http://remesd:26657
CHAIN_ID=remes-mainnet-1

# Security
JWT_SECRET=YOUR_JWT_SECRET_HERE
API_SECRET_KEY=YOUR_API_SECRET_HERE

# CORS
CORS_ALLOWED_ORIGINS=https://r3mes.network,https://www.r3mes.network

# Monitoring
SENTRY_DSN=YOUR_SENTRY_DSN
EOF
```

### Adım 3: Docker Secrets Oluşturma

```bash
# Docker Swarm başlat
docker swarm init

# Secrets oluştur
echo "YOUR_POSTGRES_PASSWORD" | docker secret create postgres_password -
echo "r3mes_user" | docker secret create postgres_user -
echo "YOUR_REDIS_PASSWORD" | docker secret create redis_password -
echo "YOUR_JWT_SECRET" | docker secret create jwt_secret -
echo "YOUR_API_SECRET" | docker secret create api_secret_key -
echo "YOUR_GRAFANA_PASSWORD" | docker secret create grafana_admin_password -
```

### Adım 4: Docker Images Build

```bash
# Backend image build
cd backend
docker build -t r3mes/backend:latest .

# Nginx image build (opsiyonel)
cd ../nginx
docker build -t r3mes/nginx:latest .
```

### Adım 5: Docker Compose ile Başlatma

```bash
# Basit deployment için docker-compose.yml
cat > docker-compose.prod.yml << 'EOF'
version: '3.8'

services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: r3mes_user
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: r3mes
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U r3mes_user"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    restart: unless-stopped

  backend:
    image: r3mes/backend:latest
    env_file:
      - backend/.env.production
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - /etc/letsencrypt:/etc/letsencrypt:ro
    depends_on:
      - backend
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
EOF

# Başlat
docker compose -f docker-compose.prod.yml up -d
```

### Adım 6: Nginx Yapılandırması

```bash
cat > nginx/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream backend {
        server backend:8000;
    }

    # HTTP -> HTTPS redirect
    server {
        listen 80;
        server_name api.r3mes.network;
        return 301 https://$server_name$request_uri;
    }

    # HTTPS server
    server {
        listen 443 ssl http2;
        server_name api.r3mes.network;

        ssl_certificate /etc/letsencrypt/live/api.r3mes.network/fullchain.pem;
        ssl_certificate_key /etc/letsencrypt/live/api.r3mes.network/privkey.pem;

        # SSL settings
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
        ssl_prefer_server_ciphers off;

        # Security headers
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;

        location / {
            proxy_pass http://backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_cache_bypass $http_upgrade;
        }

        location /health {
            proxy_pass http://backend/health;
        }
    }
}
EOF
```

---

## Netlify Frontend Deployment

### Adım 1: Netlify Hesabı ve Proje Oluşturma

1. https://netlify.com adresine gidin
2. GitHub ile giriş yapın
3. "Add new site" > "Import an existing project"
4. GitHub repository'nizi seçin

### Adım 2: Build Ayarları

Netlify dashboard'da:

```
Base directory: web-dashboard
Build command: npm run build
Publish directory: web-dashboard/.next
```

### Adım 3: Environment Variables

Netlify > Site settings > Environment variables:

```
NEXT_PUBLIC_API_URL=https://api.r3mes.network
NEXT_PUBLIC_CHAIN_ID=remes-mainnet-1
NEXT_PUBLIC_SITE_URL=https://r3mes.network
NEXT_PUBLIC_RPC_URL=https://rpc.r3mes.network
NEXT_PUBLIC_REST_URL=https://api.r3mes.network
NODE_VERSION=20
```

### Adım 4: netlify.toml Oluşturma

```bash
cat > web-dashboard/netlify.toml << 'EOF'
[build]
  base = "web-dashboard"
  command = "npm run build"
  publish = ".next"

[build.environment]
  NODE_VERSION = "20"
  NPM_FLAGS = "--legacy-peer-deps"

[[plugins]]
  package = "@netlify/plugin-nextjs"

[[headers]]
  for = "/*"
  [headers.values]
    X-Frame-Options = "DENY"
    X-Content-Type-Options = "nosniff"
    X-XSS-Protection = "1; mode=block"
    Referrer-Policy = "strict-origin-when-cross-origin"

[[redirects]]
  from = "/api/*"
  to = "https://api.r3mes.network/:splat"
  status = 200
  force = true
EOF
```

### Adım 5: next.config.js Güncelleme

```javascript
// web-dashboard/next.config.js
/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  images: {
    unoptimized: true,
  },
  // Netlify için gerekli
  experimental: {
    serverActions: {
      allowedOrigins: ['r3mes.network', 'www.r3mes.network'],
    },
  },
};

module.exports = nextConfig;
```

### Adım 6: Deploy

```bash
# Local'de test et
cd web-dashboard
npm install
npm run build

# Git'e push et (Netlify otomatik deploy eder)
git add .
git commit -m "Add Netlify configuration"
git push origin main
```

---

## DNS Yapılandırması

### Cloudflare DNS Ayarları

| Type | Name | Content | Proxy |
|------|------|---------|-------|
| A | @ | VPS_IP | Proxied |
| A | api | VPS_IP | Proxied |
| A | rpc | VPS_IP | DNS only |
| CNAME | www | r3mes.network | Proxied |

### Netlify Custom Domain

1. Netlify > Domain settings > Add custom domain
2. `r3mes.network` ve `www.r3mes.network` ekleyin
3. DNS'i Netlify'a yönlendirin veya CNAME kullanın:

```
CNAME www YOUR_NETLIFY_SUBDOMAIN.netlify.app
```

---

## Monitoring ve Bakım

### Log Kontrolü

```bash
# Docker logs
docker compose -f docker-compose.prod.yml logs -f backend
docker compose -f docker-compose.prod.yml logs -f nginx

# Sistem logs
journalctl -u docker -f
```

### Health Check

```bash
# Backend health
curl https://api.r3mes.network/health

# Frontend health
curl https://r3mes.network
```

### Backup Script

```bash
cat > /home/r3mes/backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR=/home/r3mes/backups

# PostgreSQL backup
docker exec postgres pg_dump -U r3mes_user r3mes > $BACKUP_DIR/db_$DATE.sql

# Compress
gzip $BACKUP_DIR/db_$DATE.sql

# Keep last 7 days
find $BACKUP_DIR -name "*.gz" -mtime +7 -delete
EOF

chmod +x /home/r3mes/backup.sh

# Cron job ekle
echo "0 2 * * * /home/r3mes/backup.sh" >> /etc/crontab
```

### Güncelleme Prosedürü

```bash
# Yeni kod çek
cd /home/r3mes/R3MES
git pull origin main

# Backend güncelle
docker compose -f docker-compose.prod.yml build backend
docker compose -f docker-compose.prod.yml up -d backend

# Netlify otomatik deploy eder (git push ile)
```

---

## Sorun Giderme

### Yaygın Hatalar

**1. SSL Sertifikası Hatası:**
```bash
certbot renew --force-renewal
systemctl reload nginx
```

**2. Docker Container Başlamıyor:**
```bash
docker compose -f docker-compose.prod.yml logs backend
docker compose -f docker-compose.prod.yml restart backend
```

**3. Database Bağlantı Hatası:**
```bash
docker exec -it postgres psql -U r3mes_user -d r3mes
```

**4. Netlify Build Hatası:**
- Build logs'u kontrol edin
- `npm ci` yerine `npm install --legacy-peer-deps` deneyin
- Node version'ı kontrol edin

---

## Güvenlik Kontrol Listesi

- [ ] Firewall aktif (UFW)
- [ ] Fail2ban kurulu
- [ ] SSH key authentication (password disabled)
- [ ] SSL/TLS aktif
- [ ] Environment variables güvenli
- [ ] Database şifresi güçlü
- [ ] CORS doğru yapılandırılmış
- [ ] Rate limiting aktif
- [ ] Backup sistemi çalışıyor

---

## Destek

Sorularınız için:
- Discord: https://discord.gg/r3mes
- GitHub Issues: https://github.com/r3mes/issues

# R3MES Testnet Deployment Guide

## 📋 Overview

Bu rehber R3MES Testnet'in kurulumunu adım adım açıklar.

### Mimari

```
┌─────────────────────────────────────────────────────────────┐
│                    NETLIFY (Web Dashboard)                  │
│                    r3mes.network                            │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ HTTPS
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 SUNUCU (38.242.246.6)                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   NGINX     │  │  Validator  │  │    IPFS     │         │
│  │   (SSL)     │  │    Node     │  │   Gateway   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Backend    │  │  PostgreSQL │  │    Redis    │         │
│  │    API      │  │             │  │             │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### Subdomains

| Subdomain | Hedef | Port | Açıklama |
|-----------|-------|------|----------|
| r3mes.network | Netlify | - | Web Dashboard |
| api.r3mes.network | 38.242.246.6 | 8000 | Backend API |
| rpc.r3mes.network | 38.242.246.6 | 26657 | Blockchain RPC |
| rest.r3mes.network | 38.242.246.6 | 1317 | Blockchain REST |
| ipfs.r3mes.network | 38.242.246.6 | 8080 | IPFS Gateway |
| grafana.r3mes.network | 38.242.246.6 | 3001 | Monitoring |

### Wallet Adresleri (Testnet)

| Wallet | Adres | Bakiye |
|--------|-------|--------|
| Validator | remes1jrtxwayldq3l3wu4frt8eg9syzcdkzw7vmmshm | 100,000 R3MES (staked) |
| Faucet | remes19asaj7tyd9p698uqq74dwx5m2k6882cvq3c9lx | 1,000,000 R3MES |
| Treasury | remes16kykek0kkvk803mfw4e3wukykhaccla26tvxhq | 8,900,000 R3MES |

---

## 🚀 Quick Deploy (Mevcut Kurulum)

Sunucuda zaten kurulum varsa:

```bash
ssh root@38.242.246.6
cd /opt/r3mes

# Kodu güncelle
git pull origin main

# Deploy et
bash deploy/testnet/deploy-validator.sh deploy

# Durumu kontrol et
bash deploy/testnet/deploy-validator.sh status
```

### Deploy Script Komutları

```bash
# Tam deployment
bash deploy/testnet/deploy-validator.sh deploy

# Sadece validator'ı rebuild et
bash deploy/testnet/deploy-validator.sh rebuild

# Tüm servisleri restart et
bash deploy/testnet/deploy-validator.sh restart

# Logları izle
bash deploy/testnet/deploy-validator.sh logs [service]

# Durum kontrolü
bash deploy/testnet/deploy-validator.sh status

# Faucet testi
bash deploy/testnet/deploy-validator.sh test-faucet
```

---

## 🌐 Step 1: DNS Configuration

Domain sağlayıcınızda (Cloudflare, Namecheap, etc.) şu DNS kayıtlarını ekleyin:

### A Records

```
Type    Name      Value           TTL
A       @         [Netlify IP]    Auto
A       www       [Netlify IP]    Auto
A       api       38.242.246.6    Auto
A       rpc       38.242.246.6    Auto
A       rest      38.242.246.6    Auto
A       ipfs      38.242.246.6    Auto
A       grafana   38.242.246.6    Auto
```

### CNAME for Netlify (Alternative)

```
Type    Name      Value                   TTL
CNAME   @         your-site.netlify.app   Auto
CNAME   www       your-site.netlify.app   Auto
```

> ⚠️ DNS propagation 24 saate kadar sürebilir.

---

## 🖥️ Step 2: Server Setup

### 2.1 SSH ile Sunucuya Bağlan

```bash
ssh root@38.242.246.6
```

### 2.2 Eski Kurulumu Temizle

```bash
# Script'i indir ve çalıştır
curl -sSL https://raw.githubusercontent.com/YOUR_REPO/R3MES/main/deploy/testnet/scripts/01-cleanup-server.sh | bash
```

Veya manuel:

```bash
# Docker container'ları durdur
docker stop $(docker ps -aq)
docker rm $(docker ps -aq)

# Volume'ları sil
docker volume rm $(docker volume ls -q)

# Eski dizinleri sil
rm -rf /opt/r3mes /root/.remes

# Docker temizliği
docker system prune -af --volumes
```

### 2.3 Sunucuyu Hazırla

```bash
# Script'i indir ve çalıştır
curl -sSL https://raw.githubusercontent.com/YOUR_REPO/R3MES/main/deploy/testnet/scripts/02-setup-server.sh | bash
```

Bu script:
- Sistem güncellemesi yapar
- Docker ve Docker Compose kurar
- Firewall ayarlarını yapar
- Fail2ban kurar
- Gerekli dizinleri oluşturur

### 2.4 Projeyi Klonla

```bash
cd /opt/r3mes
git clone https://github.com/YOUR_REPO/R3MES.git .
```

### 2.5 Environment Dosyasını Yapılandır

```bash
cd /opt/r3mes/deploy/testnet
cp .env.example .env
nano .env
```

Şu değerleri güncelle:
- `DB_PASSWORD`: Güçlü bir şifre
- `SECRET_KEY`: 32+ karakter rastgele string
- `GRAFANA_PASSWORD`: Grafana admin şifresi

### 2.6 SSL Sertifikalarını Al

```bash
chmod +x scripts/*.sh
./scripts/03-setup-ssl.sh
```

### 2.7 Deploy Et

```bash
./scripts/04-deploy.sh
```

Bu script:
- Docker image'larını build eder
- Wallet'ları oluşturur
- Genesis dosyasını günceller
- Servisleri başlatır

---

## 🌐 Step 3: Netlify Deployment

### 3.1 Netlify'da Yeni Site Oluştur

1. [Netlify](https://app.netlify.com) hesabına giriş yap
2. "Add new site" → "Import an existing project"
3. GitHub repo'nu seç
4. Build settings:
   - Base directory: `web-dashboard`
   - Build command: `npm run build`
   - Publish directory: `web-dashboard/.next`

### 3.2 Environment Variables

Netlify dashboard'da Site settings → Environment variables:

```
NEXT_PUBLIC_API_URL = https://api.r3mes.network
NEXT_PUBLIC_RPC_URL = https://rpc.r3mes.network
NEXT_PUBLIC_REST_URL = https://rest.r3mes.network
NEXT_PUBLIC_IPFS_GATEWAY = https://ipfs.r3mes.network
NEXT_PUBLIC_CHAIN_ID = r3mes-testnet-1
NEXT_PUBLIC_CHAIN_NAME = R3MES Testnet
NEXT_PUBLIC_DENOM = ur3mes
NEXT_PUBLIC_DENOM_DISPLAY = R3MES
NEXT_PUBLIC_FAUCET_ENABLED = true
```

### 3.3 Custom Domain

1. Site settings → Domain management
2. Add custom domain: `r3mes.network`
3. DNS ayarlarını Netlify'ın verdiği değerlerle güncelle

### 3.4 Deploy

```bash
# Otomatik deploy (GitHub push ile)
git push origin main

# Manuel deploy
netlify deploy --prod
```

---

## ✅ Step 4: Verification

### Sunucu Servisleri

```bash
# Container durumu
docker-compose -f docker-compose.testnet.yml ps

# Logları kontrol et
docker-compose -f docker-compose.testnet.yml logs -f

# Blockchain durumu
curl http://localhost:26657/status | jq
```

### Endpoint'leri Test Et

```bash
# API
curl https://api.r3mes.network/health

# RPC
curl https://rpc.r3mes.network/status

# REST
curl https://rest.r3mes.network/cosmos/base/tendermint/v1beta1/node_info

# IPFS
curl https://ipfs.r3mes.network/api/v0/id
```

### Web Dashboard

Tarayıcıda https://r3mes.network adresini aç.

---

## 🔧 Maintenance

### Logları İzle

```bash
# Tüm loglar
docker-compose -f docker-compose.testnet.yml logs -f

# Belirli servis
docker-compose -f docker-compose.testnet.yml logs -f validator
```

### Restart

```bash
# Tüm servisleri restart
docker-compose -f docker-compose.testnet.yml restart

# Belirli servis
docker-compose -f docker-compose.testnet.yml restart backend
```

### Update

```bash
cd /opt/r3mes
git pull origin main
docker-compose -f deploy/testnet/docker-compose.testnet.yml build
docker-compose -f deploy/testnet/docker-compose.testnet.yml up -d
```

### Backup

```bash
# Database backup
docker exec r3mes-postgres pg_dump -U r3mes r3mes > backup_$(date +%Y%m%d).sql

# Validator data backup
docker cp r3mes-validator:/root/.remes ./validator_backup_$(date +%Y%m%d)
```

---

## 🆘 Troubleshooting

### Container başlamıyor

```bash
# Logları kontrol et
docker-compose -f docker-compose.testnet.yml logs validator

# Container'a gir
docker exec -it r3mes-validator sh
```

### SSL hatası

```bash
# Sertifikaları yenile
certbot renew --force-renewal
docker restart r3mes-nginx
```

### Database bağlantı hatası

```bash
# PostgreSQL durumu
docker exec r3mes-postgres pg_isready -U r3mes

# Bağlantıyı test et
docker exec -it r3mes-postgres psql -U r3mes -d r3mes
```

---

## 📞 Support

- GitHub Issues: https://github.com/YOUR_REPO/R3MES/issues
- Discord: https://discord.gg/r3mes
- Email: support@r3mes.network

# R3MES Deployment

## 🚀 Tek Komutla Kurulum (VPS)

### Gereksinimler
- Ubuntu 22.04 LTS
- Root erişimi
- Domain DNS ayarları yapılmış olmalı

### DNS Ayarları (Önce Yapılmalı!)

Domain sağlayıcınızda şu kayıtları ekleyin:

| Type | Name    | Value          |
|------|---------|----------------|
| A    | @       | 38.242.246.6   |
| A    | api     | 38.242.246.6   |
| A    | testnet | 38.242.246.6   |
| A    | rpc     | 38.242.246.6   |
| A    | www     | 38.242.246.6   |

### Kurulum

```bash
# VPS'e bağlan
ssh root@38.242.246.6

# Projeyi indir ve kur
git clone https://github.com/YOUR_USERNAME/R3MES.git /opt/r3mes
cd /opt/r3mes/deploy
chmod +x install.sh
./install.sh
```

**Bu kadar!** Script otomatik olarak:
- Docker kurulumu
- Firewall yapılandırması
- SSL sertifikası alma
- Tüm servisleri başlatma
- Backup ve SSL yenileme cron job'ları

---

## 🌐 Netlify Frontend Deployment

### Adım 1: Netlify'da Yeni Site

1. https://app.netlify.com adresine git
2. "Add new site" > "Import an existing project"
3. GitHub'ı bağla ve R3MES repository'sini seç

### Adım 2: Build Ayarları

```
Base directory: web-dashboard
Build command: npm run build
Publish directory: web-dashboard/.next
```

### Adım 3: Environment Variables

Netlify Dashboard > Site settings > Environment variables:

```
NEXT_PUBLIC_API_URL = https://api.r3mes.network
NEXT_PUBLIC_BACKEND_URL = https://api.r3mes.network
NEXT_PUBLIC_SITE_URL = https://r3mes.network
NEXT_PUBLIC_RPC_URL = https://rpc.r3mes.network
NEXT_PUBLIC_CHAIN_ID = r3mes-testnet-1
NODE_VERSION = 20
```

### Adım 4: Custom Domain

1. Domain settings > Add custom domain
2. `r3mes.network` ekle
3. DNS'i Netlify'a yönlendir veya:

```
CNAME www YOUR-SITE.netlify.app
```

---

## 📁 Dosya Yapısı

```
deploy/
├── docker-compose.production.yml  # Ana Docker yapılandırması
├── .env.production                # Environment değişkenleri
├── install.sh                     # Tek komut kurulum scripti
├── nginx/
│   ├── nginx.conf                 # Ana Nginx yapılandırması
│   └── conf.d/
│       └── default.conf           # Site yapılandırmaları
└── certbot/                       # SSL sertifikaları (otomatik oluşur)
```

---

## 🔐 Güvenlik Bilgileri

Şifreler `.env.production` dosyasında:

```
POSTGRES_PASSWORD=Xk9#mP2$vL7@nQ4wR8!jF5
REDIS_PASSWORD=Hy6$tN3#kW9@pM1!vB8xZ2
JWT_SECRET=aR7$kL2#mX9@pQ4!wN6vB3tY8hJ5gF1cD0eS
API_SECRET_KEY=zU4#nK8$vM2@wP6!xQ9tL3yH7jB5gF1cR0eA
```

⚠️ **ÖNEMLİ:** Production'da bu şifreleri değiştirin!

---

## 🛠️ Yönetim Komutları

```bash
# Logları görüntüle
docker compose -f /opt/r3mes/deploy/docker-compose.production.yml logs -f

# Servisleri yeniden başlat
docker compose -f /opt/r3mes/deploy/docker-compose.production.yml restart

# Servisleri durdur
docker compose -f /opt/r3mes/deploy/docker-compose.production.yml down

# Manuel backup
/usr/local/bin/r3mes-backup.sh

# SSL sertifikasını yenile
docker compose -f /opt/r3mes/deploy/docker-compose.production.yml run --rm certbot renew
```

---

## 🔗 URL'ler

| Servis  | URL                           |
|---------|-------------------------------|
| Website | https://r3mes.network         |
| API     | https://api.r3mes.network     |
| Testnet | https://testnet.r3mes.network |
| RPC     | https://rpc.r3mes.network     |

---

## ❓ Sorun Giderme

### SSL Sertifikası Alınamıyor
```bash
# DNS'in doğru ayarlandığını kontrol et
dig api.r3mes.network

# Manuel SSL al
docker run --rm -v "/opt/r3mes/deploy/certbot/conf:/etc/letsencrypt" \
  -v "/opt/r3mes/deploy/certbot/www:/var/www/certbot" \
  certbot/certbot certonly --webroot --webroot-path=/var/www/certbot \
  --email admin@r3mes.network --agree-tos --no-eff-email \
  -d r3mes.network -d api.r3mes.network
```

### Backend Başlamıyor
```bash
# Logları kontrol et
docker compose -f /opt/r3mes/deploy/docker-compose.production.yml logs backend

# Database bağlantısını test et
docker exec r3mes-postgres psql -U r3mes_admin -d r3mes -c "SELECT 1"
```

### Netlify Build Hatası
- Node version'ı kontrol et (20 olmalı)
- `npm install --legacy-peer-deps` dene
- Build loglarını incele

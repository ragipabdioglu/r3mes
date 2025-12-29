# R3MES Quick Deploy Guide

## 🚀 Tek Komutla Kurulum

R3MES'i sunucuya tek komutla kurmak için:

### Testnet Kurulumu

```bash
git clone <your-repo-url> R3MES
cd R3MES
bash scripts/quick_deploy.sh --domain testnet.r3mes.network --email admin@r3mes.network
```

### Mainnet Kurulumu

```bash
git clone <your-repo-url> R3MES
cd R3MES
bash scripts/quick_deploy.sh --domain r3mes.network --email admin@r3mes.network --mainnet
```

### Özel Chain ID ile

```bash
bash scripts/quick_deploy.sh \
  --domain testnet.r3mes.network \
  --email admin@r3mes.network \
  --chain-id remes-testnet-1
```

## ✨ Script Ne Yapıyor?

`quick_deploy.sh` script'i otomatik olarak:

1. ✅ **Docker Kontrolü**: Docker ve Docker Compose kurulu mu kontrol eder, yoksa kurar
2. ✅ **Secrets Oluşturma**: PostgreSQL, Redis ve Grafana için rastgele şifreler oluşturur
3. ✅ **Environment Yapılandırması**: `.env.production` dosyasını otomatik oluşturur ve ayarlar
4. ✅ **Deployment**: Tüm servisleri başlatır

## 📋 Gereksinimler

- Ubuntu 20.04/22.04 veya benzeri Linux dağıtımı
- Root veya sudo erişimi
- Domain adı (Let's Encrypt için)
- Minimum: 4 vCPU, 8GB RAM, 75GB disk
- Port 80 ve 443 açık (firewall)

## 🔧 Parametreler

### Zorunlu Parametreler

- `--domain DOMAIN`: Domain adınız (örn: `testnet.r3mes.network`)
- `--email EMAIL`: Let's Encrypt için email adresiniz

### Opsiyonel Parametreler

- `--chain-id CHAIN_ID`: Chain ID (varsayılan: network tipine göre otomatik)
- `--mainnet`: Mainnet için deploy (varsayılan: testnet)
- `--testnet`: Testnet için deploy (varsayılan)
- `--no-auto-secrets`: Secrets'ları otomatik oluşturma (manuel oluştur)
- `--skip-docker-check`: Docker kurulum kontrolünü atla
- `--help, -h`: Yardım mesajını göster

## 📝 Örnekler

### Basit Testnet Kurulumu

```bash
bash scripts/quick_deploy.sh \
  --domain testnet.r3mes.network \
  --email admin@r3mes.network
```

### Mainnet Kurulumu

```bash
bash scripts/quick_deploy.sh \
  --domain r3mes.network \
  --email admin@r3mes.network \
  --mainnet \
  --chain-id remes-mainnet-1
```

### Manuel Secrets ile

```bash
# Önce secrets oluştur
bash scripts/create_secrets.sh

# Sonra deploy et (secrets'ları kullanır)
bash scripts/quick_deploy.sh \
  --domain testnet.r3mes.network \
  --email admin@r3mes.network \
  --no-auto-secrets
```

## ✅ Kurulum Sonrası

Kurulum tamamlandıktan sonra:

1. **Servislerin başlamasını bekleyin** (2-5 dakika)
2. **Durumu kontrol edin:**
   ```bash
   cd docker
   docker compose -f docker-compose.prod.yml ps
   ```
3. **Logları kontrol edin:**
   ```bash
   cd docker
   docker compose -f docker-compose.prod.yml logs -f
   ```
4. **Health check yapın:**
   - Frontend: `https://your-domain/health`
   - Backend: `https://your-domain/api/health`
   - Blockchain: `https://your-domain/api/blockchain/health`

## 🌐 Erişim

- **Web Dashboard**: `https://your-domain`
- **API Docs**: `https://your-domain/api/docs`
- **Grafana**: `https://your-domain:3001` (admin şifresi: secrets'ta)

## 🔍 Sorun Giderme

### Docker Kurulumu Başarısız

Script otomatik kurmaya çalışır, ama başarısız olursa:
```bash
# Ubuntu/Debian için
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

### Secrets Zaten Var

Script mevcut secrets'ları kullanır. Yeni secrets oluşturmak için:
```bash
rm docker/secrets/*.txt
bash scripts/quick_deploy.sh --domain ... --email ...
```

### Port Çakışması

Eğer portlar kullanılıyorsa:
```bash
# Hangi servislerin çalıştığını kontrol et
docker ps
# Gerekirse durdur
docker stop <container-name>
```

## 📚 Daha Fazla Bilgi

- **[Docker Production Guide](docker/README_PRODUCTION.md)** - Detaylı Docker deployment
- **[Testnet Deployment Guide](docs/TESTNET_DEPLOYMENT.md)** - Testnet özel talimatlar
- **[Contabo VPS Guide](docker/CONTOBO_DEPLOYMENT_GUIDE.md)** - VPS kurulum rehberi


# R3MES Testnet Kurulum Rehberi

## 🚀 Hızlı Başlangıç

### Sunucuya Bağlan
```bash
ssh root@SUNUCU_IP
```

### Projeyi İndir
```bash
git clone https://github.com/YOUR_USERNAME/R3MES.git
cd R3MES/docker
```

### Tek Komutla Başlat
```bash
chmod +x start_testnet.sh
./start_testnet.sh
```

Bu script otomatik olarak:
1. ✅ Docker secrets oluşturur (şifreler)
2. ✅ Environment dosyasını kopyalar
3. ✅ Docker images build eder
4. ✅ Tüm servisleri başlatır
5. ✅ Genesis otomatik oluşturur

---

## 📋 Manuel Kurulum (Alternatif)

### 1. Secrets Oluştur
```bash
cd docker
bash ../scripts/create_secrets.sh
# Enter'a basarak random şifreler oluştur
```

### 2. Environment Dosyasını Kopyala
```bash
cp .env.testnet .env
# Gerekirse düzenle: nano .env
```

### 3. Servisleri Başlat
```bash
docker compose -f docker-compose.prod.yml up -d
```

### 4. Logları İzle
```bash
# Tüm loglar
docker compose -f docker-compose.prod.yml logs -f

# Sadece blockchain
docker logs -f r3mes-blockchain-prod
```

---

## 🌐 Servis Adresleri

| Servis | Port | URL |
|--------|------|-----|
| Frontend | 3000 | https://r3mes.network |
| Backend API | 8000 | https://r3mes.network/api |
| Blockchain RPC | 26657 | http://localhost:26657 |
| Blockchain gRPC | 9090 | localhost:9090 |
| Blockchain REST | 1317 | http://localhost:1317 |
| IPFS API | 5001 | http://localhost:5001 |
| Grafana | 3001 | https://r3mes.network/grafana |

---

## 🔧 Faydalı Komutlar

### Servis Durumu
```bash
docker compose -f docker-compose.prod.yml ps
```

### Servisleri Durdur
```bash
docker compose -f docker-compose.prod.yml down
```

### Servisleri Yeniden Başlat
```bash
docker compose -f docker-compose.prod.yml restart
```

### Logları Görüntüle
```bash
# Tüm servisler
docker compose -f docker-compose.prod.yml logs -f

# Belirli servis
docker logs -f r3mes-blockchain-prod
docker logs -f r3mes-backend-prod
docker logs -f r3mes-frontend-prod
```

### Blockchain Durumu
```bash
curl http://localhost:26657/status | jq
```

### Validator Bilgileri
```bash
docker exec r3mes-blockchain-prod remesd keys list --home /app/.remesd --keyring-backend test
```

---

## 🔐 Güvenlik

### Firewall Ayarları
```bash
# UFW ile
ufw allow 22      # SSH
ufw allow 80      # HTTP
ufw allow 443     # HTTPS
ufw allow 26656   # P2P
ufw enable
```

### SSL Sertifikası
Certbot container otomatik olarak Let's Encrypt sertifikası alır ve yeniler.

---

## 📊 Monitoring

Grafana'ya erişim:
- URL: https://r3mes.network/grafana
- Kullanıcı: admin
- Şifre: `docker/secrets/grafana_admin_password.txt` dosyasında

---

## ⚠️ Sorun Giderme

### Container başlamıyorsa
```bash
docker compose -f docker-compose.prod.yml logs [servis_adı]
```

### Genesis hatası
```bash
# Volume'u sil ve yeniden başlat
docker compose -f docker-compose.prod.yml down -v
docker compose -f docker-compose.prod.yml up -d
```

### Port çakışması
```bash
# Hangi portlar kullanılıyor?
ss -tlnp | grep -E '(26657|9090|8000|3000)'
```

---

## 🖥️ Miner Bağlantısı

Başka bir PC'den miner bağlamak için:

```bash
# Miner PC'de
export R3MES_NODE_GRPC_URL="SUNUCU_IP:9090"
export R3MES_IPFS_URL="http://SUNUCU_IP:5001"

cd miner-engine
pip install -e .
python -m r3mes.cli.commands start
```

---

## 📞 Destek

Sorun yaşarsan:
1. Docker loglarını kontrol et
2. `docker ps` ile container durumunu gör
3. Firewall ayarlarını kontrol et

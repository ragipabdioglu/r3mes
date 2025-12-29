# Contabo VPS Deployment Guide - Docker Secrets ile

## Genel Bakış

Bu rehber, Contabo VPS'de R3MES'i Docker Secrets kullanarak kurmak için adım adım talimatlar içerir.

## Ön Gereksinimler

- Ubuntu 20.04/22.04
- Root veya sudo erişimi
- Domain adı (Let's Encrypt için)
- Minimum: 4 vCPU, 8GB RAM, 75GB disk (önerilen: 6 vCPU, 12GB RAM, 200GB disk)

## Adım 1: Sunucuya Bağlan

```bash
ssh root@your-vps-ip
# veya
ssh your-username@your-vps-ip
```

## Adım 2: Docker Kurulumu

```bash
# Sistem güncellemesi
sudo apt update && sudo apt upgrade -y

# Eski Docker versiyonlarını kaldır (varsa)
sudo apt remove docker docker-engine docker.io containerd runc -y

# Docker için gerekli paketler
sudo apt install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Docker'ın resmi GPG key'ini ekle
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Docker repository ekle
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Docker kur
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Docker servisini başlat
sudo systemctl start docker
sudo systemctl enable docker

# Docker'ı sudo olmadan kullanmak için
sudo usermod -aG docker $USER

# Kurulumu kontrol et
docker --version
docker compose version
```

**Not:** Docker grup değişikliği için logout/login gerekir.

## Adım 3: Firewall Ayarları

```bash
# UFW aktifse, gerekli portları aç
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP (Let's Encrypt için)
sudo ufw allow 443/tcp   # HTTPS

# Firewall'u aktif et (eğer değilse)
sudo ufw enable

# Durumu kontrol et
sudo ufw status
```

## Adım 4: Projeyi Sunucuya Yükle

### Seçenek A: Git ile (Önerilen)

```bash
# Git kur (eğer yoksa)
sudo apt install -y git

# Projeyi clone et
git clone <your-repo-url> R3MES
cd R3MES
```

### Seçenek B: Manuel Yükleme

```bash
# Projeyi zip olarak yükle (FTP, SCP, vs.)
# Sonra aç:
unzip R3MES.zip
cd R3MES
```

## Adım 5: Docker Secrets Oluştur

### Yöntem 1: Interactive (Önerilen - İlk Kurulum)

```bash
bash scripts/create_secrets.sh
```

Bu script:
- Her secret için şifre ister
- Enter'a basarsanız otomatik güçlü şifre oluşturur
- Secrets dosyalarını `docker/secrets/` dizinine kaydeder

**Örnek çıktı:**
```
Enter PostgreSQL password (press Enter to generate random): [Enter]
Generated random password
✅ PostgreSQL password secret created

Enter Redis password (press Enter to generate random): [Enter]
Generated random password
✅ Redis password secret created

Enter Grafana admin password (press Enter to generate random): [Enter]
Generated random password
✅ Grafana admin password secret created
```

### Yöntem 2: .env.production'dan Otomatik

```bash
# Önce .env.production dosyasını oluşturun
cd docker
cp env.production.example .env.production
nano .env.production  # Şifreleri doldurun

# Sonra secrets oluşturun
bash ../scripts/create_secrets_from_env.sh
```

## Adım 6: ENV Dosyası Oluştur (Non-Secret Variables)

```bash
cd docker
cp env.production.example .env.production
nano .env.production
```

### Minimum Doldurulması Gerekenler:

```bash
# Domain adınız
DOMAIN=your-domain.com

# Email adresiniz (Let's Encrypt için)
EMAIL=your-email@example.com

# Frontend URL'leri (domain'inize göre)
NEXT_PUBLIC_API_URL=https://your-domain.com/api
NEXT_PUBLIC_BACKEND_URL=https://your-domain.com/api
NEXT_PUBLIC_SITE_URL=https://your-domain.com
NEXT_PUBLIC_RPC_URL=https://rpc.your-domain.com
NEXT_PUBLIC_REST_URL=https://api.your-domain.com

# CORS ayarları (domain'inize göre)
CORS_ALLOWED_ORIGINS=https://your-domain.com,https://www.your-domain.com
```

**Not:** Şifreler artık secrets dosyalarında, `.env.production`'da değil!

## Adım 7: DNS Ayarları

Domain'inizi Contabo VPS IP'sine yönlendirin:

- **A record:** `@` → VPS IP adresi
- **A record:** `www` → VPS IP adresi (opsiyonel)

DNS yayılması 5-30 dakika sürebilir.

## Adım 8: Sistemi Başlat

### Deployment Script ile (Önerilen)

```bash
bash scripts/deploy_production_docker.sh
```

Script otomatik olarak:
- Docker ve Docker Compose kontrolü yapar
- Secrets'ları kontrol eder (yoksa oluşturur)
- ENV dosyasını kontrol eder
- GPU desteğini kontrol eder
- Network ve volume'ları oluşturur
- Servisleri başlatır
- Health check'leri bekler

### Manuel Başlatma

```bash
cd docker
docker compose -f docker-compose.prod.yml up -d
```

## Adım 9: Durumu Kontrol Et

```bash
# Tüm servislerin durumu
docker compose -f docker-compose.prod.yml ps

# Logları kontrol et
docker compose -f docker-compose.prod.yml logs -f

# Belirli bir servisin logları
docker compose -f docker-compose.prod.yml logs -f backend
```

## Adım 10: SSL Sertifikası (Otomatik)

Let's Encrypt sertifikası otomatik alınır. İlk başlatmada:
- Port 80'in açık olması gerekir
- DNS'in yayılmış olması gerekir
- Certbot container'ı sertifikayı alır

## Adım 11: GPU Mining (Opsiyonel)

NVIDIA GPU varsa:

```bash
# NVIDIA Container Toolkit kur
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker

# GPU mining ile başlat
cd docker
docker compose -f docker-compose.prod.yml --profile miner up -d
```

## Secrets vs ENV Dosyası

### Secrets Dosyaları (`docker/secrets/`)

**Kullanım:** Sensitive data (şifreler)
- `postgres_password.txt` - PostgreSQL şifresi
- `redis_password.txt` - Redis şifresi
- `grafana_admin_password.txt` - Grafana admin şifresi

**Oluşturma:**
```bash
bash scripts/create_secrets.sh
```

### ENV Dosyası (`.env.production`)

**Kullanım:** Non-sensitive configuration
- Domain adları
- Email adresi
- Public URLs
- Feature flags

**Oluşturma:**
```bash
cd docker
cp env.production.example .env.production
nano .env.production
```

## Sorun Giderme

### Secrets Bulunamıyor

```bash
# Secrets dizinini kontrol et
ls -la docker/secrets/

# Secrets oluştur
bash scripts/create_secrets.sh
```

### Servisler Başlamıyor

```bash
# Logları kontrol et
docker compose -f docker-compose.prod.yml logs

# Secrets dosyalarını kontrol et
ls -l docker/secrets/
cat docker/secrets/postgres_password.txt  # Dikkatli!
```

### Database Bağlantı Hatası

```bash
# Backend loglarını kontrol et
docker compose -f docker-compose.prod.yml logs backend | grep -i database

# PostgreSQL loglarını kontrol et
docker compose -f docker-compose.prod.yml logs postgres
```

## Güvenlik Notları

1. **Secrets dosyalarını asla commit etmeyin** - `.gitignore` zaten ignore eder
2. **Dosya izinlerini kontrol edin:**
   ```bash
   chmod 600 docker/secrets/*.txt
   ```
3. **Secrets'ları düzenli olarak rotate edin**
4. **`.env.production` dosyasını da commit etmeyin**

## Özet Checklist

- [ ] Docker kuruldu (`docker --version`)
- [ ] Docker Compose kuruldu (`docker compose version`)
- [ ] Port 80, 443 açık (UFW)
- [ ] DNS ayarları yapıldı
- [ ] Secrets oluşturuldu (`bash scripts/create_secrets.sh`)
- [ ] `.env.production` dosyası hazırlandı
- [ ] Proje sunucuya yüklendi
- [ ] Servisler başlatıldı (`docker compose ps`)
- [ ] SSL sertifikaları alındı (otomatik)

## Hızlı Komutlar

```bash
# Servisleri başlat
cd docker && docker compose -f docker-compose.prod.yml up -d

# Servisleri durdur
cd docker && docker compose -f docker-compose.prod.yml down

# Logları görüntüle
cd docker && docker compose -f docker-compose.prod.yml logs -f

# Servisleri yeniden başlat
cd docker && docker compose -f docker-compose.prod.yml restart

# Secrets yeniden oluştur
bash scripts/create_secrets.sh
```

## Destek

Sorun yaşarsanız:
1. Logları kontrol edin: `docker compose logs`
2. Secrets dosyalarını kontrol edin: `ls -la docker/secrets/`
3. Health check'leri kontrol edin: `docker compose ps`


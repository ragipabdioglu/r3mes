# Docker Secrets Kullanım Rehberi

## Genel Bakış

R3MES production deployment'ında Docker secrets kullanılmaktadır. Bu, hassas bilgilerin (şifreler, API key'leri) güvenli bir şekilde saklanmasını sağlar.

## Secrets Dosyaları

Docker Compose, secrets'ları dosyalardan okur. Tüm secrets dosyaları `docker/secrets/` dizininde saklanır:

- `postgres_password.txt` - PostgreSQL veritabanı şifresi
- `redis_password.txt` - Redis şifresi
- `grafana_admin_password.txt` - Grafana admin şifresi

## Secrets Oluşturma

### Yöntem 1: Interactive (Önerilen - İlk Kurulum)

```bash
bash scripts/create_secrets.sh
```

Bu script:
- Her secret için sizden şifre ister
- Enter'a basarsanız otomatik güçlü şifre oluşturur
- Secrets dosyalarını `docker/secrets/` dizinine kaydeder
- Dosya izinlerini 600 (sadece owner okuyabilir) yapar

### Yöntem 2: .env.production'dan Otomatik

```bash
# Önce .env.production dosyasını oluşturun
cd docker
cp env.production.example .env.production
nano .env.production  # Şifreleri doldurun

# Sonra secrets oluşturun
bash scripts/create_secrets_from_env.sh
```

Bu script:
- `.env.production` dosyasından şifreleri okur
- Secrets dosyalarını oluşturur
- `.env.production` dosyasındaki şifreleri kontrol eder

## Secrets Kullanımı

### Docker Compose'da

Secrets otomatik olarak servislere mount edilir:

```yaml
services:
  postgres:
    secrets:
      - postgres_password
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
```

### Backend Kodunda

Backend kodu otomatik olarak secrets dosyalarını okur:

```python
# backend/app/database_config.py
# POSTGRES_PASSWORD_FILE varsa otomatik okur ve DATABASE_URL oluşturur

# backend/app/cache.py
# REDIS_PASSWORD_FILE varsa otomatik okur ve REDIS_URL oluşturur
```

## Güvenlik

### Dosya İzinleri

Secrets dosyaları otomatik olarak 600 izinleriyle oluşturulur:

```bash
chmod 600 docker/secrets/*.txt
```

### Git'te Saklamayın

`.gitignore` dosyası secrets dizinini zaten ignore eder:

```
secrets/
*.txt
```

### Production'da

- Secrets dosyalarını asla commit etmeyin
- Secrets dosyalarını sadece root veya deploy user'ı okuyabilir olmalı
- Secrets dosyalarını düzenli olarak rotate edin

## Secrets Güncelleme

### Şifre Değiştirme

1. Yeni şifreyi secrets dosyasına yazın:

```bash
echo -n "yeni-şifre" > docker/secrets/postgres_password.txt
chmod 600 docker/secrets/postgres_password.txt
```

2. Servisleri yeniden başlatın:

```bash
docker compose -f docker-compose.prod.yml restart postgres
```

### Tüm Secrets'ı Yeniden Oluşturma

```bash
# Mevcut secrets'ı silin (dikkatli!)
rm docker/secrets/*.txt

# Yeni secrets oluşturun
bash scripts/create_secrets.sh
```

## Sorun Giderme

### Secrets Dosyası Bulunamıyor

```bash
# Secrets dizinini kontrol edin
ls -la docker/secrets/

# Eğer yoksa oluşturun
mkdir -p docker/secrets
bash scripts/create_secrets.sh
```

### Şifre Okunamıyor

```bash
# Dosya izinlerini kontrol edin
ls -l docker/secrets/

# İzinleri düzeltin
chmod 600 docker/secrets/*.txt
```

### Container'da Secret Görünmüyor

```bash
# Container içinde kontrol edin
docker exec r3mes-postgres-prod ls -la /run/secrets/

# Secret içeriğini kontrol edin (dikkatli!)
docker exec r3mes-postgres-prod cat /run/secrets/postgres_password
```

## ENV Dosyası vs Secrets

### ENV Dosyası (.env.production)

**Kullanım:** Non-sensitive configuration
- Domain adları
- Port numaraları
- Feature flags
- Public API URLs

**Örnek:**
```bash
DOMAIN=r3mes.network
EMAIL=admin@r3mes.network
CORS_ALLOWED_ORIGINS=https://r3mes.network
```

### Secrets (docker/secrets/)

**Kullanım:** Sensitive data
- Şifreler
- API key'leri
- Private key'ler

**Örnek:**
```bash
# docker/secrets/postgres_password.txt
GüçlüŞifre123!@#456
```

## Deployment Workflow

1. **Secrets Oluştur:**
   ```bash
   bash scripts/create_secrets.sh
   ```

2. **ENV Dosyası Oluştur:**
   ```bash
   cd docker
   cp env.production.example .env.production
   nano .env.production  # Domain, email vs. doldurun
   ```

3. **Deploy:**
   ```bash
   bash scripts/deploy_production_docker.sh
   ```

Script otomatik olarak:
- Secrets'ları kontrol eder
- Eksikse oluşturur
- Servisleri başlatır

## Özet

- ✅ Secrets dosyaları `docker/secrets/` dizininde
- ✅ Otomatik oluşturma scriptleri mevcut
- ✅ Backend kodu otomatik olarak secrets'ları okur
- ✅ Git'te saklanmaz (`.gitignore`)
- ✅ Güvenli dosya izinleri (600)


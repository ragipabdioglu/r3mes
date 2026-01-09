# Google Cloud Secret Manager Setup Guide

Bu rehber, R3MES projesi için Google Cloud Secret Manager'ın nasıl yapılandırılacağını detaylı olarak açıklar.

## Genel Bakış

Google Cloud Secret Manager, production ortamında secret'ları güvenli bir şekilde saklamak ve yönetmek için kullanılır. AWS Secrets Manager'a alternatif olarak kullanılabilir.

## Önkoşullar

- Google Cloud hesabı
- Google Cloud SDK kurulu
- Python 3.10+
- `google-cloud-secret-manager` Python paketi

---

## Adım 1: Google Cloud SDK Kurulumu

### Ubuntu/Debian

```bash
# Method 1: Official installer
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Method 2: APT repository
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
sudo apt-get update && sudo apt-get install google-cloud-sdk
```

### macOS

```bash
# Homebrew ile
brew install --cask google-cloud-sdk
```

### Windows

```bash
# Chocolatey ile
choco install gcloudsdk

# Veya resmi installer'ı indir:
# https://cloud.google.com/sdk/docs/install
```

---

## Adım 2: Google Cloud Authentication

### 2.1 Google Cloud'a Login

```bash
# Interactive login
gcloud auth login

# Browser açılacak, Google hesabınızla giriş yapın
```

### 2.2 Project Seç

```bash
# Mevcut project'leri listele
gcloud projects list

# Project seç
gcloud config set project YOUR_PROJECT_ID

# Veya yeni project oluştur
gcloud projects create r3mes-production --name="R3MES Production"
gcloud config set project r3mes-production
```

### 2.3 Application Default Credentials (ADC) Ayarla

**Development için (önerilen)**:
```bash
gcloud auth application-default login
```

Bu komut credentials'ları `~/.config/gcloud/application_default_credentials.json` dosyasına kaydeder.

**Production için (Service Account önerilen)**:
```bash
# Service account oluştur
gcloud iam service-accounts create r3mes-secret-manager \
    --display-name="R3MES Secret Manager" \
    --description="Service account for R3MES secret management"

# Secret Manager Admin role'ü ver
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:r3mes-secret-manager@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/secretmanager.admin"

# Service account key oluştur
gcloud iam service-accounts keys create ~/r3mes-gcp-key.json \
    --iam-account=r3mes-secret-manager@YOUR_PROJECT_ID.iam.gserviceaccount.com

# Environment variable olarak ayarla
export GOOGLE_APPLICATION_CREDENTIALS=~/r3mes-gcp-key.json

# Güvenlik: Key dosyasını sadece okunabilir yap
chmod 600 ~/r3mes-gcp-key.json
```

---

## Adım 3: Secret Manager API'yi Enable Et

```bash
# Secret Manager API'yi enable et
gcloud services enable secretmanager.googleapis.com

# Kontrol et
gcloud services list --enabled | grep secretmanager
```

---

## Adım 4: Secret'ları Oluşturma

### 4.1 Temel Secret Oluşturma

```bash
# Project ID'yi ayarla
export GOOGLE_CLOUD_PROJECT="your-project-id"

# 1. Database URL
echo -n "postgresql://r3mes_user:STRONG_PASSWORD@db.r3mes.network:5432/r3mes" | \
    gcloud secrets create r3mes-production-database-url \
    --data-file=- \
    --replication-policy="automatic" \
    --labels=environment=production,component=database

# 2. Redis URL
echo -n "redis://STRONG_PASSWORD@redis.r3mes.network:6379/0" | \
    gcloud secrets create r3mes-production-redis-url \
    --data-file=- \
    --replication-policy="automatic" \
    --labels=environment=production,component=cache

# 3. API Key Secret (minimum 32 karakter)
echo -n "YOUR_32_CHARACTER_MINIMUM_SECRET_KEY_HERE" | \
    gcloud secrets create r3mes-production-api-key-secret \
    --data-file=- \
    --replication-policy="automatic" \
    --labels=environment=production,component=security

# 4. JWT Secret (minimum 32 karakter)
echo -n "YOUR_32_CHARACTER_MINIMUM_JWT_SECRET_HERE" | \
    gcloud secrets create r3mes-production-jwt-secret \
    --data-file=- \
    --replication-policy="automatic" \
    --labels=environment=production,component=security

# 5. Blockchain RPC URL
echo -n "https://rpc.r3mes.network:26657" | \
    gcloud secrets create r3mes-production-blockchain-rpc-url \
    --data-file=- \
    --replication-policy="automatic" \
    --labels=environment=production,component=blockchain

# 6. Blockchain gRPC URL
echo -n "rpc.r3mes.network:9090" | \
    gcloud secrets create r3mes-production-blockchain-grpc-url \
    --data-file=- \
    --replication-policy="automatic" \
    --labels=environment=production,component=blockchain

# 7. Blockchain REST URL
echo -n "https://rpc.r3mes.network:1317" | \
    gcloud secrets create r3mes-production-blockchain-rest-url \
    --data-file=- \
    --replication-policy="automatic" \
    --labels=environment=production,component=blockchain
```

### 4.2 Secret'ları Listele

```bash
# Tüm secret'ları listele
gcloud secrets list

# Belirli prefix ile filtrele
gcloud secrets list --filter="name:r3mes-production-*"

# Secret detaylarını görüntüle
gcloud secrets describe r3mes-production-database-url
```

### 4.3 Secret Versiyonları

```bash
# Secret'ın tüm versiyonlarını listele
gcloud secrets versions list r3mes-production-database-url

# Yeni versiyon ekle (secret rotation için)
echo -n "new_secret_value" | \
    gcloud secrets versions add r3mes-production-database-url \
    --data-file=-

# Belirli versiyonu disable et
gcloud secrets versions disable 1 \
    --secret=r3mes-production-database-url

# Belirli versiyonu destroy et (geri alınamaz!)
gcloud secrets versions destroy 1 \
    --secret=r3mes-production-database-url
```

---

## Adım 5: Secret Erişim Kontrolü

### 5.1 Service Account'a Erişim Ver

```bash
# Tek bir secret için
gcloud secrets add-iam-policy-binding r3mes-production-database-url \
    --member="serviceAccount:r3mes-secret-manager@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"

# Tüm production secret'ları için (script)
for secret in r3mes-production-database-url \
              r3mes-production-redis-url \
              r3mes-production-api-key-secret \
              r3mes-production-jwt-secret \
              r3mes-production-blockchain-rpc-url \
              r3mes-production-blockchain-grpc-url \
              r3mes-production-blockchain-rest-url; do
    gcloud secrets add-iam-policy-binding $secret \
        --member="serviceAccount:r3mes-secret-manager@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
        --role="roles/secretmanager.secretAccessor"
done
```

### 5.2 IAM Policy Kontrolü

```bash
# Secret'ın IAM policy'sini görüntüle
gcloud secrets get-iam-policy r3mes-production-database-url
```

---

## Adım 6: Python Dependency Kurulumu

```bash
# Backend dizinine git
cd /home/rabdi/R3MES/backend

# Virtual environment aktif et (eğer kullanıyorsan)
source venv/bin/activate

# Google Cloud Secret Manager paketini yükle
pip install google-cloud-secret-manager>=2.18.0

# Veya requirements.txt'ten yükle
pip install -r requirements.txt
```

---

## Adım 7: Environment Variables Ayarlama

### 7.1 Systemd Service için

```bash
sudo nano /etc/systemd/system/r3mes-backend.service
```

Şu environment variables'ları ekle:

```ini
[Service]
Environment="R3MES_ENV=production"
Environment="GOOGLE_CLOUD_PROJECT=your-project-id"
Environment="GOOGLE_APPLICATION_CREDENTIALS=/home/remes/r3mes-gcp-key.json"
Environment="LOG_LEVEL=INFO"
Environment="ENABLE_FILE_LOGGING=true"
```

**Not**: `GOOGLE_APPLICATION_CREDENTIALS` opsiyoneldir. Eğer Application Default Credentials (ADC) kullanıyorsanız, bu satırı eklemeyin.

### 7.2 Environment File Kullan (Önerilen)

```bash
# Production environment file oluştur
sudo mkdir -p /etc/r3mes
sudo nano /etc/r3mes/production.env
```

İçeriği:
```bash
R3MES_ENV=production
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/home/remes/r3mes-gcp-key.json
LOG_LEVEL=INFO
ENABLE_FILE_LOGGING=true
```

Systemd service'te kullan:
```ini
[Service]
EnvironmentFile=/etc/r3mes/production.env
```

### 7.3 Docker Compose için

```yaml
services:
  backend:
    environment:
      - R3MES_ENV=production
      - GOOGLE_CLOUD_PROJECT=your-project-id
      - GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
    volumes:
      - /path/to/r3mes-gcp-key.json:/path/to/key.json:ro
```

---

## Adım 8: Secret'ları Kodda Kullanma

### 8.1 Otomatik Kullanım (Önerilen)

Kod otomatik olarak `GOOGLE_CLOUD_PROJECT` environment variable'ını kontrol eder ve Google Cloud Secret Manager'ı kullanır:

```python
from app.secrets import get_secret

# Secret'ları otomatik olarak Google Cloud'tan alır
database_url = get_secret("r3mes-production-database-url")
redis_url = get_secret("r3mes-production-redis-url")
api_key_secret = get_secret("r3mes-production-api-key-secret")
```

### 8.2 Manuel Kullanım

```python
from app.secrets import GoogleCloudSecretManager

# Initialize
secret_manager = GoogleCloudSecretManager(project_id="your-project-id")

# Get secret
database_url = secret_manager.get_secret("r3mes-production-database-url")

# Get multiple secrets
secrets = secret_manager.get_secrets("r3mes-production")
```

---

## Adım 9: Test Etme

### 9.1 Connection Test

```bash
# Environment variables ayarla
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_APPLICATION_CREDENTIALS=~/r3mes-gcp-key.json
export R3MES_ENV=production

# Python test script'i
python3 -c "
import os
os.environ['R3MES_ENV'] = 'production'
os.environ['GOOGLE_CLOUD_PROJECT'] = 'your-project-id'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '~/r3mes-gcp-key.json'

from backend.app.secrets import get_secret_manager
sm = get_secret_manager()
print(f'Secret Manager: {type(sm).__name__}')
print(f'Connection: {sm.test_connection()}')

# Test secret okuma
try:
    db_url = sm.get_secret('r3mes-production-database-url')
    print(f'Database URL retrieved: {db_url[:30]}...')
except Exception as e:
    print(f'Error: {e}')
"
```

### 9.2 Production Validation Script

```bash
# Production environment validation
python scripts/validate_production_env.py

# Başarılı çıktı:
# ✅ Environment variables validated successfully
# ✅ Secret management service connection successful (GoogleCloudSecretManager)
# ✅ All production environment validations passed!
```

### 9.3 Health Check Endpoint

```bash
# Backend başlatıldıktan sonra
curl http://localhost:8000/health/secrets

# Başarılı response:
# {"status":"healthy","secret_manager":"GoogleCloudSecretManager","type":"GoogleCloudSecretManager"}
```

---

## Adım 10: Secret Rotation (Opsiyonel)

### 10.1 Yeni Versiyon Ekle

```bash
# Yeni secret değeri oluştur
echo -n "new_database_password" | \
    gcloud secrets versions add r3mes-production-database-url \
    --data-file=-

# Versiyonları listele
gcloud secrets versions list r3mes-production-database-url
```

### 10.2 Eski Versiyonları Temizle

```bash
# Eski versiyonları disable et
gcloud secrets versions disable 1 \
    --secret=r3mes-production-database-url

# Veya destroy et (geri alınamaz!)
gcloud secrets versions destroy 1 \
    --secret=r3mes-production-database-url
```

---

## Troubleshooting

### Problem: "Permission denied" hatası

**Çözüm**:
```bash
# Service account'a gerekli permission'ları ver
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:r3mes-secret-manager@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

### Problem: "Secret not found" hatası

**Çözüm**:
```bash
# Secret'ın var olduğunu kontrol et
gcloud secrets list --filter="name:r3mes-production-database-url"

# Secret ismini doğru yazdığından emin ol
# Google Cloud Secret Manager'da secret isimleri kebab-case kullanır
```

### Problem: "Authentication failed" hatası

**Çözüm**:
```bash
# Application Default Credentials'ı yenile
gcloud auth application-default login

# Veya service account key dosyasının path'ini kontrol et
echo $GOOGLE_APPLICATION_CREDENTIALS
ls -la $GOOGLE_APPLICATION_CREDENTIALS
```

### Problem: "Project not found" hatası

**Çözüm**:
```bash
# Project ID'yi kontrol et
gcloud config get-value project

# Doğru project'i seç
gcloud config set project YOUR_PROJECT_ID

# Environment variable'ı kontrol et
echo $GOOGLE_CLOUD_PROJECT
```

---

## Best Practices

1. **Secret Naming**: Kebab-case kullan (`r3mes-production-database-url`)
2. **Labels**: Secret'lara label ekle (environment, component)
3. **Replication**: Production için `automatic` replication kullan
4. **Access Control**: Service account kullan, user credentials değil
5. **Rotation**: Düzenli olarak secret'ları rotate et
6. **Monitoring**: Secret access'leri logla ve monitor et
7. **Backup**: Kritik secret'ları yedekle (encrypted)

---

## Cost Optimization

Google Cloud Secret Manager pricing:
- **Secret storage**: $0.06 per secret per month
- **Secret access**: $0.03 per 10,000 operations
- **Secret versions**: $0.06 per version per month

**Optimization Tips**:
- Gereksiz versiyonları sil
- Secret'ları cache'le (kod zaten 5 dakika cache yapıyor)
- Gereksiz secret'ları oluşturma

---

## Security Best Practices

1. **Service Account**: Production'da user credentials yerine service account kullan
2. **Least Privilege**: Service account'a sadece gerekli permission'ları ver
3. **Key Rotation**: Service account key'lerini düzenli rotate et
4. **Audit Logging**: Secret access'leri audit et
5. **Encryption**: Secret'lar otomatik olarak encrypted, ekstra bir şey yapmana gerek yok

---

## Kaynaklar

- [Google Cloud Secret Manager Documentation](https://cloud.google.com/secret-manager/docs)
- [Python Client Library](https://cloud.google.com/python/docs/reference/secretmanager/latest)
- [IAM Best Practices](https://cloud.google.com/iam/docs/using-iam-securely)


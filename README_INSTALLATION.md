# R3MES Kurulum Kılavuzu

## Sistem Gereksinimleri

- **İşletim Sistemi**: Windows 10+, macOS 10.15+, veya Linux (Ubuntu 20.04+)
- **Python**: 3.8 veya üzeri (sadece geliştirme için)
- **Docker**: 20.10+ (önerilir)
- **CUDA**: 11.0+ (GPU hızlandırma için, opsiyonel)
- **RAM**: Minimum 8GB, önerilen 16GB+
- **Disk**: Minimum 10GB boş alan

## Hızlı Başlangıç

### 1. Bağımlılıkları Kontrol Et

```bash
# Backend dizininde
cd backend
python check_dependencies.py
```

Bu script Docker ve CUDA kurulumunu kontrol eder ve eksik olanlar için yükleme sayfalarını açar.

### 2. Backend Binary Oluşturma (Opsiyonel)

Standalone binary oluşturmak için:

```bash
cd backend
python build_binary.py
```

Binary dosyası `backend/dist/r3mes_backend` (veya `.exe` Windows'ta) konumunda oluşturulur.

### 3. Backend'i Başlat

#### Geliştirme Modu

```bash
cd backend
pip install -r requirements.txt
python -m app.main
```

#### Production Modu (Binary ile)

```bash
# Windows
dist\r3mes_backend.exe

# Linux/macOS
./dist/r3mes_backend
```

### 4. Frontend'i Başlat

```bash
cd web-dashboard
npm install
npm run dev
```

Uygulama `http://localhost:3000` adresinde çalışacaktır.

## Docker Kurulumu

### Windows/macOS

1. [Docker Desktop](https://www.docker.com/products/docker-desktop) indirin ve kurun
2. Docker Desktop'ı başlatın
3. Kurulumu doğrulayın:
   ```bash
   docker --version
   ```

### Linux

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Fedora
sudo dnf install docker
sudo systemctl start docker
sudo systemctl enable docker
```

## CUDA Kurulumu

### Windows

1. [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) indirin
2. Kurulum sihirbazını takip edin
3. Kurulumu doğrulayın:
   ```bash
   nvidia-smi
   ```

### Linux

```bash
# Ubuntu
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda
```

## Yapılandırma

### Environment Variables

Backend için `.env` dosyası oluşturun:

```env
# Database
DATABASE_PATH=backend/database.db
CHAIN_JSON_PATH=chain.json

# Model
BASE_MODEL_PATH=checkpoints/base_model
MODEL_DOWNLOAD_DIR=~/.r3mes/models

# Mining
MINING_DIFFICULTY=1234.0
GPU_MEMORY_LIMIT_MB=8192
P2P_PORT=9090

# API
RATE_LIMIT_CHAT=10/minute
RATE_LIMIT_GET=30/minute

# Network
BLOCKCHAIN_RPC_URL=http://localhost:26657
BLOCKCHAIN_GRPC_URL=localhost:9090

# Features
AUTO_START_MINING=false
ENABLE_NOTIFICATIONS=true

# Security
R3MES_ENV=production
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

### UI'dan Yapılandırma

1. Uygulamayı başlatın
2. Cüzdanınızı bağlayın
3. Settings sayfasına gidin (`/settings`)
4. İstediğiniz ayarları yapın

## Sorun Giderme

### Backend başlamıyor

1. Port'un kullanılabilir olduğundan emin olun (varsayılan: 8000)
2. Python bağımlılıklarının kurulu olduğunu kontrol edin:
   ```bash
   pip install -r requirements.txt
   ```
3. Log dosyalarını kontrol edin: `~/.r3mes/logs/r3mes_backend.log`

### GPU algılanmıyor

1. CUDA'nın kurulu olduğunu doğrulayın:
   ```bash
   nvidia-smi
   ```
2. PyTorch'un CUDA desteği ile kurulu olduğundan emin olun
3. GPU bellek limitini ayarlardan kontrol edin

### Docker hatası

1. Docker Desktop'ın çalıştığından emin olun
2. Docker servisinin başlatıldığını kontrol edin:
   ```bash
   docker ps
   ```

## Daha Fazla Yardım

- [Yardım Sayfası](/help) - Uygulama içi SSS ve destek
- [GitHub Issues](https://github.com/r3mes/r3mes/issues) - Sorun bildirimi
- [Dokümantasyon](/docs) - Detaylı kullanım kılavuzu

## 📚 İlgili Dokümantasyon

- [COMPREHENSIVE_DOCUMENTATION.md](./COMPREHENSIVE_DOCUMENTATION.md) - Tüm API endpoint'leri, component'ler ve environment variables
- [TEST_GUIDE.md](./TEST_GUIDE.md) - Kapsamlı test kılavuzu
- [QUICK_START.md](./QUICK_START.md) - Hızlı başlangıç kılavuzu
- [PROJECT_STATUS_REPORT.md](./PROJECT_STATUS_REPORT.md) - Proje durum raporu


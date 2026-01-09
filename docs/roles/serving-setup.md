# Serving Node Setup Guide

R3MES Network'te Serving Node olarak AI inference hizmeti sunmak için bu rehberi takip edin.

## Gereksinimler

### Donanım
- **GPU**: NVIDIA GPU (minimum 16GB VRAM, önerilen 24GB+)
- **CPU**: 8+ core
- **RAM**: Minimum 32GB
- **Depolama**: 100GB+ NVMe SSD
- **İnternet**: Stabil bağlantı (minimum 100 Mbps, düşük latency)

### Yazılım
- Ubuntu 20.04+ veya Debian 11+
- Docker 24.0+
- NVIDIA Container Toolkit
- CUDA 11.8+

## Kurulum Adımları

### 1. Sistem Hazırlığı

```bash
# Sistem güncellemesi
sudo apt update && sudo apt upgrade -y

# Docker kurulumu
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2. R3MES Serving Node Kurulumu

```bash
# Repository klonlama
git clone https://github.com/r3mes/r3mes.git
cd r3mes

# Environment dosyası oluşturma
cp deploy/.env.example deploy/.env.serving
```

### 3. Konfigürasyon

`deploy/.env.serving` dosyasını düzenleyin:

```env
# Serving Node Configuration
R3MES_ENV=production
R3MES_INFERENCE_MODE=local

# Wallet (stake için)
SERVING_WALLET_ADDRESS=remes1your_wallet_address

# API Endpoint (public erişim için)
SERVING_ENDPOINT_URL=https://your-domain.com:8080

# GPU Settings
CUDA_VISIBLE_DEVICES=0
MAX_BATCH_SIZE=8
MAX_CONCURRENT_REQUESTS=4

# Model Settings
BASE_MODEL_PATH=/models/bitnet_b1_58
LORA_ADAPTERS_PATH=/models/adapters
```

### 4. Cüzdan ve Stake

Serving Node olmak için minimum **1,000 REMES** stake etmeniz gerekiyor.

```bash
# Cüzdan oluşturma (yoksa)
remesd keys add serving-node

# Stake işlemi
remesd tx remes stake-serving 1000000000uremes \
  --from serving-node \
  --chain-id remes-mainnet-1 \
  --gas auto \
  --gas-adjustment 1.5
```

### 5. Docker ile Başlatma

```bash
# Serving Node başlatma
docker compose -f deploy/docker-compose.serving.yml up -d

# Logları kontrol etme
docker logs -f r3mes-serving
```

### 6. Endpoint Kaydı

Serving Node'unuzu ağa kaydedin:

```bash
remesd tx remes register-serving-endpoint \
  "https://your-domain.com:8080" \
  --from serving-node \
  --chain-id remes-mainnet-1
```

## Docker Compose Örneği

```yaml
# docker-compose.serving.yml
version: '3.8'

services:
  serving:
    build:
      context: ../backend
      dockerfile: Dockerfile
    container_name: r3mes-serving
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - R3MES_INFERENCE_MODE=local
    volumes:
      - ./models:/models
    ports:
      - "8080:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Kazanç Modeli

| Aktivite | Ödül |
|----------|------|
| Inference İsteği | 0.01 REMES/istek |
| Uptime Bonusu | %10 ekstra (99.9% uptime) |
| Hız Bonusu | %5 ekstra (< 100ms latency) |

## Monitoring

```bash
# GPU kullanımı
nvidia-smi -l 1

# Container durumu
docker stats r3mes-serving

# API health check
curl http://localhost:8080/health
```

## Güvenlik

1. HTTPS kullanın (Let's Encrypt önerilir)
2. Rate limiting aktif edin
3. Firewall kuralları ayarlayın
4. Düzenli güvenlik güncellemeleri yapın

## Destek

- Discord: [discord.gg/r3mes](https://discord.gg/r3mes)
- Docs: [docs.r3mes.network](https://docs.r3mes.network)

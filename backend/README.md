# R3MES Backend Inference Service

FastAPI tabanlı AI inference servisi. Multi-LoRA desteği, kredi tabanlı ekonomi ve akıllı yönlendirme özellikleri içerir.

## Kurulum

### 1. Bağımlılıkları Yükle

```bash
cd backend
pip install -r requirements.txt
```

### 2. Environment Variables (Opsiyonel)

`.env` dosyası oluştur:

```bash
BASE_MODEL_PATH=checkpoints/base_model
DATABASE_PATH=backend/database.db
CHAIN_JSON_PATH=chain.json

# Semantic Router Configuration
USE_SEMANTIC_ROUTER=true          # Semantic router kullanılsın mı? (true/false)
SEMANTIC_ROUTER_THRESHOLD=0.7    # Minimum similarity threshold (0.0-1.0)

# Rate Limiting
RATE_LIMIT_CHAT=10/minute         # Chat endpoint rate limit
RATE_LIMIT_GET=30/minute           # GET endpoint rate limit

# CORS Configuration
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173  # Allowed origins (comma-separated)
CORS_ALLOW_ALL=false              # Allow all origins (development only, set to true)
```

### 3. Model Dosyalarını Hazırla

```bash
mkdir -p checkpoints
# Base model ve LoRA adaptörlerini checkpoints/ klasörüne kopyala
```

## Kullanım

### Sunucuyu Başlat

```bash
# Ana dizinden
python run_backend.py

# Veya backend klasöründen
cd backend
python -m app.main
```

### API Endpoint'leri

- `POST /chat` - Chat endpoint (AI inference)
- `GET /user/info/{wallet_address}` - Kullanıcı bilgileri
- `GET /network/stats` - Ağ istatistikleri
- `GET /health` - Health check
- `GET /docs` - API dokümantasyonu (Swagger UI)

#### API Key Management Endpoints

- `POST /api-keys/create` - Yeni API key oluştur
- `GET /api-keys/list/{wallet_address}` - API key'leri listele
- `POST /api-keys/revoke` - API key'i iptal et
- `DELETE /api-keys/delete` - API key'i sil

### Örnek Kullanım

```bash
# Chat endpoint (wallet address ile)
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I write a Python function?", "wallet_address": "remes1abc..."}'

# Chat endpoint (API key ile)
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: r3mes_your_api_key_here" \
  -d '{"message": "How do I write a Python function?"}'

# API key oluştur
curl -X POST "http://localhost:8000/api-keys/create" \
  -H "Content-Type: application/json" \
  -d '{"wallet_address": "remes1abc...", "name": "My API Key", "expires_days": 30}'

# API key'leri listele
curl "http://localhost:8000/api-keys/list/remes1abc..."

# User info
curl "http://localhost:8000/user/info/remes1abc..."

# Network stats
curl "http://localhost:8000/network/stats"
```

## Semantic Router

Backend servisi, mesajları semantic similarity ile analiz ederek uygun LoRA adaptörünü seçer.

### Özellikler

- **Embedding-based Routing**: Sentence transformers kullanarak semantic similarity hesaplar
- **Threshold Mekanizması**: Similarity skoru belirli bir eşiğin altındaysa `default_adapter` kullanır
- **Fallback**: Semantic router başarısız olursa keyword-based router'a düşer
- **Performance**: <30ms inference süresi

### Test

```bash
# Semantic router'ı test et
python -m backend.app.test_semantic_router
```

### Yapılandırma

- `USE_SEMANTIC_ROUTER=true`: Semantic router'ı aktif et (varsayılan: **true**)
- `USE_SEMANTIC_ROUTER=false`: Keyword-based router kullan
- `SEMANTIC_ROUTER_THRESHOLD=0.7`: Minimum similarity threshold (varsayılan: 0.7)

**Not**: Semantic router varsayılan olarak aktiftir (`USE_SEMANTIC_ROUTER=true`). Semantic router embedding model gerektirdiği için `sentence-transformers` kütüphanesine ihtiyaç duyar. Semantic router CPU'da çalışır ve CUDA gerektirmez (sadece bitsandbytes CUDA gerektirir, semantic router için gerekli değil).

## Troubleshooting

### Model Yüklenemiyor

- `BASE_MODEL_PATH` environment variable'ını kontrol et
- Model dosyalarının `checkpoints/base_model/` klasöründe olduğundan emin ol

### Database Hatası

- `DATABASE_PATH` klasörünün yazılabilir olduğundan emin ol
- SQLite3 kurulu olduğundan emin ol (Python ile birlikte gelir)

### CUDA/GPU Hatası

- CUDA kurulu olduğundan emin ol
- `bitsandbytes` CUDA desteği ile kurulmuş olmalı

### Semantic Router Hatası

- `sentence-transformers` paketinin kurulu olduğundan emin ol: `pip install sentence-transformers`
- İlk çalıştırmada embedding modeli indirilecek (~80MB)
- Eğer semantic router çalışmazsa otomatik olarak keyword router'a düşer

## Detaylı Dokümantasyon

`docs/14_backend_inference_service.md` dosyasına bakın.


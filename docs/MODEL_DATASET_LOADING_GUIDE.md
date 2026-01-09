# R3MES Model ve Dataset Yükleme Rehberi

## 1. Sistem Mimarisi

```
┌─────────────────────────────────────────────────────────────────────┐
│                         BLOCKCHAIN                                   │
│  (Onaylı model/dataset/adapter hash'leri - immutable)               │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      UnifiedRegistry                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │ModelRegistry│  │DatasetReg.  │  │AdapterReg.  │                 │
│  └─────────────┘  └─────────────┘  └─────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DownloadManager                                 │
│  HuggingFace → IPFS → HTTP (fallback sırası)                        │
│  + Retry (exponential backoff)                                       │
│  + Atomic download (rollback on failure)                             │
│  + MANDATORY checksum verification                                   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Yerel Depolama                                  │
│  ~/.r3mes/                                                           │
│  ├── models/v1.0.0/                                                  │
│  ├── datasets/dataset_id/                                            │
│  └── adapters/general_dora.pt                                        │
└─────────────────────────────────────────────────────────────────────┘
```

## 2. Dosya Konumları

### Windows
```
C:\Users\<username>\.r3mes\
├── models\           # AI modelleri
├── datasets\         # Eğitim datasetleri
├── adapters\         # DoRA/LoRA adaptörleri
└── cache\            # Registry cache
```

### Linux/macOS
```
~/.r3mes/
├── models/
├── datasets/
├── adapters/
└── cache/
```

## 3. Model Yükleme

### 3.1 Blockchain'den Otomatik Yükleme (Önerilen)

```python
from utils.unified_registry import UnifiedRegistry
from bridge.blockchain_client import BlockchainClient
from utils.ipfs_client import IPFSClient

# Client'ları başlat
blockchain_client = BlockchainClient(
    node_url="localhost:9090",
    chain_id="remes-testnet-1",
    private_key="YOUR_PRIVATE_KEY"
)
ipfs_client = IPFSClient()

# Registry başlat
registry = UnifiedRegistry(
    blockchain_client=blockchain_client,
    ipfs_client=ipfs_client,
    auto_download=True,
    verify_on_load=True,
)

# Blockchain'den sync et ve yükle
result = registry.ensure_ready_atomic(
    required_adapters=["general_dora"],
    require_dataset=False,
)

if result.success:
    print(f"✅ Model yüklendi: {result.status.model_path}")
    print(f"✅ Adaptörler: {result.status.loaded_adapters}")
else:
    print(f"❌ Hata: {result.error_message}")
```

### 3.2 Manuel Model Yükleme

```python
from utils.model_registry import ModelRegistry

registry = ModelRegistry(
    blockchain_client=blockchain_client,
    ipfs_client=ipfs_client,
)

# Blockchain'den sync
registry.sync_from_blockchain()

# Model indir ve doğrula
success, path = registry.download_model(model_version="1.0.0")
if success:
    print(f"Model indirildi: {path}")
```

### 3.3 HuggingFace'den Yükleme (Fallback)

Model blockchain'de HuggingFace repo bilgisi varsa otomatik fallback:

```python
# ModelInfo içinde:
model_info = ModelInfo(
    model_id="bitnet-b1.58",
    version="1.0.0",
    ipfs_hash="QmXXX...",
    checksum="abc123...",
    huggingface_repo="R3MES-Network/bitnet-b1.58",  # Fallback
    huggingface_revision="main",
    ...
)
```

## 4. Dataset Yükleme

### 4.1 Otomatik Yükleme

```python
from utils.dataset_registry import DatasetRegistry

registry = DatasetRegistry(
    blockchain_client=blockchain_client,
    ipfs_client=ipfs_client,
)

# Sync ve yükle
registry.sync_from_blockchain()
success, path = registry.download_dataset("training_dataset_v1")
```

### 4.2 Streaming Mode (Büyük Datasetler)

```python
# Belleğe yüklemeden streaming
dataset, error = registry.load_dataset(
    dataset_id="large_dataset",
    streaming=True,  # Memory-efficient
)

if dataset:
    for record in dataset:
        # Her kayıt tek tek işlenir
        process(record)
```

## 5. Adaptör Yükleme

### 5.1 Uyumluluk Kontrolü (İndirmeden ÖNCE)

```python
from utils.adapter_registry import AdapterRegistry

registry = AdapterRegistry(
    blockchain_client=blockchain_client,
    ipfs_client=ipfs_client,
    current_model_version="1.0.0",
)

# Uyumluluk kontrolü
is_compatible, reason = registry.check_compatibility("coder_adapter")
if not is_compatible:
    print(f"Adaptör uyumsuz: {reason}")
    # İndirme yapılmaz!
```

### 5.2 Çakışma Kontrolü

```python
# Yüklü adaptörlerle çakışma kontrolü
has_conflicts, conflicts = registry.check_conflicts("new_adapter")
if has_conflicts:
    print(f"Çakışan adaptörler: {conflicts}")
```

## 6. Environment Yapılandırması

### .env Dosyası

```bash
# Model Yapılandırması
R3MES_MODEL_IPFS_HASH=QmXXX...          # Blockchain'den alınır
R3MES_MODEL_PATH=~/.r3mes/models        # Yerel depolama
R3MES_ADAPTER_DIR=~/.r3mes/adapters     # Adaptör dizini

# IPFS Yapılandırması
IPFS_GATEWAY=http://localhost:8080      # IPFS gateway
IPFS_API_URL=http://localhost:5001      # IPFS API

# Blockchain Yapılandırması
BLOCKCHAIN_GRPC_URL=localhost:9090      # gRPC endpoint
CHAIN_ID=remes-testnet-1

# Backend Inference Mode
R3MES_INFERENCE_MODE=local              # local, remote, mock
```

## 7. Web Dashboard Entegrasyonu

### Chat Sayfası Akışı

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Frontend   │────▶│   Backend    │────▶│ServingEngine │
│  (Chat UI)   │     │  (/chat)     │     │(InferencePipe│
└──────────────┘     └──────────────┘     └──────────────┘
                                                  │
                                                  ▼
                                          ┌──────────────┐
                                          │UnifiedRegistry│
                                          │(Model/Adapter)│
                                          └──────────────┘
```

### Backend Başlatma

```bash
# Development
cd backend
uvicorn app.main:app --reload --port 8000

# Production
R3MES_ENV=production uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Frontend Başlatma

```bash
cd web-dashboard
npm run dev
```

## 8. Güvenlik Özellikleri

| Özellik | Açıklama |
|---------|----------|
| Zorunlu Checksum | Her indirmede SHA256 doğrulaması |
| Atomic Download | Başarısız indirmelerde rollback |
| Retry Mechanism | Exponential backoff ile yeniden deneme |
| Uyumluluk Kontrolü | İndirmeden ÖNCE versiyon kontrolü |
| Çakışma Tespiti | Adaptör modül çakışması kontrolü |
| Trust Chain | Blockchain → IPFS → Local doğrulama |

## 9. Sorun Giderme

### Model Yüklenmiyor

```bash
# Cache temizle
rm -rf ~/.r3mes/cache/

# Blockchain sync
python -c "
from utils.unified_registry import UnifiedRegistry
registry = UnifiedRegistry(...)
registry.sync_all()
"
```

### Checksum Hatası

```bash
# Model'i yeniden indir
python -c "
from utils.model_registry import ModelRegistry
registry = ModelRegistry(...)
registry.download_model('1.0.0', force=True)
"
```

### IPFS Bağlantı Hatası

```bash
# IPFS daemon kontrol
ipfs daemon

# Gateway test
curl http://localhost:8080/ipfs/QmTest
```

## 10. Test Komutu

```bash
cd miner-engine
python -m pytest tests/test_registry_security.py -v
```

# Hibrit Model İndirme Sistemi

## Genel Bakış

R3MES artık **Hibrit İndirme Sistemi** kullanıyor. Bu sistem, modelleri HuggingFace'den doğrudan indirir ve SHA256 hash kontrolü ile güvenliği sağlar. IPFS sadece fallback olarak kullanılır.

## Yeni Yapı

### Genesis/Config Yapısı

Eski yapı (IPFS-only):
```json
{
  "model_ipfs": "QmEskiVeCokBuyukDosyaHashi..."
}
```

Yeni yapı (Hibrit):
```json
{
  "model_config": {
    "model_name": "Llama-3-8B-R3MES-Optimized",
    "file_name": "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
    "download_source": "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
    "verification_sha256": "BURAYA_DOSYANIN_SHA256_KODUNU_YAZACAKSIN",
    "required_disk_space_gb": 6,
    "ipfs_fallback_hash": "QmEskiVeCokBuyukDosyaHashi..."
  }
}
```

## Model Format: GGUF

R3MES artık **GGUF formatı** kullanıyor. Bu format:
- ✅ Daha az RAM gerektirir (8GB minimum, 16GB önerilen)
- ✅ Daha az disk alanı gerektirir (6GB minimum, 10GB önerilen)
- ✅ Daha hızlı yüklenir
- ✅ CPU ve GPU'da çalışır
- ✅ llama-cpp-python ile optimize edilmiştir

## İndirme Süreci

### 1. Dosya Kontrolü
```python
# Dosya zaten var mı?
if os.path.exists(model_file):
    # Hash kontrolü yap
    if verify_sha256(model_file, expected_hash):
        print("✅ Dosya zaten mevcut ve güvenli")
        return
```

### 2. HuggingFace'den İndirme
```python
# HuggingFace'den indir
response = requests.get(download_url, stream=True)
# Progress bar göster
# Dosyayı kaydet
```

### 3. SHA256 Doğrulama
```python
# İndikten sonra hash kontrolü
downloaded_hash = calculate_sha256(model_file)
if downloaded_hash == expected_hash:
    print("✅ Dosya orijinal ve güvenli")
    start_mining()
else:
    print("❌ HATA: Dosya bozuk veya değiştirilmiş!")
    os.remove(model_file)
```

## Mining Engine: llama-cpp-python

### Kurulum
```bash
pip install llama-cpp-python
```

### Kullanım
```python
from llama_cpp import Llama

# Model yükle
llm = Llama(
    model_path="./model.gguf",
    n_gpu_layers=-1,  # Tüm işi GPU'ya yık
    n_ctx=2048  # Hafıza boyutu
)

# Inference
output = llm("Kullanıcının sorduğu soru...", max_tokens=100)
```

## Sistem Gereksinimleri (Güncellenmiş)

| Bileşen | Eski | Yeni |
|---------|------|------|
| **RAM** | 32 GB | 8 GB (minimum), 16 GB (önerilen) |
| **Disk** | 50 GB | 10 GB (minimum), 20 GB (önerilen) |
| **GPU** | RTX 3090 | GTX 1660+ (CPU mode da desteklenir) |

## Launcher Entegrasyonu

Launcher artık:
1. ✅ Genesis'ten model config'i okur
2. ✅ HuggingFace'den model indirir
3. ✅ SHA256 hash kontrolü yapar
4. ✅ IPFS'ye fallback yapar (gerekirse)
5. ✅ Model durumunu gösterir

## Mining Engine Entegrasyonu

Mining engine:
1. ✅ GGUF dosyasını arar (`~/.r3mes/models/*.gguf`)
2. ✅ llama-cpp-python ile yükler
3. ✅ GPU'da inference yapar
4. ✅ PyTorch'a fallback yapar (gerekirse)

## Güvenlik

- ✅ SHA256 hash kontrolü zorunlu
- ✅ Dosya bozuksa otomatik silinir
- ✅ Genesis'teki hash değiştirilemez (blockchain güvenliği)
- ✅ IPFS fallback sadece HuggingFace başarısız olursa kullanılır

## Geçiş Rehberi

### Mevcut Kullanıcılar İçin

1. **Model Dosyasını İndir**
   ```bash
   # HuggingFace'den manuel indirme (opsiyonel)
   wget https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
   ```

2. **SHA256 Hash Al**
   ```bash
   sha256sum Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
   ```

3. **Genesis'e Ekle**
   - Hash'i genesis.json'a ekle
   - Model config'i güncelle

4. **Launcher'ı Güncelle**
   - Yeni launcher versiyonunu indir
   - Model otomatik indirilecek

## Teknik Detaylar

### Model.proto Güncellemeleri

```protobuf
message ModelConfig {
  // ... existing fields ...
  
  // Hybrid Download Configuration
  string model_name = 10;
  string file_name = 11;
  string download_source = 12;
  string verification_sha256 = 13;
  double required_disk_space_gb = 14;
  string ipfs_fallback_hash = 15;
}
```

### Launcher: model_downloader.rs

Yeni `ModelDownloader` modülü:
- HuggingFace'den indirme
- SHA256 doğrulama
- Progress tracking
- Disk space kontrolü

### Mining Engine: gguf_loader.py

Yeni `GGUFModelLoader` modülü:
- llama-cpp-python entegrasyonu
- GPU/CPU otomatik algılama
- Context window yönetimi

## Sorun Giderme

### Model İndirme Başarısız

1. İnternet bağlantısını kontrol et
2. Disk alanını kontrol et
3. HuggingFace erişimini kontrol et
4. IPFS fallback'i dene

### Hash Doğrulama Başarısız

1. Dosyayı tekrar indir
2. SHA256 hash'i kontrol et
3. Genesis'teki hash'i doğrula

### llama-cpp-python Kurulumu

```bash
# CUDA desteği ile
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# CPU-only
pip install llama-cpp-python
```

## Kaynaklar

- [llama-cpp-python Documentation](https://llama-cpp-python.readthedocs.io/)
- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [HuggingFace Model Hub](https://huggingface.co/models)


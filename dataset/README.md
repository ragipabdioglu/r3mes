# R3MES Dataset Directory

Bu klasör R3MES AI training için kullanılan dataset'leri içerir.

## Dosyalar

- **haberler.csv**: Türkçe haber dataset'i (41,847 satır)
  - Sütunlar: `haber` (metin), `kategori` (kategori bilgisi)
  
- **haberler.jsonl**: BitNet training için JSONL formatında dönüştürülmüş dataset
  - Format: Her satır bir JSON objesi
  - Örnek: `{"text": "...", "category": "..."}`

## JSONL Formatı

JSONL (JSON Lines) formatı, her satırın bağımsız bir JSON objesi olduğu bir formatdır. BitNet ve diğer LLM modelleri için standart training formatıdır.

### Örnek Format

```json
{"text": "luis enrique sampiyonlar ligini kizi icin kazandi", "category": "futbol"}
{"text": "3 uluslararasi aile sempozyumu aile kurumuna kuresel tehditler masaya yatirildi", "category": "yasam"}
```

### Format Seçenekleri

1. **Text Format** (varsayılan):
   ```json
   {"text": "...", "category": "..."}
   ```
   - En yaygın format
   - LLM'ler için doğrudan kullanılabilir

2. **Instruction Format**:
   ```json
   {"instruction": "Bu haberin kategorisini belirle:", "input": "...", "output": "..."}
   ```
   - Instruction-following fine-tuning için

## Dönüştürme

CSV'yi JSONL'e dönüştürmek için:

```bash
cd ~/R3MES/dataset
python3 convert_csv_to_jsonl.py haberler.csv -o haberler.jsonl
```

### Seçenekler

```bash
# Sadece text (kategori olmadan)
python3 convert_csv_to_jsonl.py haberler.csv -o haberler.jsonl --no-category

# Instruction format
python3 convert_csv_to_jsonl.py haberler.csv -o haberler.jsonl --format instruction

# Özel sütun isimleri
python3 convert_csv_to_jsonl.py haberler.csv -o haberler.jsonl \
  --text-column haber --category-column kategori
```

## Dataset Kullanımı

### Shard-Based Distribution

R3MES sistemi dataset'i otomatik olarak shard'lara böler:

```
Dataset (41,847 örnek)
├─ Shard 0: Örnek 0, 100, 200, ... (~418 örnek)
├─ Shard 1: Örnek 1, 101, 201, ... (~418 örnek)
...
└─ Shard 99: Örnek 99, 199, 299, ... (~418 örnek)
```

Her miner kendi shard'ına düşen veriyi kullanır.

### IPFS'te Saklama

Dataset'ler IPFS'te saklanır ve governance ile onaylanır:

```bash
# Dataset'i IPFS'e yükle
ipfs add -r dataset/

# IPFS hash'i blockchain'e kaydet (governance ile)
```

## Kategoriler

Dataset'teki kategoriler:
- `futbol`
- `yasam`
- `gundem`
- `ekonomi`
- `basketbol`
- `dunya`
- `tenis`
- `magazin`
- `bilgi`
- `ic-haber`
- ... ve daha fazlası

## Notlar

- Dataset UTF-8 encoding kullanır
- Türkçe karakterler korunur
- Her satır bağımsız bir training örneğidir
- JSONL formatı streaming için optimize edilmiştir

---

**Son Güncelleme**: 2025-01-27


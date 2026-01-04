# R3MES AI Architecture: BitNet + DoRA + RAG

> **Version:** 3.4 (Integration Tests Tamamlandı)
> **Son Güncelleme:** Ocak 2026
> **Implementation Status:** 14/14 temel bileşen tamamlandı (215 test geçti)

## Genel Bakış

R3MES, üç temel teknoloji üzerine kurulu merkezi olmayan bir AI sistemidir:

1. **BitNet** - 1.58-bit quantized base model (frozen backbone)
2. **DoRA** - Weight-Decomposed Low-Rank Adaptation (trainable experts)
3. **RAG** - Retrieval-Augmented Generation (güncel bilgi erişimi)

## Tamamlanan Bileşenler

| # | Bileşen | Dosya | Test Sayısı | Durum |
|---|---------|-------|-------------|-------|
| 1 | BitLinear | `core/bitlinear.py` | - | ✅ |
| 2 | DoRA Layer | `core/dora.py` | 19 | ✅ |
| 3 | Inference Backend | `core/inference_backend.py` | 15 | ✅ |
| 4 | PyTorch Backend | `core/backends/pytorch_backend.py` | - | ✅ |
| 5 | Tiered Cache | `cache/tiered_cache.py` | 19 | ✅ |
| 6 | VRAM Manager | `cache/vram_manager.py` | - | ✅ |
| 7 | Keyword Router | `router/keyword_router.py` | 22 | ✅ |
| 8 | Semantic Router | `router/semantic_router.py` | 19 | ✅ |
| 9 | Hybrid Router | `router/hybrid_router.py` | 19 | ✅ |
| 10 | VRAM Adaptive Gating | `router/vram_adaptive_gating.py` | - | ✅ |
| 11 | FAISS Store | `rag/faiss_store.py` | 17 | ✅ |
| 12 | RAG Embedder | `rag/embedder.py` | 15 | ✅ |
| 13 | RAG Retriever | `rag/retriever.py` | 17 | ✅ |
| 14 | **Inference Pipeline** | `r3mes/serving/inference_pipeline.py` | 27 | ✅ |
| 15 | **Integration Tests** | `tests/test_integration_pipeline.py` | 26 | ✅ **YENİ** |

**Toplam: 215 test geçti ✅**

---

## Tasarım Kararları (Design Decisions)

Bu bölüm, mimari tartışmalar sonucu alınan kritik kararları içerir.

### Karar 1: Hybrid Router Strategy (Keyword + Semantic + VRAM-Adaptive)

**Problem:** 
- Sadece keyword router: Hızlı ama edge case'lerde zayıf
- Sadece semantic router: Doğru ama yavaş (~10-20ms)
- Multi-adapter inference VRAM'i şişirebilir

**Çözüm:** 4 aşamalı Hybrid Router Pipeline:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HYBRID ROUTER PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT: User Query                                                          │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  STAGE 1: KEYWORD ROUTER (Fast Pre-filter)                          │   │
│  │  ├── Latency: <1ms                                                  │   │
│  │  ├── Method: Regex patterns, domain/language/task detection         │   │
│  │  ├── Output: candidate_experts + confidence                         │   │
│  │  │                                                                   │   │
│  │  │  IF confidence >= 0.85 → SKIP Stage 2 (fast path) ────────────┐  │   │
│  │  │  ELSE → Continue to Stage 2                                   │  │   │
│  │  └──────────────────────────────────────────────────────────────────│   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                               │   │
│         ▼ (confidence < 0.85)                                           │   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  STAGE 2: SEMANTIC ROUTER (Deep Understanding)                      │   │
│  │  ├── Latency: ~10-15ms                                              │   │
│  │  ├── Model: all-MiniLM-L6-v2 (22M params, 384 dim)                  │   │
│  │  ├── Method: Query embedding → Cosine sim with expert embeddings    │   │
│  │  ├── Expert Embeddings: 15-20 representative queries per expert     │   │
│  │  └── Similarity: Max pooling over expert embeddings                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                               │   │
│         ▼                                                               │   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  STAGE 3: SCORE FUSION                                              │   │
│  │  ├── Keyword weight: 0.3                                            │   │
│  │  ├── Semantic weight: 0.7                                           │   │
│  │  │                                                                   │   │
│  │  │  final_score = 0.3 × keyword_score + 0.7 × semantic_score        │   │
│  │  │                                                                   │   │
│  │  │  Adaptive (optional):                                            │   │
│  │  │  - Keyword confidence yüksekse → keyword weight artır            │   │
│  │  │  - Ambiguous query → semantic weight artır                       │   │
│  │  └──────────────────────────────────────────────────────────────────│   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                               │   │
│         ▼                                                               │   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  STAGE 4: VRAM-ADAPTIVE GATING                                      │   │
│  │  ├── VRAM < 8GB  → Top-1 expert                                     │   │
│  │  ├── VRAM 8-16GB → Top-2 experts                                    │   │
│  │  ├── VRAM > 16GB → Top-3 experts                                    │   │
│  │  │                                                                   │   │
│  │  │  Fallback: general_dora if max_score < 0.5                       │   │
│  │  └──────────────────────────────────────────────────────────────────│   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  OUTPUT: [(expert_id, weight), ...]                                         │
│          Example: [("medical_dora", 0.6), ("turkish_dora", 0.4)]           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Neden Hybrid?**
| Yaklaşım | Latency | Accuracy | Edge Cases | Resource |
|----------|---------|----------|------------|----------|
| Keyword Only | <1ms | Düşük-Orta | Zayıf | Minimal |
| Semantic Only | ~15ms | Yüksek | İyi | Model yükleme |
| **Hybrid** | **~5-10ms** | **En Yüksek** | **Çok İyi** | **Orta** |

**Fast Path Optimization:**
- Keyword confidence >= 0.85 → Semantic router atlanır
- Örnek: "Python'da for loop nasıl yazılır?" → coding_dora (0.95) → Fast path
- Örnek: "Bu konuyu açıklar mısın?" → (0.3) → Semantic router çalışır

**Uygulama:**
```python
class HybridRouter:
    def __init__(self):
        self.keyword_router = KeywordRouter()
        self.semantic_router = SemanticRouter()
        self.gating = VRAMAdaptiveGating()
        
        # Weights
        self.keyword_weight = 0.3
        self.semantic_weight = 0.7
        self.fast_path_threshold = 0.85
    
    def route(self, query: str, vram_gb: float) -> List[Tuple[str, float]]:
        # Stage 1: Keyword Router
        keyword_results = self.keyword_router.route(query)
        max_keyword_conf = max((r.confidence for r in keyword_results), default=0)
        
        # Fast path: Skip semantic if keyword is confident enough
        if max_keyword_conf >= self.fast_path_threshold:
            scores = [(r.expert_id, r.confidence) for r in keyword_results]
        else:
            # Stage 2: Semantic Router
            semantic_results = self.semantic_router.route(query)
            
            # Stage 3: Score Fusion
            scores = self._fuse_scores(keyword_results, semantic_results)
        
        # Stage 4: VRAM-Adaptive Gating
        return self.gating.select(scores, vram_gb)
    
    def _fuse_scores(self, keyword, semantic) -> List[Tuple[str, float]]:
        combined = {}
        for r in keyword:
            combined[r.expert_id] = self.keyword_weight * r.confidence
        for r in semantic:
            if r.expert_id in combined:
                combined[r.expert_id] += self.semantic_weight * r.score
            else:
                combined[r.expert_id] = self.semantic_weight * r.score
        return sorted(combined.items(), key=lambda x: x[1], reverse=True)
```


---

### Karar 2: Tiered Caching - Cold Start Çözümü

**Problem:** Diskten DoRA adapter yüklemek (cold start) latency'yi artırır.

**Çözüm:** 3 katmanlı cache sistemi + predictive loading:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TIERED CACHING SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  TIER 1: VRAM (Hot Cache)                                           │   │
│  │  ├── Latency: 0ms (zaten yüklü)                                     │   │
│  │  ├── Capacity: 2-4 adapter (VRAM'e bağlı)                           │   │
│  │  ├── Contents: turkish_dora, general_dora (her zaman)               │   │
│  │  └── Policy: Startup'ta preload                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                             │                                               │
│                             ▼ (cache miss)                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  TIER 2: RAM (Warm Cache)                                           │   │
│  │  ├── Latency: ~5ms (memory copy)                                    │   │
│  │  ├── Capacity: 10-20 adapter                                        │   │
│  │  ├── Contents: Sık kullanılan (medical, coding, legal)              │   │
│  │  └── Policy: LRU eviction                                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                             │                                               │
│                             ▼ (cache miss)                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  TIER 3: DISK (Cold Cache)                                          │   │
│  │  ├── Latency: ~50-100ms (disk I/O)                                  │   │
│  │  ├── Capacity: Unlimited                                            │   │
│  │  ├── Contents: Nadir kullanılan (cobol_dora, sanskrit_dora)         │   │
│  │  └── Policy: IPFS'ten indir, local cache                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  PREDICTIVE LOADING:                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Router "medical" dedi → Inference başlarken arka planda:           │   │
│  │  asyncio.create_task(preload_to_vram("medical_dora"))               │   │
│  │                                                                      │   │
│  │  Sonraki sorgu muhtemelen aynı domain'den gelecek.                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Uygulama:**
```python
class TieredDoRACache:
    def __init__(self, vram_capacity_mb: int):
        self.vram_cache = {}      # Tier 1: GPU memory
        self.ram_cache = {}       # Tier 2: CPU memory  
        self.disk_cache_dir = ".r3mes/dora_cache"  # Tier 3
        
        # Startup preload
        self._preload_hot_adapters(["turkish_dora", "general_dora"])
    
    async def get_adapter(self, adapter_id: str) -> DoRAAdapter:
        # Tier 1: VRAM
        if adapter_id in self.vram_cache:
            return self.vram_cache[adapter_id]
        
        # Tier 2: RAM
        if adapter_id in self.ram_cache:
            adapter = self.ram_cache[adapter_id]
            await self._promote_to_vram(adapter_id, adapter)
            return adapter
        
        # Tier 3: Disk
        adapter = await self._load_from_disk(adapter_id)
        self.ram_cache[adapter_id] = adapter
        return adapter
    
    async def predictive_load(self, likely_adapters: List[str]):
        """Router sonucuna göre arka planda yükle."""
        for adapter_id in likely_adapters:
            if adapter_id not in self.vram_cache:
                asyncio.create_task(self._warm_up(adapter_id))
```

---

### Karar 3: Inference Backend Abstraction (Phased Approach)

**Problem:** BitNet 1.58-bit için native optimized inference yok.

**Çözüm:** Abstract backend interface + phased implementation:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INFERENCE ENGINE ABSTRACTION                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    InferenceBackend (Abstract)                       │   │
│  │                                                                      │   │
│  │  interface:                                                         │   │
│  │  ├── load_base_model(ipfs_hash) -> Model                            │   │
│  │  ├── load_dora_adapter(adapter_id) -> DoRAAdapter                   │   │
│  │  ├── inference(input, adapters, weights) -> Tensor                  │   │
│  │  ├── get_capabilities() -> {vram, speed, precision}                 │   │
│  │  └── supports_feature(feature) -> bool                              │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                             │                                               │
│         ┌───────────────────┼───────────────────┐                          │
│         │                   │                   │                          │
│         ▼                   ▼                   ▼                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
│  │  PyTorch    │    │  Triton     │    │  BitNet-cpp │                     │
│  │  Backend    │    │  Backend    │    │  Backend    │                     │
│  │             │    │             │    │             │                     │
│  │ ✅ Phase 1  │    │ 🔄 Phase 2  │    │ 📅 Phase 3  │                     │
│  │ (Şimdi)     │    │ (3-6 ay)    │    │ (6-12 ay)   │                     │
│  │             │    │             │    │             │                     │
│  │ Features:   │    │ Features:   │    │ Features:   │                     │
│  │ • Kolay dev │    │ • Custom    │    │ • Native    │                     │
│  │ • Debug     │    │   kernels   │    │   1.58-bit  │                     │
│  │ • Fallback  │    │ • 2-3x hız  │    │ • 5-10x hız │                     │
│  └─────────────┘    └─────────────┘    └─────────────┘                     │
│                                                                             │
│  AUTO-SELECTION:                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  def get_best_backend() -> InferenceBackend:                        │   │
│  │      if triton_available() and has_nvidia_gpu():                    │   │
│  │          return TritonBackend()                                     │   │
│  │      elif bitnet_cpp_available():                                   │   │
│  │          return BitNetCppBackend()                                  │   │
│  │      else:                                                          │   │
│  │          return PyTorchBackend()  # Always available                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Phase 1 (Şimdi):** PyTorch Backend
- Tüm mimariyi kur (DoRA, Router, RAG)
- End-to-end çalışan sistem
- Development ve testing için ideal

**Phase 2 (3-6 ay):** Triton Kernels
- BitLinear + DoRA için custom kernels
- 2-3x performans artışı
- PyTorch backend fallback olarak kalır

**Phase 3 (6-12 ay):** BitNet-cpp veya Custom C++
- Native 1.58-bit support
- 5-10x performans artışı
- Production-grade optimization


---

### Karar 4: Custom DoRA Layer (BitLinear Entegrasyonu)

**Problem:** PEFT kütüphanesi `nn.Linear` bekliyor, bizim `BitLinear` custom layer.

**Çözüm:** Custom DoRA layer yazacağız (PEFT wrapper yerine):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CUSTOM DoRA LAYER DESIGN                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DoRA Formula:                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │  output = W₀x + m * (V / ||V||) * x                                 │   │
│  │                                                                      │   │
│  │  Burada:                                                            │   │
│  │  • W₀ = BitLinear backbone (frozen, {-1, 0, +1})                    │   │
│  │  • m  = magnitude (learnable scalar per output dim)                 │   │
│  │  • V  = direction matrix = B @ A (low-rank)                         │   │
│  │  • ||V|| = column-wise L2 norm                                      │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Class Structure:                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  class BitLinearDoRA(nn.Module):                                    │   │
│  │      def __init__(self, bitlinear: BitLinear, rank: int = 16):      │   │
│  │          self.backbone = bitlinear           # Frozen                │   │
│  │          self.backbone.requires_grad_(False)                        │   │
│  │                                                                      │   │
│  │          # DoRA components (trainable)                              │   │
│  │          self.magnitude = nn.Parameter(                             │   │
│  │              torch.ones(bitlinear.out_features)                     │   │
│  │          )                                                          │   │
│  │          self.direction_A = nn.Parameter(                           │   │
│  │              torch.randn(rank, bitlinear.in_features) * 0.01        │   │
│  │          )                                                          │   │
│  │          self.direction_B = nn.Parameter(                           │   │
│  │              torch.zeros(bitlinear.out_features, rank)              │   │
│  │          )                                                          │   │
│  │                                                                      │   │
│  │      def forward(self, x: Tensor) -> Tensor:                        │   │
│  │          # Backbone (frozen BitNet)                                 │   │
│  │          backbone_out = self.backbone(x)                            │   │
│  │                                                                      │   │
│  │          # Direction: V = B @ A                                     │   │
│  │          V = self.direction_B @ self.direction_A  # [out, in]       │   │
│  │                                                                      │   │
│  │          # Normalize direction (column-wise)                        │   │
│  │          V_norm = V / (V.norm(dim=1, keepdim=True) + 1e-8)          │   │
│  │                                                                      │   │
│  │          # DoRA output: m * normalized_direction * x                │   │
│  │          dora_out = self.magnitude.unsqueeze(0) * F.linear(x, V_norm)│   │
│  │                                                                      │   │
│  │          return backbone_out + dora_out                             │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Neden PEFT Wrapper Değil?                                                 │
│  ├── Tam kontrol: BitLinear'ın quantized weights'i ile uyumlu             │
│  ├── Performans: Gereksiz abstraction yok                                  │
│  ├── Debug: Kolay inspect ve modify                                        │
│  └── Future-proof: Triton kernels için hazır                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Trainable vs Frozen Parameters:**
```python
# Model parametreleri
Total Parameters:
├── BitLinear backbone: ~1B params (FROZEN, {-1,0,+1})
└── DoRA adapters: ~10-50M params (TRAINABLE)
    ├── magnitude: out_features per layer
    ├── direction_A: rank × in_features per layer
    └── direction_B: out_features × rank per layer

# Örnek (rank=16, hidden=4096):
# Per layer: 4096 + (16×4096) + (4096×16) = 4096 + 65536 + 65536 = 135,168 params
# 32 layer: 32 × 135,168 = 4.3M trainable params
```

---

### Karar 5: Hybrid RAG Architecture (Merkezi + Yerel)

**Problem:** RAG tamamen merkezi mi yoksa dağıtık mı olmalı?

**Çözüm:** Hibrit mimari - Serving Node merkezi, Miner Node opsiyonel yerel:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       HYBRID RAG ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    SERVING NODE (Merkezi)                            │   │
│  │                                                                      │   │
│  │  Responsibilities:                                                  │   │
│  │  ├── Global indices (news, general, medical, code)                  │   │
│  │  ├── Sık güncellenen veriler (news: her 6 saat)                     │   │
│  │  ├── API endpoint sağlar (/api/v1/rag/search)                       │   │
│  │  └── Index versioning ve sync                                       │   │
│  │                                                                      │   │
│  │  Vector Store: FAISS (tercih edilen)                                │   │
│  │  ├── Hızlı (C++ backend)                                            │   │
│  │  ├── Az dependency                                                  │   │
│  │  ├── IPFS'e serialize edilebilir                                    │   │
│  │  └── GPU acceleration (IVF-PQ)                                      │   │
│  │                                                                      │   │
│  │  Storage:                                                           │   │
│  │  ├── Index files: Local SSD                                         │   │
│  │  ├── Documents: IPFS (content-addressed)                            │   │
│  │  └── Embeddings: Numpy arrays (memory-mapped)                       │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                             │                                               │
│                             │ API calls                                     │
│                             ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    MINER NODE (Yerel - Opsiyonel)                    │   │
│  │                                                                      │   │
│  │  Use Cases:                                                         │   │
│  │  ├── Kendi training data'sı için mini index                         │   │
│  │  ├── Offline çalışabilme (internet kesintisi)                       │   │
│  │  ├── Privacy-sensitive data (şirket içi dokümanlar)                 │   │
│  │  └── Low-latency local retrieval                                    │   │
│  │                                                                      │   │
│  │  Sync Strategy:                                                     │   │
│  │  ├── Serving Node'dan index snapshot indir (IPFS)                   │   │
│  │  ├── Delta updates (sadece değişenler)                              │   │
│  │  └── Offline mode: Local cache kullan                               │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                             │                                               │
│                             │ IPFS sync                                     │
│                             ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    IPFS (Dağıtık Storage)                            │   │
│  │                                                                      │   │
│  │  Stored Data:                                                       │   │
│  │  ├── Index snapshots (weekly)                                       │   │
│  │  ├── Document chunks (content-addressed)                            │   │
│  │  ├── Embedding cache (per document)                                 │   │
│  │  └── Index metadata (version, stats)                                │   │
│  │                                                                      │   │
│  │  Benefits:                                                          │   │
│  │  ├── Decentralized: Tek nokta arızası yok                           │   │
│  │  ├── Content-addressed: Aynı doküman = aynı hash                    │   │
│  │  ├── Cacheable: CDN-like distribution                               │   │
│  │  └── Verifiable: Hash ile integrity check                           │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Neden FAISS (ChromaDB değil)?**
| Özellik | FAISS | ChromaDB |
|---------|-------|----------|
| Hız | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Memory efficiency | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| GPU support | ✅ Native | ❌ |
| Serialize to IPFS | ✅ Kolay | ⚠️ Zor |
| Dependencies | Minimal | SQLite, etc. |
| Production-ready | ✅ Meta kullanıyor | ⚠️ Daha yeni |


---

### Karar 6: Semantic Router - Embedding Model ve Expert Embeddings

**Problem:** Semantic router için hangi embedding model kullanılmalı ve expert embeddings nasıl oluşturulmalı?

**Çözüm:** Hafif multilingual model + pre-computed expert embeddings:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SEMANTIC ROUTER ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  EMBEDDING MODEL SEÇİMİ:                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ✅ SELECTED: all-MiniLM-L6-v2                                      │   │
│  │  ├── Size: 22M params, ~80MB                                        │   │
│  │  ├── Dimension: 384                                                 │   │
│  │  ├── Speed: ~5ms/query (CPU), ~1ms (GPU)                            │   │
│  │  ├── Quality: Good for short queries                                │   │
│  │  └── Multilingual: Temel seviye (İngilizce ağırlıklı)               │   │
│  │                                                                      │   │
│  │  Alternatifler (gerekirse):                                         │   │
│  │  ├── multilingual-e5-small: Daha iyi Türkçe, 118M params            │   │
│  │  └── all-mpnet-base-v2: Daha yüksek kalite, 110M params             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  EXPERT EMBEDDING STRATEGY:                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Her expert için 15-20 representative query:                        │   │
│  │                                                                      │   │
│  │  medical_dora:                                                      │   │
│  │  ├── "hastalık belirtileri nelerdir"                                │   │
│  │  ├── "bu ilacın yan etkileri var mı"                                │   │
│  │  ├── "tedavi seçenekleri nelerdir"                                  │   │
│  │  ├── "what are the symptoms of diabetes"                            │   │
│  │  ├── "how to treat high blood pressure"                             │   │
│  │  └── ... (15-20 total)                                              │   │
│  │                                                                      │   │
│  │  coding_dora:                                                       │   │
│  │  ├── "Python'da liste nasıl sıralanır"                              │   │
│  │  ├── "JavaScript async await kullanımı"                             │   │
│  │  ├── "how to implement binary search"                               │   │
│  │  ├── "fix null pointer exception"                                   │   │
│  │  └── ... (15-20 total)                                              │   │
│  │                                                                      │   │
│  │  turkish_dora:                                                      │   │
│  │  ├── "Türkçe dilbilgisi kuralları"                                  │   │
│  │  ├── "Türk kültürü ve gelenekleri"                                  │   │
│  │  ├── "İstanbul'un tarihi yerleri"                                   │   │
│  │  └── ... (15-20 total)                                              │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  SIMILARITY CALCULATION:                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Method: Max Pooling (Simple & Effective)                           │   │
│  │                                                                      │   │
│  │  score(query, expert) = max(                                        │   │
│  │      cosine_sim(query_embed, expert_embed_i)                        │   │
│  │      for expert_embed_i in expert_embeddings                        │   │
│  │  )                                                                  │   │
│  │                                                                      │   │
│  │  Neden Max Pooling?                                                 │   │
│  │  ├── En yakın örneği yakalar                                        │   │
│  │  ├── Outlier'lara dayanıklı                                         │   │
│  │  └── Hesaplama basit                                                │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  STORAGE & CACHING:                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Expert Embeddings:                                                 │   │
│  │  ├── Pre-computed at build time                                     │   │
│  │  ├── Stored as .npy files (~50KB per expert)                        │   │
│  │  ├── Loaded to RAM at startup                                       │   │
│  │  └── Version controlled with expert adapters                        │   │
│  │                                                                      │   │
│  │  Embedding Model:                                                   │   │
│  │  ├── Loaded once at startup (~80MB)                                 │   │
│  │  ├── Cached in RAM                                                  │   │
│  │  └── Optional: GPU acceleration                                     │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Uygulama:**
```python
class SemanticRouter:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.expert_embeddings = self._load_expert_embeddings()
    
    def _load_expert_embeddings(self) -> Dict[str, np.ndarray]:
        """Load pre-computed expert embeddings."""
        embeddings = {}
        for expert_id in EXPERT_IDS:
            path = f"router/expert_embeddings/{expert_id}.npy"
            embeddings[expert_id] = np.load(path)
        return embeddings
    
    def route(self, query: str) -> List[ExpertScore]:
        # Embed query
        query_embed = self.model.encode(query, normalize_embeddings=True)
        
        # Calculate similarity with each expert
        scores = []
        for expert_id, expert_embeds in self.expert_embeddings.items():
            # Max pooling over expert embeddings
            similarities = np.dot(expert_embeds, query_embed)
            max_sim = float(np.max(similarities))
            scores.append(ExpertScore(expert_id, max_sim, "semantic"))
        
        return sorted(scores, key=lambda x: x.score, reverse=True)
```

**Expert Embedding Generation Script:**
```python
# scripts/generate_expert_embeddings.py
def generate_expert_embeddings():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    expert_queries = {
        "medical_dora": [
            "hastalık belirtileri nelerdir",
            "bu ilacın yan etkileri",
            "tedavi seçenekleri",
            # ... 15-20 queries
        ],
        "coding_dora": [
            "Python'da liste nasıl sıralanır",
            "JavaScript async await",
            "how to implement binary search",
            # ... 15-20 queries
        ],
        # ... other experts
    }
    
    for expert_id, queries in expert_queries.items():
        embeddings = model.encode(queries, normalize_embeddings=True)
        np.save(f"router/expert_embeddings/{expert_id}.npy", embeddings)
```

---

## Ana Mimari

### 1. BitNet Base Model

BitNet, Microsoft Research tarafından geliştirilen 1.58-bit quantized model mimarisidir.
Ağırlıklar sadece {-1, 0, +1} değerlerini alır.

**Avantajları:**
- 10x daha az bellek (FP16'ya göre)
- Hızlı inference (çarpma yerine toplama/çıkarma)
- Enerji verimli (mobil/edge cihazlarda çalışabilir)
- Deterministic (blockchain verification için kritik)

**R3MES'te Kullanımı:**
```
BitNet Base Model (Frozen)
├── Tüm kullanıcılarda aynı
├── IPFS'te saklanır (tek hash)
├── Güncellenmez (immutable)
└── Sadece inference için kullanılır
```

---

### 2. DoRA Expert System

```
┌─────────────────────────────────────────────────────────────────┐
│                      DoRA EXPERT REGISTRY                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  DOMAIN EXPERTS (Uzmanlık Alanları)                            │
│  ├── medical_dora      - Tıp, sağlık, hastalıklar              │
│  ├── legal_dora        - Hukuk, kanunlar, davalar              │
│  ├── coding_dora       - Programlama, debugging                │
│  ├── finance_dora      - Finans, yatırım, ekonomi              │
│  ├── science_dora      - Bilim, fizik, kimya, biyoloji         │
│  ├── history_dora      - Tarih, olaylar, kişiler               │
│  └── education_dora    - Eğitim, öğretim, pedagoji             │
│                                                                 │
│  LANGUAGE EXPERTS (Dil Adaptörleri)                            │
│  ├── turkish_dora      - Türkçe dil ve kültür                  │
│  ├── german_dora       - Almanca                               │
│  ├── french_dora       - Fransızca                             │
│  ├── spanish_dora      - İspanyolca                            │
│  ├── arabic_dora       - Arapça                                │
│  └── chinese_dora      - Çince                                 │
│                                                                 │
│  TASK EXPERTS (Görev Adaptörleri)                              │
│  ├── summarization_dora - Özetleme                             │
│  ├── translation_dora   - Çeviri                               │
│  ├── qa_dora           - Soru-Cevap                            │
│  ├── creative_dora     - Yaratıcı yazarlık                     │
│  └── analysis_dora     - Analiz ve değerlendirme               │
│                                                                 │
│  GENERAL (Fallback - sadece gerektiğinde)                      │
│  └── general_dora      - Genel amaçlı                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### 3. DoRA Router System (Hybrid)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      HYBRID DoRA ROUTER SYSTEM                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT: User Query                                                          │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    STAGE 1: KEYWORD ROUTER                           │   │
│  │                                                                      │   │
│  │  ├── Regex-based pattern matching                                   │   │
│  │  ├── Domain detection: "hastalık" → medical_dora                    │   │
│  │  ├── Language detection: "merhaba" → turkish_dora                   │   │
│  │  ├── Task detection: "özetle" → summarization_dora                  │   │
│  │  └── Latency: <1ms                                                  │   │
│  │                                                                      │   │
│  │  Output: [(expert_id, confidence), ...]                             │   │
│  │                                                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │  IF max_confidence >= 0.85:                                  │    │   │
│  │  │      → FAST PATH: Skip semantic router                       │    │   │
│  │  │  ELSE:                                                       │    │   │
│  │  │      → Continue to Stage 2                                   │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼ (confidence < 0.85)                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    STAGE 2: SEMANTIC ROUTER                          │   │
│  │                                                                      │   │
│  │  ├── Model: all-MiniLM-L6-v2 (22M params)                           │   │
│  │  ├── Query → 384-dim embedding                                      │   │
│  │  ├── Cosine similarity with expert embeddings                       │   │
│  │  ├── Max pooling over 15-20 representative queries per expert       │   │
│  │  └── Latency: ~10-15ms                                              │   │
│  │                                                                      │   │
│  │  Output: [(expert_id, similarity_score), ...]                       │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    STAGE 3: SCORE FUSION                             │   │
│  │                                                                      │   │
│  │  final_score = 0.3 × keyword_score + 0.7 × semantic_score           │   │
│  │                                                                      │   │
│  │  Sorted by final_score descending                                   │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    STAGE 4: VRAM-ADAPTIVE GATING                     │   │
│  │                                                                      │   │
│  │  VRAM < 8GB  → Top-1 (tek expert)                                   │   │
│  │  VRAM 8-16GB → Top-2 (2 expert)                                     │   │
│  │  VRAM > 16GB → Top-3 (max 3 expert)                                 │   │
│  │                                                                      │   │
│  │  Fallback: general_dora if max_score < 0.5                          │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  OUTPUT: Selected DoRA Expert(s) + Weights                                  │
│          [("medical_dora", 0.65), ("turkish_dora", 0.35)]                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Latency Breakdown:**
| Senaryo | Keyword | Semantic | Fusion | Gating | Total |
|---------|---------|----------|--------|--------|-------|
| Fast Path (conf >= 0.85) | <1ms | SKIP | - | <1ms | **~1-2ms** |
| Full Pipeline | <1ms | ~10ms | <1ms | <1ms | **~12-15ms** |

---

### 4. RAG Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            RAG PIPELINE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DOCUMENT INGESTION:                                                       │
│  ├── Sources: Web, API feeds, User uploads, IPFS                           │
│  ├── Processing: Extract → Chunk (512 tokens) → Embed                      │
│  └── Storage: FAISS index + IPFS documents                                 │
│                                                                             │
│  VECTOR STORE (FAISS):                                                     │
│  ├── general_index    - Genel bilgi                                        │
│  ├── news_index       - Güncel haberler (TTL: 7 gün)                       │
│  ├── medical_index    - Tıbbi bilgiler                                     │
│  ├── legal_index      - Hukuki dokümanlar                                  │
│  ├── code_index       - Kod örnekleri, dokümantasyon                       │
│  └── user_index       - Kullanıcı özel dokümanları                         │
│                                                                             │
│  RETRIEVAL:                                                                │
│  ├── Dense retrieval (embedding similarity)                                │
│  ├── Sparse retrieval (BM25)                                               │
│  ├── Hybrid (RRF - Reciprocal Rank Fusion)                                 │
│  └── Re-ranking (cross-encoder, optional)                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```


---

### 5. Full Inference Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SERVING NODE PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  User Request                                                               │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. QUERY ANALYSIS                                                    │   │
│  │    ├── Tokenization                                                 │   │
│  │    ├── Language detection                                           │   │
│  │    ├── Intent classification                                        │   │
│  │    └── Query embedding                                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 2. PARALLEL PROCESSING                                               │   │
│  │                                                                      │   │
│  │    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │   │
│  │    │ DoRA Router  │    │ RAG Retrieval│    │ Cache Check  │         │   │
│  │    │              │    │              │    │              │         │   │
│  │    │ Select       │    │ Fetch        │    │ Preload      │         │   │
│  │    │ experts      │    │ documents    │    │ adapters     │         │   │
│  │    │ (Top-K)      │    │ (Top-5)      │    │ (async)      │         │   │
│  │    └──────┬───────┘    └──────┬───────┘    └──────┬───────┘         │   │
│  │           │                   │                   │                  │   │
│  │           └───────────────────┼───────────────────┘                  │   │
│  │                               ▼                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 3. INFERENCE (Backend Abstraction)                                   │   │
│  │                                                                      │   │
│  │    Input: [System Prompt] + [RAG Context] + [User Query]            │   │
│  │                                                                      │   │
│  │    ┌─────────────────────────────────────────────────────────────┐  │   │
│  │    │  Backend.inference(                                          │  │   │
│  │    │      input_ids,                                              │  │   │
│  │    │      adapters=["medical_dora", "turkish_dora"],              │  │   │
│  │    │      weights=[0.85, 0.72]                                    │  │   │
│  │    │  )                                                           │  │   │
│  │    │                                                              │  │   │
│  │    │  Internally:                                                 │  │   │
│  │    │  output = bitnet(x)                                         │  │   │
│  │    │         + 0.85 × medical_dora(x)                            │  │   │
│  │    │         + 0.72 × turkish_dora(x)                            │  │   │
│  │    └─────────────────────────────────────────────────────────────┘  │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 4. POST-PROCESSING                                                   │   │
│  │    ├── Token decoding                                               │   │
│  │    ├── Safety filtering                                             │   │
│  │    ├── Citation injection (RAG sources)                             │   │
│  │    └── Response formatting                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼                                                                     │
│  Response to User                                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Dosya Yapısı (Güncellenmiş)

```
miner-engine/
├── core/
│   ├── bitlinear.py              # ✅ Mevcut - BitNet layer
│   ├── dora.py                   # ✅ TAMAMLANDI - Custom DoRA layer
│   ├── inference_backend.py      # ✅ TAMAMLANDI - Backend abstraction
│   ├── backends/
│   │   ├── __init__.py           # ✅ TAMAMLANDI
│   │   ├── pytorch_backend.py    # ✅ TAMAMLANDI - Phase 1 backend
│   │   ├── triton_backend.py     # 📅 Phase 2 backend (placeholder)
│   │   └── bitnet_cpp_backend.py # 📅 Phase 3 backend (placeholder)
│   └── trainer.py                # ✅ Mevcut - DoRA training eklenecek
│
├── cache/
│   ├── __init__.py               # ✅ TAMAMLANDI
│   ├── tiered_cache.py           # ✅ TAMAMLANDI - 3-tier caching
│   ├── vram_manager.py           # ✅ TAMAMLANDI - VRAM allocation
│   └── predictive_loader.py      # 📅 Async preloading (MVP'de kapalı)
│
├── router/
│   ├── __init__.py               # ✅ TAMAMLANDI
│   ├── keyword_router.py         # ✅ TAMAMLANDI - Rule-based routing
│   ├── semantic_router.py        # ✅ TAMAMLANDI - Embedding-based routing
│   ├── hybrid_router.py          # ✅ TAMAMLANDI - Orchestrator
│   ├── vram_adaptive_gating.py   # ✅ TAMAMLANDI - VRAM-based Top-K
│   └── expert_embeddings/        # ✅ TAMAMLANDI - Pre-computed embeddings
│       ├── medical_dora.npy
│       ├── coding_dora.npy
│       ├── turkish_dora.npy
│       └── ...
│
├── rag/
│   ├── __init__.py               # ✅ TAMAMLANDI
│   ├── faiss_store.py            # ✅ TAMAMLANDI - FAISS wrapper
│   ├── embedder.py               # ✅ TAMAMLANDI - Document/query embedding
│   ├── retriever.py              # ✅ TAMAMLANDI - Hybrid retrieval
│   ├── reranker.py               # 📅 Cross-encoder reranking
│   ├── document_processor.py     # 🆕 Chunking, extraction
│   └── index_manager.py          # 🆕 Index lifecycle management
│
├── r3mes/
│   ├── miner/
│   │   ├── engine.py             # ✅ Mevcut - DoRA training eklenecek
│   │   ├── lora_manager.py       # ✅ Mevcut - DoRA manager olacak
│   │   └── dora_trainer.py       # 🆕 DoRA-specific training
│   │
│   ├── serving/
│   │   ├── __init__.py           # ✅ TAMAMLANDI - Export'lar
│   │   ├── engine.py             # ✅ Mevcut - Blockchain entegrasyonu
│   │   └── inference_pipeline.py # ✅ TAMAMLANDI - Full inference pipeline
│   │
│   └── proposer/
│       ├── aggregator.py         # ✅ Mevcut
│       └── dora_aggregator.py    # 🆕 DoRA-specific aggregation
│
└── tests/
    ├── test_dora.py              # ✅ TAMAMLANDI - 19 test
    ├── test_inference_backend.py # ✅ TAMAMLANDI - 15 test
    ├── test_cache.py             # ✅ TAMAMLANDI - 19 test
    ├── test_router.py            # ✅ TAMAMLANDI - 22 test
    ├── test_semantic_router.py   # ✅ TAMAMLANDI - 19 test
    ├── test_hybrid_router.py     # ✅ TAMAMLANDI - 19 test
    ├── test_rag.py               # ✅ TAMAMLANDI - 17 test
    ├── test_rag_embedder.py      # ✅ TAMAMLANDI - 15 test
    ├── test_rag_retriever.py     # ✅ TAMAMLANDI - 17 test
    ├── test_inference_pipeline.py # ✅ TAMAMLANDI - 27 test
    └── test_integration_pipeline.py # ✅ TAMAMLANDI - 26 test (YENİ)
```

**Test Durumu:** 215 test geçti ✅

---

## Konfigürasyon

```yaml
# config/dora_config.yaml

# DoRA Settings
dora:
  default_rank: 16
  default_alpha: 32
  
# Hybrid Router Settings (Karar 1 + Karar 6)
router:
  strategy: "hybrid"  # keyword, semantic, hybrid
  
  # Keyword Router
  keyword:
    confidence_threshold: 0.3
    
  # Semantic Router
  semantic:
    model: "all-MiniLM-L6-v2"  # 22M params, 384 dim
    device: "cpu"  # cpu, cuda
    embeddings_dir: "router/expert_embeddings"
    
  # Hybrid Fusion
  hybrid:
    keyword_weight: 0.3
    semantic_weight: 0.7
    fast_path_threshold: 0.85  # Skip semantic if keyword conf >= this
    
  # VRAM-Adaptive Gating
  gating:
    vram_8gb_max_experts: 1
    vram_16gb_max_experts: 2
    vram_24gb_max_experts: 3
    general_fallback_threshold: 0.5

# Cache Settings (Karar 2)
cache:
  tier1_vram:
    preload: ["turkish_dora", "general_dora"]
    max_adapters: 4
  tier2_ram:
    max_adapters: 20
    eviction_policy: "lru"
  tier3_disk:
    cache_dir: ".r3mes/dora_cache"
    max_size_gb: 10
  predictive_loading: false  # MVP'de kapalı

# Backend Settings (Karar 3)
inference:
  backend: "auto"  # auto, pytorch, triton, bitnet_cpp
  fallback_backend: "pytorch"

# RAG Settings (Karar 5)
rag:
  vector_store: "faiss"
  embedding_model: "all-MiniLM-L6-v2"
  chunk_size: 512
  chunk_overlap: 50
  top_k: 5
  hybrid_search: true
  rerank: false  # MVP'de kapalı
  
  # Serving Node (merkezi)
  serving_node:
    indices: ["general", "news", "medical", "code"]
    update_interval_hours: 6
  
  # Miner Node (yerel, opsiyonel)
  miner_node:
    enabled: false
    sync_from_serving: true
    local_indices: ["user"]
```

---

## Implementation Roadmap

### Tamamlanan (✅)
| # | Bileşen | Dosya | Durum | Test |
|---|---------|-------|-------|------|
| 1 | DoRA Layer | `core/dora.py` | ✅ Tamamlandı | 19 test |
| 2 | Backend Abstraction | `core/inference_backend.py` | ✅ Tamamlandı | 15 test |
| 3 | PyTorch Backend | `core/backends/pytorch_backend.py` | ✅ Tamamlandı | - |
| 4 | Tiered Cache | `cache/tiered_cache.py` | ✅ Tamamlandı | 19 test |
| 5 | VRAM Manager | `cache/vram_manager.py` | ✅ Tamamlandı | - |
| 6 | Keyword Router | `router/keyword_router.py` | ✅ Tamamlandı | 22 test |
| 7 | Semantic Router | `router/semantic_router.py` | ✅ Tamamlandı | 19 test |
| 8 | Hybrid Router | `router/hybrid_router.py` | ✅ Tamamlandı | 19 test |
| 9 | VRAM-Adaptive Gating | `router/vram_adaptive_gating.py` | ✅ Tamamlandı | - |
| 10 | FAISS Store | `rag/faiss_store.py` | ✅ Tamamlandı | 17 test |
| 11 | RAG Embedder | `rag/embedder.py` | ✅ Tamamlandı | 15 test |
| 12 | RAG Retriever | `rag/retriever.py` | ✅ Tamamlandı | 17 test |
| 13 | **Inference Pipeline** | `r3mes/serving/inference_pipeline.py` | ✅ Tamamlandı | 27 test |
| 14 | **Integration Tests** | `tests/test_integration_pipeline.py` | ✅ Tamamlandı | 26 test |

**Toplam: 215 test geçti ✅**

### Sıradaki (🆕)
| # | Bileşen | Dosya | Tahmini Süre | Bağımlılık |
|---|---------|-------|--------------|------------|
| 15 | End-to-End Tests | `tests/test_e2e.py` | 3-4 saat | Gerçek model |
| 16 | Performance Benchmarks | `benchmarks/` | 2-3 saat | Pipeline |
| 17 | Production Deployment | ServingEngine entegrasyonu | 4-5 saat | Pipeline |

### Gelecek (📅)
| # | Bileşen | Dosya | Hedef |
|---|---------|-------|-------|
| 18 | Triton Backend | `core/backends/triton_backend.py` | Phase 2 (3-6 ay) |
| 19 | BitNet-cpp Backend | `core/backends/bitnet_cpp_backend.py` | Phase 3 (6-12 ay) |
| 20 | Predictive Loader | `cache/predictive_loader.py` | Post-MVP |
| 21 | Cross-encoder Reranker | `rag/reranker.py` | Post-MVP |

**Toplam İlerleme:** 14/14 temel bileşen tamamlandı (100%)

---

## Örnek Kullanım Senaryoları

### Senaryo 1: Tıbbi Soru (Türkçe, 8GB VRAM)
```
User: "Diyabet hastalarında insülin direnci nasıl tedavi edilir?"

1. Router:
   - Keyword: "diyabet", "insülin", "tedavi" → medical_dora (0.9)
   - Language: Türkçe → turkish_dora (0.8)
   - VRAM: 8GB → Top-1 Gating → medical_dora seçilir

2. Cache:
   - medical_dora Tier 2'de (RAM) → Tier 1'e (VRAM) promote
   - Predictive: turkish_dora arka planda yüklenir

3. RAG:
   - medical_index'ten ilgili dokümanlar
   - "İnsülin direnci tedavisinde metformin..."

4. Inference:
   - BitNet + medical_dora(1.0)
   - RAG context ile zenginleştirilmiş cevap
```

### Senaryo 2: Kod Sorusu (İngilizce, 24GB VRAM)
```
User: "How to implement a binary search tree in Python?"

1. Router:
   - Keyword: "implement", "Python" → coding_dora (0.95)
   - Language: English → (no language adapter)
   - VRAM: 24GB → Top-3 Gating → coding_dora + general_dora

2. Cache:
   - coding_dora Tier 1'de (VRAM) → Hemen kullan
   - general_dora Tier 1'de (VRAM) → Hemen kullan

3. RAG:
   - code_index'ten Python BST örnekleri

4. Inference:
   - BitNet + coding_dora(0.95) + general_dora(0.3)
   - Kod örnekleri ile cevap
```


---

## Inference Pipeline (YENİ - v3.3)

### Pipeline Mimarisi

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      INFERENCE PIPELINE (v3.3)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  User Query                                                                 │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 1: RAG CONTEXT RETRIEVAL                                       │   │
│  │                                                                      │   │
│  │  ├── Query embedding (all-MiniLM-L6-v2)                             │   │
│  │  ├── FAISS similarity search (top-k=3)                              │   │
│  │  ├── Context augmentation                                           │   │
│  │  │                                                                   │   │
│  │  │  Template:                                                        │   │
│  │  │  "Context:\n{retrieved_docs}\n\nQuery: {user_query}"             │   │
│  │  │                                                                   │   │
│  │  └── Latency: ~10-20ms                                              │   │
│  │                                                                      │   │
│  │  Skip if: enable_rag=False or skip_rag=True                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 2: EXPERT ROUTING (HybridRouter)                               │   │
│  │                                                                      │   │
│  │  ├── Keyword Router (<1ms)                                          │   │
│  │  │   └── IF confidence >= 0.85 → FAST PATH                          │   │
│  │  │                                                                   │   │
│  │  ├── Semantic Router (~10ms) - if not fast path                     │   │
│  │  │                                                                   │   │
│  │  ├── Score Fusion                                                   │   │
│  │  │   └── 0.3 × keyword + 0.7 × semantic                             │   │
│  │  │                                                                   │   │
│  │  └── VRAM-Adaptive Gating                                           │   │
│  │      ├── <8GB: Top-1                                                │   │
│  │      ├── 8-16GB: Top-2                                              │   │
│  │      └── >16GB: Top-3                                               │   │
│  │                                                                      │   │
│  │  Output: [(expert_id, weight), ...]                                 │   │
│  │                                                                      │   │
│  │  Skip if: force_experts provided                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 3: ADAPTER LOADING (TieredCache)                               │   │
│  │                                                                      │   │
│  │  For each selected expert:                                          │   │
│  │  ├── Check VRAM cache (Tier 1) → 0ms                                │   │
│  │  ├── Check RAM cache (Tier 2) → ~5ms                                │   │
│  │  ├── Load from Disk (Tier 3) → ~50-100ms                            │   │
│  │  └── Promote to higher tier if space available                      │   │
│  │                                                                      │   │
│  │  Metrics: cache_hits, cache_misses                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 4: INFERENCE EXECUTION (InferenceBackend)                      │   │
│  │                                                                      │   │
│  │  Backend.inference(                                                 │   │
│  │      input_ids,                                                     │   │
│  │      adapter_ids=["medical_dora", "turkish_dora"],                  │   │
│  │      adapter_weights=[0.65, 0.35]                                   │   │
│  │  )                                                                  │   │
│  │                                                                      │   │
│  │  Internally:                                                        │   │
│  │  output = bitnet(x)                                                 │   │
│  │         + 0.65 × medical_dora(x)                                    │   │
│  │         + 0.35 × turkish_dora(x)                                    │   │
│  │                                                                      │   │
│  │  Backend Priority: BitNet-cpp > Triton > PyTorch                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼                                                                     │
│  OUTPUT: PipelineResult                                                     │
│  ├── output: torch.Tensor                                                  │
│  ├── metrics: PipelineMetrics                                              │
│  ├── experts_used: [(expert_id, weight), ...]                              │
│  ├── rag_context: Optional[str]                                            │
│  └── success: bool                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Pipeline Kullanımı

```python
from r3mes.serving import (
    InferencePipeline,
    PipelineConfig,
    create_pipeline,
)

# 1. Basit kullanım
pipeline = create_pipeline(enable_rag=True)
await pipeline.initialize()
await pipeline.load_model("path/to/model")

result = await pipeline.run("Diyabet tedavisi hakkında bilgi ver")
print(result.output)
print(result.experts_used)  # [("medical_dora", 0.8), ("turkish_dora", 0.2)]
print(result.metrics.total_time_ms)  # ~50ms

# 2. Özelleştirilmiş konfigürasyon
config = PipelineConfig(
    enable_rag=True,
    rag_top_k=5,
    router_strategy="hybrid",
    keyword_weight=0.3,
    semantic_weight=0.7,
    fast_path_threshold=0.85,
    vram_capacity_mb=4096,
)

pipeline = InferencePipeline(config=config)
await pipeline.initialize()

# 3. RAG dokümanları ekleme
pipeline.add_rag_document(
    doc_id="med_001",
    content="Diyabet tedavisinde metformin ilk tercih ilaçtır...",
    metadata={"domain": "medical", "source": "guidelines"}
)

# 4. Batch inference
queries = ["Soru 1", "Soru 2", "Soru 3"]
results = await pipeline.run_batch(queries)

# 5. Streaming inference
async for token in pipeline.run_streaming("Uzun bir cevap ver"):
    print(token, end="", flush=True)

# 6. Force specific experts
result = await pipeline.run(
    "Test query",
    force_experts=["coding_dora", "general_dora"]
)

# 7. İstatistikler
stats = pipeline.get_stats()
print(stats["router"]["fast_path_rate"])  # 0.65
print(stats["cache"]["vram"]["utilization"])  # 0.75
```

### Pipeline Metrikleri

```python
@dataclass
class PipelineMetrics:
    # Timing
    total_time_ms: float      # Toplam süre
    rag_time_ms: float        # RAG retrieval süresi
    routing_time_ms: float    # Router süresi
    loading_time_ms: float    # Adapter yükleme süresi
    inference_time_ms: float  # Inference süresi
    
    # RAG
    rag_docs_retrieved: int   # Bulunan doküman sayısı
    rag_context_length: int   # Context karakter sayısı
    
    # Routing
    used_fast_path: bool      # Fast path kullanıldı mı
    keyword_confidence: float # Keyword router güveni
    
    # Cache
    adapters_loaded: List[str]  # Yüklenen adapter'lar
    cache_hits: int           # Cache hit sayısı
    cache_misses: int         # Cache miss sayısı
    
    # Inference
    tokens_generated: int     # Üretilen token sayısı
    backend_used: str         # Kullanılan backend
```

### Dosya Konumu

```
miner-engine/r3mes/serving/
├── __init__.py              # Export'lar
├── engine.py                # Mevcut ServingEngine (blockchain entegrasyonu)
└── inference_pipeline.py    # YENİ - Full inference pipeline
```

---

## Sonraki Adımlar

### Tamamlanan (v3.4)
- ✅ Inference Pipeline (`r3mes/serving/inference_pipeline.py`) - 27 test
- ✅ Integration Tests (`tests/test_integration_pipeline.py`) - 26 test
- ✅ Toplam 215 test geçti
- ✅ **14/14 temel bileşen tamamlandı (100%)**

### Sıradaki
1. **End-to-End Tests** - Gerçek model ile test
2. **Performance Benchmarks** - Latency ve throughput ölçümü
3. **Production Deployment** - ServingEngine entegrasyonu

---

*Son güncelleme: Ocak 2026 - v3.4 (Integration Tests)*

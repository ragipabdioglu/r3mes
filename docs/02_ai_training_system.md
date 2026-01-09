# R3MES AI Training System Detaylı Döküman

## Multi-GPU Support

### Overview

R3MES miner engine supports multi-GPU training for increased throughput and efficiency. The system uses PyTorch's DataParallel and DistributedDataParallel for parallel training across multiple GPUs.

### Implementation

**Multi-GPU Trainer** (`core/multi_gpu_trainer.py`):
- Extends `LoRATrainer` with multi-GPU capabilities
- Supports DataParallel (single-node, multi-GPU)
- Supports DistributedDataParallel (multi-node, multi-GPU)
- Automatic GPU detection and device selection
- GPU utilization monitoring

**Usage**:
```python
from core.multi_gpu_trainer import create_multi_gpu_trainer

# Use all available GPUs
trainer = create_multi_gpu_trainer(model, use_all_gpus=True)

# Use specific GPUs
trainer = create_multi_gpu_trainer(model, device_ids=[0, 1, 2])

# Use DistributedDataParallel
trainer = create_multi_gpu_trainer(model, use_ddp=True)
```

**Features**:
- Automatic batch splitting across GPUs
- Gradient synchronization (DDP)
- GPU utilization monitoring
- Fallback to single-GPU if insufficient GPUs available

### Performance Benefits

- **Throughput**: Linear scaling with number of GPUs (up to ~90% efficiency)
- **Memory**: Distributed memory across GPUs
- **Training Speed**: Faster gradient computation with parallel processing

---

# R3MES AI Training System Detaylı Döküman

## Genel Bakış

R3MES, BitNet 1.58-bit model architecture ile LoRA (Low-Rank Adaptation) tabanlı distributed training sistemi kullanır. Bu sistem 99.6%+ bandwidth reduction sağlar.

## BitNet 1.58-bit Architecture

### Core Concept
BitNet, neural network weights'leri {-1, 0, +1} değerlerine quantize eder, extreme efficiency sağlar.

### LoRA Integration (DEFAULT MECHANISM)
```python
class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, lora_rank=8):
        super().__init__()
        
        # Frozen backbone weights (quantized {-1, 0, +1} - NEVER updated)
        self.register_buffer('frozen_weight', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.ones(out_features))
        
        # LoRA adapters (trainable, small rank matrices)
        self.lora_A = nn.Parameter(torch.zeros(lora_rank, in_features, dtype=torch.float16))
        self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank, dtype=torch.float16))
        self.lora_alpha = nn.Parameter(torch.tensor(float(lora_rank), dtype=torch.float16))
        
    def forward(self, x):
        # Frozen backbone computation
        backbone_output = self.deterministic_bitwise_matmul(x, self.frozen_weight)
        
        # LoRA adaptation: (alpha/rank) * x @ A.T @ B.T
        lora_scaling = self.lora_alpha / self.lora_rank
        lora_output = torch.matmul(torch.matmul(x, self.lora_A.t()), self.lora_B.t()) * lora_scaling
        
        return backbone_output + lora_output
```

### Bandwidth Optimization

#### Traditional vs LoRA Approach
- **Traditional**: Full model weights (28GB+) transfer
- **LoRA**: Only adapter weights (10-100MB) transfer
- **Reduction**: 99.6%+ bandwidth savings

#### Training Workflow
1. **Initial Setup**: Miners download frozen backbone once (28GB)
2. **Training**: Only LoRA adapters trained locally
3. **Sharing**: Only adapter weights shared (MB scale)
4. **Aggregation**: Adapter aggregation (MB scale operations)

## Distributed Training Architecture

### Training Coordinator
```python
class OffChainDistributedCoordinator:
    def __init__(self, global_model, aggregation_threshold=0.6):
        self.global_model = global_model
        self.pending_gradient_hashes = []  # Only hashes
        self.ipfs_client = IPFSClient()
        self.proposer_rotation = ProposerRotation()
        
    def aggregate_gradients_multiproposer(self, gradient_hashes, round_id):
        # Multi-proposer system with commit-reveal
        proposers = self.proposer_rotation.select_proposers(round_id, count=3)
        
        # Each proposer downloads from IPFS for aggregation
        commitments = []
        for proposer in proposers:
            gradient_data_list = []
            for gradient_hash in gradient_hashes:
                # PASSIVE RETRIEVAL: Download from IPFS only when needed
                gradient_data = self.ipfs_client.retrieve_gradients(gradient_hash)
                gradient_data_list.append(gradient_data)
            
            # Compute aggregation
            aggregation = self.compute_aggregation(gradient_data_list, proposer.seed)
            commitment = self.commit_aggregation(aggregation, proposer.nonce)
            commitments.append(commitment)
        
        # Reveal phase and robust combination
        revealed_aggregations = []
        for i, proposer in enumerate(proposers):
            if self.verify_commitment(commitments[i], proposer):
                revealed_aggregations.append(proposer.aggregation)
        
        # Filter trap jobs and use median/trimmed mean
        filtered_aggregations = self.filter_trap_jobs(revealed_aggregations)
        final_aggregation = self.robust_combine(filtered_aggregations)
        
        return final_aggregation
```

### Layer-Based Sharding (Subnet Architecture)

#### Problem Solution
- **Problem**: Full model downloads (28GB+) create network congestion
- **Solution**: Divide network into specialized subnets training layer subsets

#### Subnet Configuration
```go
type SubnetConfig struct {
    SubnetID        uint64     `json:"subnet_id"`
    LayerRange      LayerRange `json:"layer_range"`      // layers 0-10, 11-20, etc.
    AssignedMiners  []string   `json:"assigned_miners"`
    ActivationHash  string     `json:"activation_hash"`  // IPFS hash
    NextSubnetID    uint64     `json:"next_subnet_id"`
    Status          string     `json:"status"`
}

type LayerRange struct {
    StartLayer uint64 `json:"start_layer"`
    EndLayer   uint64 `json:"end_layer"`
}
```

#### Workflow
1. **Subnet 1**: Trains layers 0-10, produces activations, stores on IPFS
2. **Subnet 1**: Submits activation hash to blockchain (only hash)
3. **Subnet 2**: Downloads activations from IPFS, trains layers 11-20
4. **Process continues** through all subnets sequentially
5. **Final subnet**: Produces complete model update

#### Benefits
- 90% bandwidth reduction per miner
- Parallel training across subnets
- Scalable to thousands of miners
- IPFS handles data distribution

### Window-Based Async Rollup

#### Problem Solution
- **Problem**: Cosmos blocks every 5 seconds, AI training takes minutes
- **Solution**: Asynchronous gradient submission with lazy aggregation

#### Training Window Structure
```go
type TrainingWindow struct {
    WindowID        uint64    `json:"window_id"`
    StartHeight     int64     `json:"start_height"`
    EndHeight       int64     `json:"end_height"`        // +100 blocks (~8-10 min)
    Status          string    `json:"status"`            // "collecting", "aggregating", "finalized"
    GradientHashes  []string  `json:"gradient_hashes"`   // IPFS hashes
    AggregatorNode  string    `json:"aggregator_node"`   // Assigned proposer
    AggregationHash string    `json:"aggregation_hash"`  // Final result
    GlobalSeed      []byte    `json:"global_seed"`       // Deterministic seed
}
```

#### Workflow
1. **Window Opens** (Block N): Training window starts
2. **Async Submission** (Blocks N to N+100):
   - Python miners upload gradients DIRECTLY to IPFS
   - Python miners send ONLY IPFS hash via gRPC to Go node
   - Go node stores only hash + metadata on-chain (KB scale)
   - No model update yet (non-blocking)
3. **Window Closes** (Block N+100): Window boundary reached
4. **Lazy Aggregation**: Aggregator downloads from IPFS, aggregates off-chain
5. **Model Update**: Single aggregation transaction updates global model

## Deterministic Training

### Execution Environment
```go
type ExecutionEnvironment struct {
    DockerImage     string            `json:"docker_image"`      // r3mes/training:cuda12.1-pytorch2.1.0
    RequiredVersion string            `json:"required_version"`  // Locked version hash
    GPUDrivers     map[string]string `json:"gpu_drivers"`       // NVIDIA: 525.60.11
    PythonPackages map[string]string `json:"python_packages"`   // torch==2.1.0
    Checksum       []byte            `json:"checksum"`          // Environment integrity
}
```

### Seed Synchronization
```go
type GlobalSeedLock struct {
    WindowID        uint64   `json:"window_id"`
    BlockHash       []byte   `json:"block_hash"`         // Block hash at window start
    DerivedSeed     uint64   `json:"derived_seed"`       // Deterministic seed
    PyTorchSeed     uint64   `json:"pytorch_seed"`       // torch.manual_seed(seed)
    NumPySeed       uint64   `json:"numpy_seed"`         // np.random.seed(seed)
    CUDASeed        uint64   `json:"cuda_seed"`          // torch.cuda.manual_seed_all(seed)
}
```

### GPU Architecture Handling
```python
def detect_gpu_architecture():
    if not torch.cuda.is_available():
        return {"architecture": "cpu", "compute_capability": "0.0"}
    
    device = torch.cuda.current_device()
    compute_capability = torch.cuda.get_device_capability(device)
    
    arch_map = {
        (6, 0): "pascal",
        (7, 0): "volta", 
        (7, 5): "turing",
        (8, 0): "ampere",
        (8, 6): "ampere",
        (8, 9): "ada",
        (9, 0): "blackwell",
    }
    
    architecture = arch_map.get(compute_capability, "unknown")
    return {"architecture": architecture, "compute_capability": f"{compute_capability[0]}.{compute_capability[1]}"}
```

## Data Flow Architecture

### Python Miner Side (Active IPFS Upload)
```python
class PythonMinerEngine:
    def submit_gradient(self, gradient_data, metadata):
        # Step 1: Detect GPU architecture
        gpu_arch = self.detect_gpu_architecture()
        
        # Step 2: Upload gradient DIRECTLY to IPFS
        ipfs_hash = self.ipfs_client.add_gradients(gradient_data)
        
        # Step 3: Generate proof-of-work
        proof_of_work = self.generate_proof_of_work(gradient_data, metadata)
        
        # Step 4: Create message with authentication
        msg = MsgSubmitGradient(
            miner=self.miner_address,
            ipfs_hash=ipfs_hash,  # Only hash, not data
            metadata=gradient_metadata,
            proof_of_work=proof_of_work,
            nonce=self.nonce,
            timestamp=int(time.time())
        )
        
        # Step 5: Sign message (prevent bypass attacks)
        msg.signature = self.sign_message(msg, self.private_key)
        
        # Step 6: Send via gRPC (only hash + metadata, KB scale)
        response = self.grpc_client.SubmitGradient(msg, tls_cert=self.tls_client_cert)
        return response
```

### Go Node Side (Passive Validation)
```go
func (k keeper) HandleMsgSubmitGradient(ctx sdk.Context, msg MsgSubmitGradient) (*MsgSubmitGradientResponse, error) {
    // Authentication checks
    if err := k.VerifyMessageSignature(msg, msg.Miner); err != nil {
        return nil, err
    }
    
    // Staking requirement
    if err := k.CheckStakingRequirement(ctx, msg.Miner); err != nil {
        return nil, err
    }
    
    // Rate limiting
    if err := k.CheckRateLimit(ctx, msg.Miner); err != nil {
        return nil, err
    }
    
    // IPFS hash validation
    if err := k.ValidateIPFSHash(msg.IPFSHash); err != nil {
        return nil, err
    }
    
    // Store only hash + metadata on-chain (KB scale)
    storedGradient := StoredGradient{
        Miner:           msg.Miner,
        IPFSHash:        msg.IPFSHash,
        Metadata:        msg.Metadata,
        BlockHeight:     ctx.BlockHeight(),
        GPUArchitecture: msg.Metadata.GPUArchitecture,
    }
    
    k.SetStoredGradient(ctx, msg.IPFSHash, storedGradient)
    
    return &MsgSubmitGradientResponse{
        IPFSHash:    msg.IPFSHash,
        BlockHeight: ctx.BlockHeight(),
    }, nil
}
```

## Model Versioning

### Version Registry
```go
type ModelVersion struct {
    VersionNumber      uint64    `json:"version_number"`     // 1, 2, 3
    ModelHash          string    `json:"model_hash"`         // IPFS hash
    IPFSPath           string    `json:"ipfs_path"`          // v1/, v2/
    Architecture       string    `json:"architecture"`       // bitnet-v1, bitnet-v2
    Status             string    `json:"status"`             // active, deprecated, migration
    ActivationHeight   int64     `json:"activation_height"`
    MigrationWindowStart int64   `json:"migration_window_start"`
    MigrationWindowEnd   int64   `json:"migration_window_end"`
}
```

### Upgrade Process
1. **Governance Proposal**: New model version proposed
2. **Voting**: Token holders vote on upgrade
3. **Migration Window**: Dual model support (old + new)
4. **Deprecation**: Old version deprecated after grace period

## Performance Metrics

### Bandwidth Comparison
| Approach | Initial Download | Regular Updates | Total Bandwidth |
|----------|------------------|-----------------|-----------------|
| Traditional | 28GB | 28GB per epoch | 28GB × epochs |
| LoRA | 28GB (one-time) | 10-100MB per epoch | 28GB + (100MB × epochs) |
| **Savings** | Same | **99.6% reduction** | **~95% total reduction** |

### Training Efficiency
- **Convergence**: Maintained with LoRA adapters
- **Quality**: Comparable to full fine-tuning
- **Speed**: Faster due to smaller parameter updates
- **Memory**: Reduced memory requirements

## Adaptive VRAM Scaling (Oto-VRAM Ölçekleme)

### Problem Statement
GTX 1650 (4GB) ile RTX 4090 (24GB) aynı kodu çalıştıramaz. Sistem kullanıcının kartını tahmin etmeyecek, ona göre şekil alacak.

### Production Solution: Dynamic Profile System

Miner başlatılırken `torch.cuda.get_device_properties` ile VRAM miktarını okuyan bir fonksiyon yazılır ve otomatik profil uygulanır.

#### VRAM-Based Profile Configuration

```python
# miner-engine/r3mes/miner/vram_profiler.py
import torch
from transformers import PagedAdamW8bit

def detect_vram_profile():
    """Detect GPU VRAM and return appropriate training profile"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    
    device = torch.cuda.current_device()
    vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    
    if vram_gb < 6:
        # Entry Level (GTX 1650, RTX 3050, etc.)
        return {
            "batch_size": 1,
            "gradient_accumulation": 32,
            "optimizer": "PagedAdamW8bit",  # RAM spillover allowed
            "mixed_precision": True,
            "max_memory": {0: f"{int(vram_gb * 0.9)}GB"}  # 90% VRAM usage
        }
    elif vram_gb < 12:
        # Mid Range (RTX 3060, RTX 3070, etc.)
        return {
            "batch_size": 4,
            "gradient_accumulation": 8,
            "optimizer": "AdamW",
            "mixed_precision": True,
            "max_memory": {0: f"{int(vram_gb * 0.85)}GB"}  # 85% VRAM usage
        }
    else:
        # High End (RTX 3090, RTX 4090, etc.)
        return {
            "batch_size": 16,
            "gradient_accumulation": 1,
            "optimizer": "AdamW",
            "mixed_precision": False,  # Full precision for high-end cards
            "max_memory": {0: f"{int(vram_gb * 0.8)}GB"}  # 80% VRAM usage
        }

def apply_profile(model, optimizer, dataloader, profile):
    """Apply detected profile to training configuration"""
    # Update batch size
    dataloader.batch_size = profile["batch_size"]
    
    # Update optimizer
    if profile["optimizer"] == "PagedAdamW8bit":
        from transformers import PagedAdamW8bit
        optimizer = PagedAdamW8bit(
            model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    # Update gradient accumulation
    gradient_accumulation_steps = profile["gradient_accumulation"]
    
    return model, optimizer, dataloader, gradient_accumulation_steps
```

#### Automatic Profile Application

```python
# miner-engine/r3mes/miner/engine.py
class MinerEngine:
    def __init__(self, private_key, blockchain_url, chain_id, ...):
        # Detect VRAM and apply profile automatically
        self.vram_profile = detect_vram_profile()
        
        # Model initialization with profile
        self.model = self._initialize_model_with_profile()
        
        # Optimizer with profile
        self.optimizer = self._initialize_optimizer_with_profile()
        
        # Gradient accumulation from profile
        self.gradient_accumulation_steps = self.vram_profile["gradient_accumulation"]
        
        # Log profile to user
        print(f"✅ Detected VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
        print(f"✅ Applied Profile: batch_size={self.vram_profile['batch_size']}, "
              f"gradient_accumulation={self.vram_profile['gradient_accumulation']}")
    
    def _initialize_model_with_profile(self):
        """Initialize model with VRAM-aware configuration"""
        model = SimpleBitNetModel(...)
        
        # Apply memory optimization for low VRAM
        if self.vram_profile["batch_size"] == 1:
            # Enable gradient checkpointing for memory efficiency
            model.gradient_checkpointing_enable()
        
        return model
```

**Key Features**:
- **Zero Configuration**: Kullanıcıya sorulmaz, otomatik uygulanır
- **VRAM Detection**: `torch.cuda.get_device_properties` ile gerçek VRAM miktarı okunur
- **Profile Lock**: Profil seçildikten sonra değiştirilemez (deterministic training için)
- **PagedAdamW Support**: Düşük VRAM kartlar için RAM spillover izni verilir

## LoRA-Enforced Architecture (Zorunlu LoRA)

### Problem Statement
Yanlışlıkla Full-Finetuning açılırsa sistem çöker. GTX 1650'nin hayatta kalma garantisi için model mimarisini kod seviyesinde kilitlemek gerekir.

### Production Solution: Code-Level Architecture Lock

Model yükleme fonksiyonunda PEFT kütüphanesini zorunlu kıl, ana modeli dondur, sadece LoRA adaptörlerini trainable yap.

#### Enforced LoRA Implementation

```python
# miner-engine/r3mes/miner/model_loader.py
from peft import LoraConfig, get_peft_model, TaskType
import torch.nn as nn

def load_model_with_enforced_lora(model_path, lora_rank=8, lora_alpha=16):
    """
    Load model with MANDATORY LoRA architecture.
    Full fine-tuning is BLOCKED at code level.
    """
    # Load base model
    base_model = SimpleBitNetModel.from_pretrained(model_path)
    
    # MANDATORY: Freeze all base model parameters
    for param in base_model.parameters():
        param.requires_grad = False
    
    # MANDATORY: Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Attention layers
        lora_dropout=0.1,
        bias="none"
    )
    
    # Apply LoRA to model
    model = get_peft_model(base_model, lora_config)
    
    # VERIFY: Ensure no base parameters are trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    if trainable_params > (total_params * 0.1):  # More than 10% trainable = error
        raise RuntimeError(
            "LoRA enforcement failed: Too many trainable parameters detected. "
            "Full fine-tuning is not supported on this hardware architecture."
        )
    
    print(f"✅ LoRA Enforced: {trainable_params:,} trainable / {total_params:,} total parameters")
    return model

def validate_lora_only_training(model):
    """Validate that only LoRA parameters are trainable"""
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "lora" not in name.lower():
                raise RuntimeError(
                    f"Non-LoRA parameter '{name}' is trainable. "
                    "This hardware architecture does not support full fine-tuning."
                )
    return True
```

#### Configuration Blocking

```python
# miner-engine/r3mes/miner/config.py
class TrainingConfig:
    def __init__(self, config_dict):
        # Check if full fine-tuning is attempted
        if config_dict.get("full_finetune", False):
            raise ValueError(
                "Full fine-tuning is not supported. "
                "This hardware architecture only supports LoRA training. "
                "Please remove 'full_finetune' from config."
            )
        
        # Force LoRA mode
        self.lora_enabled = True
        self.lora_rank = config_dict.get("lora_rank", 8)
        self.lora_alpha = config_dict.get("lora_alpha", 16)
```

**Key Features**:
- **Code-Level Enforcement**: Full fine-tuning config'de olsa bile engellenir
- **Automatic Freezing**: Base model parametreleri otomatik dondurulur
- **Validation**: Training başlamadan önce LoRA-only training doğrulanır
- **Error Messages**: Kullanıcıya açık hata mesajları gösterilir

## Fixed Chunk / Variable Speed (Sabit Veri, Değişken Hız)

### Problem Statement
Zayıf kartlara küçük veri yollarsak doğrulama (validation) karmaşıklaşır. Veri bütünlüğünü korurken hızı serbest bırakmak gerekir.

### Production Solution: Standard Chunk Size with Local Micro-Batching

Veri dağıtım protokolünde 'Chunk Size'ı sabit bir değere (örneğin 2048 token) kilitle. Madencilerin bu paketi bölmesine izin verme. Ancak madenci tarafında (local), bu büyük paketi VRAM'e sığdırmak için 'Micro-Batching' yapmasına izin ver.

#### Fixed Chunk Protocol

```python
# miner-engine/r3mes/miner/data_protocol.py
CHUNK_SIZE_TOKENS = 2048  # Fixed chunk size (standardized across all miners)

class FixedChunkDataLoader:
    """
    Fixed chunk size data loader with local micro-batching.
    
    Protocol:
    - Server sends fixed 2048-token chunks (never split)
    - Miner receives full chunk
    - Miner can micro-batch locally for VRAM efficiency
    - Gradient computed on full chunk (for validation)
    """
    
    def __init__(self, chunk_data, local_batch_size=1):
        self.chunk_data = chunk_data  # Full 2048-token chunk
        self.chunk_size = len(chunk_data)
        self.local_batch_size = local_batch_size  # Local micro-batch size
        
        # Verify chunk size matches protocol
        if self.chunk_size != CHUNK_SIZE_TOKENS:
            raise ValueError(
                f"Invalid chunk size: {self.chunk_size} (expected {CHUNK_SIZE_TOKENS})"
            )
    
    def __iter__(self):
        """Iterate over chunk with local micro-batching"""
        # Split chunk into micro-batches for local processing
        for i in range(0, self.chunk_size, self.local_batch_size):
            micro_batch = self.chunk_data[i:i + self.local_batch_size]
            yield micro_batch
    
    def compute_full_chunk_gradient(self, model, loss_fn):
        """
        Compute gradient on full chunk (for validation).
        This ensures gradient is computed on complete chunk as required by protocol.
        """
        model.train()
        total_loss = 0.0
        
        # Process in micro-batches but accumulate gradients
        for micro_batch in self:
            output = model(micro_batch["input_ids"])
            loss = loss_fn(output, micro_batch["labels"])
            loss = loss / (self.chunk_size / self.local_batch_size)  # Scale for accumulation
            loss.backward()
            total_loss += loss.item()
        
        # Gradient is now computed on full chunk (via accumulation)
        return total_loss
```

#### Blockchain Protocol Enforcement

```go
// remes/x/remes/keeper/msg_server_submit_gradient.go
const FIXED_CHUNK_SIZE = 2048  // Fixed chunk size in tokens

func (k Keeper) ValidateChunkSize(chunkData []byte) error {
    tokenCount := countTokens(chunkData)
    
    if tokenCount != FIXED_CHUNK_SIZE {
        return sdkerrors.Wrapf(
            ErrInvalidChunkSize,
            "chunk size must be exactly %d tokens, got %d",
            FIXED_CHUNK_SIZE,
            tokenCount,
        )
    }
    
    return nil
}
```

**Key Features**:
- **Fixed Chunk Size**: Tüm minerlar aynı boyutta chunk alır (2048 token)
- **Local Micro-Batching**: Miner kendi VRAM'ine göre chunk'ı mikro batch'lere bölebilir
- **Full Chunk Gradient**: Gradient her zaman tam chunk üzerinden hesaplanır (validation için)
- **Protocol Compliance**: Blockchain tarafında chunk boyutu doğrulanır

## Miner Engine Implementation

### MinerEngine Class

```python
# miner-engine/r3mes/miner/engine.py
class MinerEngine:
    def __init__(self, private_key, blockchain_url, chain_id, ...):
        # VRAM profile detection (automatic)
        self.vram_profile = detect_vram_profile()
        
        # Model initialization with enforced LoRA
        self.model = load_model_with_enforced_lora(model_path, ...)
        
        # IPFS client initialization (embedded IPFS daemon)
        # Blockchain client initialization (gRPC)
        # Stats collector initialization
    
    def train_and_submit(self, num_iterations=1):
        # Training loop with VRAM-aware batching
        # Fixed chunk loading with local micro-batching
        # Gradient computation on full chunk
        # IPFS upload
        # Blockchain submission
        # Stats update
```

### Continuous Mining

```python
# miner-engine/r3mes/cli/continuous_mining.py
# Batch-based mining execution
# Stats HTTP server integration
# Graceful shutdown handling
```

**Özellikler**:
- Batch size: Configurable (default: 5 iterations)
- Pause between batches: Configurable (default: 2 seconds)
- Stats server: Otomatik başlatılır (port 8080)
- Graceful shutdown: Ctrl+C handling

### Stats Collection

```python
# miner-engine/r3mes/miner/stats_server.py
class StatsCollector:
    def get_gpu_stats(self):
        # GPU temperature, fan speed, VRAM usage
        # pynvml veya PyTorch fallback
    
    def get_training_metrics(self):
        # Loss, epoch, gradient norm
        # Training step count
    
    def calculate_hash_rate(self):
        # Gradients per hour
```

### Stats HTTP Server

```python
# miner-engine/r3mes/miner/stats_http_server.py
# HTTP server (port 8080)
# Endpoints:
#   - GET /stats: Miner statistics
#   - GET /health: Health check
```

**Integration**: Stats server, miner engine oluşturulduktan SONRA başlatılır (stats collector initialize edildikten sonra).

### Embedded IPFS Daemon

```python
# miner-engine/r3mes/utils/ipfs_manager.py
# Otomatik IPFS binary indirme (Windows/macOS/Linux)
# IPFS daemon otomatik başlatma
# Repository initialization
```

**Özellikler**:
- Platform-specific binary (v0.24.0)
- Otomatik daemon başlatma
- Port: 5001 (default)
- Kullanıcıya soru sorulmaz (otomatik)

Bu AI training system, R3MES'in core innovation'ını oluşturur ve massive scalability sağlar.
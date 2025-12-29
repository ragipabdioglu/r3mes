# R3MES Blockchain Infrastructure Detaylı Döküman

## Genel Bakış

R3MES, Cosmos SDK v0.50.x LTS tabanlı özel bir blockchain uygulamasıdır. Proof of Useful Work (PoUW) konsensüs mekanizması ile AI model eğitimini birleştirir.

## Teknik Stack

### Core Blockchain Components
- **Cosmos SDK**: v0.50.x LTS (Long Term Support)
- **CometBFT**: v0.38.27 (En kararlı versiyon)
- **Go Version**: 1.22 (Cosmos SDK v0.50.x ile tam uyumlu)
- **Binary Name**: `remesd`

### Blockchain Modülleri

#### 1. x/remes Modülü (Ana PoUW Modülü)
```go
// Ana modül yapısı
type Keeper struct {
    // Gradient submission ve aggregation koordinasyonu
    // Mining reward hesaplama ve dağıtım
    // IPFS hash doğrulama ve saklama
    // Merkle proof verification
}
```

**Temel Fonksiyonlar:**
- Gradient submission transaction validation
- IPFS hash storage (sadece hash, gradient data değil)
- Off-chain aggregation coordination
- Mining reward calculation ve distribution

#### 2. Standard Cosmos SDK Modülleri
- `x/bank`: Token transfer ve balance management
- `x/staking`: Validator staking ve consensus
- `x/gov`: Protocol governance ve parameter updates
- `x/slashing`: Malicious behavior penalties

### Protocol Buffer Schemas

#### Message Types
```protobuf
// Gradient submission (sadece hash + metadata)
message MsgSubmitGradient {
    string miner = 1;
    string ipfs_hash = 2;  // Python miner IPFS'e upload etti
    GradientMetadata metadata = 3;
    bytes proof_of_work = 4;
    uint64 nonce = 5;
    bytes signature = 6;
    int64 timestamp = 7;
    uint64 token_count = 17;  // MUST be exactly 2048 (Fixed Chunk / Variable Speed Protocol)
}
```

**TokenCount Validation**: The `token_count` field is mandatory and must be exactly 2048 tokens per chunk. This enforces the Fixed Chunk / Variable Speed Protocol where:
- Each chunk submitted must contain exactly 2048 tokens
- Validation occurs at the message handler level before processing
- Invalid token counts result in transaction rejection

// Off-chain aggregation results
message MsgSubmitAggregation {
    string proposer = 1;
    string aggregated_hash = 2;
    bytes merkle_root = 3;
    repeated string gradient_hashes = 4;
    uint64 participant_count = 5;
}

// Challenge/dispute mechanism
message MsgChallengeAggregation {
    string challenger = 1;
    string aggregation_hash = 2;
    repeated uint64 sample_indices = 3;
    cosmos.base.v1beta1.Coin deposit = 4;
}
```

#### Query Types
- `GetGradient`: Stored gradient bilgilerini getir
- `GetModelParams`: Global model state
- `GetAggregation`: Aggregation records
- `GetMinerScore`: Miner reputation ve performance
- `ListStoredGradient`: Gradient listesi

### Genesis State

```go
type GenesisState struct {
    // Initial model configuration (Hybrid Download: HuggingFace + IPFS fallback)
    ModelConfig ModelConfig `json:"model_config"`
    // DEPRECATED: Use ModelConfig instead
    ModelHash string `json:"model_hash"`
    ModelVersion string `json:"model_version"`
    
    // Initial network participants
    InitialParticipants []string `json:"initial_participants"`
    
    // Protocol parameters
    Params Params `json:"params"`
}

// ModelConfig contains hybrid download configuration
type ModelConfig struct {
    ModelName string `json:"model_name"`           // e.g., "Llama-3-8B-R3MES-Optimized"
    FileName string `json:"file_name"`             // e.g., "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
    DownloadSource string `json:"download_source"` // HuggingFace direct URL
    VerificationSHA256 string `json:"verification_sha256"` // SHA256 hash for integrity check
    RequiredDiskSpaceGB float64 `json:"required_disk_space_gb"` // Minimum disk space needed
    IPFSFallbackHash string `json:"ipfs_fallback_hash"` // IPFS hash as fallback
}
```

### Open Buffet / Task Pool Architecture (Asenkron İş Havuzu)

#### Problem Statement
Blok süresine bağlı kalmak güçlü kartları yavaşlatıyor, zayıf kartları sistemden atıyor. RTX 4090'ın potansiyelini serbest bırakmak için tekli görev sisteminden "Havuz Sistemi"ne geçiş gerekir.

#### Production Solution: Task Pool System

Blockchain/Keeper yapısını 'Task Pool' mantığına çevir. Her blokta 1 değil, örneğin 50 adet 'Chunk' (Veri Paketi) yayınla. Prefetching (Önceden Çekme) ile miner kodu, mevcut işi bitirmeden sıradaki işi hafızaya çekebilsin.

#### Task Pool Structure

```go
// remes/x/remes/types/task_pool.go
type TaskPool struct {
    PoolID        uint64    `json:"pool_id"`
    WindowID      uint64    `json:"window_id"`
    TotalChunks   uint64    `json:"total_chunks"`    // e.g., 50 chunks per block
    AvailableChunks []TaskChunk `json:"available_chunks"`
    CompletedChunks []string   `json:"completed_chunks"`  // IPFS hashes of completed tasks
    Status        string    `json:"status"`          // "active", "aggregating", "closed"
    CreatedHeight int64     `json:"created_height"`
    ExpiryHeight  int64     `json:"expiry_height"`   // Window closes after N blocks
}

type TaskChunk struct {
    ChunkID       uint64   `json:"chunk_id"`
    DataHash      string   `json:"data_hash"`       // IPFS hash of chunk data
    ShardID       uint64   `json:"shard_id"`
    AssignedMiner string   `json:"assigned_miner"`  // Optional: can be nil for open pool
    Status        string   `json:"status"`          // "available", "in_progress", "completed"
    ClaimedAt     int64    `json:"claimed_at"`
    CompletedAt   int64    `json:"completed_at"`
    GradientHash  string   `json:"gradient_hash"`   // IPFS hash of result
    // Internal fields (NOT exposed to miners via query responses):
    IsTrap        bool     `json:"is_trap"`         // Internal flag for trap jobs from genesis vault
    VaultEntryID  uint64   `json:"vault_entry_id"` // Reference to genesis vault entry (if trap)
}

// Task assignment message
type MsgClaimTask struct {
    Miner    string `json:"miner"`
    PoolID   uint64 `json:"pool_id"`
    ChunkID  uint64 `json:"chunk_id"`
    Nonce    uint64 `json:"nonce"`
}

// Task completion message
type MsgCompleteTask struct {
    Miner       string `json:"miner"`
    PoolID      uint64 `json:"pool_id"`
    ChunkID     uint64 `json:"chunk_id"`
    GradientHash string `json:"gradient_hash"`  // IPFS hash
    ProofOfWork  []byte `json:"proof_of_work"`
}
```

#### Task Pool Workflow

```go
// remes/x/remes/keeper/task_pool.go
func (k Keeper) CreateTaskPool(ctx sdk.Context, windowID uint64) TaskPool {
    // Create pool with 50 chunks per block
    chunksPerBlock := uint64(50)
    totalChunks := chunksPerBlock * 100  // 100 blocks = 5000 chunks
    
    pool := TaskPool{
        PoolID:        k.GetNextPoolID(ctx),
        WindowID:      windowID,
        TotalChunks:   totalChunks,
        AvailableChunks: make([]TaskChunk, 0, totalChunks),
        Status:        "active",
        CreatedHeight: ctx.BlockHeight(),
        ExpiryHeight:  ctx.BlockHeight() + 100,  // 100 blocks window
    }
    
    // Generate chunks with deterministic shard assignment
    for i := uint64(0); i < totalChunks; i++ {
        chunk := TaskChunk{
            ChunkID:      i,
            DataHash:     k.GenerateChunkDataHash(windowID, i),
            ShardID:      i % 100,  // 100 shards
            Status:       "available",
        }
        pool.AvailableChunks = append(pool.AvailableChunks, chunk)
    }
    
    k.SetTaskPool(ctx, pool)
    return pool
}

func (k Keeper) ClaimTask(ctx sdk.Context, miner string, poolID uint64, chunkID uint64) error {
    pool := k.GetTaskPool(ctx, poolID)
    
    // Find available chunk
    for i, chunk := range pool.AvailableChunks {
        if chunk.ChunkID == chunkID && chunk.Status == "available" {
            // Claim chunk
            pool.AvailableChunks[i].Status = "in_progress"
            pool.AvailableChunks[i].AssignedMiner = miner
            pool.AvailableChunks[i].ClaimedAt = ctx.BlockTime().Unix()
            
            k.SetTaskPool(ctx, pool)
            return nil
        }
    }
    
    return sdkerrors.Wrapf(ErrTaskNotAvailable, "chunk %d not available", chunkID)
}

func (k Keeper) CompleteTask(ctx sdk.Context, msg MsgCompleteTask) error {
    pool := k.GetTaskPool(ctx, msg.PoolID)
    
    // Find and complete chunk
    for i, chunk := range pool.AvailableChunks {
        if chunk.ChunkID == msg.ChunkID && chunk.AssignedMiner == msg.Miner {
            pool.AvailableChunks[i].Status = "completed"
            pool.AvailableChunks[i].GradientHash = msg.GradientHash
            pool.AvailableChunks[i].CompletedAt = ctx.BlockTime().Unix()
            
            pool.CompletedChunks = append(pool.CompletedChunks, msg.GradientHash)
            
            k.SetTaskPool(ctx, pool)
            
            // Reward miner per completed task (not per block)
            k.DistributeTaskReward(ctx, msg.Miner, msg.ChunkID)
            
            return nil
        }
    }
    
    return sdkerrors.Wrapf(ErrTaskNotFound, "chunk %d not found", msg.ChunkID)
}
```

#### Prefetching Support

```python
# miner-engine/r3mes/miner/task_pool_client.py
class TaskPoolClient:
    def __init__(self, grpc_client):
        self.grpc_client = grpc_client
        self.prefetch_queue = []  # Local prefetch queue
        self.max_prefetch = 5     # Prefetch up to 5 tasks ahead
    
    def claim_and_prefetch_tasks(self, pool_id):
        """Claim current task and prefetch next tasks"""
        # Claim current task
        current_task = self.grpc_client.ClaimTask(pool_id, chunk_id=None)  # Auto-assign
        
        # Prefetch next tasks
        for _ in range(self.max_prefetch):
            next_task = self.grpc_client.ClaimTask(pool_id, chunk_id=None)
            if next_task:
                self.prefetch_queue.append(next_task)
        
        return current_task
    
    def get_next_task(self):
        """Get next task from prefetch queue or claim new one"""
        if self.prefetch_queue:
            return self.prefetch_queue.pop(0)
        else:
            return self.claim_and_prefetch_tasks(self.current_pool_id)
```

#### Reward Mechanism Update

```go
// remes/x/remes/keeper/rewards.go
func (k Keeper) DistributeTaskReward(ctx sdk.Context, miner string, chunkID uint64) {
    // Reward per completed task (not per block)
    baseReward := k.GetBaseTaskReward(ctx)
    
    // Quality multiplier based on gradient quality
    qualityMultiplier := k.CalculateQualityMultiplier(ctx, miner, chunkID)
    
    reward := baseReward.MulInt64(int64(qualityMultiplier))
    
    // Mint and distribute
    k.MintTokens(ctx, miner, reward)
}
```

**Key Features**:
- **Task Pool**: Her blokta 50 chunk yayınlanır (tek görev değil)
- **Prefetching**: Miner mevcut işi bitirmeden sıradaki işi alabilir
- **Variable Speed**: RTX 4090 20 iş yaparken, GTX 1650 1 iş yapabilir
- **Per-Task Rewards**: Ödül blok başına değil, tamamlanan task başına
- **Blind Delivery**: Miner'lara gönderilen chunk'lar %90 gerçek iş, %10 tuzak (trap) içerir
  - Trap flag'leri (`is_trap`, `vault_entry_id`) miner'lara expose edilmez (güvenlik)
  - Query handler'larda `GetAvailableChunksForMiner()` kullanılır (sanitized response)
  - `ToMinerResponse()` metodu ile trap flag'leri kaldırılır
  - Miner'lar gerçek iş ile tuzakları ayırt edemez (Panoptikon etkisi)

### Genesis Vault ve Proof of Reuse

#### Genel Bakış

Genesis Vault, Proof of Reuse + Ghost Jobs sisteminin temelidir. Bu sistem, kendi kendini besleyen bir güvenlik mekanizmasıdır:

1. **Genesis Vault**: Başlangıçta 5000 tuzak (trap) problemi ve doğru cevapları içeren güvenli depo
2. **Blind Delivery**: Miner'lara %90 gerçek iş + %10 tuzak karışık olarak gönderilir
3. **Tolerant Verification**: Cosine similarity ile donanım farklarına toleranslı doğrulama
4. **Proof of Reuse**: Doğrulanmış gerçek işler vault'a eklenerek gelecekte tuzak olarak kullanılır

#### Genesis Vault Structure

```go
// remes/x/remes/types/genesis_vault.go
type GenesisVaultEntry struct {
    EntryID                 uint64 `json:"entry_id"`
    DataHash                string `json:"data_hash"`                // Input data IPFS hash
    ExpectedGradientHash    string `json:"expected_gradient_hash"`   // SHA256 hash (exact match için)
    ExpectedGradientIPFSHash string `json:"expected_gradient_ipfs_hash"` // IPFS hash
    ExpectedFingerprint     string `json:"expected_fingerprint"`     // JSON: Top-K (100) indices + values
    GPUArchitecture         string `json:"gpu_architecture"`         // GPU that solved correctly
    CreatedHeight           int64  `json:"created_height"`
    UsageCount              uint64 `json:"usage_count"`              // Kaç kez tuzak olarak kullanıldı
    LastUsedHeight          int64  `json:"last_used_height"`
    Encrypted               bool   `json:"encrypted"`                // Optional encryption
}

// GradientFingerprint JSON format:
// {
//   "top_k": 100,
//   "indices": [5, 100, 999, ...],     // ZORUNLU - pozisyonlar
//   "values": [0.9, 0.5, 0.3, ...],   // ZORUNLU - değerler (indices ile aynı sırada)
//   "shape": [128, 128]
// }
```

#### Blind Delivery System

Miner'lara gönderilen chunk'lar:
- **%90 Gerçek İş**: Normal task pool'dan seçilen gerçek müşteri işleri
- **%10 Tuzak**: Genesis vault'tan seçilen random trap entry'ler
- **Random Shuffle**: Fisher-Yates algoritması ile karıştırılır
- **Security**: `is_trap` ve `vault_entry_id` flag'leri miner'lara gönderilmez (sanitization)

```go
// remes/x/remes/keeper/task_pool.go
func (k Keeper) GetAvailableChunks(ctx context.Context, poolID uint64, limit uint64) ([]types.TaskChunk, error) {
    // 90% real jobs
    realJobCount := uint64(float64(limit) * 0.9)
    realJobs := getRealJobs(pool, realJobCount)
    
    // 10% traps from vault
    trapCount := limit - realJobCount
    traps := k.SelectRandomTraps(ctx, trapCount)
    
    // Shuffle and return
    allChunks := append(realJobs, traps...)
    return shuffleChunks(allChunks), nil
}
```

#### Tolerant Verification (Maskeleme Yöntemi)

Cosine similarity hesaplama için **maskeleme yöntemi** kullanılır:

1. Vault'taki `ExpectedFingerprint.Indices` alınır (örn: [5, 100, 999])
2. Miner'ın full gradient tensor'ü IPFS'ten indirilir
3. Miner'ın gradient'inden vault indices'lerindeki değerler extract edilir
4. Vault values ile miner'ın maskelenmiş vektörü arasında cosine similarity hesaplanır

**Neden Maskeleme?**
- Miner'ın Top-K listesi donanım gürültüsü nedeniyle farklı olabilir
- Vault'un önemli gördüğü noktalara (indices) bakarak doğruluğu ölçeriz
- Miner dürüst olsa bile Top-K listesi farklı olsa sorun olmaz

#### Proof of Reuse

Doğrulanmış gerçek işler vault'a eklenir:

```go
// remes/x/remes/keeper/task_pool.go
func (k Keeper) addToVaultIfValid(ctx context.Context, chunk types.TaskChunk, gradientHash string, gradientIPFSHash string, minerGPU string) error {
    // Real job kontrolü
    if chunk.IsTrap {
        return fmt.Errorf("cannot add trap to vault")
    }
    
    // Gradient'i IPFS'ten indir
    fullGradient, err := k.RetrieveGradientTensor(ctx, gradientIPFSHash)
    
    // Top-K fingerprint extract et (indices + values ZORUNLU - aynı sırada)
    // NOT: Miner'ın Top-K listesi kullanılmaz, sadece vault indices'leri kullanılır (maskeleme yöntemi)
    fingerprint, err := ExtractTopKFingerprint(fullGradient, 100)
    
    // Vault'a ekle
    vaultEntry := types.GenesisVaultEntry{
        EntryID: nextEntryID,
        DataHash: chunk.DataHash,
        ExpectedGradientHash: gradientHash,
        ExpectedGradientIPFSHash: gradientIPFSHash,
        ExpectedFingerprint: serializeFingerprint(fingerprint),
        GPUArchitecture: minerGPU,
        // ...
    }
    
    return k.AddToVault(ctx, vaultEntry)
}
```

#### Vault Pruning

Vault boyutunu kontrol etmek için pruning mekanizması:
- LRU (Least Recently Used) policy
- Minimum vault size korunur (örn: 1000 entry)
- Maximum vault size limiti (örn: 100,000 entry)
- Eski/kullanılmayan entry'ler temizlenir

### State Management

#### Global Model State
```go
type GlobalModelState struct {
    ModelHash     string   `json:"model_hash"`      // IPFS hash
    Version       uint64   `json:"version"`         // Model version
    UpdateHeight  int64    `json:"update_height"`   // Last update block
    Participants  []string `json:"participants"`    // Active miners
}
```

#### Stored Gradient
```go
type StoredGradient struct {
    Miner         string            `json:"miner"`
    IPFSHash      string            `json:"ipfs_hash"`
    Metadata      GradientMetadata  `json:"metadata"`
    ProofOfWork   []byte            `json:"proof_of_work"`
    BlockHeight   int64             `json:"block_height"`
    Timestamp     int64             `json:"timestamp"`
}
```

#### Task Pool State
```go
type TaskPoolState struct {
    ActivePools   []TaskPool `json:"active_pools"`
    CompletedPools []TaskPool `json:"completed_pools"`
    NextPoolID    uint64     `json:"next_pool_id"`
}
```

### Network Architecture

#### Node Types
1. **Mining Nodes**: BitNet model training
2. **Serving Nodes**: AI inference services
3. **Validator Nodes**: Transaction validation ve consensus

#### Communication Flow
```
Python Miner → IPFS (gradient upload) → Go Node (hash submission) → Blockchain
```

**Önemli**: Go node gradient data'yı memory'de tutmaz, sadece IPFS hash'lerini saklar.

### Consensus Mechanism

**Clarification**: R3MES Tendermint consensus kullanır block production ve finality için. PoUW consensus mechanism değil, block content validation ve token inflation sistemidir.

- **Block Production**: Tendermint validators
- **PoUW Role**: Training contribution validation ve reward distribution
- **Finality**: Tendermint BFT consensus

### Security Features

#### Authentication
- Message signing (miner private key)
- TLS mutual authentication (mTLS)
- Nonce/challenge-response mechanism
- Rate limiting per miner
- Staking requirements

#### Verification
- Three-layer optimistic verification
- CPU Iron Sandbox for disputes
- Merkle proof verification
- IPFS content verification

### Deployment Configuration

#### Docker Setup
```dockerfile
FROM golang:1.22-alpine AS builder
# Cosmos SDK v0.50.x build
# CometBFT v0.38.27 integration
```

#### Network Parameters
- Block time: ~5 seconds
- Unbonding time: 21 days
- Max validators: 100 (başlangıç)
- Training epochs: 100 blocks (~8-10 minutes)

### Integration Points

#### IPFS Integration
- Content addressing
- Pinning incentives
- Data availability challenges
- Passive retrieval (Go node)
- Active upload (Python miner)

#### Python-Go Bridge
- gRPC communication
- Protocol buffer serialization
- Apache Arrow Flight (optional, zero-copy)
- Authentication ve authorization

#### Web Dashboard Backend Integration
- **REST API Endpoints**: ServeHTTP implementation in Cosmos SDK
  - `/api/dashboard/status`: Network status
  - `/api/dashboard/miners`: Miner list (pagination)
  - `/api/dashboard/blocks`: Block list (pagination)
  - `/api/dashboard/statistics`: Network statistics
  - `/api/dashboard/ipfs/health`: IPFS health check
- **WebSocket Streaming**: Real-time data push to frontend
  - `topic:miner_stats`: GPU Temp, Fan Speed, VRAM Usage (2s intervals, Python miner stats HTTP server'dan)
  - `topic:training_metrics`: Current Loss, Epoch Progress (per step, Python miner stats HTTP server'dan)
  - `topic:network_status`: Network status (5s intervals, on-chain data)
- **gRPC Query Endpoints**: SDK context erişimi için
  - `QueryMiners`: Miner listesi
  - `QueryStatistics`: Network statistics
  - `QueryBlocks`: Block listesi (pagination)
  - `QueryBlock`: Belirli blok detayları
- **gRPC-Web Support**: Frontend blockchain communication
- **CORS Configuration**: Next.js frontend access (environment variable ile yapılandırılabilir)
- **Authentication**: Keplr wallet signature verification
- **Caching**: In-memory caching (30 seconds TTL, configurable)
- **Rate Limiting**: API abuse prevention

#### Transaction Hash Retrieval

**Implementation**: `MsgSubmitGradientResponse` içinde `tx_hash` field'ı eklendi.

```go
// remes/x/remes/keeper/msg_server_submit_gradient.go
txHash := hex.EncodeToString(sdkCtx.TxBytes())

return &types.MsgSubmitGradientResponse{
    StoredGradientId: gradientID,
    TxHash:           txHash,  // Transaction hash
}, nil
```

**Python Miner Side**:
```python
# miner-engine/r3mes/bridge/blockchain_client.py
response = self.stub.SubmitGradient(msg)
tx_hash = getattr(response, 'tx_hash', '')
if not tx_hash:
    tx_hash = "pending"  # Fallback
```

#### Block Time Calculation

**Implementation**: Block timestamps state'de saklanır ve average block time hesaplanır.

```go
// remes/x/remes/keeper/end_blocker.go
// Store current block timestamp
k.BlockTimestamps.Set(ctx, uint64(currentHeight), ctx.BlockTime())

// Prune old block timestamps (keep last 100 blocks)
if currentHeight > 100 {
    k.BlockTimestamps.Remove(ctx, uint64(currentHeight-100))
}
```

```go
// remes/x/remes/keeper/data_availability.go
func (k Keeper) GetAverageBlockTime(ctx sdk.Context) time.Duration {
    // Iterate over last 100 block timestamps
    // Calculate average block time
}
```

## Keeper Modülleri (Detaylı Liste)

R3MES blockchain'inin `x/remes/keeper` paketinde 97+ Go dosyası bulunmaktadır. Tüm modüller aşağıda listelenmiştir:

### Core Keeper Modülleri

1. **`keeper.go`** - Ana keeper struct ve modül initialization
2. **`auth.go`** - Authentication ve authorization logic
3. **`verification.go`** - Verification logic (PoUW, PoRep)
4. **`security_validation.go`** - Production environment validation, test mode checks
5. **`ipfs_manager.go`** - IPFS integration ve hash management

### Dataset ve Governance Modülleri

6. **`dataset_governance.go`** - Dataset integrity verification, IPFS hash validation, checksum verification
7. **`dataset_governance_integration_test.go`** - Integration tests
8. **`dataset_governance_test.go`** - Unit tests

### Verification Modülleri

9. **`optimistic_verification.go`** - PoUW verification, loss-based spot check, forward pass execution
10. **`verification_client.go`** - gRPC verification client, health check, forward pass execution
11. **`proof_of_replication.go`** - PoRep verification logic
12. **`proof_of_replication_test.go`** - PoRep tests
13. **`proof_of_work.go`** - Proof of Work implementation
14. **`cpu_verification_panel.go`** - CPU verification interface
15. **`container_validation.go`** - Container validation logic
16. **`environment_validation.go`** - Environment validation

### Model Management Modülleri

17. **`model_registry.go`** - Model registry management
18. **`model_registry_test.go`** - Model registry tests
19. **`global_model.go`** - Global model state management
20. **`model_sync.go`** - Model synchronization
21. **`model_versioning.go`** - Model version management
22. **`deterministic_quantization.go`** - Deterministic quantization support

### Training ve Epoch Modülleri

23. **`epoch_training.go`** - Epoch-based training management
24. **`training_window.go`** - Training window management
25. **`convergence_monitoring.go`** - Training convergence tracking
    - **Loss History Tracking**: Stores last N loss values in blockchain state for trend analysis
    - **Convergence Rate Calculation**: Calculates rate of loss reduction from historical data
    - **Convergence Detection**: Checks if model has converged based on loss change threshold and variance
    - **State Storage**: Uses `ConvergenceMetrics` collection to persist metrics per training round
    - **Adaptive Compression**: Adjusts compression based on convergence status
    - **Real-time Monitoring**: Updates metrics on each gradient submission
26. **`partition_handling.go`** - Data partitioning logic

### Task Pool ve Coordination Modülleri

27. **`task_pool.go`** - Task pool yönetimi (asenkron iş havuzu)
28. **`seed_locking.go`** - Seed locking mechanism
29. **`shard_assignment.go`** - Shard assignment calculation

### Economic ve Reward Modülleri

30. **`economic_incentives.go`** - Economic incentives logic
31. **`rewards.go`** - Rewards distribution
32. **`miners_reward.go`** - Miner-specific reward calculation
33. **`pinning_rewards.go`** - IPFS pinning rewards
34. **`treasury.go`** - Treasury management
35. **`slashing.go`** - Slashing mechanism

### Trust ve Reputation Modülleri

36. **`trust_score.go`** - Trust score calculation
37. **`scalability.go`** - Scalability features

### Network ve Subnet Modülleri

38. **`subnet.go`** - Subnet management
39. **`proposer_rotation.go`** - Proposer rotation logic
40. **`resource_enforcement.go`** - Resource limits enforcement
41. **`gpu_whitelist.go`** - GPU whitelisting
42. **`gpu_whitelist_test.go`** - GPU whitelist tests

### Message Server Modülleri

43. **`msg_server.go`** - Main message handlers
44. **`msg_server_submit_gradient.go`** - Gradient submission handler
45. **`msg_server_submit_aggregation.go`** - Aggregation submission handler
46. **`msg_server_commit_aggregation.go`** - Aggregation commit handler
47. **`msg_server_reveal_aggregation.go`** - Aggregation reveal handler
48. **`msg_server_challenge_aggregation.go`** - Challenge aggregation handler
49. **`msg_server_trap_job.go`** - Trap job handler
50. **`federated_trap_jobs.go`** - Federated trap job system
51. **`trap_job_blinding.go`** - Trap job security
52. **`msg_server_random_verifier.go`** - Random verification handler
53. **`msg_server_random_verifier_test.go`** - Random verifier tests
54. **`msg_server_cpu_verification.go`** - CPU verification messages
55. **`msg_server_execution_environment.go`** - Execution environment management
56. **`msg_server_model.go`** - Model-related messages
57. **`msg_server_model_upgrade.go`** - Model upgrade messages
58. **`msg_server_register_node.go`** - Node registration handler
59. **`msg_server_submit_resource_usage.go`** - Resource usage tracking
60. **`msg_server_serving.go`** - Model serving messages
61. **`msg_server_subnet.go`** - Subnet message handling
62. **`msg_server_training_window.go`** - Training window messages
63. **`msg_server_propose_dataset.go`** - Dataset proposal handler
64. **`msg_server_vote_dataset.go`** - Dataset voting handler
65. **`msg_server_mark_dataset_official.go`** - Mark dataset official handler
66. **`msg_server_remove_dataset.go`** - Remove dataset handler
67. **`msg_server_pinning.go`** - IPFS pinning messages
68. **`msg_server_validator_proposer_slashing.go`** - Validator proposer slashing
69. **`msg_update_params.go`** - Parameter update handler
70. **`msg_update_params_test.go`** - Parameter update tests

### Query Modülleri

71. **`query.go`** - Main query handlers
72. **`query_dashboard.go`** - Dashboard queries
73. **`query_dataset.go`** - Dataset queries
74. **`query_economic.go`** - Economic queries
75. **`query_gradient.go`** - Gradient queries
76. **`query_node.go`** - Node queries
77. **`query_params.go`** - Parameter queries
78. **`query_params_test.go`** - Parameter query tests
79. **`query_serving.go`** - Serving queries
80. **`query_stored_gradient.go`** - Stored gradient queries
81. **`query_sync.go`** - Sync queries

### API ve WebSocket Modülleri

82. **`dashboard_api.go`** - Dashboard API implementation
83. **`websocket.go`** - WebSocket streaming handler
84. **`data_availability.go`** - Data availability logic
85. **`data_availability_test.go`** - Data availability tests

### End Block ve Genesis Modülleri

86. **`end_blocker.go`** - End block logic
87. **`genesis.go`** - Genesis state initialization
88. **`genesis_test.go`** - Genesis tests

### Merkle ve Crypto Modülleri

89. **`merkle.go`** - Merkle tree implementation

### Rate Limiting ve Security Modülleri

90. **`rate_limiting.go`** - Rate limiting implementation
91. **`tls_config.go`** - TLS configuration

### Test Modülleri

92. **`keeper_test.go`** - Keeper unit tests
93. **`integration_test.go`** - Integration tests
94. **`loss_verification_test.go`** - Loss verification tests
95. **`performance_test.go`** - Performance tests
96. **`block_time_test.go`** - Block time tests
97. **`property_test.go`** - Property tests

### Önemli Notlar

- Tüm modüller production-ready durumda
- Test coverage yüksek (integration, unit, property tests)
- Security validation ve fail-closed model aktif
- IPFS entegrasyonu tam
- WebSocket streaming desteği var
- Dashboard API tam implementasyon

## Key Management

### Key Creation

```bash
# Yeni key oluştur
./build/remesd keys add validator

# Key'leri listele
./build/remesd keys list

# Key address'ini al
./build/remesd keys show validator -a
```

### Key Auto-Detection

Terminal komutları ve CLI araçları, otomatik olarak key bulur:
1. "validator" key'i arar
2. Bulamazsa, listedeki ilk key'i kullanır
3. Key yoksa hata gösterir

**Not**: Web Dashboard Keplr wallet kullanır, key auto-detection yapmaz.

### Key Export/Import

```bash
# Key export (encrypted)
./build/remesd keys export validator --output-file validator-backup.json

# Key import
./build/remesd keys import validator validator-backup.json
```

## Implementation Priority

1. **Phase 1**: Basic blockchain scaffold
2. **Phase 2**: Core PoUW module
3. **Phase 3**: IPFS integration
4. **Phase 4**: Security mechanisms
5. **Phase 5**: Production optimization

Bu blockchain infrastructure, R3MES protokolünün temelini oluşturur ve diğer tüm bileşenlerin üzerine inşa edilir.
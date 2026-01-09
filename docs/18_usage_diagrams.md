# R3MES Kullanım Şemaları

Bu dokümantasyon, R3MES ekosistemindeki farklı kullanıcı rollerinin sistemle nasıl etkileşime girdiğini, hangi işlemleri yapabileceklerini ve bunları nasıl gerçekleştireceklerini detaylı şemalarla açıklar.

## Genel Bakış

R3MES ekosisteminde dört ana kullanıcı rolü bulunmaktadır:

1. **Kurucu (Founder/Creator)**: Network'ün başlangıç konfigürasyonunu yapan, genesis state'i oluşturan
2. **Miner**: AI model eğitimi yaparak gradient submit eden, token kazanan
3. **Validator**: Transaction validation ve consensus'a katılan, network'ü güvende tutan
4. **Developer**: Kod geliştiren, test eden, sistem üzerinde çalışan

Her rol için aşağıda detaylı şemalar, workflow'lar ve kullanım kılavuzları bulunmaktadır.

---

## Rol Seçim Rehberi

Aşağıdaki akış diyagramı, hangi rolün size uygun olduğunu belirlemenize yardımcı olur:

```mermaid
flowchart TD
    Start([R3MES'e Hoş Geldiniz]) --> HasGPU{GPU'nuz var mı?}
    
    HasGPU -->|Evet, RTX 3060+| MinerPath[Miner Ol]
    HasGPU -->|Hayır| HasServer{Sunucunuz var mı?}
    
    HasServer -->|Evet, 24/7 çalışan| ValidatorPath[Validator Ol]
    HasServer -->|Hayır| HasDataset{Dataset'iniz var mı?}
    
    HasDataset -->|Evet| DatasetProvider[Dataset Provider Ol]
    HasDataset -->|Hayır| IsDeveloper{Kod geliştirme<br/>yapabilir misiniz?}
    
    IsDeveloper -->|Evet| DeveloperPath[Developer Ol]
    IsDeveloper -->|Hayır| Start
    
    MinerPath --> MinerSetup[GPU ile Model Eğitimi<br/>Token Kazanma]
    ValidatorPath --> ValidatorSetup[Network Security<br/>Consensus Katılımı]
    DatasetProvider --> GovernanceFlow[Governance'e<br/>Dataset Önerisi]
    DeveloperPath --> DevSetup[Kod Geliştirme<br/>Contribution]
    
    MinerSetup --> End([Başlayın])
    ValidatorSetup --> End
    GovernanceFlow --> End
    DevSetup --> End
```

---

## 1. Kurucu (Founder/Creator) Kullanım Şeması

Kurucu, R3MES network'ünün ilk kurulumunu yapan, genesis state'i oluşturan ve network parametrelerini ayarlayan kişidir.

### 1.1 Kurucu Workflow

```mermaid
sequenceDiagram
    participant Founder as Kurucu
    participant Genesis as Genesis Config
    participant Chain as Blockchain Node
    participant Vault as Genesis Vault
    participant Model as Model Registry
    
    Founder->>Genesis: 1. Genesis state oluştur
    Note over Genesis: Initial validators<br/>Initial balances<br/>Network parameters
    
    Founder->>Model: 2. İlk model kaydı
    Note over Model: BitNet b1.58<br/>Model config<br/>Architecture
    
    Founder->>Vault: 3. Genesis Vault başlat
    Note over Vault: 5000 initial trap entries<br/>Pre-solved problems<br/>Fingerprints
    
    Founder->>Chain: 4. Genesis file'i build et
    Chain->>Chain: 5. Chain ID: remes-1<br/>Initial height: 1
    
    Founder->>Chain: 6. İlk validator'ı başlat
    Chain->>Chain: 7. Network başlatıldı
    
    Founder->>Genesis: 8. Network parametrelerini ayarla
    Note over Genesis: Staking requirements<br/>Slashing rates<br/>Reward formulas
```

### 1.2 Genesis State Yapısı

```mermaid
flowchart LR
    Genesis[Genesis State] --> Validators[Initial Validators]
    Genesis --> Balances[Initial Balances]
    Genesis --> Params[Network Parameters]
    Genesis --> ModelConfig[Model Configuration]
    Genesis --> VaultEntries[Genesis Vault Entries]
    
    Validators --> V1[Validator 1]
    Validators --> V2[Validator 2]
    Validators --> VN[Validator N]
    
    ModelConfig --> BitNet[BitNet b1.58 Config]
    ModelConfig --> Architecture[Architecture Config]
    
    VaultEntries --> Trap1[Trap Entry 1]
    VaultEntries --> Trap2[Trap Entry 2]
    VaultEntries --> TrapN[Trap Entry 5000]
```

### 1.3 Kurucu İşlemleri

#### Genesis State Oluşturma

```bash
# Genesis file oluştur
./build/remesd init mynode --chain-id remes-1

# Genesis'teki parametreleri düzenle
# ~/.remesd/config/genesis.json
```

**Yapılandırılacak Parametreler:**
- Initial validator addresses ve stakes
- Initial token distribution
- Network parameters (block time, epoch duration)
- Model configuration (BitNet b1.58)
- Genesis vault entries (5000 trap jobs)

#### İlk Model Kaydı

```bash
# Model kayıt transaction'ı
./build/remesd tx remes register-model \
  --model-type MODEL_TYPE_BITNET \
  --model-version "b1.58" \
  --architecture-config "..." \
  --from founder \
  --chain-id remes-1 \
  --yes
```

#### Genesis Vault Başlatma

Genesis vault, 5000 önceden çözülmüş problem içerir. Bu problemler trap job olarak kullanılır.

```go
// Genesis vault entry structure
type GenesisVaultEntry struct {
    EntryID             uint64
    ExpectedGradientHash string
    ExpectedFingerprint  TopKFingerprint  // Top-K indices + values
    GpuArchitecture      string
    Encrypted            bool
    EncryptionKey        []byte  // Optional AES-256-GCM
}
```

---

## 2. Miner Kullanım Şeması

Miner, AI model eğitimi yaparak gradient submit eden ve token kazanan kullanıcıdır.

### 2.1 Miner Kayıt ve Başlangıç Workflow

```mermaid
sequenceDiagram
    participant Miner as Miner
    participant Wallet as Wallet System
    participant Blockchain as Blockchain Node
    participant IPFS as IPFS Network
    participant Model as Model Manager
    participant Training as Training Engine
    
    Miner->>Wallet: 1. Wallet oluştur
    Wallet->>Miner: Wallet address: remes1...
    
    Miner->>Blockchain: 2. Node kaydı (MsgRegisterNode)
    Note over Blockchain: Role: MINER<br/>Stake: 1000remes<br/>GPU Architecture
    
    Blockchain->>Miner: 3. Registration confirmed
    
    Miner->>IPFS: 4. IPFS node başlat (embedded)
    IPFS->>Miner: 5. IPFS ready (localhost:5001)
    
    Miner->>Model: 6. Model indir (BitNet b1.58)
    Note over Model: HuggingFace + IPFS fallback<br/>28GB frozen backbone<br/>LoRA adapters
    
    Miner->>Blockchain: 7. Training round bilgisi al
    Blockchain->>Miner: 8. Shard assignment<br/>Dataset hash<br/>Training params
    
    Miner->>Training: 9. Training başlat
    Training->>Training: 10. Gradient hesapla
    
    Training->>IPFS: 11. Gradient'i IPFS'e yükle
    IPFS->>Training: 12. IPFS hash (CID)
    
    Training->>Blockchain: 13. MsgSubmitGradient
    Note over Blockchain: IPFS hash only<br/>Gradient hash<br/>Metadata
    
    Blockchain->>Blockchain: 14. Validation<br/>Rate limiting<br/>Stake check
    
    Blockchain->>Miner: 15. Reward dağıt
    Note over Miner: Mining reward<br/>Trust score update
```

### 2.2 Gradient Submission Detaylı Akış

```mermaid
flowchart TD
    Start([Training Tamamlandı]) --> Compute[Gradient Hesapla]
    Compute --> Hash[Gradient Hash SHA256]
    Hash --> IPFSUpload[IPFS'e Yükle]
    
    IPFSUpload --> GetCID[IPFS CID Al]
    GetCID --> PrepareMsg[MsgSubmitGradient Hazırla]
    
    PrepareMsg --> Validate1[Stake Kontrolü]
    Validate1 -->|Yetersiz| StakeError[Stake Hatası]
    Validate1 -->|Yeterli| Validate2[Rate Limit Kontrolü]
    
    Validate2 -->|Aşım| RateLimitError[Rate Limit Hatası]
    Validate2 -->|OK| Validate3[Nonce Kontrolü]
    
    Validate3 -->|Kullanılmış| NonceError[Nonce Hatası]
    Validate3 -->|Yeni| Sign[İmzala]
    
    Sign --> Submit[Blockchain'e Submit Et]
    Submit --> OptimisticVerify[Optimistic Verification<br/>Layer 1]
    
    OptimisticVerify -->|Geçerli| Store[On-chain Store<br/>Hash only]
    OptimisticVerify -->|Şüpheli| Challenge[Challenge Mekanizması<br/>Layer 2]
    
    Store --> Reward[Reward Dağıt]
    Challenge --> Verify[Full Verification<br/>Layer 3]
    
    Verify -->|Geçerli| Reward
    Verify -->|Geçersiz| Slash[Slashing]
    
    Reward --> End([Başarılı])
    StakeError --> End
    RateLimitError --> End
    NonceError --> End
    Slash --> End
```

### 2.3 Miner İşlemleri

#### Node Kaydı

```bash
# Miner node kaydı
./build/remesd tx remes register-node \
  --node-address remes1abc... \
  --node-type NODE_TYPE_MINER \
  --roles MINER \
  --stake 1000remes \
  --resources "gpu:nvidia,ram:16gb" \
  --from miner \
  --chain-id remes-1 \
  --yes
```

#### PyPI ile Miner Kurulumu

```bash
# Miner engine kurulumu
pip install r3mes

# Setup wizard
r3mes-miner setup

# Mining başlatma
r3mes-miner start --continuous
```

#### Gradient Submission

```python
# Python miner engine içinde
gradient_hash = compute_gradient_hash(gradients)
ipfs_hash = upload_to_ipfs(gradients)  # Direct upload

# Submit to blockchain
msg = MsgSubmitGradient(
    miner="remes1abc...",
    ipfs_hash=ipfs_hash,
    gradient_hash=gradient_hash,
    model_version="b1.58",
    training_round_id=123,
    shard_id=23,
    token_count=2048,  # MUST be 2048
    nonce=next_nonce(),
    signature=sign_message(msg)
)

response = submit_transaction(msg)
```

### 2.4 Miner Dashboard Kullanımı

```mermaid
graph LR
    Dashboard[Miner Dashboard] --> Stats[İstatistikler]
    Dashboard --> Training[Training Monitor]
    Dashboard --> Rewards[Rewards]
    Dashboard --> Settings[Settings]
    
    Stats --> ActiveRound[Aktif Round]
    Stats --> ShardAssignment[Shard Assignment]
    Stats --> TrustScore[Trust Score]
    
    Training --> CurrentTraining[Mevcut Training]
    Training --> History[Geçmiş Training'ler]
    Training --> Gradients[Gradient Listesi]
    
    Rewards --> Pending[Pending Rewards]
    Rewards --> History[Reward History]
    Rewards --> Withdraw[Withdraw]
```

---

## 3. Validator Kullanım Şeması

Validator, transaction validation ve consensus'a katılan, network'ü güvende tutan node operatörüdür.

### 3.1 Validator Setup ve Consensus Workflow

```mermaid
sequenceDiagram
    participant Validator as Validator
    participant Staking as Staking Module
    participant Consensus as Consensus Engine
    participant Proposer as Block Proposer
    participant Network as P2P Network
    
    Validator->>Staking: 1. Validator kaydı
    Note over Staking: Minimum stake<br/>Self-delegation<br/>Commission rate
    
    Staking->>Validator: 2. Validator address
    
    Validator->>Network: 3. P2P network'e bağlan
    Network->>Validator: 4. Peer connections
    
    Validator->>Consensus: 5. Consensus'a katıl
    Consensus->>Validator: 6. Voting power assigned
    
    loop Her Block
        Proposer->>Validator: 7. Block proposal
        Validator->>Validator: 8. Transactions validate et
        
        alt Tüm tx geçerli
            Validator->>Consensus: 9. Pre-vote: YES
        else Geçersiz tx var
            Validator->>Consensus: 10. Pre-vote: NO
        end
        
        Consensus->>Validator: 11. Pre-commit request
        Validator->>Consensus: 12. Pre-commit
        
        Consensus->>Validator: 13. Block committed
    end
```

### 3.2 Trap Job Oluşturma Workflow

```mermaid
sequenceDiagram
    participant Validator as Validator
    participant Vault as Genesis Vault
    participant Blockchain as Blockchain
    participant Miner as Target Miner
    participant Verification as Verification System
    
    Validator->>Vault: 1. Vault entry seç
    Vault->>Validator: 2. Expected fingerprint
    
    Validator->>Blockchain: 3. MsgCreateTrapJob
    Note over Blockchain: Target miner<br/>Dataset hash<br/>Vault entry ID
    
    Blockchain->>Blockchain: 4. Trap job oluştur
    Blockchain->>Miner: 5. Task assignment<br/>(Blind delivery)
    Note over Miner: Miner trap olduğunu<br/>bilmiyor (90% real, 10% trap)
    
    Miner->>Miner: 6. Training yap
    Miner->>Blockchain: 7. Gradient submit
    
    Blockchain->>Verification: 8. Tolerant verification
    Note over Verification: Cosine similarity<br/>Top-K fingerprint<br/>Masking method
    
    alt Verification Geçerli
        Verification->>Blockchain: 9. Similarity > threshold
        Blockchain->>Validator: 10. Miner güvenilir
        Blockchain->>Miner: 11. Reward
    else Verification Geçersiz
        Verification->>Blockchain: 12. Similarity < threshold
        Blockchain->>Blockchain: 13. Fraud detected
        Blockchain->>Miner: 14. Slashing
        Blockchain->>Validator: 15. Fraud score update
    end
```

### 3.3 Validator İşlemleri

#### Validator Kaydı

```bash
# Validator node kaydı
./build/remesd tx remes register-node \
  --node-address remes1validator... \
  --node-type NODE_TYPE_VALIDATOR \
  --roles VALIDATOR \
  --stake 10000remes \
  --resources "cpu:8cores,ram:32gb,disk:500gb" \
  --from validator \
  --chain-id remes-1 \
  --yes

# Validator'ı staking modülüne kaydet
./build/remesd tx staking create-validator \
  --amount 10000remes \
  --pubkey $(./build/remesd tendermint show-validator) \
  --moniker "MyValidator" \
  --commission-rate "0.1" \
  --commission-max-rate "0.2" \
  --commission-max-change-rate "0.01" \
  --min-self-delegation "1000" \
  --from validator \
  --chain-id remes-1 \
  --yes
```

#### Trap Job Oluşturma

```bash
# Trap job oluştur (sadece validator'lar yapabilir)
./build/remesd tx remes create-trap-job \
  --target-miner remes1miner... \
  --dataset-ipfs-hash QmXxxx... \
  --vault-entry-id 123 \
  --from validator \
  --chain-id remes-1 \
  --yes
```

#### Governance Oylama

```bash
# Proposal'a oy ver
./build/remesd tx gov vote 1 yes \
  --from validator \
  --chain-id remes-1 \
  --yes
```

### 3.4 Validator Monitoring

```mermaid
graph TD
    ValidatorNode[Validator Node] --> ConsensusMetrics[Consensus Metrics]
    ValidatorNode --> StakingMetrics[Staking Metrics]
    ValidatorNode --> PerformanceMetrics[Performance Metrics]
    
    ConsensusMetrics --> VotingPower[Voting Power]
    ConsensusMetrics --> Uptime[Uptime %]
    ConsensusMetrics --> BlocksProposed[Blocks Proposed]
    ConsensusMetrics --> BlocksVoted[Blocks Voted]
    
    StakingMetrics --> SelfDelegation[Self Delegation]
    StakingMetrics --> TotalDelegations[Total Delegations]
    StakingMetrics --> Commission[Commission Rate]
    StakingMetrics --> Rewards[Validator Rewards]
    
    PerformanceMetrics --> CPUUsage[CPU Usage]
    PerformanceMetrics --> MemoryUsage[Memory Usage]
    PerformanceMetrics --> NetworkIO[Network I/O]
    PerformanceMetrics --> DiskIO[Disk I/O]
```

---

## 4. Developer Kullanım Şeması

Developer, R3MES kodunu geliştiren, test eden ve sisteme katkıda bulunan kişidir.

### 4.1 Development Environment Setup

```mermaid
sequenceDiagram
    participant Dev as Developer
    participant Repo as Git Repository
    participant GoEnv as Go Environment
    participant PythonEnv as Python Environment
    participant NodeEnv as Node.js Environment
    participant Docker as Docker
    participant Services as Local Services
    
    Dev->>Repo: 1. Repository clone
    Repo->>Dev: 2. Source code
    
    Dev->>GoEnv: 3. Go 1.22+ kur
    GoEnv->>Dev: 4. Go ready
    
    Dev->>PythonEnv: 5. Python 3.10+ kur
    PythonEnv->>Dev: 6. Python ready
    
    Dev->>NodeEnv: 7. Node.js 18+ kur
    NodeEnv->>Dev: 8. Node ready
    
    Dev->>Docker: 9. Docker kur (opsiyonel)
    Docker->>Dev: 10. Docker ready
    
    Dev->>Services: 11. Local services başlat
    Note over Services: IPFS daemon<br/>PostgreSQL<br/>Redis
    
    Services->>Dev: 12. Services ready
    
    Dev->>GoEnv: 13. Blockchain node build
    GoEnv->>Dev: 14. remesd binary
    
    Dev->>PythonEnv: 15. Backend dependencies
    PythonEnv->>Dev: 16. Backend ready
    
    Dev->>PythonEnv: 17. Miner engine dependencies
    PythonEnv->>Dev: 18. Miner engine ready
    
    Dev->>NodeEnv: 19. Frontend dependencies
    NodeEnv->>Dev: 20. Frontend ready
```

### 4.2 Development Workflow

```mermaid
flowchart TD
    Start([Development Başlat]) --> Clone[Repository Clone]
    Clone --> Setup[Environment Setup]
    Setup --> Branch[Feature Branch Oluştur]
    
    Branch --> Code[Kod Geliştir]
    Code --> Test[Test Yaz]
    Test --> RunTests[Testleri Çalıştır]
    
    RunTests -->|Başarısız| Fix[Fixes]
    Fix --> Code
    RunTests -->|Başarılı| Lint[Linting]
    
    Lint -->|Hata var| FixLint[Lint Fixes]
    FixLint --> Code
    Lint -->|OK| Build[Build Test]
    
    Build -->|Başarısız| FixBuild[Build Fixes]
    FixBuild --> Code
    Build -->|Başarılı| Commit[Commit]
    
    Commit --> Push[Push to Remote]
    Push --> PR[Pull Request]
    PR --> Review[Code Review]
    
    Review -->|Değişiklik gerekli| Code
    Review -->|Onaylandı| Merge[Merge]
    Merge --> End([Tamamlandı])
```

### 4.3 Developer İşlemleri

#### Local Development Setup

```bash
# 1. Repository clone
git clone https://github.com/r3mes/r3mes.git
cd r3mes

# 2. Blockchain node build
cd remes
go mod download
make build

# 3. Backend setup
cd ../backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Miner engine setup
cd ../miner-engine
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 5. Frontend setup
cd ../web-dashboard
npm install

# 6. Services başlat (Docker)
docker-compose up -d postgres redis ipfs

# 7. Local node başlat
cd remes
./build/remesd start --home ~/.remesd-local
```

#### Debug Mode Kullanımı

```bash
# Debug mode environment variables
export R3MES_DEBUG_MODE=true
export R3MES_DEBUG_LEVEL=verbose
export R3MES_DEBUG_COMPONENTS=blockchain,backend,miner
export R3MES_DEBUG_LOG_LEVEL=TRACE
export R3MES_DEBUG_LOG_FORMAT=json
export R3MES_DEBUG_LOG_FILE=~/.r3mes/debug.log

# Debug script ile başlat
./scripts/debug/start_debug_mode.sh
```

#### Testing

```bash
# Go tests
cd remes
go test ./x/remes/keeper/... -v

# Python tests
cd backend
pytest tests/ -v

# Miner engine tests
cd miner-engine
pytest tests/ -v

# Integration tests
cd tests/integration
python test_full_workflow.py
```

### 4.4 API Development

```mermaid
sequenceDiagram
    participant Dev as Developer
    participant API as FastAPI Backend
    participant DB as Database
    participant Blockchain as Blockchain Node
    participant Frontend as Frontend
    
    Dev->>API: 1. Endpoint tasarla
    Note over API: OpenAPI schema<br/>Request/Response models
    
    Dev->>DB: 2. Database migration
    DB->>Dev: 3. Schema updated
    
    Dev->>API: 4. Endpoint implement et
    API->>Blockchain: 5. gRPC query
    Blockchain->>API: 6. Response
    
    API->>DB: 7. Cache/store
    DB->>API: 8. Data
    
    API->>Frontend: 9. API test (Swagger UI)
    Frontend->>API: 10. Request
    API->>Frontend: 11. Response
    
    Dev->>API: 12. Unit tests
    Dev->>API: 13. Integration tests
```

### 4.5 Contributing Workflow

```mermaid
graph LR
    Fork[Fork Repository] --> Clone[Clone Fork]
    Clone --> Branch[Create Branch]
    Branch --> Dev[Develop Feature]
    Dev --> Test[Write Tests]
    Test --> Commit[Commit Changes]
    Commit --> Push[Push to Fork]
    Push --> PR[Create Pull Request]
    PR --> Review[Code Review]
    Review -->|Approve| Merge[Merge]
    Review -->|Request Changes| Dev
```

---

## Ortak İşlemler ve Araçlar

### Transaction Types

Tüm kullanıcılar aşağıdaki transaction türlerini kullanabilir:

```mermaid
graph TD
    Transactions[Transaction Types] --> SubmitGradient[MsgSubmitGradient]
    Transactions --> RegisterNode[MsgRegisterNode]
    Transactions --> ProposeDataset[MsgProposeDataset]
    Transactions --> VoteDataset[MsgVoteDataset]
    Transactions --> SubmitAggregation[MsgSubmitAggregation]
    Transactions --> ChallengeAggregation[MsgChallengeAggregation]
    
    SubmitGradient --> MinerTx[Miner: Gradient submit]
    RegisterNode --> AllTx[Tüm Roller: Node kaydı]
    ProposeDataset --> DatasetTx[Dataset Provider: Dataset öner]
    VoteDataset --> GovTx[Governance: Oyla]
    SubmitAggregation --> ProposerTx[Proposer: Aggregation submit]
    ChallengeAggregation --> ValidatorTx[Validator: Challenge]
```

### CLI Araçları

```bash
# Wallet yönetimi
r3mes wallet create
r3mes wallet import <mnemonic|private_key>
r3mes wallet balance <address>

# Miner işlemleri
r3mes miner start
r3mes miner stop
r3mes miner status

# Node işlemleri
r3mes node start
r3mes node stop
r3mes node status

# Governance işlemleri
r3mes governance vote <proposal_id> <yes|no>
r3mes governance proposals
```

### Web Dashboard

Tüm kullanıcılar web dashboard'u kullanabilir:

- **Ana Sayfa**: Network overview
- **Chat**: AI inference (backend servisi)
- **Mine**: Miner console (miner'lar için)
- **Wallet**: Wallet yönetimi
- **Network**: Network statistics ve governance
- **Settings**: Kullanıcı ayarları
- **Help**: Dokümantasyon ve destek
- **Onboarding**: İlk kurulum rehberi

### API Endpoints

Tüm kullanıcılar REST API'yi kullanabilir:

```bash
# User info
GET /user/info/{wallet_address}

# Network stats
GET /network/stats

# Blocks
GET /blocks

# Chat (inference)
POST /chat

# Metrics
GET /metrics
```

---

## Sonuç

Bu dokümantasyon, R3MES ekosistemindeki farklı rollerin sistemle nasıl etkileşime girdiğini ve hangi işlemleri yapabileceklerini detaylı şemalarla açıklar. Her rol için özel workflow'lar, transaction akışları ve kullanım kılavuzları sağlanmıştır.

Daha detaylı bilgi için ilgili dokümantasyonlara bakınız:

- [User Onboarding Guides](./09_user_onboarding_guides.md)
- [Blockchain Infrastructure](./01_blockchain_infrastructure.md)
- [Governance System](./06_governance_system.md)
- [API Reference](./13_api_reference.md)
- [Debug Mode](./17_debug_mode.md)

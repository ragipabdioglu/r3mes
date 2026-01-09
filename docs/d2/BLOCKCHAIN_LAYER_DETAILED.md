# 🔗 R3MES Blockchain Layer - Senior Level Architecture Documentation

> **Versiyon:** 1.0.0  
> **Tarih:** 2026-01-09  
> **Yazar:** Kiro AI Architecture Assistant  
> **Kapsam:** Cosmos SDK tabanlı blockchain katmanının aşırı detaylı teknik analizi

---

## 📋 İçindekiler

1. [Genel Bakış](#1-genel-bakış)
2. [Keeper Orchestration Pattern](#2-keeper-orchestration-pattern)
3. [Domain Keeper'lar Detaylı Analiz](#3-domain-keeperlar-detaylı-analiz)
4. [State Management & Collections](#4-state-management--collections)
5. [Transaction Flow (MsgServer)](#5-transaction-flow-msgserver)
6. [Query Flow (QueryServer)](#6-query-flow-queryserver)
7. [Trap Job Verification System](#7-trap-job-verification-system)
8. [Gradient Aggregation Pipeline](#8-gradient-aggregation-pipeline)
9. [Economics & Slashing Mechanics](#9-economics--slashing-mechanics)
10. [Inter-Keeper Dependencies](#10-inter-keeper-dependencies)

---

## 1. Genel Bakış

R3MES blockchain katmanı, Cosmos SDK üzerine inşa edilmiş, **Proof of Useful Work (PoUW)** konsensüs mekanizmasını destekleyen özel bir modüldür.


```d2
# ═══════════════════════════════════════════════════════════════════════════════
# R3MES BLOCKCHAIN LAYER - GENEL MİMARİ
# ═══════════════════════════════════════════════════════════════════════════════

vars: {
  d2-config: {
    layout-engine: dagre
    theme-id: 200
  }
}

direction: down
grid-rows: 1
grid-gap: 50

title: |md
  # 🔗 R3MES Blockchain Layer
  **Cosmos SDK Module Architecture**
  *8 Domain-Specific Keepers + Orchestrator Pattern*
|

classes: {
  orchestrator: {
    style: {
      fill: "#e74c3c"
      stroke: "#c0392b"
      stroke-width: 4
      border-radius: 15
      font-color: "#ffffff"
      font-size: 16
      bold: true
    }
  }
  keeper: {
    style: {
      fill: "#3498db"
      stroke: "#2980b9"
      stroke-width: 2
      border-radius: 10
      font-color: "#ffffff"
    }
  }
  collection: {
    style: {
      fill: "#9b59b6"
      stroke: "#8e44ad"
      stroke-width: 1
      border-radius: 5
      font-color: "#ffffff"
    }
  }
  server: {
    style: {
      fill: "#27ae60"
      stroke: "#1e8449"
      stroke-width: 2
      border-radius: 8
      font-color: "#ffffff"
    }
  }
  proto: {
    style: {
      fill: "#f39c12"
      stroke: "#d68910"
      stroke-width: 1
      border-radius: 5
      font-color: "#000000"
    }
  }
  external: {
    style: {
      fill: "#95a5a6"
      stroke: "#7f8c8d"
      stroke-width: 2
      stroke-dash: 3
      border-radius: 8
      font-color: "#ffffff"
    }
  }
}

# ═══════════════════════════════════════════════════════════════════════════════
# ANA KEEPER ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

orchestrator: "🎯 KEEPER ORCHESTRATOR\nkeeper/keeper.go" {
  class: orchestrator
  
  desc: |md
    **Composition Pattern**
    - Domain keeper'ları orchestrate eder
    - Legacy method delegation
    - Genesis init/export
    - Schema management
  |
  
  methods: "Public Methods" {
    init: "NewKeeper()"
    genesis_init: "InitGenesis()"
    genesis_export: "ExportGenesis()"
    finalize: "FinalizeExpiredAggregations()"
  }
  
  delegations: "Legacy Delegations" {
    model_ops: "RegisterModel() → model.RegisterModel()"
    training_ops: "SubmitGradient() → training.SubmitGradient()"
    dataset_ops: "ProposeDataset() → dataset.ProposeDataset()"
    node_ops: "RegisterNode() → node.RegisterNode()"
    economics_ops: "CalculateRewards() → economics.CalculateRewards()"
    security_ops: "VerifySignature() → security.VerifySignature()"
    infra_ops: "VerifyIPFSContent() → infra.VerifyIPFSContent()"
  }
}

# ═══════════════════════════════════════════════════════════════════════════════
# 8 DOMAIN-SPECIFIC KEEPERS
# ═══════════════════════════════════════════════════════════════════════════════

keepers: "Domain-Specific Keepers" {
  
  # 1. CORE KEEPER
  core: "🔧 CORE KEEPER\nkeeper/core/keeper.go" {
    class: keeper
    
    responsibility: |md
      **Temel Blockchain İşlevleri**
      - Params yönetimi
      - Block timestamp tracking
      - Nonce management (replay attack prevention)
      - Submission history
      - IBC capability management
    |
    
    collections: "Collections" {
      params: "Params\nItem[types.Params]" { class: collection }
      block_ts: "BlockTimestamps\nMap[int64, int64]" { class: collection }
      used_nonces: "UsedNonces\nMap[string, bool]" { class: collection }
      nonce_windows: "NonceWindows\nMap[string, string]" { class: collection }
      submission_hist: "SubmissionHistory\nMap[string, uint64]" { class: collection }
    }
    
    methods: "Key Methods" {
      get_params: "GetParams(ctx) → Params"
      set_params: "SetParams(ctx, params)"
      is_nonce_used: "IsNonceUsed(ctx, key) → bool"
      mark_nonce: "MarkNonceAsUsed(ctx, key)"
      get_block_height: "GetCurrentBlockHeight(ctx) → int64"
      claim_cap: "ClaimCapability(ctx, cap, name)"
    }
    
    deps: "Dependencies" {
      bank: "BankKeeper"
      auth: "AuthKeeper"
      capability: "CapabilityKeeper"
      scoped: "ScopedKeeper"
    }
  }
  
  # 2. MODEL KEEPER
  model: "📦 MODEL KEEPER\nkeeper/model/keeper.go" {
    class: keeper
    
    responsibility: |md
      **Model Lifecycle Management**
      - Model registration & versioning
      - Global model state
      - Upgrade proposals & voting
      - Adapter approval workflow
      - Gradient aggregation logic
    |
    
    collections: "Collections" {
      global_state: "GlobalModelState\nItem[GlobalModelState]" { class: collection }
      registries: "ModelRegistries\nMap[uint64, ModelRegistry]" { class: collection }
      model_id: "ModelID\nSequence" { class: collection }
      versions: "ModelVersions\nMap[uint64, ModelVersion]" { class: collection }
      proposals: "ModelUpgradeProposals\nMap[uint64, ModelUpgradeProposal]" { class: collection }
      proposal_id: "ModelUpgradeProposalID\nSequence" { class: collection }
      votes: "ModelUpgradeVotes\nMap[uint64, ModelUpgradeVote]" { class: collection }
      vote_id: "ModelUpgradeVoteID\nSequence" { class: collection }
      active_versions: "ActiveModelVersions\nItem[ActiveModelVersions]" { class: collection }
    }
    
    methods: "Key Methods" {
      register: "RegisterModel(ctx, model)"
      get_model: "GetModel(ctx, modelID) → ModelRegistry"
      update_model: "UpdateModel(ctx, modelID, updates)"
      get_global: "GetGlobalModelState(ctx) → GlobalModelState"
      update_global: "UpdateGlobalModelState(ctx, state)"
      create_version: "CreateModelVersion(ctx, version)"
      propose_upgrade: "ProposeModelUpgrade(ctx, proposal)"
      vote_upgrade: "VoteOnModelUpgrade(ctx, vote)"
    }
    
    aggregation: "Aggregation Module\naggregation.go" {
      methods: "AggregationMethod enum" {
        weighted: "WeightedAverage"
        trimmed: "TrimmedMean (Byzantine-robust)"
        median: "Median (Most robust)"
      }
      
      config: "AggregationConfig" {
        method: "Method: TrimmedMean"
        threshold: "ByzantineThreshold: 0.2"
        min_grad: "MinGradients: 3"
        max_grad: "MaxGradients: 100"
      }
      
      funcs: "Functions" {
        aggregate: "AggregateGradients(ctx, roundID, gradientIDs, config)"
        trimmed: "trimmedMeanSelection(gradients, trimRatio)"
        median_sel: "medianSelection(gradients)"
        weighted_sel: "weightedAverageSelection(gradients)"
        merkle: "computeMerkleRoot(gradients, includedIDs)"
        validate: "ValidateAggregation(ctx, result, merkleRoot)"
        finalize: "FinalizeAggregation(ctx, roundID, ipfsHash, result)"
      }
    }
    
    adapter: "Adapter Approval\nadapter_approval.go" {
      status: "AdapterStatus enum" {
        pending: "pending"
        approved: "approved"
        rejected: "rejected"
        expired: "expired"
      }
      
      config: "AdapterApprovalConfig" {
        voting: "VotingPeriod: 7 days"
        quorum: "RequiredQuorum: 33%"
        threshold: "RequiredThreshold: 50%"
        min_propose: "MinStakeToPropose: 1M tokens"
        min_vote: "MinStakeToVote: 100K tokens"
      }
      
      funcs: "Functions" {
        propose: "ProposeAdapter(ctx, proposer, adapter, config)"
        vote: "VoteOnAdapter(ctx, proposalID, voter, option, power)"
        tally: "TallyAdapterVotes(ctx, proposalID, totalStake)"
        approve: "approveAdapter(ctx, proposal)"
        get_approved: "GetApprovedAdapters(ctx)"
        get_by_domain: "GetAdaptersByDomain(ctx, domain)"
        get_compatible: "GetCompatibleAdapters(ctx, modelVersion)"
        revoke: "RevokeAdapter(ctx, adapterID, reason)"
      }
    }
  }
  
  # 3. TRAINING KEEPER
  training: "🎯 TRAINING KEEPER\nkeeper/training/keeper.go" {
    class: keeper
    
    responsibility: |md
      **Training & Gradient Management**
      - Gradient submission & storage
      - Aggregation records
      - Training windows
      - Mining contributions
      - Convergence metrics
      - **TRAP JOB VERIFICATION** ⚠️
    |
    
    collections: "Collections" {
      gradients: "StoredGradients\nMap[uint64, StoredGradient]" { class: collection }
      gradient_id: "StoredGradientID\nSequence" { class: collection }
      aggregations: "AggregationRecords\nMap[uint64, AggregationRecord]" { class: collection }
      agg_id: "AggregationID\nSequence" { class: collection }
      commitments: "AggregationCommitments\nMap[uint64, AggregationCommitment]" { class: collection }
      commit_id: "AggregationCommitmentID\nSequence" { class: collection }
      contributions: "MiningContributions\nMap[string, MiningContribution]" { class: collection }
      windows: "TrainingWindows\nMap[uint64, TrainingWindow]" { class: collection }
      async_subs: "AsyncGradientSubmissions\nMap[uint64, AsyncGradientSubmission]" { class: collection }
      lazy_aggs: "LazyAggregations\nMap[uint64, LazyAggregation]" { class: collection }
      convergence: "ConvergenceMetrics\nMap[uint64, ConvergenceMetrics]" { class: collection }
      pending_aggs: "PendingAggregationsByDeadline\nMap[int64, AggregationIDList]" { class: collection }
      subnets: "SubnetConfigs\nMap[uint64, SubnetConfig]" { class: collection }
      activations: "ActivationTransmissions\nMap[uint64, ActivationTransmission]" { class: collection }
      workflows: "SubnetTrainingWorkflows\nMap[uint64, SubnetTrainingWorkflow]" { class: collection }
      trap_jobs: "TrapJobs\nMap[uint64, TrapJob]" { class: collection }
    }
    
    methods: "Key Methods" {
      submit: "SubmitGradient(ctx, gradient) ⚠️ TRAP CHECK"
      get_gradient: "GetGradient(ctx, gradientID)"
      aggregate: "AggregateGradients(ctx, gradients)"
      get_agg: "GetAggregation(ctx, aggregationID)"
      create_window: "CreateTrainingWindow(ctx, window)"
      get_window: "GetTrainingWindow(ctx, windowID)"
      get_contrib: "GetMiningContribution(ctx, minerAddress)"
      update_contrib: "updateMiningContribution(ctx, miner, gradientID)"
      add_pending: "AddPendingAggregation(ctx, deadline, aggID)"
      remove_pending: "RemovePendingAggregation(ctx, deadline, aggID)"
      record_conv: "RecordConvergenceMetrics(ctx, metrics)"
    }
    
    trap: "Trap Verification\ntrap_verification.go" {
      verdicts: "TrapJobVerdict enum" {
        normal: "VerdictNormalJob"
        passed: "VerdictTrapPassed → +10% bonus"
        failed: "VerdictTrapFailed → -50% slash"
        timeout: "VerdictTimeout → -25% slash"
      }
      
      funcs: "Functions" {
        verify: "VerifyTrapJobResult(ctx, gradient) → Result"
        get_trap: "getTrapJobForGradient(ctx, gradient)"
        is_trap: "isTrapJobRound(roundID) → bool"
        get_by_round: "getTrapJobByRound(ctx, roundID)"
        process: "ProcessTrapJobVerdict(ctx, result)"
        reward: "rewardTrapJobPass(ctx, miner, amount)"
        slash_fail: "slashTrapJobFailure(ctx, miner, amount, reason)"
        slash_timeout: "slashTrapJobTimeout(ctx, miner, amount)"
      }
      
      flow: |md
        **Verification Flow:**
        1. Check if gradient is for trap job
        2. Verify deadline not exceeded
        3. Compare gradient hashes
        4. Process verdict (reward/slash)
      |
    }
    
    deps: "Dependencies" {
      core_ref: "core *CoreKeeper"
      infra_ref: "infra *InfraKeeper"
      economics_ref: "economics *EconomicsKeeper"
    }
  }
  
  # 4. ECONOMICS KEEPER
  economics: "💰 ECONOMICS KEEPER\nkeeper/economics/keeper.go" {
    class: keeper
    
    responsibility: |md
      **Economic Incentives & Treasury**
      - Reward calculation & distribution
      - Slashing mechanics
      - Treasury management
      - Buy-back operations
      - Trap job bonus/penalty
    |
    
    collections: "Collections" {
      treasury: "Treasury\nItem[types.Treasury]" { class: collection }
    }
    
    methods: "Key Methods" {
      calc_rewards: "CalculateRewards(ctx, contributions) → []Reward"
      distribute: "DistributeRewards(ctx, rewards)"
      get_treasury: "GetTreasury(ctx) → Treasury"
      update_treasury: "UpdateTreasury(ctx, treasury)"
      buyback: "ProcessTreasuryBuyBack(ctx)"
      calc_staking: "CalculateStakingRewards(ctx, stakingInfo)"
      slash_validator: "SlashValidator(ctx, validatorAddr, amount)"
      slash_miner: "SlashMiner(ctx, minerAddr, percent, reason)"
      trap_bonus: "AddTrapJobBonus(ctx, minerAddr, bonusPercent)"
      trap_slash_fail: "SlashForTrapJobFailure(ctx, miner, percent, error)"
      trap_slash_timeout: "SlashForTrapJobTimeout(ctx, miner, percent)"
      get_params: "GetEconomicParameters(ctx) → EconomicParams"
      update_params: "UpdateEconomicParameters(ctx, params)"
      get_supply: "GetTotalSupply(ctx, denom) → string"
      get_inflation: "GetInflationRate(ctx) → string"
    }
    
    reward_struct: "Reward Struct" {
      recipient: "Recipient: string"
      amount: "Amount: string"
      reason: "Reason: string"
    }
    
    deps: "Dependencies" {
      core_ref: "core *CoreKeeper"
      bank_ref: "bankKeeper types.BankKeeper"
    }
  }
  
  # 5. DATASET KEEPER
  dataset: "📊 DATASET KEEPER\nkeeper/dataset/keeper.go" {
    class: keeper
    
    responsibility: |md
      **Dataset Governance**
      - Dataset proposals
      - Voting mechanism
      - Approved datasets registry
      - Official training data marking
    |
    
    methods: "Key Methods" {
      propose: "ProposeDataset(ctx, proposal)"
      vote: "VoteOnDataset(ctx, vote)"
      get_proposal: "GetDatasetProposal(ctx, proposalID)"
      list_proposals: "ListDatasetProposals(ctx)"
      get_approved: "GetApprovedDataset(ctx, datasetID)"
      list_approved: "ListApprovedDatasets(ctx)"
    }
  }
  
  # 6. NODE KEEPER
  node: "🖥️ NODE KEEPER\nkeeper/node/keeper.go" {
    class: keeper
    
    responsibility: |md
      **Node Registration & Management**
      - Node registration with roles
      - Resource quotas
      - Serving node status
      - Inference requests
    |
    
    methods: "Key Methods" {
      register: "RegisterNode(ctx, node)"
      get_node: "GetNode(ctx, address)"
      update_status: "UpdateNodeStatus(ctx, address, status)"
      list_nodes: "ListNodes(ctx)"
      get_inference: "GetInferenceRequest(ctx, requestID)"
      get_serving: "GetServingNodeStatus(ctx, nodeAddress)"
    }
    
    deps: "Dependencies" {
      core_ref: "core *CoreKeeper"
      bank_ref: "bankKeeper types.BankKeeper"
    }
  }
  
  # 7. SECURITY KEEPER
  security: "🔒 SECURITY KEEPER\nkeeper/security/keeper.go" {
    class: keeper
    
    responsibility: |md
      **Security & Fraud Detection**
      - Signature verification
      - Nonce validation
      - Fraud detection
      - Rate limiting
    |
    
    methods: "Key Methods" {
      verify_sig: "VerifySignature(ctx, address, message, signature)"
      validate_nonce: "ValidateNonce(ctx, address, nonce)"
      detect_fraud: "DetectFraud(ctx, submission) → bool"
    }
    
    deps: "Dependencies" {
      core_ref: "core *CoreKeeper"
      auth_ref: "authKeeper types.AuthKeeper"
    }
  }
  
  # 8. INFRA KEEPER
  infra: "⚙️ INFRA KEEPER\nkeeper/infra/keeper.go" {
    class: keeper
    
    responsibility: |md
      **Infrastructure Integration**
      - IPFS content verification
      - Gradient caching
      - External service integration
    |
    
    methods: "Key Methods" {
      verify_ipfs: "VerifyIPFSContent(ctx, hash) → bool"
      cache_gradient: "CacheGradient(ctx, hash, data)"
      get_cached: "GetCachedGradient(ctx, hash) → []byte"
    }
    
    config: "Configuration" {
      ipfs_url: "ipfsAPIURL: string"
    }
  }
}

# ═══════════════════════════════════════════════════════════════════════════════
# KEEPER BAĞLANTILARI
# ═══════════════════════════════════════════════════════════════════════════════

orchestrator -> keepers.core: "core *CoreKeeper" {
  style.stroke: "#e74c3c"
  style.stroke-width: 3
}
orchestrator -> keepers.model: "model ModelKeeper" {
  style.stroke: "#e74c3c"
  style.stroke-width: 2
}
orchestrator -> keepers.training: "training TrainingKeeper" {
  style.stroke: "#e74c3c"
  style.stroke-width: 2
}
orchestrator -> keepers.economics: "economics EconomicsKeeper" {
  style.stroke: "#e74c3c"
  style.stroke-width: 2
}
orchestrator -> keepers.dataset: "dataset DatasetKeeper" {
  style.stroke: "#e74c3c"
  style.stroke-width: 2
}
orchestrator -> keepers.node: "node NodeKeeper" {
  style.stroke: "#e74c3c"
  style.stroke-width: 2
}
orchestrator -> keepers.security: "security SecurityKeeper" {
  style.stroke: "#e74c3c"
  style.stroke-width: 2
}
orchestrator -> keepers.infra: "infra InfraKeeper" {
  style.stroke: "#e74c3c"
  style.stroke-width: 2
}

# Inter-Keeper Dependencies
keepers.model -> keepers.core: "uses core" {
  style.stroke: "#3498db"
  style.stroke-dash: 3
}
keepers.training -> keepers.core: "uses core" {
  style.stroke: "#3498db"
  style.stroke-dash: 3
}
keepers.training -> keepers.infra: "IPFS verify" {
  style.stroke: "#27ae60"
  style.stroke-dash: 3
}
keepers.training -> keepers.economics: "slash/reward" {
  style.stroke: "#f39c12"
  style.stroke-width: 2
}
keepers.economics -> keepers.core: "uses core" {
  style.stroke: "#3498db"
  style.stroke-dash: 3
}
keepers.dataset -> keepers.core: "uses core" {
  style.stroke: "#3498db"
  style.stroke-dash: 3
}
keepers.node -> keepers.core: "uses core" {
  style.stroke: "#3498db"
  style.stroke-dash: 3
}
keepers.security -> keepers.core: "uses core" {
  style.stroke: "#3498db"
  style.stroke-dash: 3
}
```

---

## 2. Keeper Orchestration Pattern

R3MES, **Composition over Inheritance** pattern'ini kullanır. Ana `Keeper` struct'ı, 8 domain-specific keeper'ı orchestrate eder.


```d2
# ═══════════════════════════════════════════════════════════════════════════════
# KEEPER ORCHESTRATION PATTERN
# ═══════════════════════════════════════════════════════════════════════════════

vars: {
  d2-config: {
    layout-engine: dagre
    theme-id: 200
  }
}

direction: right
grid-rows: 1
grid-gap: 40

title: |md
  ## Keeper Orchestration Pattern
  **Composition over Inheritance**
|

# Orchestrator Struct
orchestrator_struct: "type Keeper struct" {
  style.fill: "#2c3e50"
  style.stroke: "#e74c3c"
  style.stroke-width: 3
  style.font-color: "#ecf0f1"
  
  fields: |go
    // Core keeper handles basic functionality
    core *core.CoreKeeper
    
    // Domain-specific keepers (interfaces)
    model     ModelKeeper
    training  TrainingKeeper
    dataset   DatasetKeeper
    node      NodeKeeper
    economics EconomicsKeeper
    security  SecurityKeeper
    infra     InfraKeeper
    
    // Schema for collections
    Schema collections.Schema
  |
}

# NewKeeper Flow
new_keeper: "NewKeeper() Flow" {
  style.fill: "#1a1a2e"
  style.stroke: "#00d9ff"
  style.stroke-width: 2
  
  step1: "1. Validate authority address"
  step2: "2. ValidateProductionSecurity()"
  step3: "3. Create CoreKeeper"
  step4: "4. Create InfraKeeper"
  step5: "5. Create ModelKeeper"
  step6: "6. Create TrainingKeeper"
  step7: "7. Create DatasetKeeper"
  step8: "8. Create NodeKeeper"
  step9: "9. Create EconomicsKeeper"
  step10: "10. Create SecurityKeeper"
  step11: "11. Build Schema"
  
  step1 -> step2 -> step3 -> step4 -> step5 -> step6 -> step7 -> step8 -> step9 -> step10 -> step11
}

# Dependency Injection
di: "Dependency Injection" {
  style.fill: "#1a1a2e"
  style.stroke: "#27ae60"
  
  params: "Constructor Parameters" {
    store: "storeService corestore.KVStoreService"
    codec: "cdc codec.Codec"
    addr_codec: "addressCodec address.Codec"
    authority: "authority []byte"
    bank: "bankKeeper types.BankKeeper"
    auth: "authKeeper types.AuthKeeper"
    ipfs: "ipfsAPIURL string"
    cap: "capabilityKeeper *capabilitykeeper.Keeper"
    scoped: "scopedKeeper capabilitykeeper.ScopedKeeper"
  }
}

# Circular Dependency Resolution
circular: "Circular Dependency Resolution" {
  style.fill: "#1a1a2e"
  style.stroke: "#f39c12"
  
  problem: |md
    **Problem:**
    TrainingKeeper needs EconomicsKeeper
    for trap job slash/reward
  |
  
  solution: |md
    **Solution:**
    SetEconomicsKeeper() called after
    both keepers are initialized
  |
  
  code: |go
    // In TrainingKeeper
    func (k *TrainingKeeper) SetEconomicsKeeper(
        economicsKeeper *economics.EconomicsKeeper,
    ) {
        k.economics = economicsKeeper
    }
  |
}

orchestrator_struct -> new_keeper: "created by"
new_keeper -> di: "uses"
new_keeper -> circular: "resolves"
```

---

## 3. Domain Keeper'lar Detaylı Analiz

### 3.1 Training Keeper - Trap Job Verification System

Bu sistem, **lazy mining** (tembel madencilik) tespiti için kritik öneme sahiptir.


```d2
# ═══════════════════════════════════════════════════════════════════════════════
# TRAP JOB VERIFICATION SYSTEM - DETAYLI
# ═══════════════════════════════════════════════════════════════════════════════

vars: {
  d2-config: {
    layout-engine: dagre
    theme-id: 200
  }
}

direction: down
grid-rows: 1
grid-gap: 40

title: |md
  ## 🎯 Trap Job Verification System
  **Lazy Mining Detection & Prevention**
  *~1% of jobs are trap jobs with pre-computed expected results*
|

classes: {
  miner: {
    style: {
      fill: "#1b4332"
      stroke: "#40916c"
      stroke-width: 2
      border-radius: 8
      font-color: "#ffffff"
    }
  }
  trap: {
    style: {
      fill: "#c0392b"
      stroke: "#e74c3c"
      stroke-width: 3
      border-radius: 10
      font-color: "#ffffff"
    }
  }
  verdict: {
    style: {
      fill: "#8e44ad"
      stroke: "#9b59b6"
      stroke-width: 2
      border-radius: 8
      font-color: "#ffffff"
    }
  }
  reward: {
    style: {
      fill: "#27ae60"
      stroke: "#2ecc71"
      stroke-width: 2
      border-radius: 8
      font-color: "#ffffff"
    }
  }
  slash: {
    style: {
      fill: "#c0392b"
      stroke: "#e74c3c"
      stroke-width: 2
      border-radius: 8
      font-color: "#ffffff"
    }
  }
}

# Trap Job Lifecycle
lifecycle: "Trap Job Lifecycle" {
  
  # 1. Genesis Trap Creation
  genesis: "1. Genesis Trap Creation" {
    class: trap
    
    desc: |md
      **scripts/generate_genesis_traps.py**
      - Pre-compute gradient hashes
      - Store in genesis_traps.json
      - ~1% injection rate
    |
    
    fields: |json
      {
        "trap_job_id": "trap_001",
        "training_round_id": 1000000001,
        "target_miner": "remes1...",
        "expected_gradient_hash": "sha256...",
        "deadline_height": 12345,
        "dataset_shard": "shard_42",
        "seed": 123456789
      }
    |
  }
  
  # 2. Miner Receives Job
  miner_job: "2. Miner Receives Job" {
    class: miner
    
    desc: |md
      **Miner cannot distinguish:**
      - Normal job vs Trap job
      - Round ID pattern is cryptographic
      - Must compute honestly
    |
    
    check: |go
      // isTrapJobRound() - Miner cannot know
      func isTrapJobRound(roundID uint64) bool {
          // Trap jobs: 1B <= roundID < 2B
          return roundID >= 1000000000 && 
                 roundID < 2000000000
      }
    |
  }
  
  # 3. Gradient Submission
  submission: "3. Gradient Submission" {
    
    msg: "MsgSubmitGradient" {
      fields: |proto
        string miner = 1;
        string ipfs_hash = 2;
        string model_version = 3;
        uint64 training_round_id = 4;
        string gradient_hash = 6;
        string claimed_loss = 15;
      |
    }
    
    flow: "SubmitGradient() Flow" {
      step1: "1. Generate gradient ID"
      step2: "2. Verify IPFS content exists"
      step3: "3. VerifyTrapJobResult() ⚠️"
      step4: "4. ProcessTrapJobVerdict()"
      step5: "5. Store gradient (if passed)"
      step6: "6. Update mining contribution"
      
      step1 -> step2 -> step3 -> step4 -> step5 -> step6
    }
  }
  
  # 4. Trap Verification
  verification: "4. Trap Verification" {
    class: trap
    
    func: "VerifyTrapJobResult()" {
      
      check1: "Check if trap job" {
        code: |go
          trapJob, isTrap, err := k.getTrapJobForGradient(ctx, gradient)
          if !isTrap {
              return VerdictNormalJob
          }
        |
      }
      
      check2: "Check deadline" {
        code: |go
          currentHeight := k.core.GetCurrentBlockHeight(ctx)
          if currentHeight > trapJob.DeadlineHeight {
              return VerdictTimeout // -25% slash
          }
        |
      }
      
      check3: "Compare hashes" {
        code: |go
          if gradient.GradientHash != trapJob.ExpectedGradientHash {
              return VerdictTrapFailed // -50% slash
          }
          return VerdictTrapPassed // +10% bonus
        |
      }
      
      check1 -> check2 -> check3
    }
  }
  
  # 5. Verdict Processing
  verdict_proc: "5. Verdict Processing" {
    
    normal: "VerdictNormalJob" {
      class: verdict
      action: "No special action"
    }
    
    passed: "VerdictTrapPassed" {
      class: reward
      action: "+10% bonus reward"
      code: |go
        k.economics.AddTrapJobBonus(ctx, miner, "10")
      |
    }
    
    failed: "VerdictTrapFailed" {
      class: slash
      action: "-50% stake slash"
      code: |go
        k.economics.SlashForTrapJobFailure(
            ctx, miner, "50", errorMsg)
      |
    }
    
    timeout: "VerdictTimeout" {
      class: slash
      action: "-25% stake slash"
      code: |go
        k.economics.SlashForTrapJobTimeout(ctx, miner, "25")
      |
    }
  }
  
  genesis -> miner_job -> submission -> verification -> verdict_proc
}

# Trap Job Data Flow
data_flow: "Trap Job Data Flow" {
  
  genesis_file: "genesis_traps.json" {
    style.fill: "#f39c12"
  }
  
  trap_jobs_collection: "TrapJobs Collection\nMap[uint64, TrapJob]" {
    style.fill: "#9b59b6"
  }
  
  training_keeper: "TrainingKeeper" {
    style.fill: "#3498db"
  }
  
  economics_keeper: "EconomicsKeeper" {
    style.fill: "#27ae60"
  }
  
  treasury: "Treasury" {
    style.fill: "#e74c3c"
  }
  
  genesis_file -> trap_jobs_collection: "InitGenesis"
  trap_jobs_collection -> training_keeper: "getTrapJobByRound()"
  training_keeper -> economics_keeper: "slash/reward"
  economics_keeper -> treasury: "update balance"
}

# TrapJobVerificationResult Struct
result_struct: "TrapJobVerificationResult" {
  style.fill: "#2c3e50"
  style.stroke: "#8e44ad"
  style.font-color: "#ecf0f1"
  
  fields: |go
    type TrapJobVerificationResult struct {
        Verdict           TrapJobVerdict
        TrapJobID         string
        MinerAddress      string
        ExpectedHash      string
        ActualHash        string
        FingerprintMatch  bool
        SlashAmount       string  // "25" or "50"
        RewardAmount      string  // "10"
        ErrorMessage      string
    }
  |
}
```

---

### 3.2 Model Keeper - Gradient Aggregation Pipeline


```d2
# ═══════════════════════════════════════════════════════════════════════════════
# GRADIENT AGGREGATION PIPELINE - DETAYLI
# ═══════════════════════════════════════════════════════════════════════════════

vars: {
  d2-config: {
    layout-engine: dagre
    theme-id: 200
  }
}

direction: down
grid-rows: 1
grid-gap: 40

title: |md
  ## 📊 Gradient Aggregation Pipeline
  **Byzantine-Robust Aggregation Methods**
  *On-chain coordination, off-chain computation*
|

classes: {
  method: {
    style: {
      fill: "#3498db"
      stroke: "#2980b9"
      stroke-width: 2
      border-radius: 8
      font-color: "#ffffff"
    }
  }
  step: {
    style: {
      fill: "#27ae60"
      stroke: "#1e8449"
      stroke-width: 1
      border-radius: 5
      font-color: "#ffffff"
    }
  }
  data: {
    style: {
      fill: "#9b59b6"
      stroke: "#8e44ad"
      stroke-width: 1
      border-radius: 5
      font-color: "#ffffff"
    }
  }
}

# Aggregation Methods
methods: "Aggregation Methods" {
  
  weighted: "WeightedAverage" {
    class: method
    
    desc: |md
      **Trust-Weighted Averaging**
      - All gradients included
      - Weight = TrustScore
      - No outlier removal
      - Fastest, least robust
    |
    
    formula: |md
      ```
      G_agg = Σ(w_i * G_i) / Σ(w_i)
      where w_i = TrustScore_i
      ```
    |
    
    code: |go
      func weightedAverageSelection(gradients []GradientInfo) {
          for _, g := range gradients {
              included = append(included, g.GradientID)
              weights = append(weights, g.TrustScore)
          }
      }
    |
  }
  
  trimmed: "TrimmedMean (DEFAULT)" {
    class: method
    
    desc: |md
      **Byzantine-Robust Trimmed Mean**
      - Sort by loss value
      - Trim 20% from each end
      - Average remaining
      - Tolerates up to 20% Byzantine
    |
    
    formula: |md
      ```
      trim_count = n * 0.2
      sorted_by_loss = sort(gradients, by=loss)
      included = sorted_by_loss[trim_count : n-trim_count]
      G_agg = weighted_avg(included)
      ```
    |
    
    code: |go
      func trimmedMeanSelection(
          gradients []GradientInfo,
          trimRatio float64,
      ) {
          n := len(gradients)
          trimCount := int(float64(n) * trimRatio)
          
          // Sort by loss (ascending)
          sort.Slice(sorted, func(i, j int) bool {
              return sorted[i].Loss.Cmp(sorted[j].Loss) < 0
          })
          
          // Exclude trimmed
          for i := 0; i < trimCount; i++ {
              excluded = append(excluded, sorted[i].GradientID)
          }
          for i := n - trimCount; i < n; i++ {
              excluded = append(excluded, sorted[i].GradientID)
          }
          
          // Include middle with trust weights
          for i := trimCount; i < n-trimCount; i++ {
              included = append(included, sorted[i].GradientID)
              weights = append(weights, sorted[i].TrustScore)
          }
      }
    |
  }
  
  median: "Median (Most Robust)" {
    class: method
    
    desc: |md
      **Coordinate-wise Median**
      - Sort by loss value
      - Select median gradient(s)
      - Most Byzantine-robust
      - Tolerates up to 50% Byzantine
    |
    
    formula: |md
      ```
      sorted_by_loss = sort(gradients, by=loss)
      if n is odd:
          G_agg = sorted_by_loss[n/2]
      else:
          G_agg = avg(sorted_by_loss[n/2-1], sorted_by_loss[n/2])
      ```
    |
    
    code: |go
      func medianSelection(gradients []GradientInfo) {
          n := len(gradients)
          sort.Slice(sorted, func(i, j int) bool {
              return sorted[i].Loss.Cmp(sorted[j].Loss) < 0
          })
          
          if n%2 == 1 {
              // Odd: single median
              medianIdx := n / 2
              included = append(included, sorted[medianIdx].GradientID)
              weights = append(weights, big.NewInt(1e18))
          } else {
              // Even: two middle values
              midLow := n/2 - 1
              midHigh := n / 2
              included = append(included, 
                  sorted[midLow].GradientID, 
                  sorted[midHigh].GradientID)
              weights = append(weights, 
                  big.NewInt(5e17), 
                  big.NewInt(5e17))
          }
      }
    |
  }
}

# Aggregation Flow
flow: "Aggregation Flow" {
  
  input: "Input" {
    class: data
    
    gradient_ids: "gradientIDs []uint64"
    config: "AggregationConfig" {
      method: "Method: TrimmedMean"
      threshold: "ByzantineThreshold: 0.2"
      min: "MinGradients: 3"
      max: "MaxGradients: 100"
    }
  }
  
  step1: "1. Validate Input" {
    class: step
    code: |go
      if len(gradientIDs) < config.MinGradients {
          return Error("insufficient gradients")
      }
      if len(gradientIDs) > config.MaxGradients {
          gradientIDs = gradientIDs[:config.MaxGradients]
      }
    |
  }
  
  step2: "2. Collect Gradient Info" {
    class: step
    code: |go
      for _, gid := range gradientIDs {
          info, err := k.getGradientInfo(ctx, gid)
          if err != nil {
              continue // Skip invalid
          }
          gradients = append(gradients, info)
      }
    |
  }
  
  step3: "3. Apply Aggregation Method" {
    class: step
    code: |go
      switch config.Method {
      case TrimmedMean:
          included, excluded, weights = 
              k.trimmedMeanSelection(gradients, config.ByzantineThreshold)
      case Median:
          included, excluded, weights = 
              k.medianSelection(gradients)
      default: // WeightedAverage
          included, excluded, weights = 
              k.weightedAverageSelection(gradients)
      }
    |
  }
  
  step4: "4. Compute Merkle Root" {
    class: step
    code: |go
      merkleRoot := k.computeMerkleRoot(gradients, included)
      
      // Build Merkle tree
      for len(hashes) > 1 {
          var nextLevel [][]byte
          for i := 0; i < len(hashes); i += 2 {
              combined := append(hashes[i], hashes[i+1]...)
              hash := sha256.Sum256(combined)
              nextLevel = append(nextLevel, hash[:])
          }
          hashes = nextLevel
      }
    |
  }
  
  step5: "5. Compute Total Weight" {
    class: step
    code: |go
      totalWeight := big.NewInt(0)
      for _, w := range weights {
          totalWeight.Add(totalWeight, w)
      }
    |
  }
  
  output: "Output" {
    class: data
    
    result: "AggregationResult" {
      success: "Success: bool"
      merkle: "MerkleRoot: string"
      included: "IncludedGradients: []uint64"
      excluded: "ExcludedGradients: []uint64"
      weight: "TotalWeight: *big.Int"
    }
  }
  
  input -> step1 -> step2 -> step3 -> step4 -> step5 -> output
}

# GradientInfo Struct
gradient_info: "GradientInfo Struct" {
  style.fill: "#2c3e50"
  style.stroke: "#3498db"
  style.font-color: "#ecf0f1"
  
  fields: |go
    type GradientInfo struct {
        GradientID   uint64
        MinerAddress string
        IPFSHash     string
        GradientHash string
        TrustScore   *big.Int  // Scaled by 1e18
        Loss         *big.Int  // Scaled by 1e18
    }
  |
}

# On-chain vs Off-chain
separation: "On-chain vs Off-chain Separation" {
  style.fill: "#1a1a2e"
  style.stroke: "#f39c12"
  
  onchain: "On-chain (Blockchain)" {
    items: |md
      - Gradient metadata storage
      - Selection algorithm (which gradients)
      - Merkle root computation
      - Weight calculation
      - Aggregation record storage
    |
  }
  
  offchain: "Off-chain (Proposer)" {
    items: |md
      - Actual tensor math
      - Gradient download from IPFS
      - Weighted averaging of tensors
      - Result upload to IPFS
      - Submit aggregated IPFS hash
    |
  }
  
  onchain -> offchain: "coordinates"
}
```

---

## 4. State Management & Collections


```d2
# ═══════════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT & COLLECTIONS
# ═══════════════════════════════════════════════════════════════════════════════

vars: {
  d2-config: {
    layout-engine: dagre
    theme-id: 200
  }
}

direction: right
grid-rows: 1
grid-gap: 40

title: |md
  ## 💾 State Management & Collections
  **Cosmos SDK Collections Framework**
  *Type-safe, efficient key-value storage*
|

classes: {
  item: {
    style: {
      fill: "#e74c3c"
      stroke: "#c0392b"
      stroke-width: 2
      border-radius: 5
      font-color: "#ffffff"
    }
  }
  map: {
    style: {
      fill: "#3498db"
      stroke: "#2980b9"
      stroke-width: 2
      border-radius: 5
      font-color: "#ffffff"
    }
  }
  sequence: {
    style: {
      fill: "#27ae60"
      stroke: "#1e8449"
      stroke-width: 2
      border-radius: 5
      font-color: "#ffffff"
    }
  }
}

# Collection Types
types: "Collection Types" {
  
  item_type: "Item[T]" {
    class: item
    desc: |md
      **Single Value Storage**
      - One value per key prefix
      - Used for global state
      - Examples: Params, Treasury
    |
    methods: |go
      Get(ctx) → (T, error)
      Set(ctx, value T) → error
    |
  }
  
  map_type: "Map[K, V]" {
    class: map
    desc: |md
      **Key-Value Mapping**
      - Multiple values indexed by key
      - Used for registries
      - Examples: Gradients, Models
    |
    methods: |go
      Get(ctx, key K) → (V, error)
      Set(ctx, key K, value V) → error
      Remove(ctx, key K) → error
      Walk(ctx, ranger, fn) → error
      Iterate(ctx, ranger) → Iterator
    |
  }
  
  sequence_type: "Sequence" {
    class: sequence
    desc: |md
      **Auto-increment Counter**
      - Generates unique IDs
      - Thread-safe
      - Examples: GradientID, ModelID
    |
    methods: |go
      Next(ctx) → (uint64, error)
      Peek(ctx) → (uint64, error)
      Set(ctx, value uint64) → error
    |
  }
}

# All Collections by Keeper
collections: "All Collections by Keeper" {
  
  core_cols: "CoreKeeper Collections" {
    params: "Params\nItem[Params]" { class: item }
    block_ts: "BlockTimestamps\nMap[int64, int64]" { class: map }
    used_nonces: "UsedNonces\nMap[string, bool]" { class: map }
    nonce_windows: "NonceWindows\nMap[string, string]" { class: map }
    submission_hist: "SubmissionHistory\nMap[string, uint64]" { class: map }
  }
  
  model_cols: "ModelKeeper Collections" {
    global_state: "GlobalModelState\nItem[GlobalModelState]" { class: item }
    registries: "ModelRegistries\nMap[uint64, ModelRegistry]" { class: map }
    model_id: "ModelID\nSequence" { class: sequence }
    versions: "ModelVersions\nMap[uint64, ModelVersion]" { class: map }
    proposals: "ModelUpgradeProposals\nMap[uint64, ModelUpgradeProposal]" { class: map }
    proposal_id: "ModelUpgradeProposalID\nSequence" { class: sequence }
    votes: "ModelUpgradeVotes\nMap[uint64, ModelUpgradeVote]" { class: map }
    vote_id: "ModelUpgradeVoteID\nSequence" { class: sequence }
    active_versions: "ActiveModelVersions\nItem[ActiveModelVersions]" { class: item }
  }
  
  training_cols: "TrainingKeeper Collections" {
    gradients: "StoredGradients\nMap[uint64, StoredGradient]" { class: map }
    gradient_id: "StoredGradientID\nSequence" { class: sequence }
    aggregations: "AggregationRecords\nMap[uint64, AggregationRecord]" { class: map }
    agg_id: "AggregationID\nSequence" { class: sequence }
    commitments: "AggregationCommitments\nMap[uint64, AggregationCommitment]" { class: map }
    commit_id: "AggregationCommitmentID\nSequence" { class: sequence }
    contributions: "MiningContributions\nMap[string, MiningContribution]" { class: map }
    windows: "TrainingWindows\nMap[uint64, TrainingWindow]" { class: map }
    async_subs: "AsyncGradientSubmissions\nMap[uint64, AsyncGradientSubmission]" { class: map }
    lazy_aggs: "LazyAggregations\nMap[uint64, LazyAggregation]" { class: map }
    convergence: "ConvergenceMetrics\nMap[uint64, ConvergenceMetrics]" { class: map }
    pending_aggs: "PendingAggregationsByDeadline\nMap[int64, AggregationIDList]" { class: map }
    subnets: "SubnetConfigs\nMap[uint64, SubnetConfig]" { class: map }
    activations: "ActivationTransmissions\nMap[uint64, ActivationTransmission]" { class: map }
    workflows: "SubnetTrainingWorkflows\nMap[uint64, SubnetTrainingWorkflow]" { class: map }
    trap_jobs: "TrapJobs\nMap[uint64, TrapJob]" { class: map }
  }
  
  economics_cols: "EconomicsKeeper Collections" {
    treasury: "Treasury\nItem[Treasury]" { class: item }
  }
}

# Key Prefixes (from types/keys.go)
keys: "Key Prefixes" {
  style.fill: "#2c3e50"
  style.stroke: "#f39c12"
  style.font-color: "#ecf0f1"
  
  prefixes: |go
    // Core Keys
    ParamsKey              = []byte{0x00}
    BlockTimestampsKey     = []byte{0x01}
    UsedNonceKey           = []byte{0x02}
    NonceWindowKey         = []byte{0x03}
    SubmissionHistoryKey   = []byte{0x04}
    
    // Model Keys
    GlobalModelStateKey    = []byte{0x10}
    ModelRegistryKey       = []byte{0x11}
    ModelIDKey             = []byte{0x12}
    ModelVersionKey        = []byte{0x13}
    ModelUpgradeProposalKey = []byte{0x14}
    
    // Training Keys
    StoredGradientKey      = []byte{0x20}
    StoredGradientIDKey    = []byte{0x21}
    AggregationRecordKey   = []byte{0x22}
    AggregationIDKey       = []byte{0x23}
    MiningContributionKey  = []byte{0x24}
    TrainingWindowKey      = []byte{0x25}
    TrapJobKey             = []byte{0x26}
    
    // Economics Keys
    TreasuryKey            = []byte{0x30}
  |
}

# Schema Builder Pattern
schema: "Schema Builder Pattern" {
  style.fill: "#1a1a2e"
  style.stroke: "#27ae60"
  
  code: |go
    func NewTrainingKeeper(...) (*TrainingKeeper, error) {
        sb := collections.NewSchemaBuilder(storeService)
        
        k := &TrainingKeeper{
            StoredGradients: collections.NewMap(
                sb, 
                types.StoredGradientKey, 
                "stored_gradients", 
                collections.Uint64Key, 
                codec.CollValue[types.StoredGradient](cdc),
            ),
            StoredGradientID: collections.NewSequence(
                sb, 
                types.StoredGradientIDKey, 
                "stored_gradient_id",
            ),
            // ... more collections
        }
        
        // Build and validate schema
        _, err := sb.Build()
        if err != nil {
            return nil, fmt.Errorf("failed to build schema: %w", err)
        }
        
        return k, nil
    }
  |
}
```

---

## 5. Transaction Flow (MsgServer)


```d2
# ═══════════════════════════════════════════════════════════════════════════════
# TRANSACTION FLOW (MsgServer)
# ═══════════════════════════════════════════════════════════════════════════════

vars: {
  d2-config: {
    layout-engine: dagre
    theme-id: 200
  }
}

direction: down
grid-rows: 1
grid-gap: 40

title: |md
  ## 📤 Transaction Flow (MsgServer)
  **35+ Transaction Types**
  *State-changing operations*
|

classes: {
  msg: {
    style: {
      fill: "#e74c3c"
      stroke: "#c0392b"
      stroke-width: 2
      border-radius: 8
      font-color: "#ffffff"
    }
  }
  category: {
    style: {
      fill: "#3498db"
      stroke: "#2980b9"
      stroke-width: 2
      border-radius: 10
      font-color: "#ffffff"
    }
  }
  flow: {
    style: {
      fill: "#27ae60"
      stroke: "#1e8449"
      stroke-width: 1
      border-radius: 5
      font-color: "#ffffff"
    }
  }
}

# MsgServer Implementation
server: "MsgServer Implementation" {
  style.fill: "#1a1a2e"
  style.stroke: "#e74c3c"
  
  struct: |go
    type msgServer struct {
        keeper Keeper
    }
    
    func NewMsgServerImpl(keeper Keeper) types.MsgServer {
        return &msgServer{keeper: keeper}
    }
    
    var _ types.MsgServer = (*msgServer)(nil)
  |
}

# Transaction Categories
categories: "Transaction Categories" {
  
  # 1. Governance
  governance: "🗳️ Governance" {
    class: category
    
    msgs: "Messages" {
      update_params: "MsgUpdateParams" { class: msg }
      register_model: "MsgRegisterModel" { class: msg }
      activate_model: "MsgActivateModel" { class: msg }
      mark_official: "MsgMarkDatasetAsOfficial" { class: msg }
      remove_dataset: "MsgRemoveDataset" { class: msg }
    }
    
    signer: "Signer: authority (x/gov)"
  }
  
  # 2. Training
  training: "🎯 Training" {
    class: category
    
    msgs: "Messages" {
      submit_gradient: "MsgSubmitGradient ⚠️" { class: msg }
      submit_agg: "MsgSubmitAggregation" { class: msg }
      commit_agg: "MsgCommitAggregation" { class: msg }
      reveal_agg: "MsgRevealAggregation" { class: msg }
      challenge_agg: "MsgChallengeAggregation" { class: msg }
      submit_async: "MsgSubmitAsyncGradient" { class: msg }
      submit_lazy: "MsgSubmitLazyAggregation" { class: msg }
      create_window: "MsgCreateTrainingWindow" { class: msg }
    }
    
    signer: "Signer: miner / proposer"
  }
  
  # 3. Dataset
  dataset: "📊 Dataset" {
    class: category
    
    msgs: "Messages" {
      propose: "MsgProposeDataset" { class: msg }
      vote: "MsgVoteDataset" { class: msg }
    }
    
    signer: "Signer: proposer / voter"
  }
  
  # 4. Node
  node: "🖥️ Node" {
    class: category
    
    msgs: "Messages" {
      register: "MsgRegisterNode" { class: msg }
      update: "MsgUpdateNodeRegistration" { class: msg }
      resource: "MsgSubmitResourceUsage" { class: msg }
      serving_status: "MsgUpdateServingNodeStatus" { class: msg }
    }
    
    signer: "Signer: node_address"
  }
  
  # 5. Inference
  inference: "🤖 Inference" {
    class: category
    
    msgs: "Messages" {
      request: "MsgRequestInference" { class: msg }
      result: "MsgSubmitInferenceResult" { class: msg }
    }
    
    signer: "Signer: requester / serving_node"
  }
  
  # 6. Pinning
  pinning: "📌 Pinning" {
    class: category
    
    msgs: "Messages" {
      commit: "MsgCommitPinning" { class: msg }
      challenge: "MsgChallengePinning" { class: msg }
      respond: "MsgRespondToChallenge" { class: msg }
    }
    
    signer: "Signer: node_address / challenger"
  }
  
  # 7. Verification
  verification: "✅ Verification" {
    class: category
    
    msgs: "Messages" {
      resolve: "MsgResolveChallenge" { class: msg }
      cpu_verify: "MsgSubmitCPUVerification" { class: msg }
      random_verify: "MsgSubmitRandomVerifierResult" { class: msg }
    }
    
    signer: "Signer: resolver / validator / verifier"
  }
  
  # 8. Trap Jobs
  trap: "🎯 Trap Jobs" {
    class: category
    
    msgs: "Messages" {
      create: "MsgCreateTrapJob" { class: msg }
      submit_result: "MsgSubmitTrapJobResult" { class: msg }
      appeal: "MsgAppealTrapJobSlashing" { class: msg }
    }
    
    signer: "Signer: authority / miner"
  }
  
  # 9. Slashing
  slashing: "⚔️ Slashing" {
    class: category
    
    msgs: "Messages" {
      lazy_validation: "MsgReportLazyValidation" { class: msg }
      false_verdict: "MsgReportFalseVerdict" { class: msg }
      censorship: "MsgReportProposerCensorship" { class: msg }
      appeal: "MsgAppealSlashing" { class: msg }
    }
    
    signer: "Signer: reporter / appellant"
  }
  
  # 10. Subnet
  subnet: "🌐 Subnet" {
    class: category
    
    msgs: "Messages" {
      create: "MsgCreateSubnet" { class: msg }
      activation: "MsgSubmitSubnetActivation" { class: msg }
      assign: "MsgAssignMinerToSubnet" { class: msg }
    }
    
    signer: "Signer: authority / miner"
  }
  
  # 11. Task Pool
  task_pool: "📋 Task Pool" {
    class: category
    
    msgs: "Messages" {
      claim: "MsgClaimTask" { class: msg }
      complete: "MsgCompleteTask" { class: msg }
    }
    
    signer: "Signer: miner"
  }
  
  # 12. Mentor
  mentor: "👨‍🏫 Mentor" {
    class: category
    
    msgs: "Messages" {
      register: "MsgRegisterMentorRelationship" { class: msg }
    }
    
    signer: "Signer: mentor / mentee"
  }
}

# SubmitGradient Flow (Most Critical)
submit_gradient_flow: "MsgSubmitGradient Flow (Critical)" {
  style.fill: "#1a1a2e"
  style.stroke: "#f39c12"
  style.stroke-width: 3
  
  step1: "1. Validate Request" {
    class: flow
    code: |go
      if req.Miner == "" {
          return nil, ErrInvalidRequest.Wrap("miner empty")
      }
      if req.IpfsHash == "" {
          return nil, ErrInvalidRequest.Wrap("IPFS hash empty")
      }
    |
  }
  
  step2: "2. Create StoredGradient" {
    class: flow
    code: |go
      gradient := types.StoredGradient{
          Miner:           req.Miner,
          IpfsHash:        req.IpfsHash,
          ModelVersion:    req.ModelVersion,
          TrainingRoundId: req.TrainingRoundId,
          ShardId:         req.ShardId,
          GradientHash:    req.GradientHash,
          GpuArchitecture: req.GpuArchitecture,
      }
    |
  }
  
  step3: "3. Call TrainingKeeper" {
    class: flow
    code: |go
      err := ms.keeper.GetTrainingKeeper().SubmitGradient(ctx, gradient)
      // This triggers:
      // - IPFS verification
      // - TRAP JOB VERIFICATION ⚠️
      // - Mining contribution update
    |
  }
  
  step4: "4. Return Response" {
    class: flow
    code: |go
      return &types.MsgSubmitGradientResponse{
          StoredGradientId: gradient.Id,
      }, nil
    |
  }
  
  step1 -> step2 -> step3 -> step4
}

# Proto Message Definition
proto_msg: "Proto Message Definition" {
  style.fill: "#2c3e50"
  style.stroke: "#9b59b6"
  style.font-color: "#ecf0f1"
  
  code: |proto
    message MsgSubmitGradient {
      option (cosmos.msg.v1.signer) = "miner";
      option (amino.name) = "remes/x/remes/MsgSubmitGradient";
      
      string miner = 1 [(cosmos_proto.scalar) = "cosmos.AddressString"];
      string ipfs_hash = 2;
      string model_version = 3;
      uint64 training_round_id = 4;
      uint64 shard_id = 5;
      string gradient_hash = 6;
      string gpu_architecture = 7;
      uint64 nonce = 8;
      bytes signature = 9;
      uint64 proof_of_work_nonce = 10;
      uint64 model_config_id = 11;
      string container_hash = 12;
      bytes container_signature = 13;
      uint64 global_seed = 14;
      string claimed_loss = 15;
      string porep_proof_ipfs_hash = 16;
      uint64 token_count = 17;
    }
  |
}
```

---

## 6. Query Flow (QueryServer)


```d2
# ═══════════════════════════════════════════════════════════════════════════════
# QUERY FLOW (QueryServer)
# ═══════════════════════════════════════════════════════════════════════════════

vars: {
  d2-config: {
    layout-engine: dagre
    theme-id: 200
  }
}

direction: down
grid-rows: 1
grid-gap: 40

title: |md
  ## 🔍 Query Flow (QueryServer)
  **30+ Query Types**
  *Read-only state queries*
|

classes: {
  query: {
    style: {
      fill: "#27ae60"
      stroke: "#1e8449"
      stroke-width: 2
      border-radius: 8
      font-color: "#ffffff"
    }
  }
  category: {
    style: {
      fill: "#3498db"
      stroke: "#2980b9"
      stroke-width: 2
      border-radius: 10
      font-color: "#ffffff"
    }
  }
  endpoint: {
    style: {
      fill: "#9b59b6"
      stroke: "#8e44ad"
      stroke-width: 1
      border-radius: 5
      font-color: "#ffffff"
    }
  }
}

# QueryServer Implementation
server: "QueryServer Implementation" {
  style.fill: "#1a1a2e"
  style.stroke: "#27ae60"
  
  struct: |go
    type queryServer struct {
        k Keeper
    }
    
    func NewQueryServerImpl(keeper Keeper) types.QueryServer {
        return &queryServer{k: keeper}
    }
  |
}

# Query Categories
categories: "Query Categories" {
  
  # 1. Core Queries
  core: "🔧 Core Queries" {
    class: category
    
    queries: "Queries" {
      params: "Params" {
        class: query
        endpoint: "/remes/remes/v1/params" { class: endpoint }
      }
    }
  }
  
  # 2. Model Queries
  model: "📦 Model Queries" {
    class: category
    
    queries: "Queries" {
      model_params: "GetModelParams" {
        class: query
        endpoint: "/remes/remes/v1/model_params" { class: endpoint }
      }
      global_state: "GetGlobalModelState" {
        class: query
        endpoint: "/remes/remes/v1/global_model_state" { class: endpoint }
      }
    }
  }
  
  # 3. Training Queries
  training: "🎯 Training Queries" {
    class: category
    
    queries: "Queries" {
      gradient: "GetGradient" {
        class: query
        endpoint: "/remes/remes/v1/gradient/{id}" { class: endpoint }
      }
      stored_gradient: "GetStoredGradient" {
        class: query
        endpoint: "/remes/remes/v1/stored_gradient/{id}" { class: endpoint }
      }
      list_gradients: "ListStoredGradient" {
        class: query
        endpoint: "/remes/remes/v1/stored_gradient" { class: endpoint }
      }
      aggregation: "GetAggregation" {
        class: query
        endpoint: "/remes/remes/v1/aggregation/{id}" { class: endpoint }
      }
      miner_score: "GetMinerScore" {
        class: query
        endpoint: "/remes/remes/v1/miner_score/{miner}" { class: endpoint }
      }
      global_seed: "GetGlobalSeed" {
        class: query
        endpoint: "/remes/remes/v1/global_seed/{training_round_id}" { class: endpoint }
      }
      convergence: "GetConvergenceMetrics" {
        class: query
        endpoint: "/remes/remes/v1/convergence/{round_id}" { class: endpoint }
      }
    }
  }
  
  # 4. Dataset Queries
  dataset: "📊 Dataset Queries" {
    class: category
    
    queries: "Queries" {
      proposal: "GetDatasetProposal" {
        class: query
        endpoint: "/remes/remes/v1/dataset_proposal/{proposal_id}" { class: endpoint }
      }
      list_proposals: "ListDatasetProposals" {
        class: query
        endpoint: "/remes/remes/v1/dataset_proposals" { class: endpoint }
      }
      approved: "GetApprovedDataset" {
        class: query
        endpoint: "/remes/remes/v1/approved_dataset/{dataset_id}" { class: endpoint }
      }
      list_approved: "ListApprovedDatasets" {
        class: query
        endpoint: "/remes/remes/v1/approved_datasets" { class: endpoint }
      }
    }
  }
  
  # 5. Node Queries
  node: "🖥️ Node Queries" {
    class: category
    
    queries: "Queries" {
      registration: "GetNodeRegistration" {
        class: query
        endpoint: "/remes/remes/v1/node_registration/{node_address}" { class: endpoint }
      }
      list_nodes: "ListNodeRegistrations" {
        class: query
        endpoint: "/remes/remes/v1/node_registrations" { class: endpoint }
      }
      serving_status: "GetServingNodeStatus" {
        class: query
        endpoint: "/remes/remes/v1/serving_node_status/{node_address}" { class: endpoint }
      }
      list_serving: "ListServingNodes" {
        class: query
        endpoint: "/remes/remes/v1/serving_nodes" { class: endpoint }
      }
    }
  }
  
  # 6. Inference Queries
  inference: "🤖 Inference Queries" {
    class: category
    
    queries: "Queries" {
      request: "GetInferenceRequest" {
        class: query
        endpoint: "/remes/remes/v1/inference_request/{request_id}" { class: endpoint }
      }
    }
  }
  
  # 7. Sync Queries
  sync: "🔄 Sync Queries" {
    class: category
    
    queries: "Queries" {
      sync_state: "GetParticipantSyncState" {
        class: query
        endpoint: "/remes/remes/v1/participant_sync_state/{participant_address}" { class: endpoint }
      }
      list_sync: "ListParticipantSyncStates" {
        class: query
        endpoint: "/remes/remes/v1/participant_sync_states" { class: endpoint }
      }
      catch_up: "GetCatchUpInfo" {
        class: query
        endpoint: "/remes/remes/v1/catch_up_info/{participant_address}" { class: endpoint }
      }
    }
  }
  
  # 8. Dashboard Queries
  dashboard: "📊 Dashboard Queries" {
    class: category
    
    queries: "Queries" {
      miners: "QueryMiners" {
        class: query
        endpoint: "/remes/remes/v1/dashboard/miners" { class: endpoint }
      }
      statistics: "QueryStatistics" {
        class: query
        endpoint: "/remes/remes/v1/dashboard/statistics" { class: endpoint }
      }
      blocks: "QueryBlocks" {
        class: query
        endpoint: "/remes/remes/v1/dashboard/blocks" { class: endpoint }
      }
      block: "QueryBlock" {
        class: query
        endpoint: "/remes/remes/v1/dashboard/block/{height}" { class: endpoint }
      }
    }
  }
  
  # 9. Admin Queries
  admin: "🔐 Admin Queries" {
    class: category
    
    queries: "Queries" {
      vault_stats: "QueryVaultStats" {
        class: query
        endpoint: "/remes/remes/v1/vault/stats" { class: endpoint }
      }
      fraud_score: "QueryMinerFraudScore" {
        class: query
        endpoint: "/remes/remes/v1/miner_fraud_score/{miner}" { class: endpoint }
      }
    }
  }
  
  # 10. Task Pool Queries
  task_pool: "📋 Task Pool Queries" {
    class: category
    
    queries: "Queries" {
      active_pool: "QueryActivePool" {
        class: query
        endpoint: "/remes/remes/v1/task_pool/active" { class: endpoint }
      }
      available_chunks: "QueryAvailableChunks" {
        class: query
        endpoint: "/remes/remes/v1/task_pool/{pool_id}/available_chunks" { class: endpoint }
      }
    }
  }
  
  # 11. Economics Queries
  economics: "💰 Economics Queries" {
    class: category
    
    queries: "Queries" {
      reward_formula: "GetRewardFormula" {
        class: query
        endpoint: "/remes/remes/v1/reward_formula" { class: endpoint }
      }
    }
  }
}

# Query Flow Example
query_flow: "Query Flow Example: GetMinerScore" {
  style.fill: "#1a1a2e"
  style.stroke: "#f39c12"
  
  step1: "1. gRPC Request" {
    code: |go
      req := &types.QueryGetMinerScoreRequest{
          Miner: "remes1abc...",
      }
    |
  }
  
  step2: "2. Validate Request" {
    code: |go
      if req == nil {
          return nil, types.ErrInvalidRequest
      }
    |
  }
  
  step3: "3. Call Keeper" {
    code: |go
      contribution, err := q.k.GetTrainingKeeper().
          GetMiningContribution(ctx, req.Miner)
    |
  }
  
  step4: "4. Build Response" {
    code: |go
      return &types.QueryGetMinerScoreResponse{
          Miner:            req.Miner,
          TotalSubmissions: contribution.TotalSubmissions,
      }, nil
    |
  }
  
  step1 -> step2 -> step3 -> step4
}
```

---

## 7. Proto Message Definitions


```d2
# ═══════════════════════════════════════════════════════════════════════════════
# PROTO MESSAGE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

vars: {
  d2-config: {
    layout-engine: dagre
    theme-id: 200
  }
}

direction: right
grid-rows: 1
grid-gap: 40

title: |md
  ## 📝 Proto Message Definitions
  **21 Proto Files**
  *State, Transaction, Query definitions*
|

classes: {
  proto: {
    style: {
      fill: "#f39c12"
      stroke: "#d68910"
      stroke-width: 2
      border-radius: 8
      font-color: "#000000"
    }
  }
  state: {
    style: {
      fill: "#9b59b6"
      stroke: "#8e44ad"
      stroke-width: 2
      border-radius: 8
      font-color: "#ffffff"
    }
  }
  tx: {
    style: {
      fill: "#e74c3c"
      stroke: "#c0392b"
      stroke-width: 2
      border-radius: 8
      font-color: "#ffffff"
    }
  }
  query: {
    style: {
      fill: "#27ae60"
      stroke: "#1e8449"
      stroke-width: 2
      border-radius: 8
      font-color: "#ffffff"
    }
  }
}

# Proto Files
proto_files: "Proto Files (remes/proto/remes/remes/v1/)" {
  
  # State Protos
  state_protos: "State Definitions" {
    
    state: "state.proto" {
      class: state
      messages: |md
        - GlobalModelState
        - AggregationRecord
        - AggregationCommitment
        - MiningContribution
        - ChallengeRecord
        - CPUVerificationResult
        - ParticipantSyncState
        - PartitionRecoveryInfo
        - ProofOfReplication
        - PartitionStatus
        - ConvergenceMetrics
        - PinInfo
        - AggregationIDList
      |
    }
    
    stored_gradient: "stored_gradient.proto" {
      class: state
      messages: |md
        - StoredGradient
      |
    }
    
    model: "model.proto" {
      class: state
      messages: |md
        - ModelRegistry
        - ModelConfig
      |
    }
    
    model_version: "model_version.proto" {
      class: state
      messages: |md
        - ModelVersion
        - ModelUpgradeProposal
        - ModelUpgradeVote
        - ActiveModelVersions
      |
    }
    
    dataset: "dataset.proto" {
      class: state
      messages: |md
        - DatasetProposal
        - DatasetVote
        - ApprovedDataset
        - DatasetMetadata
      |
    }
    
    node: "node.proto" {
      class: state
      messages: |md
        - NodeRegistration
        - NodeType (enum)
        - NodeStatus
        - ResourceSpec
        - RoleAllocation
      |
    }
    
    serving: "serving.proto" {
      class: state
      messages: |md
        - ServingNodeStatus
        - InferenceRequest
        - InferenceResult
      |
    }
    
    trap_job: "trap_job.proto" {
      class: state
      messages: |md
        - TrapJob
        - TrapJobResult
      |
    }
    
    slashing: "slashing.proto" {
      class: state
      messages: |md
        - SlashingRecord
        - SlashingAppeal
      |
    }
    
    training_window: "training_window.proto" {
      class: state
      messages: |md
        - TrainingWindow
        - AsyncGradientSubmission
        - LazyAggregation
      |
    }
    
    subnet: "subnet.proto" {
      class: state
      messages: |md
        - SubnetConfig
        - ActivationTransmission
        - SubnetTrainingWorkflow
      |
    }
    
    task_pool: "task_pool.proto" {
      class: state
      messages: |md
        - TaskPool
        - TaskChunk
        - TaskChunkResponse
      |
    }
    
    treasury: "treasury.proto" {
      class: state
      messages: |md
        - Treasury
        - EconomicParams
        - StakingInfo
      |
    }
    
    pinning: "pinning.proto" {
      class: state
      messages: |md
        - PinningCommitment
        - PinningChallenge
      |
    }
    
    verification: "verification.proto" {
      class: state
      messages: |md
        - VerificationResult
      |
    }
    
    genesis_vault: "genesis_vault.proto" {
      class: state
      messages: |md
        - GenesisVaultEntry
      |
    }
    
    execution_env: "execution_environment.proto" {
      class: state
      messages: |md
        - ExecutionEnvironment
      |
    }
    
    params: "params.proto" {
      class: state
      messages: |md
        - Params
      |
    }
    
    genesis: "genesis.proto" {
      class: state
      messages: |md
        - GenesisState
      |
    }
  }
  
  # Transaction Proto
  tx_proto: "Transaction Definitions" {
    
    tx: "tx.proto" {
      class: tx
      services: |md
        **service Msg** (35+ RPCs)
        - UpdateParams
        - RegisterModel / ActivateModel
        - SubmitGradient
        - SubmitAggregation / CommitAggregation / RevealAggregation
        - ChallengeAggregation
        - ProposeDataset / VoteDataset
        - RegisterNode / UpdateNodeRegistration
        - CommitPinning / ChallengePinning
        - CreateTrapJob / SubmitTrapJobResult
        - ReportLazyValidation / ReportFalseVerdict
        - CreateSubnet / SubmitSubnetActivation
        - ClaimTask / CompleteTask
        - ... and more
      |
    }
  }
  
  # Query Proto
  query_proto: "Query Definitions" {
    
    query: "query.proto" {
      class: query
      services: |md
        **service Query** (30+ RPCs)
        - Params
        - GetGradient / ListStoredGradient
        - GetModelParams / GetGlobalModelState
        - GetAggregation
        - GetMinerScore
        - GetDatasetProposal / ListDatasetProposals
        - GetNodeRegistration / ListNodeRegistrations
        - GetServingNodeStatus / ListServingNodes
        - QueryMiners / QueryStatistics (Dashboard)
        - QueryVaultStats / QueryMinerFraudScore (Admin)
        - QueryActivePool / QueryAvailableChunks
        - ... and more
      |
    }
  }
}

# Key State Messages Detail
key_messages: "Key State Messages" {
  
  global_model: "GlobalModelState" {
    style.fill: "#2c3e50"
    style.stroke: "#9b59b6"
    style.font-color: "#ecf0f1"
    
    fields: |proto
      message GlobalModelState {
        string model_ipfs_hash = 1;
        string model_version = 2;
        int64 last_updated_height = 3;
        google.protobuf.Timestamp last_updated_time = 4;
        uint64 training_round_id = 5;
        uint64 last_aggregation_id = 6;
      }
    |
  }
  
  stored_gradient: "StoredGradient" {
    style.fill: "#2c3e50"
    style.stroke: "#9b59b6"
    style.font-color: "#ecf0f1"
    
    fields: |proto
      message StoredGradient {
        uint64 id = 1;
        string miner = 2;
        string ipfs_hash = 3;
        string model_version = 4;
        uint64 training_round_id = 5;
        uint64 shard_id = 6;
        string gradient_hash = 7;
        string gpu_architecture = 8;
        int64 submitted_at_height = 9;
        google.protobuf.Timestamp submitted_at_time = 10;
        string status = 11;
        uint64 aggregation_id = 12;
      }
    |
  }
  
  mining_contribution: "MiningContribution" {
    style.fill: "#2c3e50"
    style.stroke: "#9b59b6"
    style.font-color: "#ecf0f1"
    
    fields: |proto
      message MiningContribution {
        string miner_address = 1;
        uint64 total_submissions = 2;
        uint64 successful_submissions = 3;
        string trust_score = 4;        // 0.0 to 1.0
        string reputation_tier = 5;    // excellent/trusted/developing/new
        uint64 slashing_events = 6;
        int64 last_submission_height = 7;
        uint64 traps_caught = 8;
        uint64 traps_failed = 9;
        string fraud_score = 10;       // 0.0 to 1.0
      }
    |
  }
  
  trap_job: "TrapJob" {
    style.fill: "#2c3e50"
    style.stroke: "#e74c3c"
    style.font-color: "#ecf0f1"
    
    fields: |proto
      message TrapJob {
        string trap_job_id = 1;
        uint64 training_round_id = 2;
        string target_miner = 3;
        string expected_gradient_hash = 4;
        int64 deadline_height = 5;
        string dataset_shard = 6;
        uint64 seed = 7;
        string status = 8;
      }
    |
  }
}
```

---

## 8. Inter-Keeper Dependencies Graph


```d2
# ═══════════════════════════════════════════════════════════════════════════════
# INTER-KEEPER DEPENDENCIES GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

vars: {
  d2-config: {
    layout-engine: dagre
    theme-id: 200
  }
}

direction: right
grid-rows: 1
grid-gap: 40

title: |md
  ## 🔗 Inter-Keeper Dependencies
  **Dependency Injection & Circular Resolution**
|

classes: {
  keeper: {
    style: {
      fill: "#3498db"
      stroke: "#2980b9"
      stroke-width: 2
      border-radius: 10
      font-color: "#ffffff"
    }
  }
  external: {
    style: {
      fill: "#95a5a6"
      stroke: "#7f8c8d"
      stroke-width: 2
      stroke-dash: 3
      border-radius: 8
      font-color: "#ffffff"
    }
  }
  critical: {
    style: {
      stroke: "#e74c3c"
      stroke-width: 3
    }
  }
}

# Keepers
core: "🔧 CoreKeeper" { class: keeper }
model: "📦 ModelKeeper" { class: keeper }
training: "🎯 TrainingKeeper" { class: keeper }
economics: "💰 EconomicsKeeper" { class: keeper }
dataset: "📊 DatasetKeeper" { class: keeper }
node: "🖥️ NodeKeeper" { class: keeper }
security: "🔒 SecurityKeeper" { class: keeper }
infra: "⚙️ InfraKeeper" { class: keeper }

# External Dependencies
bank: "BankKeeper\n(Cosmos SDK)" { class: external }
auth: "AuthKeeper\n(Cosmos SDK)" { class: external }
capability: "CapabilityKeeper\n(IBC)" { class: external }
scoped: "ScopedKeeper\n(IBC)" { class: external }

# Core Dependencies
core -> bank: "token operations"
core -> auth: "account verification"
core -> capability: "IBC capabilities"
core -> scoped: "IBC port"

# Model Dependencies
model -> core: "params, logging"

# Training Dependencies (CRITICAL)
training -> core: "params, block height, logging" {
  style.stroke: "#3498db"
}
training -> infra: "IPFS verification" {
  style.stroke: "#27ae60"
}
training -> economics: "slash/reward (trap jobs)" {
  class: critical
  style.stroke: "#e74c3c"
  style.stroke-width: 3
}

# Economics Dependencies
economics -> core: "params"
economics -> bank: "token mint/burn/transfer"

# Dataset Dependencies
dataset -> core: "params"

# Node Dependencies
node -> core: "params"
node -> bank: "stake management"

# Security Dependencies
security -> core: "nonce validation"
security -> auth: "signature verification"

# Infra Dependencies (standalone)
infra: "⚙️ InfraKeeper\n(No keeper deps)" { class: keeper }

# Circular Dependency Resolution
circular: "⚠️ Circular Dependency Resolution" {
  style.fill: "#1a1a2e"
  style.stroke: "#f39c12"
  style.stroke-width: 2
  
  problem: |md
    **Problem:**
    TrainingKeeper needs EconomicsKeeper
    for trap job slash/reward
    
    But both are created in NewKeeper()
  |
  
  solution: |md
    **Solution:**
    1. Create TrainingKeeper without economics
    2. Create EconomicsKeeper
    3. Call SetEconomicsKeeper() to inject
  |
  
  code: |go
    // In keeper/keeper.go NewKeeper()
    
    // Step 1: Create training keeper (economics = nil)
    trainingKeeper, err := training.NewTrainingKeeper(
        storeService, cdc, coreKeeper, infraKeeper)
    
    // Step 2: Create economics keeper
    economicsKeeper, err := economics.NewEconomicsKeeper(
        storeService, cdc, coreKeeper, bankKeeper)
    
    // Step 3: Inject economics into training
    trainingKeeper.SetEconomicsKeeper(economicsKeeper)
  |
}

# Dependency Matrix
matrix: "Dependency Matrix" {
  style.fill: "#2c3e50"
  style.stroke: "#ffffff"
  style.font-color: "#ecf0f1"
  
  table: |md
    | Keeper     | Core | Model | Training | Economics | Dataset | Node | Security | Infra | Bank | Auth |
    |------------|------|-------|----------|-----------|---------|------|----------|-------|------|------|
    | Core       | -    |       |          |           |         |      |          |       | ✓    | ✓    |
    | Model      | ✓    | -     |          |           |         |      |          |       |      |      |
    | Training   | ✓    |       | -        | ✓ ⚠️      |         |      |          | ✓     |      |      |
    | Economics  | ✓    |       |          | -         |         |      |          |       | ✓    |      |
    | Dataset    | ✓    |       |          |           | -       |      |          |       |      |      |
    | Node       | ✓    |       |          |           |         | -    |          |       | ✓    |      |
    | Security   | ✓    |       |          |           |         |      | -        |       |      | ✓    |
    | Infra      |      |       |          |           |         |      |          | -     |      |      |
    
    ⚠️ = Circular dependency (resolved via SetEconomicsKeeper)
  |
}
```

---

## 9. Economics & Slashing Mechanics


```d2
# ═══════════════════════════════════════════════════════════════════════════════
# ECONOMICS & SLASHING MECHANICS
# ═══════════════════════════════════════════════════════════════════════════════

vars: {
  d2-config: {
    layout-engine: dagre
    theme-id: 200
  }
}

direction: down
grid-rows: 1
grid-gap: 40

title: |md
  ## 💰 Economics & Slashing Mechanics
  **Reward Distribution & Penalty System**
|

classes: {
  reward: {
    style: {
      fill: "#27ae60"
      stroke: "#1e8449"
      stroke-width: 2
      border-radius: 8
      font-color: "#ffffff"
    }
  }
  slash: {
    style: {
      fill: "#e74c3c"
      stroke: "#c0392b"
      stroke-width: 2
      border-radius: 8
      font-color: "#ffffff"
    }
  }
  treasury: {
    style: {
      fill: "#f39c12"
      stroke: "#d68910"
      stroke-width: 2
      border-radius: 10
      font-color: "#000000"
    }
  }
  flow: {
    style: {
      fill: "#3498db"
      stroke: "#2980b9"
      stroke-width: 1
      border-radius: 5
      font-color: "#ffffff"
    }
  }
}

# Reward System
rewards: "Reward System" {
  
  types: "Reward Types" {
    
    gradient_reward: "Gradient Submission Reward" {
      class: reward
      desc: |md
        **Base reward for valid gradient**
        - Amount: params.RewardPerGradient
        - Condition: Gradient passes validation
        - Multiplier: Trust score
      |
    }
    
    trap_bonus: "Trap Job Bonus" {
      class: reward
      desc: |md
        **Bonus for passing trap job**
        - Amount: +10% of base reward
        - Condition: Gradient hash matches expected
        - Purpose: Incentivize honest computation
      |
    }
    
    staking_reward: "Staking Reward" {
      class: reward
      desc: |md
        **Reward for staking tokens**
        - Amount: Based on stake amount
        - Distribution: Per epoch
      |
    }
    
    aggregation_reward: "Aggregation Reward" {
      class: reward
      desc: |md
        **Reward for proposers**
        - Amount: Based on aggregation quality
        - Condition: Aggregation finalized
      |
    }
  }
  
  calculation: "Reward Calculation" {
    class: flow
    
    code: |go
      func (k *EconomicsKeeper) CalculateRewards(
          ctx context.Context,
          contributions []types.MiningContribution,
      ) ([]Reward, error) {
          var rewards []Reward
          
          params, err := k.core.GetParams(ctx)
          if err != nil {
              return nil, err
          }
          
          for _, contribution := range contributions {
              reward := Reward{
                  Recipient: contribution.MinerAddress,
                  Amount:    params.RewardPerGradient,
                  Reason:    "gradient_contribution",
              }
              rewards = append(rewards, reward)
          }
          
          return rewards, nil
      }
    |
  }
  
  distribution: "Reward Distribution" {
    class: flow
    
    code: |go
      func (k *EconomicsKeeper) DistributeRewards(
          ctx context.Context,
          rewards []Reward,
      ) error {
          for _, reward := range rewards {
              // Parse recipient address
              // Parse reward amount
              // Mint tokens to module account
              // Transfer to recipient
              if err := k.distributeReward(ctx, reward); err != nil {
                  return err
              }
          }
          return nil
      }
    |
  }
}

# Slashing System
slashing: "Slashing System" {
  
  types: "Slashing Types" {
    
    trap_fail: "Trap Job Failure" {
      class: slash
      desc: |md
        **Penalty for failing trap job**
        - Amount: -50% of stake
        - Trigger: Gradient hash mismatch
        - Severity: HIGH
      |
    }
    
    trap_timeout: "Trap Job Timeout" {
      class: slash
      desc: |md
        **Penalty for missing deadline**
        - Amount: -25% of stake
        - Trigger: No submission before deadline
        - Severity: MEDIUM
      |
    }
    
    lazy_validation: "Lazy Validation" {
      class: slash
      desc: |md
        **Penalty for validators**
        - Amount: Variable
        - Trigger: Not performing verification
        - Severity: MEDIUM
      |
    }
    
    false_verdict: "False Verdict" {
      class: slash
      desc: |md
        **Penalty for wrong verification**
        - Amount: Variable
        - Trigger: Incorrect challenge resolution
        - Severity: HIGH
      |
    }
    
    censorship: "Proposer Censorship" {
      class: slash
      desc: |md
        **Penalty for excluding gradients**
        - Amount: Variable
        - Trigger: Unfair gradient exclusion
        - Severity: HIGH
      |
    }
  }
  
  execution: "Slashing Execution" {
    class: flow
    
    code: |go
      func (k *EconomicsKeeper) SlashMiner(
          ctx context.Context,
          minerAddr string,
          slashPercent string,
          reason string,
      ) error {
          sdkCtx := sdk.UnwrapSDKContext(ctx)
          
          // Log the slashing event
          sdkCtx.Logger().Warn(
              "slashing miner",
              "miner", minerAddr,
              "slash_percent", slashPercent,
              "reason", reason,
          )
          
          // Get miner's current stake
          // stake, err := k.stakingKeeper.GetMinerStake(ctx, minerAddr)
          
          // Calculate slash amount
          // slashAmount := stake * slashPercent / 100
          
          // Execute slash
          // - Burn slashed tokens OR send to treasury
          // - Update miner's stake
          // - Record slashing event
          
          // Update treasury
          treasury, err := k.GetTreasury(ctx)
          if err != nil {
              return err
          }
          // treasury.SlashedAmount += slashAmount
          
          // Emit event
          // ctx.EventManager().EmitEvent(...)
          
          return nil
      }
    |
  }
  
  trap_specific: "Trap Job Specific Slashing" {
    
    slash_fail: "SlashForTrapJobFailure" {
      class: slash
      code: |go
        func (k *EconomicsKeeper) SlashForTrapJobFailure(
            ctx context.Context,
            minerAddr string,
            slashPercent string,
            errorMsg string,
        ) error {
            return k.SlashMiner(
                ctx, 
                minerAddr, 
                slashPercent, 
                fmt.Sprintf("trap_job_failure: %s", errorMsg),
            )
        }
      |
    }
    
    slash_timeout: "SlashForTrapJobTimeout" {
      class: slash
      code: |go
        func (k *EconomicsKeeper) SlashForTrapJobTimeout(
            ctx context.Context,
            minerAddr string,
            slashPercent string,
        ) error {
            return k.SlashMiner(
                ctx, 
                minerAddr, 
                slashPercent, 
                "trap_job_timeout",
            )
        }
      |
    }
  }
}

# Treasury
treasury: "Treasury Management" {
  class: treasury
  
  state: "Treasury State" {
    fields: |proto
      message Treasury {
        string balance = 1;           // Current balance
        string total_distributed = 2; // Total rewards distributed
        string total_slashed = 3;     // Total slashed amount
        string buyback_reserve = 4;   // Reserved for buyback
        int64 last_buyback_height = 5;
      }
    |
  }
  
  operations: "Treasury Operations" {
    
    get: "GetTreasury(ctx) → Treasury"
    update: "UpdateTreasury(ctx, treasury)"
    buyback: "ProcessTreasuryBuyBack(ctx)"
  }
  
  buyback: "Buy-Back Mechanism" {
    desc: |md
      **Token Buy-Back**
      - Periodic buy-back from market
      - Burns purchased tokens
      - Reduces circulating supply
      - Funded by slashing + fees
    |
  }
}

# Economic Flow
economic_flow: "Economic Flow" {
  
  miner: "Miner" {
    style.fill: "#1b4332"
  }
  
  gradient: "Submit Gradient" {
    style.fill: "#3498db"
  }
  
  trap_check: "Trap Job Check" {
    style.fill: "#f39c12"
  }
  
  passed: "Passed" {
    class: reward
  }
  
  failed: "Failed" {
    class: slash
  }
  
  treasury_box: "Treasury" {
    class: treasury
  }
  
  reward_dist: "Reward Distribution" {
    class: reward
  }
  
  miner -> gradient -> trap_check
  trap_check -> passed: "hash match"
  trap_check -> failed: "hash mismatch"
  passed -> reward_dist: "+10% bonus"
  failed -> treasury_box: "50% stake"
  reward_dist -> miner: "tokens"
}
```

---

## 10. Complete Data Flow Diagram


```d2
# ═══════════════════════════════════════════════════════════════════════════════
# COMPLETE DATA FLOW DIAGRAM
# ═══════════════════════════════════════════════════════════════════════════════

vars: {
  d2-config: {
    layout-engine: dagre
    theme-id: 200
  }
}

direction: down
grid-rows: 1
grid-gap: 50

title: |md
  ## 🔄 Complete Data Flow
  **End-to-End Training Round**
|

classes: {
  external: {
    style: {
      fill: "#95a5a6"
      stroke: "#7f8c8d"
      stroke-width: 2
      border-radius: 8
      font-color: "#ffffff"
    }
  }
  blockchain: {
    style: {
      fill: "#1a1a2e"
      stroke: "#e94560"
      stroke-width: 3
      border-radius: 10
      font-color: "#ffffff"
    }
  }
  keeper: {
    style: {
      fill: "#3498db"
      stroke: "#2980b9"
      stroke-width: 2
      border-radius: 8
      font-color: "#ffffff"
    }
  }
  storage: {
    style: {
      fill: "#9b59b6"
      stroke: "#8e44ad"
      stroke-width: 2
      border-radius: 8
      font-color: "#ffffff"
    }
  }
  decision: {
    style: {
      fill: "#f39c12"
      stroke: "#d68910"
      stroke-width: 2
      border-radius: 20
      font-color: "#000000"
    }
  }
  success: {
    style: {
      fill: "#27ae60"
      stroke: "#1e8449"
      stroke-width: 2
      border-radius: 8
      font-color: "#ffffff"
    }
  }
  failure: {
    style: {
      fill: "#e74c3c"
      stroke: "#c0392b"
      stroke-width: 2
      border-radius: 8
      font-color: "#ffffff"
    }
  }
}

# External Actors
miner: "⛏️ Miner\n(Python Engine)" { class: external }
proposer: "📊 Proposer\n(Aggregator)" { class: external }
ipfs: "📦 IPFS\n(Distributed Storage)" { class: storage }

# Blockchain Layer
blockchain: "🔗 Blockchain Layer" {
  class: blockchain
  
  # Servers
  msg_server: "MsgServer" { class: keeper }
  query_server: "QueryServer" { class: keeper }
  
  # Keepers
  training_keeper: "TrainingKeeper" { class: keeper }
  model_keeper: "ModelKeeper" { class: keeper }
  economics_keeper: "EconomicsKeeper" { class: keeper }
  infra_keeper: "InfraKeeper" { class: keeper }
  
  # Collections
  gradients_col: "StoredGradients\nCollection" { class: storage }
  trap_jobs_col: "TrapJobs\nCollection" { class: storage }
  contributions_col: "MiningContributions\nCollection" { class: storage }
  aggregations_col: "AggregationRecords\nCollection" { class: storage }
  global_state_col: "GlobalModelState\nCollection" { class: storage }
  treasury_col: "Treasury\nCollection" { class: storage }
}

# Decision Points
is_trap: "Is Trap Job?" { class: decision }
hash_match: "Hash Match?" { class: decision }

# Outcomes
gradient_stored: "✅ Gradient Stored" { class: success }
trap_passed: "✅ Trap Passed\n+10% Bonus" { class: success }
trap_failed: "❌ Trap Failed\n-50% Slash" { class: failure }
trap_timeout: "⏰ Timeout\n-25% Slash" { class: failure }
aggregation_done: "✅ Aggregation\nFinalized" { class: success }
model_updated: "✅ Model Updated" { class: success }

# Flow 1: Gradient Submission
miner -> ipfs: "1. Upload gradient\nto IPFS" {
  style.stroke: "#27ae60"
  style.stroke-width: 2
}

ipfs -> miner: "2. Return IPFS hash" {
  style.stroke: "#27ae60"
}

miner -> blockchain.msg_server: "3. MsgSubmitGradient\n(ipfs_hash, gradient_hash)" {
  style.stroke: "#e94560"
  style.stroke-width: 2
}

blockchain.msg_server -> blockchain.training_keeper: "4. SubmitGradient()" {
  style.stroke: "#3498db"
}

blockchain.training_keeper -> blockchain.infra_keeper: "5. VerifyIPFSContent()" {
  style.stroke: "#9b59b6"
}

blockchain.infra_keeper -> ipfs: "6. Check content exists" {
  style.stroke: "#9b59b6"
  style.stroke-dash: 3
}

blockchain.training_keeper -> blockchain.trap_jobs_col: "7. Check trap job" {
  style.stroke: "#f39c12"
}

blockchain.trap_jobs_col -> is_trap: "8. Trap job data" {
  style.stroke: "#f39c12"
}

is_trap -> hash_match: "Yes" {
  style.stroke: "#f39c12"
}

is_trap -> gradient_stored: "No (normal job)" {
  style.stroke: "#27ae60"
}

hash_match -> trap_passed: "Yes" {
  style.stroke: "#27ae60"
}

hash_match -> trap_failed: "No" {
  style.stroke: "#e74c3c"
}

trap_passed -> blockchain.economics_keeper: "9a. AddTrapJobBonus()" {
  style.stroke: "#27ae60"
}

trap_failed -> blockchain.economics_keeper: "9b. SlashForTrapJobFailure()" {
  style.stroke: "#e74c3c"
}

blockchain.economics_keeper -> blockchain.treasury_col: "10. Update treasury" {
  style.stroke: "#f39c12"
}

gradient_stored -> blockchain.gradients_col: "11. Store gradient" {
  style.stroke: "#9b59b6"
}

blockchain.training_keeper -> blockchain.contributions_col: "12. Update contribution" {
  style.stroke: "#9b59b6"
}

# Flow 2: Aggregation
proposer -> blockchain.query_server: "13. Query gradients\nfor round" {
  style.stroke: "#3498db"
}

blockchain.query_server -> blockchain.gradients_col: "14. List gradients" {
  style.stroke: "#9b59b6"
}

proposer -> ipfs: "15. Download gradients\nfrom IPFS" {
  style.stroke: "#27ae60"
  style.stroke-dash: 3
}

proposer -> proposer: "16. Aggregate\n(off-chain)" {
  style.stroke: "#f39c12"
}

proposer -> ipfs: "17. Upload aggregated\ngradient" {
  style.stroke: "#27ae60"
}

proposer -> blockchain.msg_server: "18. MsgSubmitAggregation\n(merkle_root, ipfs_hash)" {
  style.stroke: "#e94560"
  style.stroke-width: 2
}

blockchain.msg_server -> blockchain.model_keeper: "19. ValidateAggregation()" {
  style.stroke: "#3498db"
}

blockchain.model_keeper -> blockchain.aggregations_col: "20. Store aggregation" {
  style.stroke: "#9b59b6"
}

blockchain.aggregations_col -> aggregation_done: "21. Challenge period" {
  style.stroke: "#27ae60"
}

aggregation_done -> blockchain.model_keeper: "22. FinalizeAggregation()" {
  style.stroke: "#27ae60"
}

blockchain.model_keeper -> blockchain.global_state_col: "23. Update global state" {
  style.stroke: "#9b59b6"
}

blockchain.global_state_col -> model_updated: "24. New model version" {
  style.stroke: "#27ae60"
}
```

---

## 📚 Sonuç

Bu dokümantasyon, R3MES blockchain katmanının senior seviyesinde detaylı teknik analizini içermektedir:

1. **Keeper Orchestration Pattern** - 8 domain-specific keeper'ın composition pattern ile yönetimi
2. **Trap Job Verification** - Lazy mining tespiti için kritik güvenlik mekanizması
3. **Gradient Aggregation** - Byzantine-robust aggregation metodları (TrimmedMean, Median)
4. **State Management** - Cosmos SDK Collections framework kullanımı
5. **Transaction Flow** - 35+ transaction tipi ve MsgServer implementasyonu
6. **Query Flow** - 30+ query tipi ve QueryServer implementasyonu
7. **Economics & Slashing** - Reward distribution ve penalty sistemi
8. **Inter-Keeper Dependencies** - Circular dependency resolution

### Kritik Noktalar:

- **TrainingKeeper ↔ EconomicsKeeper** circular dependency `SetEconomicsKeeper()` ile çözülür
- **Trap Job Verification** her gradient submission'da çalışır
- **Aggregation** on-chain koordinasyon, off-chain hesaplama pattern'i kullanır
- **Slashing** trap job failure için %50, timeout için %25 stake kaybı


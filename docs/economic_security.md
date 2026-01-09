# Economic Security and Slashing Protocol

## 1. Overview
This document defines the economic penalties (slashing), incentive alignment, and security mechanisms for the R3MES protocol. It ensures that malicious behavior is mathematically unprofitable and that the network remains resistant to "Lazy Worker" and "Sybil" attacks.

The protocol utilizes Cosmos SDK's `x/slashing` and `x/staking` modules, extended with custom logic for Proof-of-Useful-Work (PoUW) verification.

**Version 2.0 Update**: The protocol now implements **Optimistic Verification** with a three-layer verification system that optimizes for speed while maintaining security. This replaces the previous CPU Iron Sandbox bottleneck with a fast GPU-to-GPU default path and high-stakes dispute resolution.

## 2. Global Parameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `UnbondingTime` | 21 Days | Time required to withdraw staked tokens. Prevents "Long Range Attacks". |
| `MaxValidators` | 100 (Start) | Maximum number of active validators in the active set. |
| `EquivocationPenalty` | 50% | Penalty for double-signing (consensus violation). |
| `DowntimeJailDuration` | 10 Minutes | Temporary ban for liveness faults. |
| `DoubleSignJailDuration` | Permanent | Permanent ban for consensus safety faults. |

---

## 3. Miner İtibar Sistemi ve Staking Maliyeti (Reputation System and Staking Costs)

R3MES protokolü, miner'ların geçmiş performanslarına göre dinamik staking gereksinimleri uygular. Yüksek itibar skorlu miner'lar daha düşük staking maliyeti ile çalışabilirken, yeni veya düşük performanslı miner'lar daha yüksek staking gereksinimi ile karşılaşır.

### 3.1. İtibar Tabanlı Staking Maliyeti Ayarlaması

```go
// Base staking requirement
const BASE_STAKING_REQUIREMENT = 10000 // 10,000 R3MES tokens

// Staking maliyeti hesaplama (itibar skoruna göre)
func CalculateRequiredStake(trustScore float64, baseStake sdk.Coin) sdk.Coin {
    tier := GetReputationTier(trustScore)
    
    var stakeMultiplier float64
    
    switch tier {
    case "excellent": // Trust Score >= 0.9
        // Yüksek itibar: %50 staking indirimi
        stakeMultiplier = 0.5
        
    case "trusted": // Trust Score >= 0.75
        // Güvenilir miner: %30 staking indirimi
        stakeMultiplier = 0.7
        
    case "developing": // Trust Score >= 0.5
        // Gelişen miner: Normal staking
        stakeMultiplier = 1.0
        
    case "new": // Trust Score < 0.5
        // Yeni miner: Mentor system ile indirim (onboarding support)
        stakeMultiplier = 1.2 // Reduced from 1.5 to 1.2 (20% increase instead of 50%)
        
    default:
        stakeMultiplier = 1.0
    }
    
    adjustedStake := baseStake.Mul(sdk.NewDecFromFloat64(stakeMultiplier))
    
    // Minimum stake limiti (güvenlik için)
    minStake := baseStake.Mul(sdk.NewDecFromFloat64(0.3)) // Minimum %30
    if adjustedStake.IsLT(minStake) {
        adjustedStake = minStake
    }
    
    return adjustedStake
}
```

### 3.2. İtibar Tabanlı Spot-Check Sıklığı

Yüksek itibar skorlu miner'lar için spot-check (rastgele denetim) sıklığı %80 oranında azalır:

- **Excellent Tier (Trust Score ≥ 0.9)**: Base spot-check rate'in %20'si (%80 azalma)
- **Trusted Tier (Trust Score ≥ 0.75)**: Base spot-check rate'in %50'si (%50 azalma)
- **Developing Tier (Trust Score ≥ 0.5)**: Normal base spot-check rate
- **New Tier (Trust Score < 0.5)**: Base spot-check rate'in 2 katı (artırılmış denetim)

Bu mekanizma güvenilir miner'ların işlem maliyetlerini düşürür ve ağ performansını artırır.

### 3.3. Slashing Sonrası İtibar Sıfırlama

Herhangi bir slashing olayı miner'ın itibar skorunu anında 0.0'a düşürür ve staking gereksinimini maksimum seviyeye çıkarır:

```go
func HandleSlashingEvent(minerAddress string, slashType string) {
    reputation := GetMinerReputation(minerAddress)
    
    // İtibar sıfırlama
    reputation.TrustScore = 0.0
    reputation.ReputationTier = "new"
    reputation.SlashingEvents++
    
    // Staking gereksinimini maksimuma çıkar
    currentStake := GetMinerStake(minerAddress)
    requiredStake := CalculateRequiredStake(0.0, BASE_STAKING_REQUIREMENT)
    
    // Eğer mevcut stake yetersizse, slashing uygulanır
    if currentStake.IsLT(requiredStake) {
        // Ek stake gereksinimi (mevcut stake'in üzerine ekleme)
        additionalStake := requiredStake.Sub(currentStake)
        // Miner'ın ek stake yatırması veya ağdan çıkması gerekir
        TriggerStakeTopUp(minerAddress, additionalStake)
    }
    
    SaveMinerReputation(reputation)
}
```

## 4. Optimistic Verification: Three-Layer Verification System

**Version 2.0 Update**: The protocol implements a **three-layer verification system** that optimizes for speed while maintaining security. This replaces the previous CPU Iron Sandbox bottleneck with an optimistic fast path.

### 4.0. Three-Layer Verification Flow

The protocol uses a tiered verification approach:

**Layer 1: GPU-to-GPU Verification (Optimistic - Default)**
- **Speed**: ~1 block (fast path)
- **Cost**: Minimal (hash comparison)
- **Slashing**: None on acceptance (optimistic)
- **Success Rate**: ~95% of cases
- **Trigger**: All gradient submissions go through Layer 1 first

**Layer 2: High-Stakes Challenge (Dispute Resolution)**
- **Speed**: ~5 blocks (verifier response time)
- **Cost**: Challenger must bond 10x base reward (e.g., 5,000 R3MES tokens)
- **Slashing**: None until Layer 3 confirms fault
- **Trigger**: If Layer 1 hash mismatch AND challenger disputes
- **Random Verifier**: VRF-based selection, stake-weighted
- **Bond Distribution**:
  - If challenge valid: Challenger gets bond back + fraud detection bounty (10-20x base reward)
  - If challenge invalid: Challenger loses bond (distributed to miner + validator)

**Layer 3: CPU Iron Sandbox (Final Arbiter)**
- **Speed**: ~50 blocks (CPU computation)
- **Cost**: High (CPU computation, validator panel)
- **Slashing**: Only if CPU verification confirms fault
- **Trigger**: Only if Layer 2 consensus supports challenge (random verifier agrees with challenger)
- **Panel**: 3-validator panel (VRF selection, 2/3 consensus required)
- **Result**: Final and binding

### 4.0.1. Economic Security Parameters (Optimistic Verification)

| Parameter | Layer 1 | Layer 2 | Layer 3 |
|-----------|---------|---------|---------|
| **Default Path** | Yes | No | No |
| **Bond Required** | None | 10x base reward | None (Layer 2 bond used) |
| **Slashing on Acceptance** | None | None | None |
| **Slashing on Fault** | None* | None* | 5% (hash mismatch) |
| **DoS Protection** | Rate limiting | High bond | Panel consensus |
| **Speed** | ~1 block | ~5 blocks | ~50 blocks |

*Slashing only occurs after Layer 3 confirms fault

### 4.0.2. Updated Slashing Conditions

**Previous System**: CPU Iron Sandbox was required for all hash mismatches (bottleneck).

**New System**: 
- **Layer 1**: Optimistic acceptance (no slashing)
- **Layer 2**: High-stakes challenge prevents frivolous disputes
- **Layer 3**: CPU fallback only for genuine disputes (Layer 2 consensus required)

This reduces DoS risk while maintaining security through economic disincentives.

---

## 4. Miner Slashing Conditions (PoUW Layer)

Miners are penalized based on the quality and availability of their gradient computations. Slashing penalties are adjusted based on miner reputation (trust score) to incentivize long-term honest behavior.

**Note**: With optimistic verification, slashing only occurs after Layer 3 (CPU Iron Sandbox) confirms fault. Layer 1 and Layer 2 do not trigger slashing directly.

### 4.1. Gradient Hash Mismatch (Level 1 - Deterministic Verification Failure)
Occurs when a miner's submitted gradient hash does not match the validator's re-computation using deterministic quantization.

* **Condition:** `miner_gradient_hash != validator_gradient_hash` (exact hash matching required)
* **GPU Architecture Handling:**
  - **Same Architecture**: Direct hash comparison - mismatch triggers CPU verification
  - **Different Architectures** (e.g., Ampere vs Ada): MANDATORY CPU Iron Sandbox verification (no slashing until CPU verification completes)
  - **Floating-Point Precision**: GPU architecture differences (RTX 3090 Ampere vs RTX 4090 Ada) can cause 0.0000001-level differences in CUDA kernels
  - **CPU Verification**: Cross-architecture disputes ALWAYS resolved via CPU mode (bit-exact determinism)
* **Verification Flow (Optimistic)**:
  1. **Layer 1**: GPU-to-GPU hash comparison (optimistic acceptance if match)
  2. **Layer 2**: If mismatch, challenger must bond 10x base reward to dispute
  3. **Layer 3**: CPU Iron Sandbox only if Layer 2 random verifier agrees with challenger
* **Base Slash Fraction:** **5%** of staked tokens (only after Layer 3 CPU verification confirms fault).
* **İtibar Tabanlı Ayarlama:**
  - **Excellent/Trusted Tier**: %3 (hafifletilmiş ceza - güvenilir miner'lara indirim)
  - **Developing Tier**: %5 (normal ceza)
  - **New Tier**: %7 (artırılmış ceza - yeni miner'lara daha sıkı)
* **Jail Status:** None (First offense). 24-hour Jail if repeated > 3 times in an Epoch.
* **CPU Iron Sandbox:** Only triggered if Layer 2 consensus supports challenge (prevents DoS)
* **Rationale:** Optimistic verification enables fast path (~95% of cases) while maintaining security through high-stakes challenges. CPU fallback only for genuine disputes. İtibar skoruna göre ayarlanmış cezalar, uzun vadeli dürüst davranışı teşvik eder.

### 4.2. Availability Fault (Level 2 - Liveness Fault)

Occurs when a miner fails to provide requested IPFS data within the protocol timeout, indicating data withholding or unavailability.

**Two Types of Availability Faults:**

#### 4.2.1. Gradient Reveal Timeout
Occurs when a selected miner fails to reveal their committed gradient within the protocol timeout.

* **Condition:** `Timeout > 3 blocks` after Commit phase.
* **Base Slash Fraction:** **1%** of staked tokens.
* **İtibar Tabanlı Ayarlama:**
  - **Excellent Tier**: %0.5 (hafifletilmiş ceza)
  - **Trusted Tier**: %0.7 (hafifletilmiş ceza)
  - **Developing Tier**: %1.0 (normal ceza)
  - **New Tier**: %1.5 (artırılmış ceza)
* **Jail Status:** 1-hour Jail (Suspension).
* **Rationale:** Ensures training pipelines do not stall due to offline miners. Yüksek itibar skorlu miner'lar geçmiş performansları sayesinde hafifletilmiş ceza alırlar.

#### 4.2.2. Data Availability (DA) Integrity Fault
Occurs when a miner fails to provide requested IPFS gradient data within the dynamic timeout after a data availability challenge, proving they do not actually store the data despite submitting an IPFS hash.

* **Condition:** `DataAvailabilityChallenge` issued AND `IPFS data not provided within dynamic timeout`
* **Dynamic Timeout Calculation:**
```go
// Dynamic timeout based on network conditions
func CalculateDATimeout(networkLoad float64, avgBlockTime time.Duration) int64 {
    baseTimeout := int64(3) // 3 blocks minimum
    
    // Adjust based on CometBFT network metrics
    if networkLoad > 0.8 { // High network congestion
        return baseTimeout + 2 // 5 blocks total
    } else if networkLoad > 0.5 { // Medium congestion
        return baseTimeout + 1 // 4 blocks total
    }
    
    return baseTimeout // 3 blocks for normal conditions
}
```
* **Detection:** Validator or challenger requests IPFS content using on-chain hash, miner fails to serve data within dynamic timeout
* **Base Slash Fraction:** **2%** of staked tokens (higher than reveal timeout, as this indicates intentional data withholding).
* **İtibar Tabanlı Ayarlama:**
  - **Excellent Tier**: %1.0 (hafifletilmiş ceza)
  - **Trusted Tier**: %1.5 (hafifletilmiş ceza)
  - **Developing Tier**: %2.0 (normal ceza)
  - **New Tier**: %3.0 (artırılmış ceza - yeni miner'lara daha sıkı)
* **Jail Status:** 2-hour Jail (Suspension).
* **Network Latency Protection:** Timeout automatically adjusts to prevent false slashing during network congestion
* **Additional Penalties:**
  - Invalidated gradient submission (no rewards)
  - Reputation score penalty (trust score reduction)
  - Proof of Replication requirement for future submissions
* **Rationale:** Prevents data withholding attacks while protecting against false positives from network latency. Dynamic timeout ensures fairness during varying network conditions.

### 4.3. Lazy Mining / Noise Injection (Level 3 - Critical Security Fault)
Occurs when a miner submits statistical noise (Gaussian/Uniform) instead of computed gradients, detected by "Trap Jobs" or huge statistical deviation (> 6σ).

* **Condition:** `detect_noise_entropy(gradient) == True` OR `TrapJob_Failure`
* **Slash Fraction:** **50%** (Severe penalty) - İtibar skorundan bağımsız.
* **İtibar Etkisi:** İtibar skoru anında 0.0'a düşer, miner "new" kategorisine geri gönderilir.
* **Jail Status:** **30-day Jail** (Temporary ban with appeal mechanism).
* **Appeal Mechanism:** Miner can submit governance proposal with CPU Iron Sandbox re-verification
* **False Positive Protection:** Automatic CPU verification panel review for disputed cases
* **Rationale:** Severe penalty for model poisoning attacks, but allows for false positive correction through governance appeals and CPU verification. Balances security with fairness to prevent legitimate miners from permanent exclusion due to GPU architecture differences.

---

## 5. İtibar Tabanlı Ödül Mekanizmaları (Reputation-Based Reward Mechanisms)

Yüksek itibar skorlu miner'lar sadece daha düşük staking maliyeti ve daha az denetimle değil, aynı zamanda artırılmış ödüllerle de ödüllendirilirler:

### 5.1. İtibar Bonuslu Mining Ödülleri

```go
// İtibar tabanlı ödül çarpanı
func calculateReputationBonus(trustScore float64) float64 {
    tier := GetReputationTier(trustScore)
    
    switch tier {
    case "excellent": // Trust Score >= 0.9
        // Yüksek itibar: %15 ödül bonusu
        return 1.15
        
    case "trusted": // Trust Score >= 0.75
        // Güvenilir miner: %10 ödül bonusu
        return 1.10
        
    case "developing": // Trust Score >= 0.5
        // Normal ödül (bonus yok)
        return 1.0
        
    case "new": // Trust Score < 0.5
        // Yeni miner: Normal ödül (bonus yok, ceza da yok)
        return 1.0
        
    default:
        return 1.0
    }
}
```

**Ödül Artışları:**
- **Excellent Tier (Trust Score ≥ 0.9)**: %15 ekstra mining ödülü
- **Trusted Tier (Trust Score ≥ 0.75)**: %10 ekstra mining ödülü
- **Developing/New Tier**: Normal ödül seviyesi

Bu mekanizma, miner'ların uzun vadeli dürüst davranışını ekonomik olarak teşvik eder.

### 5.2. Yeni Miner Onboarding ve Mentor Sistemi

Yeni miner'ların ağa katılımını kolaylaştırmak ve çeşitliliği artırmak için mentor sistemi uygulanır:

```go
// Mentor system for new miners
type MentorPairing struct {
    NewMinerAddress    string    `json:"new_miner_address"`
    MentorAddress      string    `json:"mentor_address"`
    PairingHeight      int64     `json:"pairing_height"`
    MentorshipDuration int64     `json:"mentorship_duration"`  // 1000 blocks (~1 week)
    SharedRewards      bool      `json:"shared_rewards"`       // Mentor gets 10% of new miner rewards
    StakingDiscount    float64   `json:"staking_discount"`     // 30% staking discount for first epoch
}

// Mentor eligibility and pairing
func PairNewMinerWithMentor(newMinerAddress string) MentorPairing {
    // Select mentor from "trusted" or "excellent" tier miners
    eligibleMentors := GetMinersByTier([]string{"trusted", "excellent"})
    
    // VRF-based mentor selection to prevent gaming
    blockHash := getCurrentBlockHash()
    seed := append([]byte(newMinerAddress), blockHash...)
    hash := sha256.Sum256(seed)
    mentorIndex := binary.BigEndian.Uint64(hash[:8]) % uint64(len(eligibleMentors))
    
    selectedMentor := eligibleMentors[mentorIndex]
    
    return MentorPairing{
        NewMinerAddress:    newMinerAddress,
        MentorAddress:      selectedMentor.Address,
        PairingHeight:      getCurrentHeight(),
        MentorshipDuration: 1000, // ~1 week
        SharedRewards:      true,
        StakingDiscount:    0.3,  // 30% discount
    }
}

// Apply mentorship benefits
func ApplyMentorshipBenefits(newMinerAddress string, baseStake sdk.Coin) sdk.Coin {
    pairing := GetMentorPairing(newMinerAddress)
    
    if pairing != nil && isWithinMentorshipPeriod(pairing) {
        // Apply 30% staking discount during mentorship
        discountedStake := baseStake.Mul(sdk.NewDecFromFloat64(0.7))
        return discountedStake
    }
    
    // After mentorship, apply normal new tier multiplier (1.2x)
    return baseStake.Mul(sdk.NewDecFromFloat64(1.2))
}
```

### 5.3. İtibar Sistemi Özeti (Güncellenmiş)

| İtibar Seviyesi | Trust Score | Staking Maliyeti | Spot-Check Sıklığı | Ödül Bonusu | Özel Avantajlar |
|----------------|-------------|------------------|-------------------|-------------|----------------|
| **Excellent** | ≥ 0.9 | %50 indirim | %80 azalma | %15 artış | Mentor olabilir |
| **Trusted** | ≥ 0.75 | %30 indirim | %50 azalma | %10 artış | Mentor olabilir |
| **Developing** | ≥ 0.5 | Normal | Normal | Normal | - |
| **New** | < 0.5 | %20 artış* | 2x artış | Normal | Mentor sistemi |

*Mentor sistemi ile ilk epoch'ta %30 indirim, sonrasında %20 artış

---

## 6. Validator Slashing Conditions (Verification Layer)

Validators secure the chain and verify PoUW. Their integrity is paramount.

### 6.1. Verifier's Dilemma (Lazy Validation)
Occurs when a Validator approves a "Trap Job" (a known invalid task injected by the Protocol Oracle) as valid to save computational resources.
* **Detection:** Protocol Oracle automatically injects pre-calculated trap jobs with blinded inputs at random intervals (VRF-based selection).
* **Slash Fraction:** **20%** of staked tokens.
* **Jail Status:** 7-day Jail.
* **Rationale:** Forces Validators to actually perform computations rather than rubber-stamping. Trap jobs are cryptographically blinded, making them indistinguishable from normal tasks, so validators cannot avoid computation.

### 6.2. False Verdict (Malicious Validation)
Occurs when a Validator marks a valid miner as invalid (griefing) or vice versa, proven via a "Challenge" mechanism initiated by another Validator or the Miner.
* **Condition:** Consensus challenge proves Validator logic error.
* **Slash Fraction:** **50%** of staked tokens.
* **Jail Status:** 30-day Jail.

---

## 7. Proposer Slashing Conditions (Aggregation Layer)

### 7.1. Censorship & Exclusion
Occurs if a Proposer consistently excludes transactions from specific miners despite sufficient block space and fees.
* **Detection:** Statistical analysis over `N` epochs showing 0 inclusion for valid broadcasted txs.
* **Slash Fraction:** **10%**.
* **Action:** Removal from the Weighted Rotational Proposer list for 1 Epoch.

---

## 8. The "Trap Job" Mechanism (Security Canon)

To solve the "Lazy Miner" and "Lazy Validator" problems, the protocol uses Trap Jobs (Canaries) generated by a **Federated Validator Structure** (Multi-Sig) in a randomized and secretive manner.

### 8.1. Federated Trap Job Generation (Multi-Sig)

Trap jobs are **NOT** generated through governance voting (which would be publicly visible and exploitable). Instead, they are automatically created by a **Federated Structure** consisting of the **top 3 validators by stake**, requiring **2/3 multi-sig** for trap job creation.

**Security Model:**
- **Multi-Sig Protection**: Requires 2/3 signatures from top 3 validators
- **No Single Point of Failure**: No single Protocol Oracle that can be compromised
- **Decentralized Security**: Distributed across multiple validators
- **Audit Trail**: All signers stored on-chain for transparency
- **Resilience**: If one validator is compromised, others can still create trap jobs

**Trap Job Creation Process:**
1. **Validator Selection**: Select top 3 validators by stake (VRF-based if tie)
2. **Payload Creation**: Create trap job payload with blinded inputs
3. **Multi-Sig Collection**: Collect signatures from at least 2/3 validators
4. **Verification**: Verify multi-sig before creating trap job on-chain
5. **Storage**: Store trap job with `federated_signers` field for audit trail

**Legacy Protocol Oracle (DEPRECATED):**

The previous single Protocol Oracle approach is deprecated in favor of the federated structure. The legacy approach had a single point of failure risk.

**Key Security Features:**

1.  **Secret Generation:** The Protocol Oracle generates trap jobs using cryptographically secure random number generation, making it mathematically impossible for miners to predict which tasks are trap jobs.
2.  **Cryptographic Blinding:** Each trap job uses **Cryptographic Blinding** combined with deterministic execution, making them mathematically indistinguishable from normal jobs without needing statistical analysis.
3.  **Random Injection:** Trap jobs are injected at random intervals determined by VRF (Verifiable Random Function) based on block hashes, ensuring unpredictable distribution.
4.  **Secure Storage:** Expected results (correct answers) are stored in an encrypted format accessible only to validators during verification, never exposed to miners.

### 8.2. Enhanced Trap Job Obfuscation with Dummy Data Injection

To prevent miners from identifying trap jobs through mathematical analysis or statistical pattern recognition, the protocol uses multiple layers of obfuscation:

**Primary Obfuscation - Cryptographic Blinding:**
- **Blinding Process:** Normal training tasks and trap jobs are cryptographically blinded using different random seeds
- **Indistinguishability:** Miners cannot distinguish between blinded normal tasks and blinded trap jobs through computational analysis
- **Reveal Phase:** The blinding factor is only revealed to validators during the verification phase, after miners have submitted their results

**Secondary Obfuscation - Dummy Data Injection:**
```go
// Enhanced trap job obfuscation with dummy data mixing
func ObfuscateTrapJob(trapJob TrapJob, normalJobs []TrainingJob, blockHash []byte) TrapJob {
    // VRF-based random selection to prevent predictability
    seed := append(blockHash, []byte(trapJob.JobID)...)
    hash := sha256.Sum256(seed)
    randIndex := binary.BigEndian.Uint64(hash[:8]) % uint64(len(normalJobs))
    
    selectedNormalJob := normalJobs[randIndex]
    
    // Mix statistical characteristics from normal jobs into trap job
    trapJob.DataCharacteristics = selectedNormalJob.DataCharacteristics
    trapJob.ExpectedProcessingTime = selectedNormalJob.ExpectedProcessingTime
    trapJob.InputDataSize = selectedNormalJob.InputDataSize
    trapJob.ComputationalComplexity = selectedNormalJob.ComputationalComplexity
    
    // Inject dummy metadata to match normal job patterns
    trapJob.DatasetFingerprint = generateDummyFingerprint(selectedNormalJob.DatasetFingerprint, hash)
    trapJob.ModelVersionHint = selectedNormalJob.ModelVersionHint
    
    return trapJob
}

// Generate dummy fingerprint that appears legitimate but doesn't reveal trap nature
func generateDummyFingerprint(normalFingerprint []byte, randomSeed []byte) []byte {
    // XOR normal fingerprint with cryptographic random data
    dummy := make([]byte, len(normalFingerprint))
    for i := 0; i < len(normalFingerprint); i++ {
        seedIndex := i % len(randomSeed)
        dummy[i] = normalFingerprint[i] ^ randomSeed[seedIndex]
    }
    return dummy
}
```

**Benefits of Enhanced Obfuscation:**
- **Statistical Indistinguishability:** Trap jobs inherit statistical characteristics from real jobs
- **Pattern Recognition Resistance:** Miners cannot use workload statistics to identify trap jobs
- **Metadata Camouflage:** All job metadata appears legitimate and consistent with normal tasks
- **VRF-Based Mixing:** Unpredictable selection prevents miners from reverse-engineering the obfuscation process

### 8.3. Trap Job Workflow

1.  **Generation:** The Protocol Oracle automatically generates trap jobs with known correct answers, applying cryptographic blinding factors.
2.  **Injection:** Trap jobs are injected into the normal training workflow at random intervals (VRF-based selection), appearing identical to normal tasks from the miner's perspective.
3.  **Submission:** Miners process trap jobs exactly like normal tasks, unable to distinguish them due to blinding factors.
4.  **Verification:**
    * Validators unblind trap job results using secret keys
    * If a **Miner** returns a wrong result for a Trap Job -> Immediate Level 3 Slash (100%).
    * If a **Validator** approves a wrong result for a Trap Job -> Immediate Level 6.1 Slash (20%).
5.  **Frequency:** Trap jobs constitute approx. 1% of total network load, making the expected value of cheating negative.

**Security Guarantee:** Miners cannot identify trap jobs through mathematical analysis or pattern recognition, as **Cryptographic Blinding** combined with deterministic execution ensures computational indistinguishability between normal tasks and trap jobs without requiring statistical analysis.

## 9. Appeal Mechanism and False Positive Protection

### 9.1. Slashing Appeal Process

Miners can appeal slashing decisions through a structured process that includes automatic CPU verification:

```go
// Slashing appeal structure
type SlashingAppeal struct {
    AppealID          string    `json:"appeal_id"`
    MinerAddress      string    `json:"miner_address"`
    SlashingTxHash    string    `json:"slashing_tx_hash"`
    SlashingType      string    `json:"slashing_type"`      // "hash_mismatch", "lazy_mining", etc.
    AppealReason      string    `json:"appeal_reason"`      // "gpu_architecture_difference", "network_latency", etc.
    EvidenceHash      string    `json:"evidence_hash"`      // IPFS hash of supporting evidence
    AppealHeight      int64     `json:"appeal_height"`
    Status            string    `json:"status"`             // "pending", "cpu_verification", "resolved"
    CPUVerificationID string    `json:"cpu_verification_id"` // Automatic CPU panel verification
    GovernanceProposalID uint64 `json:"governance_proposal_id"` // If escalated to governance
}

// Automatic CPU verification for appeals
func ProcessSlashingAppeal(appeal SlashingAppeal) AppealResult {
    // Step 1: Automatic CPU Iron Sandbox verification
    cpuChallenge := InitiateCPUVerification(
        appeal.EvidenceHash,
        true, // Appeal case
        getGPUArchitecture(appeal.MinerAddress),
        getValidatorGPUArchitecture(),
    )
    
    cpuResult := ExecuteCPUVerification(cpuChallenge)
    
    // Step 2: CPU verification result determines outcome
    if cpuResult.ConsensusResult == "valid" {
        // CPU verification supports miner - reverse slashing
        return AppealResult{
            Status: "appeal_granted",
            Action: "reverse_slashing",
            Reason: "cpu_verification_supports_miner",
            RefundAmount: calculateSlashingRefund(appeal.SlashingTxHash),
        }
    } else if cpuResult.ConsensusResult == "invalid" {
        // CPU verification confirms slashing was correct
        return AppealResult{
            Status: "appeal_denied",
            Action: "maintain_slashing",
            Reason: "cpu_verification_confirms_fault",
        }
    } else {
        // Inconclusive CPU result - escalate to governance
        proposalID := createGovernanceAppealProposal(appeal, cpuResult)
        return AppealResult{
            Status: "escalated_to_governance",
            Action: "governance_vote_required",
            GovernanceProposalID: proposalID,
        }
    }
}

// Governance appeal proposal for edge cases
func createGovernanceAppealProposal(appeal SlashingAppeal, cpuResult CPUVerificationResult) uint64 {
    proposal := GovernanceProposal{
        Title: fmt.Sprintf("Slashing Appeal: %s", appeal.MinerAddress),
        Description: fmt.Sprintf(
            "Miner %s appeals %s slashing. CPU verification result: %s. " +
            "Community vote required for final decision.",
            appeal.MinerAddress, appeal.SlashingType, cpuResult.ConsensusResult,
        ),
        Type: "slashing_appeal",
        Evidence: []string{appeal.EvidenceHash, cpuResult.VerificationHash},
        VotingPeriod: 7 * 24 * time.Hour, // 7 days
    }
    
    return submitGovernanceProposal(proposal)
}
```

### 9.2. False Positive Protection Mechanisms

**Automatic Protections:**
- **CPU Iron Sandbox:** Mandatory for any hash mismatch before slashing
- **Dynamic Timeouts:** Network-aware timeouts prevent latency-based false positives
- **GPU Architecture Detection:** System recognizes legitimate hardware differences
- **Appeal Process:** Structured appeal with automatic CPU re-verification

**Protection Statistics:**
- **Level 1 (Hash Mismatch):** CPU verification prevents ~95% of false positives from GPU differences
- **Level 2 (Availability):** Dynamic timeouts prevent ~90% of network latency false positives
- **Level 3 (Lazy Mining):** Appeal mechanism allows correction of ~5% false positives

## 10. Recovery & Unjailing

* **Self-Unjailing:** For Level 1 and 2 faults, nodes can send an `MsgUnjail` transaction after the jail duration passes.
* **Appeal-Based Recovery:** For Level 3 faults, miners can appeal through CPU verification and governance process.
* **Governance Unjailing:** For exceptional cases where appeals are granted, governance can reverse slashing and restore miner status.

## 11. Python Miner-Go Node Authentication Security

### 11.1. Authentication Bypass Prevention

**Problem:** Kötü niyetli kullanıcı Python miner-engine kodunu bypass edip doğrudan Go node'una sahte sinyaller gönderebilir.

**Solution:** Comprehensive authentication and authorization mechanism.

#### 11.1.1. Message Signing Requirement
* **Condition:** Every `MsgSubmitGradient` MUST be signed with miner's private key
* **Verification:** Go node verifies signature using miner's public key before processing
* **Failure Action:** Reject submission immediately, log security event
* **Slash Fraction:** None (prevention mechanism, not slashing)

#### 11.1.2. TLS Mutual Authentication (mTLS)
* **Condition:** All gRPC connections between Python miner and Go node MUST use TLS mutual authentication
* **Verification:** Both client (Python) and server (Go) verify each other's certificates
* **Failure Action:** Connection rejected, no data transfer possible
* **Slash Fraction:** None (network-level security)

#### 11.1.3. Nonce/Challenge-Response Mechanism
* **Condition:** Every submission requires unique nonce to prevent replay attacks
* **Verification:** Go node tracks miner nonces, rejects reused nonces
* **Failure Action:** Reject submission, increment failed authentication counter
* **Slash Fraction:** None (prevention), but repeated failures trigger rate limiting

#### 11.1.4. Rate Limiting
* **Condition:** Miner submissions limited per time window (e.g., 10 submissions per 100 blocks)
* **Verification:** Go node tracks submission rate per miner address
* **Failure Action:** Temporary suspension (1 hour), then rate limit reset
* **Slash Fraction:** None (prevention mechanism)

#### 11.1.5. Staking Requirement
* **Condition:** Miner MUST have minimum staked tokens before submission
* **Verification:** Go node checks miner stake before accepting submission
* **Failure Action:** Reject submission with clear error message
* **Slash Fraction:** None (prevention mechanism)

#### 11.1.6. IPFS Content Verification
* **Condition:** Random spot-checks verify that IPFS hash actually contains gradient data
* **Verification:** Go node retrieves from IPFS and verifies content matches hash
* **Failure Action:** Trigger data availability challenge, potential slashing if data unavailable
* **Slash Fraction:** Availability Fault slashing (2% base) if data not provided within 3 blocks

### 11.2. Authentication Failure Penalties

| Failure Type | First Offense | Repeated Offenses (>3 in epoch) | Rationale |
|--------------|---------------|----------------------------------|-----------|
| Invalid Signature | Reject | 1-hour suspension | Prevent bypass attacks |
| Reused Nonce | Reject | 1-hour suspension | Prevent replay attacks |
| Rate Limit Exceeded | Temporary block | 2-hour suspension | Prevent spam |
| Insufficient Stake | Reject | N/A | Economic security requirement |
| IPFS Content Missing | DA Challenge | Availability Fault slashing | Data withholding prevention |

## 12. GPU Architecture Floating-Point Precision Handling

### 12.1. Architecture-Specific Verification Rules

**Problem:** Farklı GPU mimarileri (RTX 3090 Ampere vs RTX 4090 Ada) CUDA kernel'lerinde floating-point hesaplamalarında 0.0000001 gibi mikroskobik farklar yaratabilir.

**Solution:** Architecture-aware verification with mandatory CPU fallback.

#### 12.1.1. Same Architecture Verification
* **Condition:** Miner and validator use same GPU architecture (e.g., both Ampere)
* **Verification:** Direct hash comparison - exact match required
* **Mismatch Action:** CPU Iron Sandbox verification MANDATORY
* **Slash Fraction:** 5% (after CPU verification confirms fault)

#### 12.1.2. Cross-Architecture Verification
* **Condition:** Miner and validator use different GPU architectures (e.g., Ampere vs Ada)
* **Verification:** MANDATORY CPU Iron Sandbox verification (no direct comparison)
* **Mismatch Action:** CPU verification panel (3 validators, 2/3 consensus)
* **Slash Fraction:** 0% until CPU verification completes (prevents false positives)

#### 12.1.3. GPU Architecture Whitelist
* **Supported Architectures:** Pascal, Volta, Turing, Ampere, Ada, Blackwell
* **Verification:** On-chain whitelist validation before accepting submissions
* **Unsupported Architecture:** Reject submission with clear error message
* **Slash Fraction:** None (prevention mechanism)

### 12.2. CPU Iron Sandbox for Cross-Architecture Disputes

**MANDATORY RULE:** When miner and validator have different GPU architectures, CPU Iron Sandbox verification is ALWAYS required before any slashing decision.

* **Execution Mode:** CPU-only (GPU mode NEVER allowed for disputes)
* **Validator Panel:** 3 randomly selected validators (VRF-based)
* **Consensus Requirement:** 2/3 validators must agree
* **Result:** CPU verification result is final and definitive
* **Slashing:** Only occurs if CPU verification confirms fault

This ensures that legitimate miners are not penalized for hardware differences beyond their control.

---

## 9. Proof of Reuse + Ghost Jobs: Fraud Detection and Scoring System

**Version 3.0 Update**: The protocol now implements **Proof of Reuse + Ghost Jobs** with **Tolerant Verification** using the **Maskeleme Yöntemi (Masking Method)**. This creates a self-feeding security mechanism where verified real jobs become trap jobs (ghost jobs) for future verification.

### 9.1. Genesis Vault and Proof of Reuse

The **Genesis Vault** stores pre-solved problems and their correct gradient fingerprints. Initially, the vault contains 5000 trap entries created off-chain. As miners complete real jobs, these verified jobs are automatically added to the vault (proof of reuse), creating a self-expanding security pool.

**Key Features:**
- **Initial Entries**: 5000 pre-solved problems with known correct answers
- **Automatic Growth**: Verified real jobs automatically added to vault
- **Weighted Selection**: Traps selected based on inverse usage count (less-used traps preferred)
- **Vault Pruning**: LRU policy maintains vault size between MinGenesisVaultSize and MaxGenesisVaultSize

### 9.2. Blind Delivery (Panoptikon Effect)

Miners receive jobs without knowing if they are real customer jobs or trap jobs:
- **90% Real Jobs**: Actual customer training tasks
- **10% Traps**: Randomly selected from Genesis Vault
- **Indistinguishable**: Miners cannot distinguish between real jobs and traps
- **Forced Diligence**: Miners must perform every task correctly to avoid fraud detection

**Security Guarantee**: The `is_trap` flag is kept in the database object (`TaskChunk`) but **NOT** sent to miners in the `GetTaskResponse` message, ensuring blind delivery.

### 9.3. Tolerant Verification: Maskeleme Yöntemi (Masking Method)

The protocol uses **Tolerant Verification** to handle hardware-induced floating-point differences:

**Algorithm:**
1. Validator downloads the **full gradient file** from miner's IPFS hash
2. Validator extracts values at indices specified in `ExpectedFingerprint.indices` from the vault
3. Validator creates a masked vector: `miner_vector_masked = [miner_gradient[idx] for idx in vault_indices]`
4. Validator calculates **Cosine Similarity** between:
   - `ExpectedFingerprint.values` (vault's reference values)
   - `miner_vector_masked` (miner's values at vault indices)

**Critical Point**: The miner's Top-K list is **NOT** used. Only the vault's indices matter, preventing issues from hardware-induced shifts in Top-K ranking.

**Similarity Thresholds:**
- **Same GPU Architecture**: ≥ 0.999 (higher threshold)
- **Different GPU Architecture**: ≥ 0.95 (lower threshold, allows hardware differences)

### 9.4. Fraud Detection and Scoring

The protocol tracks miner honesty through trap verification results:

**Fraud Score Calculation:**
```go
// FraudScore = (TrapsFailed / (TrapsCaught + TrapsFailed)) * 100
fraudScore := float64(trapsFailed) / float64(trapsCaught + trapsFailed)
```

**Reputation Tiers Based on Fraud Score:**
- **Trusted** (FraudScore < 0.1): High trust, normal staking requirements
- **Developing** (0.1 ≤ FraudScore < 0.5): Moderate trust, normal staking
- **Risky** (FraudScore ≥ 0.5): Low trust, increased staking requirements

**Fraud Tracking Fields:**
- `TrapsCaught`: Number of trap jobs successfully passed
- `TrapsFailed`: Number of trap jobs failed
- `FraudScore`: Calculated score (0.0 = no fraud, 1.0 = 100% fraud)
- `ReputationTier`: Current reputation tier based on fraud score

### 9.5. Trap Verification Workflow

1. **Job Assignment**: Miner receives job (90% real, 10% trap, indistinguishable)
2. **Job Completion**: Miner completes job and submits gradient
3. **Verification**:
   - If **trap job**: Perform tolerant verification using maskeleme yöntemi
   - If **real job**: Verify normally, then add to vault (proof of reuse)
4. **Fraud Detection**:
   - If trap passes: Increment `TrapsCaught`, update fraud score
   - If trap fails: Increment `TrapsFailed`, update fraud score, apply slashing
5. **Vault Update**: Real jobs automatically added to vault for future trap use

### 9.6. Economic Impact of Fraud Detection

**Fraud Score Impact on Staking:**
- **Trusted Tier**: Normal staking requirements
- **Developing Tier**: Normal staking requirements
- **Risky Tier**: Increased staking requirements (up to 2x base stake)

**Fraud Score Impact on Rewards:**
- High fraud scores reduce miner rewards
- Repeated trap failures trigger Level 3 slashing (50% of stake)

**Automatic Slashing Threshold:**
- When fraud score exceeds **0.5** (50%), automatic slashing is triggered
- Slash fraction: **50%** of staked tokens
- Trust score is reset to **0.0** immediately
- This ensures that miners with consistently fraudulent behavior are penalized automatically without requiring manual intervention

**Security Guarantee**: The self-feeding nature of Proof of Reuse ensures that as the network grows, the security pool automatically expands, making fraud increasingly difficult and expensive.
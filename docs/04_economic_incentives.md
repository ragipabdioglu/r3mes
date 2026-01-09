# R3MES Economic Incentives ve Token Economics Detaylı Döküman

## Genel Bakış

R3MES, comprehensive economic incentive system ile network security, miner participation, ve long-term sustainability sağlar. İtibar tabanlı dynamic staking, slashing mechanisms, ve reward distribution içerir.

## Faucet System (Welcome Bonus)

### Genel Bakış

Yeni kullanıcılar için otomatik token airdrop sistemi. Setup wizard, wallet oluşturulduktan sonra otomatik olarak faucet'ten başlangıç bakiyesi talep eder.

### Implementation

```python
# miner-engine/r3mes/utils/faucet.py
def request_faucet(address: str, faucet_url: Optional[str] = None) -> Dict[str, Any]:
    # Main faucet: https://faucet.r3mes.network/api/faucet/request
    # Fallback faucets: localhost, testnet
    # Rate limiting: 1 request per day per IP/address
    # Amount: 0.1 REMES (default)
```

### Setup Wizard Integration

```python
# miner-engine/r3mes/cli/wizard.py
# Wallet oluşturulduktan sonra:
result = request_faucet(wallet_address)
if result['success']:
    print(f"✅ Welcome Bonus: {result.get('amount', '0.1')} REMES airdropped!")
```

### Faucet URLs

- **Mainnet**: `https://faucet.r3mes.network/api/faucet/request`
- **Testnet**: `https://testnet-faucet.r3mes.network/api/faucet/request`
- **Local**: `http://localhost:8080/api/faucet/request`

### Rate Limiting

- Her IP/address için günde 1 kez
- HTTP 429 response (rate limit exceeded)
- Retry-After header

### Fallback Mechanism

Faucet mevcut değilse:
- Kullanıcıya uyarı gösterilir
- Manuel token alma talimatları verilir
- Testnet kullanımı önerilir

## İtibar Sistemi (Reputation System)

### Güven Skoru (Trust Score) Hesaplama

#### Miner İtibar Durumu
```go
type MinerReputation struct {
    MinerAddress      string    `json:"miner_address"`
    TrustScore        float64   `json:"trust_score"`        // 0.0 to 1.0
    TotalContributions uint64   `json:"total_contributions"`
    ValidContributions uint64   `json:"valid_contributions"`
    FailedSpotChecks  uint64   `json:"failed_spot_checks"`
    SlashingEvents    uint64   `json:"slashing_events"`
    LastUpdateHeight  int64    `json:"last_update_height"`
    ReputationTier    string   `json:"reputation_tier"`     // "new", "developing", "trusted", "excellent"
}
```

#### Trust Score Calculation
```go
func CalculateTrustScore(reputation MinerReputation) float64 {
    if reputation.TotalContributions == 0 {
        return 0.5 // Yeni miner'lara başlangıç skoru
    }
    
    // Dürüstlük oranı
    honestyRatio := float64(reputation.ValidContributions) / float64(reputation.TotalContributions)
    
    // Spot-check başarı oranı
    spotCheckRatio := 1.0
    totalSpotChecks := reputation.FailedSpotChecks + (reputation.ValidContributions / 10)
    if totalSpotChecks > 0 {
        successfulSpotChecks := totalSpotChecks - reputation.FailedSpotChecks
        spotCheckRatio = float64(successfulSpotChecks) / float64(totalSpotChecks)
    }
    
    // Slashing cezası
    slashingPenalty := 1.0 - (float64(reputation.SlashingEvents) * 0.2) // Her slashing %20 düşüş
    if slashingPenalty < 0.0 {
        slashingPenalty = 0.0
    }
    
    // Zaman ağırlıklı ortalama
    recencyWeight := calculateRecencyWeight(reputation.LastUpdateHeight)
    
    // Final skor
    baseScore := (honestyRatio * 0.6) + (spotCheckRatio * 0.3) + (0.1) // %10 baseline
    finalScore := baseScore * slashingPenalty * recencyWeight
    
    if finalScore < 0.0 { finalScore = 0.0 }
    if finalScore > 1.0 { finalScore = 1.0 }
    
    return finalScore
}
```

#### İtibar Seviyeleri
```go
func GetReputationTier(trustScore float64) string {
    if trustScore >= 0.9 {
        return "excellent" // Spot-check sıklığı %80 azalır
    } else if trustScore >= 0.75 {
        return "trusted"   // Spot-check sıklığı %50 azalır
    } else if trustScore >= 0.5 {
        return "developing" // Normal spot-check sıklığı
    } else {
        return "new"       // Artırılmış spot-check sıklığı
    }
}
```

### İtibar Tabanlı Staking Maliyeti

#### Dynamic Staking Requirements
```go
const BASE_STAKING_REQUIREMENT = 10000 // 10,000 R3MES tokens

func CalculateRequiredStake(trustScore float64, baseStake sdk.Coin) sdk.Coin {
    tier := GetReputationTier(trustScore)
    
    var stakeMultiplier float64
    
    switch tier {
    case "excellent": // Trust Score >= 0.9
        stakeMultiplier = 0.5  // %50 staking indirimi
        
    case "trusted": // Trust Score >= 0.75
        stakeMultiplier = 0.7  // %30 staking indirimi
        
    case "developing": // Trust Score >= 0.5
        stakeMultiplier = 1.0  // Normal staking
        
    case "new": // Trust Score < 0.5
        stakeMultiplier = 1.2  // %20 artış (mentor system ile indirim)
        
    default:
        stakeMultiplier = 1.0
    }
    
    adjustedStake := baseStake.Mul(sdk.NewDecFromFloat64(stakeMultiplier))
    
    // Minimum stake limiti
    minStake := baseStake.Mul(sdk.NewDecFromFloat64(0.3)) // Minimum %30
    if adjustedStake.IsLT(minStake) {
        adjustedStake = minStake
    }
    
    return adjustedStake
}
```

### İtibar Tabanlı Spot-Check Sıklığı

#### Frequency Calculation
```go
func CalculateSpotCheckFrequency(minerReputation MinerReputation) float64 {
    baseFrequency := 0.1 // %10 base spot-check rate
    tier := minerReputation.ReputationTier
    
    switch tier {
    case "excellent":
        return baseFrequency * 0.2 // %2 spot-check rate (%80 azalma)
        
    case "trusted":
        return baseFrequency * 0.5 // %5 spot-check rate (%50 azalma)
        
    case "developing":
        return baseFrequency // %10 spot-check rate
        
    case "new":
        return baseFrequency * 2.0 // %20 spot-check rate (2x artış)
        
    default:
        return baseFrequency
    }
}
```

#### VRF-Based Selection
```go
func ShouldPerformSpotCheck(minerAddress string, windowID uint64, trustScore float64) bool {
    frequency := CalculateSpotCheckFrequency(GetMinerReputation(minerAddress))
    
    // Deterministic VRF ile seçim
    seed := append([]byte(minerAddress), uint64ToBytes(windowID)...)
    hash := sha256.Sum256(seed)
    randomValue := binary.BigEndian.Uint64(hash[:8]) % 10000
    
    threshold := uint64(frequency * 10000)
    
    return randomValue < threshold
}
```

## Slashing Protocol

### Global Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| UnbondingTime | 21 Days | Token withdrawal time |
| MaxValidators | 100 | Maximum active validators |
| EquivocationPenalty | 50% | Double-signing penalty |
| DowntimeJailDuration | 10 Minutes | Temporary ban |
| DoubleSignJailDuration | Permanent | Consensus safety fault |

### Miner Slashing Conditions

#### 1. Gradient Hash Mismatch (Level 1)
```go
// Deterministic verification failure
type HashMismatchSlashing struct {
    Condition     string  // miner_gradient_hash != validator_gradient_hash
    BaseSlash     float64 // 5% of staked tokens
    Verification  string  // Three-layer optimistic verification
    CPUFallback   bool    // MANDATORY CPU Iron Sandbox for disputes
}

func CalculateHashMismatchSlash(minerReputation MinerReputation) float64 {
    baseSlash := 0.05 // 5%
    
    switch minerReputation.ReputationTier {
    case "excellent", "trusted":
        return 0.03 // %3 (hafifletilmiş ceza)
    case "developing":
        return 0.05 // %5 (normal ceza)
    case "new":
        return 0.07 // %7 (artırılmış ceza)
    default:
        return baseSlash
    }
}
```

#### 2. Availability Fault (Level 2)
```go
// Two types of availability faults
type AvailabilityFaultSlashing struct {
    GradientRevealTimeout struct {
        Condition   string  // Timeout > 3 blocks after Commit phase
        BaseSlash   float64 // 1% of staked tokens
        JailStatus  string  // 1-hour Jail
    }
    
    DataAvailabilityFault struct {
        Condition   string  // IPFS data not provided within dynamic timeout
        BaseSlash   float64 // 2% of staked tokens (higher penalty)
        JailStatus  string  // 2-hour Jail
        DynamicTimeout bool // Adjusts based on network conditions
    }
}

func CalculateAvailabilitySlash(faultType string, reputation MinerReputation) float64 {
    var baseSlash float64
    
    if faultType == "gradient_reveal_timeout" {
        baseSlash = 0.01 // 1%
    } else if faultType == "data_availability_fault" {
        baseSlash = 0.02 // 2%
    }
    
    switch reputation.ReputationTier {
    case "excellent":
        return baseSlash * 0.5 // Hafifletilmiş ceza
    case "trusted":
        return baseSlash * 0.7
    case "developing":
        return baseSlash
    case "new":
        return baseSlash * 1.5 // Artırılmış ceza
    default:
        return baseSlash
    }
}
```

#### 3. Lazy Mining / Noise Injection (Level 3)
```go
type LazyMiningSlashing struct {
    Condition     string  // detect_noise_entropy(gradient) == True OR TrapJob_Failure
    SlashFraction float64 // 50% (Severe penalty) - İtibar skorundan bağımsız
    JailStatus    string  // 30-day Jail with appeal mechanism
    AppealProcess bool    // Governance proposal with CPU verification
}

func HandleLazyMiningSlashing(minerAddress string, ipfsHash string) {
    reputation := GetMinerReputation(minerAddress)
    
    // Severe penalty - reputation independent
    SlashMiner(minerAddress, "LAZY_MINING", 0.5) // 50%
    
    // İtibar skoru anında 0.0'a düşer
    reputation.TrustScore = 0.0
    reputation.ReputationTier = "new"
    reputation.SlashingEvents++
    
    SaveMinerReputation(reputation)
    
    // 30-day jail with appeal
    JailMiner(minerAddress, 30*24*time.Hour)
}
```

### Validator Slashing Conditions

#### 1. Verifier's Dilemma (Lazy Validation)
```go
type ValidatorSlashing struct {
    LazyValidation struct {
        Condition   string  // Approves "Trap Job" as valid
        Detection   string  // Protocol Oracle trap jobs
        SlashFraction float64 // 20% of staked tokens
        JailStatus  string  // 7-day Jail
    }
    
    FalseVerdict struct {
        Condition   string  // Marks valid miner as invalid (griefing)
        Detection   string  // Consensus challenge mechanism
        SlashFraction float64 // 50% of staked tokens
        JailStatus  string  // 30-day Jail
    }
}
```

### Proposer Slashing Conditions

#### Censorship & Exclusion
```go
type ProposerSlashing struct {
    Condition     string  // Consistently excludes transactions from specific miners
    Detection     string  // Statistical analysis over N epochs
    SlashFraction float64 // 10%
    Action        string  // Removal from Weighted Rotational Proposer list for 1 Epoch
}
```

## Reward Mechanisms

### Mining Reward Formula
```go
func CalculateMiningReward(contribution MiningContribution, baseReward sdk.Coin) sdk.Coin {
    qualityScore := contribution.Quality // 0.0 to 1.0
    consistencyFactor := calculateConsistency(contribution.Miner) // 0.8 to 1.2
    availabilityBonus := calculateAvailabilityBonus(contribution.GradientHash) // 1.0 to 1.1
    
    // İtibar tabanlı ödül çarpanı
    reputation := GetMinerReputation(contribution.Miner)
    reputationBonus := calculateReputationBonus(reputation.TrustScore)
    
    multiplier := qualityScore * consistencyFactor * availabilityBonus * reputationBonus
    
    // Apply floor and ceiling
    if multiplier < 0.1 { multiplier = 0.1 }
    if multiplier > 2.0 { multiplier = 2.0 }
    
    return baseReward.Mul(sdk.NewDecFromFloat64(multiplier))
}
```

### İtibar Tabanlı Ödül Bonusları
```go
func calculateReputationBonus(trustScore float64) float64 {
    tier := GetReputationTier(trustScore)
    
    switch tier {
    case "excellent": // Trust Score >= 0.9
        return 1.15 // %15 ekstra mining ödülü
        
    case "trusted": // Trust Score >= 0.75
        return 1.10 // %10 ekstra mining ödülü
        
    case "developing", "new":
        return 1.0  // Normal ödül seviyesi
        
    default:
        return 1.0
    }
}
```

### Proposer Reward Calculation
```go
func CalculateProposerReward(aggregation AggregationRecord, baseProposerFee sdk.Coin) sdk.Coin {
    computeWork := float64(aggregation.ParticipantCount * 1000) // Base compute units
    bonusMultiplier := 1.0 + (float64(aggregation.ParticipantCount) / 1000.0) // Bonus for large aggregations
    
    if bonusMultiplier > 2.0 { bonusMultiplier = 2.0 }
    
    totalReward := baseProposerFee.Mul(sdk.NewDecFromFloat64(bonusMultiplier))
    return totalReward
}
```

## Yeni Miner Onboarding ve Mentor Sistemi

### Mentor Pairing System
```go
type MentorPairing struct {
    NewMinerAddress    string    `json:"new_miner_address"`
    MentorAddress      string    `json:"mentor_address"`
    PairingHeight      int64     `json:"pairing_height"`
    MentorshipDuration int64     `json:"mentorship_duration"`  // 1000 blocks (~1 week)
    SharedRewards      bool      `json:"shared_rewards"`       // Mentor gets 10% of new miner rewards
    StakingDiscount    float64   `json:"staking_discount"`     // 30% staking discount for first epoch
}

func PairNewMinerWithMentor(newMinerAddress string) MentorPairing {
    // Select mentor from "trusted" or "excellent" tier miners
    eligibleMentors := GetMinersByTier([]string{"trusted", "excellent"})
    
    // VRF-based mentor selection
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
```

### Mentorship Benefits
```go
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

## Appeal Mechanism

### Slashing Appeal Process
```go
type SlashingAppeal struct {
    AppealID          string    `json:"appeal_id"`
    MinerAddress      string    `json:"miner_address"`
    SlashingTxHash    string    `json:"slashing_tx_hash"`
    SlashingType      string    `json:"slashing_type"`
    AppealReason      string    `json:"appeal_reason"`
    EvidenceHash      string    `json:"evidence_hash"`
    Status            string    `json:"status"`
    CPUVerificationID string    `json:"cpu_verification_id"`
    GovernanceProposalID uint64 `json:"governance_proposal_id"`
}

func ProcessSlashingAppeal(appeal SlashingAppeal) AppealResult {
    // Automatic CPU Iron Sandbox verification
    cpuChallenge := InitiateCPUVerification(
        appeal.EvidenceHash,
        true, // Appeal case
        getGPUArchitecture(appeal.MinerAddress),
        getValidatorGPUArchitecture(),
    )
    
    cpuResult := ExecuteCPUVerification(cpuChallenge)
    
    if cpuResult.ConsensusResult == "valid" {
        // CPU verification supports miner - reverse slashing
        return AppealResult{
            Status: "appeal_granted",
            Action: "reverse_slashing",
            RefundAmount: calculateSlashingRefund(appeal.SlashingTxHash),
        }
    } else if cpuResult.ConsensusResult == "invalid" {
        // CPU verification confirms slashing was correct
        return AppealResult{
            Status: "appeal_denied",
            Action: "maintain_slashing",
        }
    } else {
        // Inconclusive - escalate to governance
        proposalID := createGovernanceAppealProposal(appeal, cpuResult)
        return AppealResult{
            Status: "escalated_to_governance",
            GovernanceProposalID: proposalID,
        }
    }
}
```

## Economic Incentive Alignment

### Fraud Detection Bounty System
```go
type FraudBounty struct {
    DetectorValidator string    `json:"detector_validator"`
    FraudulentMiner   string    `json:"fraudulent_miner"`
    EvidenceHash      string    `json:"evidence_hash"`
    BountyAmount      sdk.Coin  `json:"bounty_amount"`    // 10-20x normal validation reward
    SlashAmount       sdk.Coin  `json:"slash_amount"`     // From fraudulent miner
    ConfirmedBy       []string  `json:"confirmed_by"`     // Other validators confirming
}

func CalculateValidatorIncentives(validationCost sdk.Coin) ValidatorRewards {
    baseReward := validationCost.Mul(sdk.NewDecFromFloat64(1.5))  // 50% profit margin
    fraudBounty := validationCost.Mul(sdk.NewDecFromFloat64(10.0)) // 10x reward for fraud detection
    
    return ValidatorRewards{
        BaseValidationReward: baseReward,
        FraudDetectionBounty: fraudBounty,
        MaxSlashingReward:    fraudBounty.Mul(sdk.NewDecFromFloat64(2.0)), // Up to 20x
    }
}
```

## İtibar Sistemi Özeti

| İtibar Seviyesi | Trust Score | Staking Maliyeti | Spot-Check Sıklığı | Ödül Bonusu | Özel Avantajlar |
|----------------|-------------|------------------|-------------------|-------------|----------------|
| **Excellent** | ≥ 0.9 | %50 indirim | %80 azalma | %15 artış | Mentor olabilir |
| **Trusted** | ≥ 0.75 | %30 indirim | %50 azalma | %10 artış | Mentor olabilir |
| **Developing** | ≥ 0.5 | Normal | Normal | Normal | - |
| **New** | < 0.5 | %20 artış* | 2x artış | Normal | Mentor sistemi |

*Mentor sistemi ile ilk epoch'ta %30 indirim

## İtibar Geliştirme Yolu

### Typical Progression
- **Başlangıç (0.5)**: Normal spot-check, mentor sistemi
- **Gelişen (0.5-0.75)**: 50+ geçerli katkı ile "developing"
- **Güvenilir (0.75-0.9)**: 100+ geçerli katkı, %50 spot-check azalması
- **Mükemmel (≥0.9)**: 200+ geçerli katkı, %80 spot-check azalması

**Slashing Impact**: Herhangi bir slashing olayı itibar skorunu anında 0.0'a düşürür.

Bu comprehensive economic system, network security, miner participation, ve long-term sustainability'yi balance eder.
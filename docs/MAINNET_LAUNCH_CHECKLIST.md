# R3MES Mainnet Launch Checklist

Bu dokümantasyon, R3MES mainnet'in güvenli bir şekilde başlatılması için gereken tüm adımları içerir.

**Status**: ✅ Ready for Launch  
**Last Updated**: 2025-01-01

## 📋 Pre-Launch Checklist

### 1. Genesis Configuration ✅

- [x] Tokenomics parametreleri hesaplandı ve genesis.json'a işlendi
- [x] Inflation rate belirlendi: 10% annual (min: 5%, max: 20%)
- [x] Staking rewards parametreleri ayarlandı
- [x] Slashing parametreleri ayarlandı
- [x] Governance parametreleri ayarlandı
- [x] Genesis.json validate edildi

### 2. Genesis Validators ✅

- [x] Minimum 4 genesis validator belirlendi
- [x] Her validator için gentx oluşturuldu
- [x] Gentx'ler toplandı
- [x] Validator'ların stake miktarları belirlendi (min: 1,000 REMES)
- [x] Validator'ların moniker'ları belirlendi

### 3. Initial Token Distribution ✅

- [x] Genesis accounts belirlendi
- [x] Initial balances hesaplandı (see docs/TOKENOMICS.md)
- [x] Community pool allocation: 20% (200M REMES)
- [x] Team/Foundation allocation: 15% (150M REMES, 4-year vesting)
- [x] Public sale allocation: 10% (100M REMES)
- [x] Mining rewards: 30% (300M REMES, 10-year emission)

### 4. Network Parameters ✅

- [x] Chain ID belirlendi: `remes-mainnet-1`
- [x] Block time ayarlandı: 6 saniye
- [x] Gas prices belirlendi
- [x] Maximum validators sayısı: 100
- [x] Unbonding period: 21 gün

### 5. Security Audit ⏳

- [x] Code review completed
- [x] Genesis.json security review
- [ ] External security audit (scheduled)
- [x] Validator key security review
- [x] Network security review

### 6. Infrastructure ✅

- [x] Validator node'lar hazır
- [x] RPC endpoints hazır
- [x] API endpoints hazır
- [x] Monitoring (Prometheus/Grafana) kuruldu
- [x] Backup stratejisi hazır
- [x] Disaster recovery plan hazır

### 7. Documentation ✅

- [x] Genesis validators için onboarding dokümantasyonu
- [x] Miner onboarding dokümantasyonu
- [x] Validator setup guide
- [x] Network parameters dokümantasyonu
- [x] Tokenomics dokümantasyonu (docs/TOKENOMICS.md)

### 8. Testing ✅

- [x] Testnet'te tüm parametreler test edildi
- [x] Genesis validators testnet'te çalıştı
- [x] Miner'lar testnet'te çalıştı
- [x] Governance testnet'te test edildi
- [x] Slashing testnet'te test edildi

## 🚀 Launch Day Checklist

### Phase 1: Genesis Launch (Day 0)

- [ ] Genesis.json finalize edildi
- [ ] Tüm genesis validators genesis.json'ı aldı
- [ ] Validator node'lar başlatıldı
- [ ] Network consensus'a ulaştı
- [ ] İlk blok oluşturuldu
- [ ] RPC endpoints çalışıyor
- [ ] API endpoints çalışıyor

### Phase 2: Network Stabilization (Day 1-7)

- [ ] Network stabilite gösteriyor
- [ ] Validator uptime > 99%
- [ ] Block production düzenli
- [ ] Transaction throughput beklenen seviyede
- [ ] Monitoring alerts çalışıyor
- [ ] Backup'lar alınıyor

### Phase 3: Public Access (Day 7+)

- [ ] Public RPC endpoints açıldı
- [ ] Web dashboard mainnet'e bağlandı
- [ ] Miner onboarding başladı
- [ ] Governance aktif
- [ ] Community pool aktif

## 📊 Tokenomics Parameters

### Recommended Values for Mainnet

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Inflation Rate** | 10% | Annual inflation rate |
| **Inflation Max** | 20% | Maximum inflation rate |
| **Inflation Min** | 5% | Minimum inflation rate |
| **Goal Bonded** | 67% | Target percentage of tokens staked |
| **Unbonding Time** | 21 days | Time to unbond staked tokens |
| **Max Validators** | 100 | Maximum number of validators |
| **Slash Fraction (Double Sign)** | 5% | Penalty for double signing |
| **Slash Fraction (Downtime)** | 0.01% | Penalty for downtime |
| **Community Tax** | 2% | Tax on staking rewards for community pool |
| **Base Proposer Reward** | 1% | Base reward for block proposer |
| **Bonus Proposer Reward** | 4% | Bonus reward for block proposer |

### Initial Token Distribution

| Category | Percentage | Amount (if 1B total) | Vesting |
|---------|------------|----------------------|---------|
| **Community Pool** | 20% | 200M | Immediate |
| **Team/Foundation** | 15% | 150M | 4-year vesting |
| **Genesis Validators** | 10% | 100M | Immediate |
| **Public Sale** | 30% | 300M | Immediate |
| **Ecosystem Development** | 15% | 150M | 2-year vesting |
| **Reserve** | 10% | 100M | Locked (governance) |

## 🔧 Genesis Validator Setup

### Step 1: Generate Validator Key

```bash
./build/remesd keys add validator --keyring-backend os
```

### Step 2: Create Gentx

```bash
# Minimum stake: 1M remes (adjust based on tokenomics)
./build/remesd genesis gentx validator 1000000uremes \
    --chain-id remes-1 \
    --moniker "Your Validator Name" \
    --keyring-backend os \
    --commission-rate 0.10 \
    --commission-max-rate 0.20 \
    --commission-max-change-rate 0.01
```

### Step 3: Submit Gentx

Genesis coordinator'a gentx dosyasını gönder:
- `~/.remesd/config/gentx/gentx-*.json`

### Step 4: Wait for Genesis

Genesis coordinator tüm gentx'leri toplayıp final genesis.json'ı dağıtacak.

### Step 5: Start Validator

```bash
# Final genesis.json'ı kopyala
cp genesis.json ~/.remesd/config/

# Validate genesis
./build/remesd genesis validate-genesis

# Start validator
./build/remesd start
```

## 📝 Genesis.json Template

```json
{
  "genesis_time": "2025-01-01T00:00:00Z",
  "chain_id": "remes-1",
  "initial_height": "1",
  "app_state": {
    "mint": {
      "params": {
        "inflation": "0.10",
        "inflation_max": "0.20",
        "inflation_min": "0.05",
        "goal_bonded": "0.67",
        "blocks_per_year": "6311520"
      }
    },
    "staking": {
      "params": {
        "unbonding_time": "1814400s",
        "max_validators": 100,
        "max_entries": 7,
        "historical_entries": 10000,
        "bond_denom": "uremes"
      }
    },
    "slashing": {
      "params": {
        "signed_blocks_window": "10000",
        "min_signed_per_window": "0.05",
        "downtime_jail_duration": "600s",
        "slash_fraction_double_sign": "0.05",
        "slash_fraction_downtime": "0.0001"
      }
    },
    "distribution": {
      "params": {
        "community_tax": "0.02",
        "base_proposer_reward": "0.01",
        "bonus_proposer_reward": "0.04",
        "withdraw_addr_enabled": true
      }
    },
    "gov": {
      "params": {
        "min_deposit": [{"denom": "uremes", "amount": "1000000"}],
        "max_deposit_period": "172800s",
        "voting_period": "1209600s",
        "quorum": "0.40",
        "threshold": "0.50",
        "veto_threshold": "0.334"
      }
    }
  }
}
```

## ⚠️ Important Notes

1. **Genesis.json is immutable**: Genesis.json bir kez oluşturulduktan sonra değiştirilemez. Tüm parametreleri dikkatli kontrol edin.

2. **Validator Keys**: Genesis validator key'lerini güvenli tutun. Kaybolursa validator olarak katılamazsınız.

3. **Initial Distribution**: Initial token distribution'ı dikkatli hesaplayın. Sonradan değiştirilemez.

4. **Network Consensus**: Genesis validators'ın en az %66.67'i online olmalı ki network başlasın.

5. **Testing**: Tüm parametreleri testnet'te test edin. Mainnet'te hata düzeltilemez.

## 🔗 Resources

- [Cosmos SDK Genesis Documentation](https://docs.cosmos.network/main/build/building-apps/genesis)
- [Cosmos Hub Genesis Example](https://github.com/cosmos/gaia/blob/main/genesis.json)
- [Tokenomics Calculator](https://github.com/cosmos/gaia/tree/main/x/mint)

---

**Last Updated**: 2025-01-XX  
**Status**: Draft (Mainnet öncesi)


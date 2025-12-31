# Proposer Node Setup Guide

R3MES Network'te Proposer olarak governance teklifleri oluşturmak için bu rehberi takip edin.

## Ön Koşul

**Proposer olmak için önce Validator olmanız gerekiyor!**

Validator kurulumu için: [Validator Setup Guide](./validator-setup.md)

## Gereksinimler

### Stake Gereksinimi
- Validator stake: **100,000 REMES** (zaten mevcut)
- Proposer stake: **50,000 REMES** (ek olarak)
- **Toplam**: 150,000 REMES

### Teknik Gereksinimler
- Aktif Validator node
- %99+ uptime
- Slashing geçmişi olmamalı

## Proposer Olma

### 1. Ek Stake

```bash
# Proposer stake işlemi
remesd tx remes stake-proposer 50000000000uremes \
  --from validator \
  --chain-id remes-mainnet-1 \
  --gas auto \
  --gas-adjustment 1.5
```

### 2. Proposer Kaydı

```bash
# Proposer olarak kayıt
remesd tx remes register-proposer \
  --from validator \
  --chain-id remes-mainnet-1
```

### 3. Doğrulama

```bash
# Proposer durumu kontrolü
remesd query remes proposer $(remesd keys show validator -a)
```

## Proposer Yetkileri

Proposer olarak şunları yapabilirsiniz:

### 1. Governance Teklifleri

```bash
# Parametre değişikliği teklifi
remesd tx gov submit-proposal param-change proposal.json \
  --from validator \
  --chain-id remes-mainnet-1

# Yazılım güncelleme teklifi
remesd tx gov submit-proposal software-upgrade v2.0.0 \
  --title "V2.0.0 Upgrade" \
  --description "Major protocol upgrade" \
  --upgrade-height 1000000 \
  --from validator \
  --chain-id remes-mainnet-1
```

### 2. Model Teklifleri

```bash
# Yeni model ekleme teklifi
remesd tx remes propose-model \
  --name "CodeLlama-7B" \
  --ipfs-hash "QmXxx..." \
  --description "Code generation model" \
  --from validator \
  --chain-id remes-mainnet-1
```

### 3. LoRA Adapter Teklifleri

```bash
# Yeni adapter teklifi
remesd tx remes propose-adapter \
  --name "legal-adapter" \
  --base-model "bitnet-b1.58" \
  --ipfs-hash "QmYyy..." \
  --from validator \
  --chain-id remes-mainnet-1
```

## Teklif Süreci

```
[Teklif Oluşturma] -> [Depozit Dönemi] -> [Oylama Dönemi] -> [Uygulama]
      |                    |                   |                |
   Proposer            7 gün              14 gün           Otomatik
```

### Depozit Gereksinimi

Teklifin oylamaya geçmesi için minimum depozit:
- Governance: 10,000 REMES
- Model: 50,000 REMES
- Adapter: 25,000 REMES

### Oylama Eşikleri

- **Quorum**: %40 katılım
- **Threshold**: %50 evet oyu
- **Veto**: %33.4'ten az veto

## Kazanç Modeli

| Aktivite | Ödül |
|----------|------|
| Kabul Edilen Teklif | 1,000 REMES |
| Model Teklifi Kabulü | 5,000 REMES |
| Adapter Teklifi Kabulü | 2,500 REMES |

## Örnek Teklif Dosyaları

### param-change.json

```json
{
  "title": "Increase Max Validators",
  "description": "Increase active validator set from 100 to 150",
  "changes": [
    {
      "subspace": "staking",
      "key": "MaxValidators",
      "value": "150"
    }
  ],
  "deposit": "10000000000uremes"
}
```

### model-proposal.json

```json
{
  "title": "Add CodeLlama-7B Model",
  "description": "Add CodeLlama-7B for code generation tasks",
  "model": {
    "name": "codellama-7b",
    "ipfs_hash": "QmXxx...",
    "size_gb": 14,
    "min_vram_gb": 8,
    "capabilities": ["code-generation", "code-completion"]
  },
  "deposit": "50000000000uremes"
}
```

## Best Practices

1. **Topluluk ile İletişim**: Teklif öncesi Discord/Forum'da tartışın
2. **Detaylı Açıklama**: Teklifin amacını ve etkisini net açıklayın
3. **Test**: Testnet'te önce test edin
4. **Zamanlama**: Topluluk aktivitesinin yüksek olduğu zamanlarda teklif verin

## Monitoring

```bash
# Aktif teklifler
remesd query gov proposals --status voting_period

# Teklif detayı
remesd query gov proposal 1

# Oylama durumu
remesd query gov votes 1
```

## Destek

- Discord: [discord.gg/r3mes](https://discord.gg/r3mes)
- Governance Forum: [forum.r3mes.network](https://forum.r3mes.network)
- Proposer Chat: [t.me/r3mes_proposers](https://t.me/r3mes_proposers)

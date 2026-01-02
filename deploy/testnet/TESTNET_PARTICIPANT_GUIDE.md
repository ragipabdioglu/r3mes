# R3MES Testnet Katılım Rehberi

## 🎯 Genel Bakış

R3MES Testnet'e hoş geldiniz! Bu rehber, testnet'e nasıl katılacağınızı adım adım açıklar.

### Testnet Bilgileri

| Parametre | Değer |
|-----------|-------|
| Chain ID | `r3mes-testnet-1` |
| Token | R3MES / ur3mes |
| RPC | https://rpc.r3mes.network |
| REST API | https://rest.r3mes.network |
| Backend API | https://api.r3mes.network |
| Web Dashboard | https://r3mes.network |
| Faucet | https://r3mes.network/faucet |

---

## 🚀 Hızlı Başlangıç

### Seçenek 1: Web Dashboard (En Kolay)

1. https://r3mes.network adresine git
2. Keplr wallet'ı bağla
3. Faucet'ten test token al
4. Mining veya staking başlat

### Seçenek 2: Desktop Launcher (GPU Mining)

1. [Desktop Launcher](https://github.com/r3mes/releases) indir
2. Kurulumu tamamla
3. Wallet oluştur veya import et
4. Mining başlat

### Seçenek 3: CLI (Gelişmiş)

```bash
# CLI'ı indir
curl -sSL https://github.com/r3mes/releases/latest/download/r3mes-cli-linux-amd64 -o r3mes-cli
chmod +x r3mes-cli

# Wallet oluştur
./r3mes-cli wallet create

# Faucet'ten token al
./r3mes-cli faucet request <wallet-address>

# Mining başlat
./r3mes-cli miner start
```

---

## 💰 Faucet Kullanımı

### Web Dashboard

1. https://r3mes.network/faucet adresine git
2. Wallet adresini gir
3. "Request Tokens" butonuna tıkla
4. 10 R3MES alacaksın

### API

```bash
curl -X POST https://api.r3mes.network/faucet/request \
  -H "Content-Type: application/json" \
  -d '{"address": "remes1..."}'
```

### Limitler

- Her istek: 10 R3MES
- Günlük limit: 100 R3MES
- IP başına: 24 saatte 1 istek

---

## ⛏️ Mining (GPU Gerekli)

### Gereksinimler

- NVIDIA GPU (RTX 2060 veya üstü önerilir)
- CUDA 12.1+
- Docker
- 8GB+ RAM
- 50GB+ disk

### Desktop Launcher ile

1. Desktop Launcher'ı aç
2. "Mining" sekmesine git
3. "Start Mining" butonuna tıkla

### CLI ile

```bash
# Miner engine'i indir
git clone https://github.com/r3mes/miner-engine
cd miner-engine

# Konfigürasyon
cp .env.example .env
nano .env  # Wallet adresini gir

# Başlat
docker-compose up -d
```

### Miner Engine Konfigürasyonu

```env
# .env dosyası
WALLET_ADDRESS=remes1...
RPC_URL=https://rpc.r3mes.network
API_URL=https://api.r3mes.network
IPFS_URL=https://ipfs.r3mes.network

# GPU ayarları
CUDA_VISIBLE_DEVICES=0
BATCH_SIZE=32
```

---

## 🔒 Staking

### Web Dashboard ile

1. https://r3mes.network/staking adresine git
2. Validator seç
3. Stake miktarını gir
4. "Delegate" butonuna tıkla

### CLI ile

```bash
# Validator listesi
./r3mes-cli staking validators

# Stake et
./r3mes-cli staking delegate <validator-address> 1000r3mes
```

### Minimum Stake

- Delegator: 10 R3MES
- Validator: 1,000 R3MES

---

## 🗳️ Governance

### Proposal Oluştur

1. https://r3mes.network/governance adresine git
2. "Create Proposal" butonuna tıkla
3. Detayları doldur
4. Deposit yap (10 R3MES)

### Oy Ver

```bash
# Proposal listesi
./r3mes-cli governance proposals

# Oy ver
./r3mes-cli governance vote <proposal-id> yes
```

---

## 🖥️ Full Node Kurulumu

Kendi full node'unuzu çalıştırmak istiyorsanız:

### Gereksinimler

- 4 CPU
- 8GB RAM
- 100GB SSD
- Ubuntu 22.04

### Kurulum

```bash
# Binary'yi indir
curl -sSL https://github.com/r3mes/releases/latest/download/remesd-linux-amd64 -o remesd
chmod +x remesd
sudo mv remesd /usr/local/bin/

# Initialize
remesd init my-node --chain-id r3mes-testnet-1

# Genesis dosyasını indir
curl -sSL https://raw.githubusercontent.com/r3mes/R3MES/main/deploy/testnet/genesis.json > ~/.remes/config/genesis.json

# Seeds ekle
sed -i 's/seeds = ""/seeds = "seed1@38.242.246.6:26656"/' ~/.remes/config/config.toml

# Başlat
remesd start
```

### Systemd Service

```bash
sudo tee /etc/systemd/system/remesd.service > /dev/null <<EOF
[Unit]
Description=R3MES Node
After=network.target

[Service]
Type=simple
User=$USER
ExecStart=/usr/local/bin/remesd start
Restart=on-failure
RestartSec=10
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable remesd
sudo systemctl start remesd
```

---

## 🛡️ Validator Olma

### Gereksinimler

- Full node sync olmuş
- Minimum 1,000 R3MES stake
- 7/24 uptime

### Validator Oluştur

```bash
# Validator key oluştur
remesd keys add validator

# Faucet'ten token al (veya transfer)
# ...

# Validator oluştur
remesd tx staking create-validator \
  --amount=1000000000ur3mes \
  --pubkey=$(remesd tendermint show-validator) \
  --moniker="my-validator" \
  --chain-id=r3mes-testnet-1 \
  --commission-rate="0.10" \
  --commission-max-rate="0.20" \
  --commission-max-change-rate="0.01" \
  --min-self-delegation="1" \
  --from=validator
```

---

## 🔧 Keplr Wallet Kurulumu

### Manuel Ekleme

Keplr'da "Add Chain" ile şu bilgileri girin:

```json
{
  "chainId": "r3mes-testnet-1",
  "chainName": "R3MES Testnet",
  "rpc": "https://rpc.r3mes.network",
  "rest": "https://rest.r3mes.network",
  "bip44": {
    "coinType": 118
  },
  "bech32Config": {
    "bech32PrefixAccAddr": "remes",
    "bech32PrefixAccPub": "remespub",
    "bech32PrefixValAddr": "remesvaloper",
    "bech32PrefixValPub": "remesvaloperpub",
    "bech32PrefixConsAddr": "remesvalcons",
    "bech32PrefixConsPub": "remesvalconspub"
  },
  "currencies": [
    {
      "coinDenom": "R3MES",
      "coinMinimalDenom": "ur3mes",
      "coinDecimals": 6
    }
  ],
  "feeCurrencies": [
    {
      "coinDenom": "R3MES",
      "coinMinimalDenom": "ur3mes",
      "coinDecimals": 6,
      "gasPriceStep": {
        "low": 0.01,
        "average": 0.025,
        "high": 0.04
      }
    }
  ],
  "stakeCurrency": {
    "coinDenom": "R3MES",
    "coinMinimalDenom": "ur3mes",
    "coinDecimals": 6
  }
}
```

### Otomatik Ekleme

Web Dashboard'a bağlandığınızda otomatik olarak chain eklenecektir.

---

## 📊 Explorer & Monitoring

- **Web Dashboard**: https://r3mes.network
- **Block Explorer**: https://r3mes.network/network
- **API Docs**: https://api.r3mes.network/docs

---

## 🆘 Destek

- **Discord**: https://discord.gg/r3mes
- **Telegram**: https://t.me/r3mes
- **GitHub Issues**: https://github.com/r3mes/R3MES/issues
- **Email**: support@r3mes.network

---

## ⚠️ Önemli Notlar

1. **Bu bir testnet'tir** - Token'ların gerçek değeri yoktur
2. **Veriler sıfırlanabilir** - Testnet periyodik olarak resetlenebilir
3. **Bug bildirin** - Bulduğunuz hataları GitHub'da raporlayın
4. **Güvenlik** - Mainnet'te kullanacağınız wallet'ları testnet'te kullanmayın

---

## 🎁 Testnet Ödülleri

Aktif testnet katılımcıları mainnet lansmanında ödüllendirilecektir:

- Mining katılımı
- Bug raporları
- Topluluk katkıları
- Validator operasyonu

Detaylar için Discord'a katılın!

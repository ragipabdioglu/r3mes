# Validator Node Setup Guide

R3MES Network'te Validator olarak blok doğrulama yapmak için bu rehberi takip edin.

## Gereksinimler

### Donanım
- **CPU**: 4+ core (önerilen 8+ core)
- **RAM**: Minimum 16GB (önerilen 32GB)
- **Depolama**: 500GB+ SSD (NVMe önerilir)
- **İnternet**: Stabil bağlantı, statik IP önerilir

> **Not**: Validator için GPU gerekmez!

### Yazılım
- Ubuntu 20.04+ veya Debian 11+
- Go 1.21+
- Docker (opsiyonel)

## Stake Gereksinimi

Validator olmak için **100,000 REMES** stake etmeniz gerekiyor.

## Kurulum Adımları

### 1. Sistem Hazırlığı

```bash
# Sistem güncellemesi
sudo apt update && sudo apt upgrade -y

# Gerekli paketler
sudo apt install -y build-essential git curl jq

# Go kurulumu
wget https://go.dev/dl/go1.21.5.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.21.5.linux-amd64.tar.gz
echo 'export PATH=$PATH:/usr/local/go/bin:$HOME/go/bin' >> ~/.bashrc
source ~/.bashrc
```

### 2. remesd Kurulumu

```bash
# Repository klonlama
git clone https://github.com/r3mes/remes.git
cd remes

# Build
make install

# Versiyon kontrolü
remesd version
```

### 3. Node Başlatma

```bash
# Node başlatma
remesd init "your-moniker" --chain-id remes-mainnet-1

# Genesis dosyası indirme
curl -o ~/.remes/config/genesis.json https://raw.githubusercontent.com/r3mes/networks/main/mainnet/genesis.json

# Seeds ve peers ekleme
SEEDS="seed1@seed1.r3mes.network:26656,seed2@seed2.r3mes.network:26656"
sed -i "s/seeds = \"\"/seeds = \"$SEEDS\"/" ~/.remes/config/config.toml
```

### 4. Konfigürasyon

`~/.remes/config/config.toml` düzenleyin:

```toml
[p2p]
laddr = "tcp://0.0.0.0:26656"
external_address = "your-public-ip:26656"
max_num_inbound_peers = 100
max_num_outbound_peers = 50

[rpc]
laddr = "tcp://127.0.0.1:26657"

[mempool]
size = 10000
max_txs_bytes = 1073741824
```

`~/.remes/config/app.toml` düzenleyin:

```toml
minimum-gas-prices = "0.025uremes"

[api]
enable = true
address = "tcp://127.0.0.1:1317"

[grpc]
enable = true
address = "0.0.0.0:9090"
```

### 5. Systemd Service

```bash
sudo tee /etc/systemd/system/remesd.service > /dev/null <<EOF
[Unit]
Description=R3MES Daemon
After=network-online.target

[Service]
User=$USER
ExecStart=$(which remesd) start
Restart=always
RestartSec=3
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable remesd
sudo systemctl start remesd
```

### 6. Cüzdan Oluşturma

```bash
# Yeni cüzdan
remesd keys add validator

# Veya mevcut cüzdanı import
remesd keys add validator --recover
```

### 7. Validator Oluşturma

Node senkronize olduktan sonra:

```bash
# Senkronizasyon kontrolü
remesd status | jq .SyncInfo.catching_up
# false olmalı

# Validator oluşturma
remesd tx staking create-validator \
  --amount=100000000000uremes \
  --pubkey=$(remesd tendermint show-validator) \
  --moniker="your-moniker" \
  --chain-id=remes-mainnet-1 \
  --commission-rate="0.10" \
  --commission-max-rate="0.20" \
  --commission-max-change-rate="0.01" \
  --min-self-delegation="1" \
  --from=validator \
  --gas=auto \
  --gas-adjustment=1.5
```

## Kazanç Modeli

| Aktivite | Ödül |
|----------|------|
| Blok Ödülü | Stake oranına göre |
| Komisyon | Delegatörlerden %10 |
| Uptime Bonusu | %5 ekstra (99.9% uptime) |

## Monitoring

```bash
# Node durumu
remesd status | jq

# Validator bilgisi
remesd query staking validator $(remesd keys show validator --bech val -a)

# Missed blocks
remesd query slashing signing-info $(remesd tendermint show-validator)
```

## Güvenlik

### Sentry Node Mimarisi

Production için sentry node kullanın:

```
[Sentry 1] <---> [Validator] <---> [Sentry 2]
     |                                  |
     +----------[Internet]-------------+
```

### Key Management

```bash
# Validator key yedekleme
cp ~/.remes/config/priv_validator_key.json ~/backup/

# Node key yedekleme
cp ~/.remes/config/node_key.json ~/backup/
```

### Firewall

```bash
# UFW kuralları
sudo ufw allow 26656/tcp  # P2P
sudo ufw allow 26657/tcp  # RPC (sadece sentry'den)
sudo ufw enable
```

## Slashing Koruması

- **Downtime**: 10,000 blok kaçırma = %0.01 slash
- **Double Sign**: %5 slash + jail

Önlemler:
1. Yedek node hazır tutun
2. Monitoring kurulumu yapın
3. Alerting sistemi kurun

## Destek

- Discord: [discord.gg/r3mes](https://discord.gg/r3mes)
- Validator Chat: [t.me/r3mes_validators](https://t.me/r3mes_validators)

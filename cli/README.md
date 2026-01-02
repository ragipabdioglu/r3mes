# R3MES CLI Tools

R3MES blockchain için komut satırı araçları. Wallet yönetimi, miner operasyonları, node yönetimi ve governance işlemleri için kapsamlı CLI interface.

## 🚀 Hızlı Başlangıç

### Kurulum

#### Option 1: Pre-built Binary İndirme
```bash
# Linux/macOS için
curl -L https://github.com/r3mes/r3mes/releases/latest/download/r3mes-cli-linux-amd64 -o r3mes-cli
chmod +x r3mes-cli
sudo mv r3mes-cli /usr/local/bin/

# Windows için PowerShell
Invoke-WebRequest -Uri "https://github.com/r3mes/r3mes/releases/latest/download/r3mes-cli-windows-amd64.exe" -OutFile "r3mes-cli.exe"
```

#### Option 2: Source'dan Build Etme
```bash
# Repository'yi clone et
git clone https://github.com/r3mes/r3mes.git
cd r3mes/cli

# Build et
make build

# Veya tüm platformlar için
make build-all
```

### Environment Variables

CLI'ı kullanmadan önce gerekli environment variable'ları ayarlayın:

```bash
# Gerekli
export R3MES_RPC_ENDPOINT="https://rpc.r3mes.network:26657"
export R3MES_GRPC_ENDPOINT="grpc.r3mes.network:9090"

# Opsiyonel
export R3MES_CHAIN_ID="remes-mainnet"
export R3MES_WALLET_PATH="$HOME/.r3mes/wallets"
export R3MES_MINER_PORT="8080"
export R3MES_ENV="production"
```

## 📖 Kullanım

### Wallet İşlemleri

```bash
# Yeni wallet oluştur
r3mes-cli wallet create

# Mnemonic ile wallet import et
r3mes-cli wallet import "word1 word2 word3 ... word12"

# Private key ile wallet import et
r3mes-cli wallet import 0x1234567890abcdef...

# Wallet balance kontrol et
r3mes-cli wallet balance

# Belirli adres için balance
r3mes-cli wallet balance remes1abc123...

# Wallet'ları listele
r3mes-cli wallet list

# Wallet export et (private key ve mnemonic)
r3mes-cli wallet export
```

### Miner İşlemleri

```bash
# Miner başlat
r3mes-cli miner start

# Miner durdur
r3mes-cli miner stop

# Miner durumu kontrol et
r3mes-cli miner status

# Miner istatistikleri
r3mes-cli miner stats
```

### Node İşlemleri

```bash
# Node başlat
r3mes-cli node start

# Node durdur
r3mes-cli node stop

# Node durumu ve sync bilgisi
r3mes-cli node status

# Sync durumu kontrol et
r3mes-cli node sync
```

### Governance İşlemleri

```bash
# Aktif proposal'ları listele
r3mes-cli governance proposals

# Belirli proposal detayı
r3mes-cli governance proposal 1

# Proposal'a oy ver
r3mes-cli governance vote 1 yes
r3mes-cli governance vote 1 no
r3mes-cli governance vote 1 abstain
r3mes-cli governance vote 1 no_with_veto
```

### Konfigürasyon

```bash
# Mevcut konfigürasyonu göster
r3mes-cli config

# Konfigürasyon değeri ayarla
r3mes-cli config set rpc_endpoint https://new-rpc.r3mes.network:26657
```

### Genel

```bash
# Versiyon bilgisi
r3mes-cli version

# Yardım
r3mes-cli --help
r3mes-cli wallet --help
r3mes-cli miner --help
```

## 🔧 Development

### Build Requirements

- Go 1.21+
- Git

### Build Commands

```bash
# Development build
make dev

# Production build
make build

# All platforms
make build-all

# Run tests
make test

# Format code
make fmt

# Lint
make lint

# Security check
make security
```

### Project Structure

```
cli/
├── r3mes-cli/
│   ├── main.go          # Ana CLI uygulaması
│   └── go.mod           # Go dependencies
├── build.sh             # Linux/macOS build script
├── build.ps1            # Windows build script
├── Makefile             # Build automation
└── README.md            # Bu dosya
```

## 🔒 Güvenlik

### Wallet Güvenliği

- **Mnemonic Phrase**: 12 kelimelik mnemonic phrase'inizi güvenli bir yerde saklayın
- **Private Key**: Private key'inizi asla paylaşmayın
- **Encryption**: Wallet'ları mutlaka şifre ile encrypt edin
- **Backup**: Wallet backup'larınızı güvenli lokasyonlarda saklayın

### Environment Güvenliği

- Production ortamında localhost endpoint'leri kullanmayın
- Environment variable'ları güvenli şekilde yönetin
- API key'leri ve private key'leri log'larda expose etmeyin

## 🐛 Troubleshooting

### Yaygın Sorunlar

#### "Error: Required environment variable R3MES_RPC_ENDPOINT not set"
```bash
export R3MES_RPC_ENDPOINT="https://rpc.r3mes.network:26657"
```

#### "Error: Miner not running or stats endpoint unavailable"
```bash
# Miner'ın çalıştığından emin olun
r3mes-cli miner status

# Port'un doğru olduğunu kontrol edin
export R3MES_MINER_PORT="8080"
```

#### "Error: Invalid mnemonic phrase"
- Mnemonic phrase'in 12 kelime olduğundan emin olun
- Kelimelerin doğru sırada olduğunu kontrol edin
- Ekstra boşluk karakterleri olmadığından emin olun

#### "Error: Node not running"
```bash
# Node'un çalıştığından emin olun
r3mes-cli node start

# RPC endpoint'in erişilebilir olduğunu kontrol edin
curl $R3MES_RPC_ENDPOINT/status
```

### Debug Mode

Debug bilgileri için environment variable ayarlayın:
```bash
export R3MES_DEBUG=true
export R3MES_LOG_LEVEL=debug
```

## 📝 License

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](../LICENSE) dosyasına bakın.

## 🤝 Contributing

Katkıda bulunmak için [CONTRIBUTING.md](../CONTRIBUTING.md) dosyasını okuyun.

## 📞 Support

- GitHub Issues: https://github.com/r3mes/r3mes/issues
- Discord: https://discord.gg/r3mes
- Documentation: https://docs.r3mes.network
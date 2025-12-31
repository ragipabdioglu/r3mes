# Miner Node Setup Guide

R3MES Network'te Miner olarak katılmak için bu rehberi takip edin.

## Gereksinimler

### Donanım
- **GPU**: NVIDIA GPU (minimum 8GB VRAM, önerilen 12GB+)
- **RAM**: Minimum 16GB
- **Depolama**: 50GB+ SSD
- **İnternet**: Stabil bağlantı (minimum 10 Mbps)

### Yazılım
- Windows 10/11, macOS, veya Linux
- NVIDIA Driver (525.0+)
- CUDA Toolkit 11.8+

## Kurulum Adımları

### 1. Desktop Launcher İndirme

R3MES Desktop Launcher'ı indirin:
- [Windows](https://github.com/r3mes/desktop-launcher/releases/latest/download/r3mes-launcher-windows.exe)
- [macOS](https://github.com/r3mes/desktop-launcher/releases/latest/download/r3mes-launcher-macos.dmg)
- [Linux](https://github.com/r3mes/desktop-launcher/releases/latest/download/r3mes-launcher-linux.AppImage)

### 2. Cüzdan Oluşturma

Desktop Launcher'ı açın ve yeni bir cüzdan oluşturun:

1. "Create New Wallet" butonuna tıklayın
2. Güçlü bir şifre belirleyin
3. **ÖNEMLİ**: 24 kelimelik seed phrase'inizi güvenli bir yere kaydedin
4. Seed phrase'i doğrulayın

### 3. REMES Token Edinme

Miner olmak için minimum **1,000 REMES** stake etmeniz gerekiyor.

Token edinme yolları:
- Testnet için: [Faucet](https://faucet.r3mes.network) kullanın
- Mainnet için: Desteklenen borsalardan satın alın

### 4. Stake İşlemi

Desktop Launcher'da:

1. "Staking" sekmesine gidin
2. "Become a Miner" seçeneğini seçin
3. Stake miktarını girin (minimum 1,000 REMES)
4. İşlemi onaylayın

```bash
# CLI ile stake (opsiyonel)
remesd tx remes stake-miner 1000000000uremes \
  --from your-wallet \
  --chain-id remes-mainnet-1 \
  --gas auto \
  --gas-adjustment 1.5
```

### 5. Mining Başlatma

Stake işlemi onaylandıktan sonra:

1. Desktop Launcher'da "Mining" sekmesine gidin
2. GPU'nuzu seçin
3. "Start Mining" butonuna tıklayın

## Kazanç Modeli

| Aktivite | Ödül |
|----------|------|
| Blok Madenciliği | 10 REMES/blok |
| Model Eğitimi | Değişken (katkıya göre) |
| Uptime Bonusu | %5 ekstra (7/24 çalışma) |

## Sorun Giderme

### GPU Algılanmıyor
```bash
# NVIDIA driver kontrolü
nvidia-smi

# CUDA kontrolü
nvcc --version
```

### Yetersiz VRAM
- Batch size'ı düşürün (Settings > Mining > Batch Size)
- Diğer GPU kullanan uygulamaları kapatın

### Bağlantı Sorunları
- Firewall ayarlarını kontrol edin (Port 26656, 26657)
- VPN kullanıyorsanız devre dışı bırakın

## Güvenlik Önerileri

1. Seed phrase'inizi asla paylaşmayın
2. Düzenli yedekleme yapın
3. Güvenilir ağlarda çalışın
4. Desktop Launcher'ı güncel tutun

## Destek

- Discord: [discord.gg/r3mes](https://discord.gg/r3mes)
- Telegram: [t.me/r3mes](https://t.me/r3mes)
- GitHub Issues: [github.com/r3mes/issues](https://github.com/r3mes/issues)

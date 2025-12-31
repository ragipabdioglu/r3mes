# R3MES Network Roles

R3MES Network'te dört farklı rol bulunmaktadır. Her rol farklı gereksinimlere ve ödüllere sahiptir.

## Rol Karşılaştırması

| Rol | GPU Gerekli | Minimum Stake | Açıklama |
|-----|-------------|---------------|----------|
| [Miner](./miner-setup.md) | ✅ Evet | 1,000 REMES | AI model eğitimi ve blok madenciliği |
| [Serving](./serving-setup.md) | ✅ Evet | 1,000 REMES | AI inference hizmeti sunma |
| [Validator](./validator-setup.md) | ❌ Hayır | 100,000 REMES | Blok doğrulama ve ağ güvenliği |
| [Proposer](./proposer-setup.md) | ❌ Hayır | 50,000 REMES* | Governance teklifleri oluşturma |

*Proposer olmak için önce Validator olmanız gerekir (toplam 150,000 REMES)

## Hangi Rolü Seçmeliyim?

### GPU'nuz Varsa

**Miner** veya **Serving** rollerini değerlendirin:

- **Miner**: Aktif olarak AI modellerini eğitmek ve blok ödülleri kazanmak istiyorsanız
- **Serving**: Pasif gelir için inference hizmeti sunmak istiyorsanız

### GPU'nuz Yoksa

**Validator** veya **Proposer** rollerini değerlendirin:

- **Validator**: Ağ güvenliğine katkıda bulunmak ve stake ödülleri kazanmak istiyorsanız
- **Proposer**: Governance'a aktif katılım ve protokol geliştirme istiyorsanız

## Hızlı Başlangıç

### GPU ile (Miner)

```bash
# Desktop Launcher indirin ve kurun
# https://github.com/r3mes/desktop-launcher/releases

# Cüzdan oluşturun ve 1,000 REMES stake edin
# Mining'i başlatın
```

### GPU ile (Serving)

```bash
git clone https://github.com/r3mes/r3mes.git
cd r3mes
docker compose -f deploy/docker-compose.serving.yml up -d
```

### GPU'suz (Validator)

```bash
# remesd kurun
git clone https://github.com/r3mes/remes.git
cd remes && make install

# Node başlatın
remesd init "my-validator" --chain-id remes-mainnet-1
remesd start
```

### GPU'suz (Backend API)

```bash
git clone https://github.com/r3mes/r3mes.git
cd r3mes/deploy
docker compose -f docker-compose.infrastructure.yml up -d
```

## Detaylı Rehberler

- [Miner Setup Guide](./miner-setup.md)
- [Serving Node Setup Guide](./serving-setup.md)
- [Validator Setup Guide](./validator-setup.md)
- [Proposer Setup Guide](./proposer-setup.md)

## Destek

- Discord: [discord.gg/r3mes](https://discord.gg/r3mes)
- Telegram: [t.me/r3mes](https://t.me/r3mes)
- Forum: [forum.r3mes.network](https://forum.r3mes.network)

# R3MES Production Deployment Checklist

Bu doküman, R3MES projesini production'a çıkarmak için gereken tüm adımları içerir.

---

## 📋 Pre-Deployment Checklist

### ✅ 1. Genesis Hazırlığı

#### 1.1. Model'i IPFS'e Yükle
- [ ] Model dosyasını IPFS'e yükle (IPFS Desktop veya CLI kullan)
- [ ] IPFS CID'yi not al (örn: `QmaB5YKSNGuMzUccBupZQrhXT8efoYyMXqE8uWRHmPX2Lf`)
- [ ] IPFS public gateway testi yap: `python scripts/test_ipfs_gateway.py <CID>`
- [ ] Model dosyasını `models/` klasörüne kopyala

#### 1.2. Genesis Trap Jobs Oluştur
```bash
cd /home/rabdi/R3MES
python scripts/generate_genesis_traps.py --count 50 --output genesis_vault_entries.json
```

#### 1.3. Genesis'i Finalize Et
```bash
python scripts/finalize_genesis.py \
  --model-hash "QmaB5YKSNGuMzUccBupZQrhXT8efoYyMXqE8uWRHmPX2Lf" \
  --model-version "v1.0.0" \
  --chain-id "remes-mainnet-1" \
  --output remes/config/genesis.json \
  --vault-entries genesis_vault_entries.json
```

#### 1.4. Genesis'i Validate Et
```bash
python scripts/validate_genesis.py remes/config/genesis.json
```

---

### ✅ 2. Faucet Cüzdanı Hazırlama

#### 2.1. Faucet Cüzdanı Oluştur
```bash
cd /home/rabdi/R3MES/remes
./build/remesd keys add faucet-key --keyring-backend os
# Çıkan adresi not al (örn: remes1abc123...)
```

#### 2.2. Faucet Cüzdanını Genesis'e Ekle (Opsiyonel)
Genesis'e faucet cüzdanını eklemek için genesis.json'ı düzenle:
```json
{
  "app_state": {
    "bank": {
      "balances": [
        {
          "address": "remes1abc123...",
          "coins": [{"denom": "uremes", "amount": "1000000000"}]
        }
      ]
    }
  }
}
```

---

### ✅ 3. Environment Dosyalarını Hazırla

#### 3.1. Backend Environment
```bash
cd /home/rabdi/R3MES/backend
cp env.production.example .env.production
# .env.production dosyasını düzenle ve gerçek değerleri gir:
# - FAUCET_TREASURY_ADDRESS
# - CHAIN_ID=remes-mainnet-1
# - BLOCKCHAIN_REST_URL
# - FAUCET_KEY_NAME
# - REMESD_PATH
# - REMESD_HOME
```

#### 3.2. Web Dashboard Environment
```bash
cd /home/rabdi/R3MES/web-dashboard
cp env.production.example .env.local
# .env.local dosyasını düzenle:
# - NEXT_PUBLIC_BACKEND_URL
# - NEXT_PUBLIC_BLOCKCHAIN_RPC_URL
# - NEXT_PUBLIC_BLOCKCHAIN_REST_URL
```

#### 3.3. Miner Engine Environment
```bash
cd /home/rabdi/R3MES/miner-engine
cp env.production.example .env.production
# .env.production dosyasını düzenle:
# - CHAIN_ID=remes-mainnet-1
# - R3MES_NODE_GRPC_URL
# - PRIVATE_KEY (miner'ın private key'i)
```

---

### ✅ 4. Blockchain Node Kurulumu

#### 4.1. Blockchain Node'u Custom Genesis ile Initialize Et
```bash
cd /home/rabdi/R3MES
python scripts/init_genesis_with_custom.py \
  --genesis remes/config/genesis.json \
  --chain-id remes-mainnet-1 \
  --moniker "production-node" \
  --home ~/.remesd
```

#### 4.2. Validator Key Oluştur (Opsiyonel)
```bash
cd /home/rabdi/R3MES/remes
./build/remesd keys add validator-key --keyring-backend os
# Validator address'i not al
```

#### 4.3. Genesis Validator Oluştur (Opsiyonel)
```bash
# Gentx oluştur
./build/remesd genesis gentx validator-key 1000000uremes \
  --chain-id remes-mainnet-1 \
  --moniker "production-validator" \
  --keyring-backend os \
  --home ~/.remesd

# Gentx'leri topla
./build/remesd genesis collect-gentxs --home ~/.remesd
```

#### 4.4. Genesis'i Tekrar Validate Et
```bash
./build/remesd genesis validate-genesis --home ~/.remesd
```

#### 4.5. Node'u Başlat
```bash
./build/remesd start --home ~/.remesd
```

#### 4.6. Systemd Service Kur (Production için)
```bash
sudo cp scripts/systemd/remesd.service /etc/systemd/system/
# Service dosyasını düzenle: ExecStart ve WorkingDirectory'yi ayarla
sudo systemctl daemon-reload
sudo systemctl enable remesd
sudo systemctl start remesd
```

---

### ✅ 5. Faucet Cüzdanını Fonla

Node başladıktan sonra, genesis validator cüzdanından faucet cüzdanına token gönder:

```bash
cd /home/rabdi/R3MES/remes

# Genesis validator key'in adresini al
GENESIS_ADDRESS=$(./build/remesd keys show validator-key -a --keyring-backend os)
FAUCET_ADDRESS="remes1abc123..."  # Yukarıda aldığın faucet adresi

# Token gönder (örn: 10,000 REMES = 10000000000uremes)
./build/remesd tx bank send \
  $GENESIS_ADDRESS \
  $FAUCET_ADDRESS \
  10000000000uremes \
  --chain-id remes-mainnet-1 \
  --from validator-key \
  --keyring-backend os \
  --yes \
  --gas auto \
  --gas-adjustment 1.5

# Bakiye kontrolü
./build/remesd query bank balances $FAUCET_ADDRESS --chain-id remes-mainnet-1
```

---

### ✅ 6. Backend Deployment

#### 6.1. Backend'i Test Et
```bash
cd /home/rabdi/R3MES/backend
R3MES_ENV=production python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

#### 6.2. Backend'i Production'da Çalıştır
- Systemd service kullan (önerilen)
- Docker kullan
- PM2 kullan

#### 6.3. Backend Health Check
```bash
curl http://localhost:8000/health
curl http://localhost:8000/system/version
curl http://localhost:8000/system/time
```

---

### ✅ 7. Web Dashboard Deployment

#### 7.1. Production Build
```bash
cd /home/rabdi/R3MES/web-dashboard
R3MES_ENV=production npm run build
npm start
```

#### 7.2. Nginx Reverse Proxy (Opsiyonel)
```bash
sudo cp nginx.conf.example /etc/nginx/sites-available/r3mes-dashboard
sudo ln -s /etc/nginx/sites-available/r3mes-dashboard /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

### ✅ 8. Network Testing

#### 8.1. Smoke Tests
```bash
bash scripts/smoke-tests.sh https://r3mes.network https://api.r3mes.network
```

#### 8.2. IPFS Gateway Test
```bash
python scripts/test_ipfs_gateway.py QmaB5YKSNGuMzUccBupZQrhXT8efoYyMXqE8uWRHmPX2Lf
```

#### 8.3. Faucet Test
```bash
curl -X POST http://localhost:8000/faucet/claim \
  -H "Content-Type: application/json" \
  -d '{"address": "remes1testaddress..."}'
```

---

### ✅ 9. DNS ve Network Visibility (Localhost Test için)

#### 9.1. /etc/hosts Güncelle (Local Test için)
```bash
sudo nano /etc/hosts
# Şunları ekle:
127.0.0.1    r3mes.network
127.0.0.1    api.r3mes.network
127.0.0.1    rpc.r3mes.network
127.0.0.1    releases.r3mes.network
```

#### 9.2. Local CDN Server (Engine Download için)
```bash
cd /home/rabdi/R3MES/miner-engine/releases
python3 -m http.server 9000
# Tauri launcher'a CDN URL olarak: http://127.0.0.1:9000/engine-v1.0.0.zip
```

---

### ✅ 10. Final Kontroller

#### 10.1. Tüm Servislerin Durumu
```bash
# Blockchain
curl http://localhost:26657/status
curl http://localhost:1317/cosmos/base/tendermint/v1beta1/node_info

# Backend
curl http://localhost:8000/health
curl http://localhost:8000/system/version

# Web Dashboard
curl http://localhost:3000
```

#### 10.2. Log Kontrolü
```bash
# Blockchain logs
journalctl -u remesd -f

# Backend logs (eğer systemd kullanıyorsan)
journalctl -u r3mes-backend -f
```

---

## 🚨 Kritik Notlar

1. **Private Keys**: Faucet private key'i ASLA commit etme. Environment variable olarak sakla.
2. **Chain ID**: Tüm componentlerde aynı chain ID kullan (`remes-mainnet-1`).
3. **Genesis**: Genesis dosyasını validate etmeden node'u başlatma.
4. **Faucet Balance**: Faucet cüzdanının yeterli balance'a sahip olduğundan emin ol.
5. **Network Ports**: Firewall'da gerekli portları aç (26656, 26657, 1317, 9090, 4001, 8000, 3000).

---

## 📚 İlgili Dokümanlar

- [Environment Variables](docs/16_environment_variables.md)
- [Production Deployment](docs/12_production_deployment.md)
- [Mainnet Launch Checklist](docs/MAINNET_LAUNCH_CHECKLIST.md)
- [Installation Guide](docs/INSTALLATION.md)


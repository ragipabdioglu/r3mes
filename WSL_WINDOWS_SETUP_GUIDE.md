# R3MES WSL + Windows Kurulum ve Test Rehberi

**Durum**: WSL'de servisler, Windows'ta launcher çalışacak şekilde yapılandırma

---

## 📋 Genel Bakış

Bu rehber, R3MES projesini **WSL (Windows Subsystem for Linux)** üzerinde çalıştırıp, **Windows'ta Tauri Desktop Launcher** kullanarak yönetmenizi sağlar.

### Mimari

```
┌─────────────────────────────────────────────────────────┐
│                    WSL (Linux)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Backend     │  │  Blockchain  │  │   Miner     │ │
│  │  (Port 8000) │  │  (Port 26657) │  │   Engine    │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
│         │                 │                  │          │
│         └─────────────────┼──────────────────┘          │
│                           │                              │
└───────────────────────────┼──────────────────────────────┘
                            │
                    (Port Forwarding)
                            │
┌───────────────────────────┼──────────────────────────────┐
│                    Windows                                │
│                           │                              │
│                  ┌────────▼────────┐                     │
│                  │  Tauri Launcher │                     │
│                  │  (Desktop App) │                     │
│                  └─────────────────┘                     │
│                                                           │
│                  ┌─────────────────┐                    │
│                  │  Web Dashboard   │                    │
│                  │  (Port 3000)     │                    │
│                  └─────────────────┘                    │
└───────────────────────────────────────────────────────────┘
```

**Önemli Notlar:**
- WSL2 otomatik olarak port'ları Windows'a forward eder
- Windows'tan `localhost:8000`, `localhost:26657` gibi port'lara erişilebilir
- Launcher WSL'deki servislere `localhost` üzerinden bağlanır

---

## 🔧 Ön Gereksinimler

### WSL'de Gerekli

1. **Python 3.8+**
   ```bash
   python3 --version
   ```

2. **Node.js 18+** (Web Dashboard için)
   ```bash
   node --version
   ```

3. **Go 1.22+** (Blockchain node için)
   ```bash
   go version
   ```

4. **IPFS** (Dağıtık depolama için)
   ```bash
   ipfs version
   ```

5. **Git**
   ```bash
   git --version
   ```

### Windows'ta Gerekli

1. **Rust** (Tauri launcher için)
   - [Rustup](https://rustup.rs/) indirin ve kurun
   - PowerShell'de: `rustc --version`

2. **Node.js 18+** (Tauri frontend için)
   - [Node.js](https://nodejs.org/) indirin ve kurun
   - PowerShell'de: `node --version`

3. **Visual Studio Build Tools** (Windows'ta Rust build için)
   - [Visual Studio Installer](https://visualstudio.microsoft.com/downloads/)
   - "C++ ile masaüstü geliştirme" workload'ını seçin

---

## 📦 Adım 1: WSL'de Projeyi Hazırlama

### 1.1 Projeyi Klonla/İndir

```bash
# WSL terminalinde
cd ~
git clone <repository-url> R3MES
# veya mevcut projeyi kullan
cd ~/R3MES
```

### 1.2 Backend Bağımlılıklarını Kur

```bash
# WSL terminalinde
cd ~/R3MES/backend

# Virtual environment oluştur
python3 -m venv venv
source venv/bin/activate

# Bağımlılıkları kur
pip install --upgrade pip
pip install -r requirements.txt
```

### 1.3 Miner Engine Bağımlılıklarını Kur

```bash
# WSL terminalinde
cd ~/R3MES/miner-engine

# Virtual environment oluştur
python3 -m venv venv
source venv/bin/activate

# Bağımlılıkları kur
pip install --upgrade pip
pip install -r requirements.txt

# Miner'ı editable mode'da kur
pip install -e .
```

### 1.4 Blockchain Node'u Build Et

```bash
# WSL terminalinde
cd ~/R3MES/remes

# Go modülleri indir
go mod download

# Node'u build et
make build
# veya
go build -o build/remesd ./cmd/remesd
```

### 1.5 Web Dashboard Bağımlılıklarını Kur

```bash
# WSL terminalinde
cd ~/R3MES/web-dashboard

# Node modules'ları kur
npm install
```

### 1.6 IPFS Kurulumu

```bash
# WSL terminalinde
# IPFS kurulumu (eğer yoksa)
wget https://dist.ipfs.tech/go-ipfs/v0.24.0/go-ipfs_v0.24.0_linux-amd64.tar.gz
tar -xvzf go-ipfs_v0.24.0_linux-amd64.tar.gz
cd go-ipfs
sudo ./install.sh

# IPFS'i initialize et (ilk kez)
ipfs init

# IPFS daemon'u başlat (arka planda)
ipfs daemon &
```

---

## 🚀 Adım 2: WSL'de Servisleri Başlatma

### 2.1 IPFS Başlat (WSL Terminal 1)

```bash
# WSL terminalinde
ipfs daemon
```

**Beklenen**: IPFS API `http://localhost:5001` adresinde çalışmalı

**Kontrol**:
```bash
# Başka bir WSL terminalinde
curl http://localhost:5001/api/v0/version
```

---

### 2.2 Blockchain Node Başlat (WSL Terminal 2)

```bash
# Yeni bir WSL terminali aç (Windows Terminal'de yeni tab)

cd ~/R3MES/remes

# Node'u başlat
./build/remesd start --home ~/.remes
```

**Beklenen**: Node başlar ve port'lar açılır:
- `localhost:26657` (RPC)
- `localhost:9090` (gRPC)
- `localhost:1317` (REST API)

**Kontrol**:
```bash
# WSL terminalinde
curl http://localhost:26657/status | jq .result.sync_info.latest_block_height
```

**Not**: WSL2 port'ları otomatik olarak Windows'a forward edilir. Windows'tan `http://localhost:26657` erişilebilir.

---

### 2.3 Backend Başlat (WSL Terminal 3)

```bash
# Yeni bir WSL terminali aç

cd ~/R3MES/backend
source venv/bin/activate

# Backend'i başlat
python3 -m app.main
# veya
python3 run_backend.py
```

**Beklenen**: Backend `http://localhost:8000` adresinde çalışmalı

**Kontrol**:
```bash
# WSL terminalinde
curl http://localhost:8000/health
```

**Windows'tan Kontrol**:
```powershell
# PowerShell'de
Invoke-WebRequest -Uri http://localhost:8000/health
```

---

### 2.4 Miner Engine Başlat (WSL Terminal 4) - Opsiyonel

```bash
# Yeni bir WSL terminali aç

cd ~/R3MES/miner-engine
source venv/bin/activate

# Miner'ı başlat
r3mes-miner start
# veya
python3 miner_engine.py
```

**Beklenen Loglar**:
```
[INFO] Miner started
[INFO] GPU detected: CPU (veya NVIDIA GPU)
[INFO] Model loaded: BitNet 1.58-bit
[INFO] Training started
```

---

### 2.5 Web Dashboard Başlat (WSL Terminal 5) - Opsiyonel

```bash
# Yeni bir WSL terminali aç

cd ~/R3MES/web-dashboard
npm run dev
```

**Beklenen**: Web Dashboard `http://localhost:3000` adresinde çalışmalı

**Windows'tan Erişim**: Tarayıcıda `http://localhost:3000` açın

---

## 🪟 Adım 3: Windows'ta Launcher'ı Hazırlama

### 3.1 Launcher Bağımlılıklarını Kur

```powershell
# PowerShell'de (Windows)

# Rust kurulumu kontrolü
rustc --version

# Node.js kurulumu kontrolü
node --version

# Launcher dizinine git
cd C:\Users\<YourUser>\R3MES\desktop-launcher-tauri
# veya WSL'den erişim:
# \\wsl$\Ubuntu\home\<user>\R3MES\desktop-launcher-tauri

# Bağımlılıkları kur
npm install
```

### 3.2 Launcher'ı Build Et

```powershell
# PowerShell'de
cd desktop-launcher-tauri

# Development mode
npm run tauri dev

# Production build
npm run tauri build
```

**Not**: İlk build uzun sürebilir (Rust toolchain indirme).

---

## 🔌 Adım 4: Launcher'ı WSL Servislerine Bağlama

### 4.1 WSL IP Adresini Bul

```bash
# WSL terminalinde
hostname -I
# veya
ip addr show eth0 | grep "inet\b" | awk '{print $2}' | cut -d/ -f1
```

**Örnek Çıktı**: `172.20.10.2`

### 4.2 Launcher Yapılandırması

Launcher'ın WSL'deki servislere bağlanması için:

**Seçenek 1: localhost Kullan (Önerilen)**

WSL2 otomatik port forwarding yapar, `localhost` kullanabilirsiniz:

```typescript
// desktop-launcher-tauri/src/config.ts
export const config = {
  backendUrl: 'http://localhost:8000',
  blockchainRpc: 'http://localhost:26657',
  blockchainGrpc: 'localhost:9090',
  ipfsApi: 'http://localhost:5001',
};
```

**Seçenek 2: WSL IP Adresi Kullan**

Eğer localhost çalışmazsa, WSL IP adresini kullanın:

```typescript
// desktop-launcher-tauri/src/config.ts
export const config = {
  backendUrl: 'http://172.20.10.2:8000',  // WSL IP adresi
  blockchainRpc: 'http://172.20.10.2:26657',
  blockchainGrpc: '172.20.10.2:9090',
  ipfsApi: 'http://172.20.10.2:5001',
};
```

---

## 🧪 Adım 5: Test Etme

### 5.1 Servis Durumu Kontrolü

**WSL'den**:
```bash
# Backend
curl http://localhost:8000/health

# Blockchain Node
curl http://localhost:26657/status

# IPFS
curl http://localhost:5001/api/v0/version
```

**Windows'tan (PowerShell)**:
```powershell
# Backend
Invoke-WebRequest -Uri http://localhost:8000/health

# Blockchain Node
Invoke-WebRequest -Uri http://localhost:26657/status

# IPFS
Invoke-WebRequest -Uri http://localhost:5001/api/v0/version
```

### 5.2 Launcher'dan Test

1. **Launcher'ı Başlat**:
   ```powershell
   cd desktop-launcher-tauri
   npm run tauri dev
   ```

2. **Launcher UI'da Kontrol**:
   - Node durumu: "Running" olmalı
   - Backend durumu: "Running" olmalı
   - IPFS durumu: "Running" olmalı

3. **Miner Başlat**:
   - Launcher UI'dan "Start Miner" butonuna tıklayın
   - veya WSL terminalinde manuel başlatın

### 5.3 Web Dashboard'dan Test

1. **Tarayıcıda Aç**: `http://localhost:3000`

2. **Kontroller**:
   - Ana sayfa yükleniyor mu?
   - Network stats görünüyor mu?
   - Chat sayfası çalışıyor mu?
   - Mine sayfası miner stats gösteriyor mu?

---

## 🔧 Sorun Giderme

### Problem 1: Windows'tan WSL Port'larına Erişilemiyor

**Çözüm 1: WSL Port Forwarding Kontrolü**

```powershell
# PowerShell'de (Admin olarak)
netsh interface portproxy show all
```

**Çözüm 2: WSL IP Adresini Kullan**

WSL IP adresini bulun ve launcher config'de kullanın:

```bash
# WSL terminalinde
hostname -I
```

### Problem 2: Launcher WSL Servislerine Bağlanamıyor

**Çözüm**: Firewall kontrolü

```powershell
# PowerShell'de (Admin olarak)
# WSL için firewall kuralı ekle
New-NetFirewallRule -DisplayName "WSL" -Direction Inbound -InterfaceAlias "vEthernet (WSL)" -Action Allow
```

### Problem 3: Miner GPU Bulamıyor

**Çözüm**: WSL'de NVIDIA GPU desteği

```bash
# WSL terminalinde
# NVIDIA driver kontrolü
nvidia-smi

# CUDA kontrolü
nvcc --version
```

**Not**: WSL'de GPU desteği için Windows'ta NVIDIA driver kurulu olmalı.

### Problem 4: Port Kullanımda Hatası

**Çözüm**: Port'u kullanan process'i bul ve durdur

```bash
# WSL terminalinde
# Port 8000'i kullanan process
lsof -i :8000

# Process'i durdur
kill -9 <PID>
```

### Problem 5: Launcher Build Hatası

**Çözüm**: Rust toolchain güncelle

```powershell
# PowerShell'de
rustup update
cargo clean
npm run tauri build
```

---

## 📊 Servis Portları Özeti

| Servis | Port | WSL | Windows Erişimi |
|--------|------|-----|------------------|
| Backend | 8000 | ✅ | `http://localhost:8000` |
| Blockchain RPC | 26657 | ✅ | `http://localhost:26657` |
| Blockchain gRPC | 9090 | ✅ | `localhost:9090` |
| Blockchain REST | 1317 | ✅ | `http://localhost:1317` |
| IPFS API | 5001 | ✅ | `http://localhost:5001` |
| IPFS Gateway | 8080 | ✅ | `http://localhost:8080` |
| Web Dashboard | 3000 | ✅ | `http://localhost:3000` |
| Miner Stats | 8080 | ✅ | `http://localhost:8080/stats` |

---

## 🎯 Hızlı Başlatma Script'i (WSL)

WSL'de tüm servisleri tek seferde başlatmak için:

```bash
# WSL terminalinde
cd ~/R3MES

# Script'i çalıştırılabilir yap
chmod +x start_test.sh

# Servisleri başlat
./start_test.sh
```

Bu script:
- ✅ Backend'i başlatır (port 8000)
- ✅ Frontend'i başlatır (port 3000)
- ✅ Health check yapar

**Not**: Blockchain node ve IPFS'i manuel başlatmanız gerekir.

---

## 🛑 Servisleri Durdurma

### WSL'de

```bash
# Tüm Python process'lerini durdur
pkill -f "app.main"
pkill -f "miner_engine"
pkill -f "remesd"
pkill -f "ipfs daemon"

# veya script ile
cd ~/R3MES
./stop_test.sh
```

### Windows'ta

```powershell
# Launcher'ı kapat
# Launcher UI'dan "Exit" butonuna tıklayın
```

---

## 📝 Notlar

1. **WSL2 Port Forwarding**: WSL2 otomatik olarak port'ları Windows'a forward eder, ancak bazen manuel yapılandırma gerekebilir.

2. **Firewall**: Windows Firewall WSL trafiğini engelleyebilir, gerekirse kural ekleyin.

3. **GPU Desteği**: WSL'de GPU desteği için Windows'ta NVIDIA driver kurulu olmalı ve WSL2 GPU desteği aktif olmalı.

4. **Performance**: WSL2'de I/O performansı native Linux'tan daha yavaş olabilir, özellikle disk I/O.

5. **Network**: WSL2 kendi network namespace'ini kullanır, ancak `localhost` üzerinden erişilebilir.

---

## 🎉 Başarılı Kurulum İşaretleri

✅ **WSL'de**:
- Backend `http://localhost:8000` çalışıyor
- Blockchain node `http://localhost:26657` çalışıyor
- IPFS `http://localhost:5001` çalışıyor
- Miner engine çalışıyor (opsiyonel)

✅ **Windows'ta**:
- Launcher başlatılabiliyor
- Launcher WSL servislerine bağlanabiliyor
- Web Dashboard `http://localhost:3000` erişilebilir

✅ **Test**:
- Miner başlatılabiliyor
- Training başlıyor
- Gradient'ler hesaplanıyor
- Blockchain'e submit ediliyor

---

## 🆘 Yardım

Detaylı bilgi için:
- [TEST_GUIDE.md](./TEST_GUIDE.md) - Kapsamlı test kılavuzu
- [README_INSTALLATION.md](./README_INSTALLATION.md) - Kurulum kılavuzu
- [QUICK_START.md](./QUICK_START.md) - Hızlı başlangıç
- [desktop-launcher-tauri/README.md](./desktop-launcher-tauri/README.md) - Launcher dokümantasyonu

---

**Son Güncelleme**: 2025-01-27


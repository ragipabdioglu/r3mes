# R3MES User Onboarding & Guides

## Overview

R3MES provides comprehensive onboarding guides for different user roles. Each role has specific installation steps and requirements.

## 🎯 Which Role Is Right for You?

**💻 Have a Powerful Graphics Card?** → **[Become a Miner](#miner-onboarding)** (Earn Tokens)  
**🖥️ Want to Provide AI Model Inference?** → **[Become a Serving Node](serving.md)** (Inference Serving)  
**🖥️ Have a Server?** → **[Become a Validator](#validator-onboarding)** (Manage the Network)  
**📊 Want to Aggregate Gradients?** → **[Become a Proposer](proposer.md)** (Gradient Aggregation)  
**📦 Have a Dataset?** → **[Become a Dataset Provider](#dataset-provider-onboarding)**

---

## 🖥️ Choosing Your Interface

Before you start, choose the interface that best fits your needs:

### Desktop Launcher (Recommended for Node Operators & Miners)

The **Desktop Launcher** is the easiest way to get started if you want to:
- Run a blockchain node, miner, or validator locally
- Manage infrastructure processes (Node, Miner, IPFS)
- Monitor system status and logs in a native desktop app
- Use embedded IPFS management

**Quick Start**: Download and install the Desktop Launcher from the releases page. The setup wizard will guide you through the process. See [Desktop Launcher Documentation](10_desktop_launcher.md) for details.

### Web Dashboard (Recommended for Users & Investors)

The **Web Dashboard** is best if you want to:
- Use AI inference (chat interface)
- Stake tokens or participate in governance
- Explore the network (3D globe, miners table)
- Access from any device/browser (no installation)

**Quick Start**: Open the Web Dashboard at `https://dashboard.r3mes.network` (production) or `http://localhost:3000` (local). Connect your wallet and start using features immediately. See [Web Dashboard Documentation](08_web_dashboard_command_center.md) for details.

### CLI/Python SDK (Recommended for Developers)

Use the **CLI/Python SDK** if you:
- Are a developer building integrations
- Prefer command-line interfaces
- Need programmatic control
- Want to automate tasks with scripts

**Quick Start**: Install via `pip install r3mes` and use commands like `r3mes-miner start`. See [Quick Start Guide](QUICK_START.md) for CLI instructions.

---

---

## 🔐 Role Registration

After selecting which roles you want to participate in, you need to register your node on the blockchain. Registration differs from role selection:

- **Role Selection**: Choosing which roles you want to participate in (done during setup)
- **Role Registration**: Actually registering your node on the blockchain with those roles (requires blockchain transaction)

### Registration Methods

1. **Web Dashboard (Recommended)**: Navigate to `/roles` page, select roles, set stake amount, and submit transaction
2. **Desktop Launcher**: Role selection is done during setup wizard, but registration should be completed via web dashboard
3. **Blockchain CLI**: Use `remesd tx remes register-node` command directly

**Detailed Guide**: See [Role Registration Guide](role_registration_guide.md) for step-by-step instructions for each method.

### Quick Registration Checklist

- [ ] Wallet connected (Keplr or compatible wallet)
- [ ] Sufficient balance for stake and transaction fees
- [ ] Selected roles meet minimum stake requirements
- [ ] Authorization obtained (for Validator/Proposer roles)
- [ ] Registration transaction submitted
- [ ] Transaction confirmed on blockchain

**Note**: Public roles (Miner, Serving) can be registered immediately. Restricted roles (Validator, Proposer) require authorization before registration. See [Role Access Control](ROLE_ACCESS_CONTROL.md) for authorization details.

---

## 💻 System Requirements

| Component | Minimum | Recommended | Critical Note |
| :--- | :--- | :--- | :--- |
| **GPU** | GTX 1660 (6GB) / CPU Mode | RTX 3060+ (12GB+) | **NVIDIA GPU recommended, CPU mode supported (slow).** |
| **RAM** | 8 GB System Memory | 16 GB+ System Memory | Sufficient for model weights (thanks to GGUF format). |
| **Disk** | 10 GB Free Space | 20 GB+ SSD | Disk requirement reduced thanks to GGUF format. |
| **OS** | Ubuntu 20.04 / WSL2 / Windows | Ubuntu 22.04 LTS | Windows users can use WSL2 or native Windows. |
| **CPU Mode** | Supported | - | CPU mode works but GPU is much faster. |

**⚠️ Important**: CPU mode is supported but very slow (100x slower). GPU is recommended.

---

## ⛏️ Miner Onboarding

### Who Is This For?

Users who contribute to the network by training AI models and earn tokens in return.

### Getting Started: Choose Your Method

You have three options to start mining:

1. **Desktop Launcher (Easiest - Recommended)**: Download and install the Desktop Launcher. The setup wizard will guide you through everything automatically.
2. **Web Dashboard**: Use the Web Dashboard to monitor mining if you're already running a miner via CLI.
3. **CLI/Python SDK**: Use command-line tools for full control and automation.

---

### Method 1: Desktop Launcher (Recommended)

The Desktop Launcher is the easiest way to get started with mining. It includes:
- Setup wizard with hardware compatibility checks
- Automatic IPFS setup
- Process management (Start/Stop Miner)
- Real-time monitoring and logs

**Steps:**

1. **Download Desktop Launcher**: Get the installer from the releases page
2. **Run Setup Wizard**: The wizard will:
   - Check your hardware (GPU, RAM, disk)
   - Help you select roles (Miner, Serving, Validator, Proposer)
   - Create or import a wallet
   - Configure blockchain connection
   - Set up IPFS automatically
3. **Start Mining**: Click "Start Miner" in the launcher
4. **Monitor Progress**: View stats in the System Status Panel and Mining Dashboard

See [Desktop Launcher Documentation](10_desktop_launcher.md) for detailed instructions.

---

### Method 2: CLI/Python SDK

For advanced users who prefer command-line interfaces or need automation.

#### Step 1: PyPI Package Installation

**Linux / macOS Users:**
```bash
# Python 3.10+ required
python3 --version  # Should be 3.10+

# Install from PyPI
pip install r3mes

# Or install specific version
pip install r3mes==0.1.0
```

**Windows Users (with NVIDIA GPU):**

⚠️ **Important**: On Windows, you must manually install PyTorch first for your graphics card to work at full performance.

```powershell
# 1. First install CUDA-enabled PyTorch:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. Then install R3MES:
pip install r3mes
```

#### Step 2: Initial Setup (Setup Wizard)

```bash
# Run setup wizard
r3mes-miner setup
```

**Wizard Flow**:

1. **System Requirements Check**
   ```
   ✅ Python 3.10+ detected
   ✅ PyTorch installed
   ✅ GPU detected: NVIDIA GeForce RTX 3090
   ⚠️  IPFS daemon not running (will use embedded)
   ```

2. **Wallet Creation**
   ```
   💼 Wallet Setup
   1. Create new wallet (recommended)
   2. Use existing private key
   
   Choose: 1
   ✅ Wallet created: ~/.r3mes/wallet.json
   ✅ Address: remes1abc...
   
   🎁 Faucet Check...
   ✅ Welcome Bonus: 0.1 REMES airdropped for gas fees! (Ready to mine)
   ```
   
   **Note**: The setup wizard automatically requests initial balance from the faucet. This covers the gas fees needed for your first transactions.

3. **Blockchain Configuration**
   ```
   🌐 Blockchain Connection
   Mainnet URL: node.r3mes.network:9090
   Chain ID: remes-1
   ```

4. **IPFS Configuration** (Automatic)
   ```
   📦 IPFS Setup
   ⚙️  Initializing embedded IPFS node...
   ⬇️  Downloading IPFS binary (v0.24.0)...
   🚀 IPFS Daemon started at localhost:5001
   ✅ IPFS connection: Active (Private Node)
   ```
   
   **Note**: IPFS is automatically installed and started. No user prompt. Embedded IPFS daemon is used (cost: 0, no central server required).

5. **Mining Parameters**
   ```
   ⛏️  Mining Configuration
   Model Size: 768 (default)
   LoRA Rank: 8 (default)
   Batch Size: 4 (default)
   ```

6. **Connection Test**
   ```
   🔌 Testing Connection...
   ✅ Blockchain connection: OK
   ✅ IPFS connection: OK
   ```

#### Step 3: Start Mining

```bash
# Miner'ı başlat
r3mes-miner start

# Veya continuous mining mode
r3mes-miner start --continuous --batch-size 5 --pause-seconds 2
```

**Beklenen Loglar**:
```
[INFO] Miner started
[INFO] GPU detected: NVIDIA GeForce RTX 3090
[INFO] Model loaded: BitNet 1.58-bit
[INFO] Shard assignment: Shard 23
[INFO] Training started
[INFO] Gradient computed
[INFO] Uploaded to IPFS: Qm...
[INFO] Submitted to blockchain: tx_hash=...
[INFO] ✅ Stats HTTP server started on http://localhost:8080/stats
```

#### Step 4: Systemd Service (Production)

```bash
# Service dosyasını kopyala
sudo cp scripts/systemd/r3mes-miner.service /etc/systemd/system/

# Service'i aktif et
sudo systemctl daemon-reload
sudo systemctl enable r3mes-miner
sudo systemctl start r3mes-miner

# Status kontrol
sudo systemctl status r3mes-miner
```

---

### Method 3: Web Dashboard Setup

The Web Dashboard is primarily for using AI inference, staking, and governance. However, you can use it to:

- Monitor mining statistics (if you're running a miner via CLI or Desktop Launcher)
- View network statistics and explorer
- Manage your wallet and transactions
- Participate in staking and governance

**Note**: The Web Dashboard does not run mining processes directly. You need to run the miner via Desktop Launcher or CLI, then monitor it through the Web Dashboard.

See [Web Dashboard Documentation](08_web_dashboard_command_center.md) for detailed instructions.

---

## 🖥️ Validator Onboarding

### Who Is This For?
Blockchain network'ünü yöneten, transaction'ları validate eden node operator.

### Step 1: Blockchain Node Installation

```bash
# Go 1.24.0+ installation
wget https://go.dev/dl/go1.24.0.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.24.0.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin

# Projeyi klonla
git clone https://github.com/r3mes/r3mes.git
cd r3mes/remes

# Build et
make build
```

### Step 2: Node Initialization

```bash
# Otomatik script ile
bash ../scripts/install_validator.sh

# Veya manuel:
./build/remesd init my-validator --chain-id remes-1
./build/remesd keys add validator
```

### Step 3: Genesis and Validator Key

```bash
# Genesis'i al (kurucudan)
# ~/.remesd/config/genesis.json dosyasına kopyala

# Validator oluştur
./build/remesd tx staking create-validator \
  --amount 1000000uremes \
  --from validator \
  --chain-id remes-1 \
  --yes
```

### Step 4: Systemd Service

```bash
# Service dosyasını kopyala
sudo cp scripts/systemd/remesd.service /etc/systemd/system/

# Service'i aktif et
sudo systemctl daemon-reload
sudo systemctl enable remesd
sudo systemctl start remesd
```

---

## 📊 Proposer Onboarding

### Who Is This For?
Gradient'leri toplayıp aggregate eden, model update'leri oluşturan kullanıcı.

### Requirements
- Validator node çalışıyor olmalı
- Yeterli stake (proposer olmak için)
- IPFS erişimi (gradient download için)

### Step 1: Proposer Registration

```bash
# Proposer olarak kayıt ol
./build/remesd tx remes register-proposer \
  --from proposer-key \
  --chain-id remes-1 \
  --yes
```

### Step 2: Start Aggregation

```bash
# Aggregation script'i çalıştır
python scripts/aggregate_gradients.py \
  --round-id 1 \
  --gradient-hashes Qm... Qm... Qm...
```

---

## 📦 Dataset Provider Onboarding

### Who Is This For?
Training dataset'leri sağlayan, blockchain'de dataset proposal oluşturan kullanıcı.

### Step 1: Dataset Preparation

```bash
# CSV'yi JSONL'e çevir
python dataset/convert_csv_to_jsonl.py \
  dataset/haberler.csv \
  -o dataset/haberler.jsonl \
  --format text \
  --no-category
```

### Step 2: Upload Dataset to IPFS

```bash
# IPFS'e yükle
ipfs add dataset/haberler.jsonl

# CID'yi kaydet
# Örnek: QmXxxx...
```

### Step 3: Create Dataset Proposal

```bash
# Terminal'den:
./build/remesd tx remes propose-dataset \
  --dataset-ipfs-hash QmXxxx... \
  --deposit 1000uremes \
  --from dataset-provider \
  --chain-id remes-1 \
  --yes

# Veya Web Dashboard'dan (Governance paneli):
# 1. Network sayfasına git
# 2. Governance panelini aç
# 3. "Create Proposal" butonuna tıkla
# 4. Dataset proposal oluştur
```

---

## 🔧 Setup Wizard Details

### System Requirements Check

Setup wizard şunları kontrol eder:
- ✅ Python 3.10+ version
- ✅ PyTorch installation (CUDA support check)
- ✅ GPU detection (NVIDIA/AMD/Intel)
- ✅ RAM amount
- ✅ Disk space
- ✅ IPFS daemon status (if embedded IPFS will be used)

### Wallet Creation

**Option 1: Create New Wallet (Recommended)**
- Wallet is automatically created
- Private key is stored securely
- Wallet address is automatically obtained
- Automatic request is sent to faucet

**Option 2: Use Existing Private Key**
- Private key is entered as hex string
- Address is automatically derived
- Faucet request is not made (must be done manually)

### Faucet Integration

Setup wizard, wallet oluşturulduktan sonra otomatik olarak faucet'e istek gönderir:

```python
# miner-engine/r3mes/utils/faucet.py
result = request_faucet(wallet_address)
if result['success']:
    print(f"✅ Welcome Bonus: {result.get('amount', '0.1')} REMES airdropped!")
```

**Faucet URLs**:
- Mainnet: `https://faucet.r3mes.network/api/faucet/request`
- Testnet: `https://testnet-faucet.r3mes.network/api/faucet/request`
- Local: `http://localhost:8080/api/faucet/request`

**Rate Limiting**: Once per day per IP/address

**Production Critical Fix**: "Chicken-Egg Problem" solved - new users now automatically receive initial balance (0.1 REMES). Faucet message is clearer and more prominent.

### Embedded IPFS Daemon

The setup wizard automatically installs and starts the IPFS daemon:

```python
# miner-engine/r3mes/utils/ipfs_manager.py
# Automatic IPFS binary download (Windows/macOS/Linux)
# Automatic IPFS daemon startup
# Repository initialization
```

**Features**:
- Platform-specific binary download (v0.24.0)
- Automatic daemon startup
- Repository initialization
- Port: 5001 (default)

**No User Prompt**: IPFS is automatically installed and started.

**Production Critical Fix**: IPFS confusion resolved - users are no longer asked "Local or Remote?" question. Embedded IPFS daemon is automatically used.

---

## 🪟 Windows + WSL Installation Guide

### On Windows

1. **WSL Installation** (PowerShell - Administrator):
   ```powershell
   wsl --install -d Ubuntu-22.04
   ```

2. **Downloads for Windows**:
   - Visual Studio Code (WSL extension ile)
   - Windows Terminal (opsiyonel)

### In WSL

1. **Python Installation**:
   ```bash
   sudo apt update
   sudo apt install python3.10 python3-pip python3-venv
   ```

2. **CUDA-Enabled PyTorch Installation**:
   ```bash
   # CUDA driver must be installed on Windows
   # WSL'de PyTorch CUDA desteği için:
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **R3MES Installation**:
   ```bash
   pip install r3mes
   r3mes-miner setup
   ```

### Port Forwarding

WSL ports are automatically forwarded to Windows:
- `localhost:5001` (IPFS)
- `localhost:26657` (Tendermint RPC)
- `localhost:1317` (REST API)
- `localhost:9090` (gRPC)
- `localhost:8080` (Stats Server)

### Test Scenario (Windows + WSL)

**Terminal 1 (WSL)**: IPFS
```bash
ipfs daemon
```

**Terminal 2 (WSL)**: Blockchain Node
```bash
cd ~/R3MES
./scripts/node_control.sh start
```

**Terminal 3 (WSL)**: Miner
```bash
cd ~/R3MES/miner-engine
source venv/bin/activate
r3mes-miner start
```

**Terminal 4 (WSL)**: Web Dashboard
```bash
cd ~/R3MES/web-dashboard
npm run dev
```

**Windows Tarayıcı**: `http://localhost:3000`

---

## 🔍 Hardware Check Utility

Setup wizard, sistem gereksinimlerini otomatik kontrol eder:

```python
# miner-engine/r3mes/utils/hardware_check.py
gpu_info = get_gpu_info()  # VRAM, compute capability
ram_info = get_ram_info()  # Total RAM
disk_info = get_disk_info()  # Available disk space
```

**Kontrol Edilenler**:
- GPU VRAM (minimum 12GB önerilir)
- Sistem RAM (minimum 16GB)
- Disk alanı (minimum 100GB)
- CUDA availability

**Uyarılar**:
- Yetersiz VRAM: "⚠️ GPU VRAM yetersiz. CPU mode kullanılacak (çok yavaş)."
- Yetersiz RAM: "⚠️ Sistem RAM yetersiz. Model yüklenemeyebilir."
- CPU mode: "⚠️ CPU mode çok yavaştır (100x). GPU önerilir."

**Production Critical Fix**: "Donanım Hayal Kırıklığı" çözüldü - detaylı sistem gereksinimleri tablosu eklendi ve başlangıçta gösteriliyor. Hardware check utility ile otomatik kontrol yapılıyor.

---

## 🌐 Endpoint Checker Utility

Setup wizard, blockchain endpoint'lerinin erişilebilirliğini kontrol eder:

```python
# miner-engine/r3mes/utils/endpoint_checker.py
# DNS resolution kontrolü
# Endpoint connectivity testi
# Fallback IP'ler (production'da set edilecek)
```

**Kontrol Edilenler**:
- DNS resolution (node.r3mes.network)
- gRPC connectivity (port 9090)
- REST API connectivity (port 1317)
- RPC connectivity (port 26657)

---

## 📝 Configuration File

Setup wizard, konfigürasyonu `~/.r3mes/config.yaml` dosyasına kaydeder:

```yaml
private_key: "0x..."
wallet_path: "~/.r3mes/wallet.json"
blockchain_url: "node.r3mes.network:9090"
chain_id: "remes-1"
model_size: 768
lora_rank: 8
gradient_accumulation_steps: 4
top_k_compression: 0.1
use_tls: false
```

---

## 🚀 Quick Start (5 Dakika)

### Adım 1: Node'u Başlat

```bash
cd ~/R3MES

# Node durumunu kontrol et
./scripts/node_control.sh status

# Node'u başlat (eğer çalışmıyorsa)
./scripts/node_control.sh start

# Node'un başladığını kontrol et (5 saniye bekle)
sleep 5
curl http://localhost:26657/status | jq .result.sync_info.latest_block_height
```

**Beklenen**: Block height sayısı görünmeli (örn: 100, 200, ...)

### Adım 2: IPFS Başlat

```bash
# IPFS daemon'u başlat (yeni terminal)
ipfs daemon

# Veya embedded IPFS kullan (otomatik, setup wizard'da)
```

**Beklenen**: IPFS API `http://localhost:5001` adresinde çalışmalı

#### Step 3: Start Miner

```bash
# Prepare miner environment
cd ~/R3MES/miner-engine
source venv/bin/activate

# Initial setup (one time only)
r3mes-miner setup

# Start miner
r3mes-miner start
```

**Expected Logs**:
```
[INFO] Miner started
[INFO] GPU detected: NVIDIA GeForce RTX 3090 (veya CPU mode)
[INFO] Model loaded: BitNet 1.58-bit
[INFO] Shard assignment: Shard 23
[INFO] Training started
[INFO] Gradient computed
[INFO] Uploaded to IPFS: Qm...
[INFO] Submitted to blockchain: tx_hash=...
[INFO] ✅ Stats HTTP server started on http://localhost:8080/stats
```

#### Step 4: Monitor with Dashboard

```bash
# Start dashboard (new terminal)
cd ~/R3MES/web-dashboard
npm run dev
```

**In Browser**: `http://localhost:3000`

**What You'll See**:
- ✅ Active miners: 1
- ✅ Block height: Artıyor
- ✅ Network hash rate: > 0
- ✅ Miner locations: 3D globe'da görünür

---

## 🔧 Troubleshooting

### "Faucet is currently unavailable"
- You can manually get tokens from the faucet
- You can use testnet
- You can run a faucet server for local development

### "IPFS binary not found"
- Embedded IPFS is automatically installed
- Manual installation: use `ipfs daemon` command

### "Blockchain connection failed"
- Check endpoint URLs
- Test DNS resolution
- Check firewall settings

### "Insufficient funds"
- Get tokens from faucet
- Use testnet
- Add balance in genesis (local development)

---

## 📦 PyPI Package Publishing

**IMPORTANT**: PyPI package publishing will be done at the END of the project. [[memory:12332172]]

### Package Status

✅ **Package structure ready**:
- `pyproject.toml` - Package configuration completed
- `build_and_upload_pypi.sh` - Build script ready
- CLI entry point (`r3mes-miner`) working
- All dependencies defined

### Publishing Steps

1. **Test Upload to TestPyPI**:
   ```bash
   cd miner-engine
   ./scripts/build_and_upload_pypi.sh
   python -m twine upload --repository testpypi dist/*
   ```

2. **Test Installation from TestPyPI**:
   ```bash
   pip install -i https://test.pypi.org/simple/ r3mes
   r3mes-miner --help
   ```

3. **Upload to Real PyPI** (AT THE END):
   ```bash
   python -m twine upload dist/*
   ```

**Details**: See `PIP_PAKET_YAYINLAMA_REHBERI.md` file.

---

This onboarding guide provides a comprehensive starting point for all users who want to join the R3MES network.

---

## 📚 Related Documentation

- [Quick Start Guide](QUICK_START.md) - Get started in 5 minutes
- [Desktop Launcher Documentation](10_desktop_launcher.md) - Desktop Launcher features
- [Web Dashboard Documentation](08_web_dashboard_command_center.md) - Web Dashboard features
- [Role Registration Guide](role_registration_guide.md) - Register your node roles
- [Installation Guide](INSTALLATION.md) - Detailed installation instructions

---

**Last Updated**: 2025-01-15  
**Maintained by**: R3MES Development Team


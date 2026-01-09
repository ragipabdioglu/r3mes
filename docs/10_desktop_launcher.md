# R3MES Desktop Launcher

## Overview

The R3MES Desktop Launcher enables users to manage the R3MES system (Node, Miner, IPFS) from a single native desktop application. It is the **recommended interface for node operators, miners, and validators** who need native desktop integration and local process management.

**Technology**: Tauri 1.5 (Rust + React)  
**Platform**: Windows, macOS, Linux

---

## 🖥️ Desktop Launcher Overview

### What Is It?

The Desktop Launcher is a native desktop application built with the modern Tauri framework. It provides system tray integration, process management, and automatic IPFS management features. The launcher is designed for users who want to run R3MES infrastructure components (blockchain nodes, miners, validators) on their local machines.

### Key Features Overview

- **Setup Wizard**: Hardware compatibility checks, role selection, automated configuration
- **Process Management**: Start/Stop Node, Miner, IPFS with real-time monitoring
- **System Status Panel**: Chain sync, IPFS status, model download, node health
- **Live Logs Viewer**: Real-time log streaming with filtering and search
- **Wallet Management**: Local keystore with OS keychain integration
- **Model Downloader**: Progress tracking, resume capability, integrity verification
- **Mining Dashboard**: Hashrate, earnings, GPU temperature, training metrics
- **System Tray Integration**: Background operation support
- **Silent Auto-Updates**: Automatic component updates without user intervention

### Technical Stack

- **Tauri 1.5**: Native desktop framework (Rust backend, Web frontend)
- **React + TypeScript**: Frontend UI
- **Rust**: Backend process management
- **Vite**: Build tool

### Features

#### 1. Setup Wizard - Pre-flight Checks

When you first launch the Desktop Launcher, it automatically checks if your computer is suitable for mining and node operations.

**Hardware Checks**:
- **Docker Check**: Is Docker running? (Download link shown if not installed)
- **GPU Check**: Is NVIDIA driver installed? (`nvidia-smi` command runs in background)
- **Disk Space**: Sufficient space for model (28GB) and chain data? (Minimum 100GB SSD)
- **RAM Check**: Minimum 16GB system memory
- **CUDA Version**: CUDA 12.1+ compatibility

**Wizard Flow**:
1. Hardware compatibility check screen
2. **Role Selection**: Select which roles your node will perform (Miner, Serving, Validator, Proposer)
3. Review selected roles and system requirements
4. Complete setup
5. **Role Registration**: After setup, register your selected roles on the blockchain via web dashboard (`/roles`) or CLI

**Note**: Role selection in the setup wizard configures your local setup. To actually participate in the network, you must complete blockchain registration via the web dashboard or blockchain CLI. See [Role Registration Guide](role_registration_guide.md) for detailed instructions.

**Automatic Installation**:
- Docker: Automatic installation script for Linux/macOS
- Python: Automatic installation via package manager
- Node.js: Automatic installation via package manager
- CUDA: User guidance (manual installation required)

#### 2. Process Management

**Node Management**:
- Start/Stop blockchain node
- Real-time status monitoring
- Log viewing

**Miner Management**:
- Start/Stop miner engine
- Training metrics display
- GPU status monitoring

**IPFS Management**:
- Automatic IPFS daemon startup (Managed Sidecar)
- IPFS status monitoring
- Embedded IPFS (no external installation required)

#### 3. System Status Panel

The main screen displays a "System Status" panel with the following indicators:

**Chain Sync Status**:
- 🟢 Chain Sync: 99.9% (Block height: 12,345)
- Real-time block height updates
- Sync progress bar

**IPFS Status**:
- 🟢 IPFS: Connected (Peer count: 42)
- IPFS daemon status
- Peer connection count

**Model Status**:
- 🟡 Model Status: Downloading (45%)...
- 🟢 Model Status: Ready (BitNet b1.58 - 28GB)
- Model download progress (critical: user should see progress while downloading 28GB)
- Model version and size information

**Node Status**:
- 🟢 Node: Running (Port: 26657)
- 🟡 Node: Syncing
- 🔴 Node: Stopped

#### 4. Live Logs Viewer

**Embedded Log Viewer**:
- Process tabs (Node, Miner, IPFS) - separate log viewing for each process
- Real-time log streaming (auto-refresh every 2 seconds)
- Python script (miner-engine) stdout outputs displayed in real-time
- Docker container logs (if used)
- Red-colored warnings in error situations
- Log level filtering (Error, Warning, Info, Debug)
- Log search feature - full-text search
- Log export (download as txt file)
- Color-coded log levels (ERROR: red, WARN: yellow, INFO: blue, DEBUG: gray)

**Log Sources**:
- Miner Engine logs (Python stdout/stderr)
- Blockchain Node logs (remesd output)
- IPFS daemon logs
- System errors and warnings

#### 5. Wallet Management (Local Keystore)

**Real Blockchain Integration**:
- **Balance Query**: Queries blockchain REST API for actual wallet balance
- **BIP39 Mnemonic**: Proper 12/24-word mnemonic generation using `bip39` crate
- **Private Key Import**: Secp256k1 private key import with validation
- **Wallet Encryption**: AES-256-GCM encryption for wallet export
- **Keychain Storage**: OS keychain integration for secure private key storage

**Wallet Operations**:
- `create_wallet()`: Generates new BIP39 mnemonic and derives address
- `import_wallet_from_mnemonic()`: Imports wallet from 12/24-word mnemonic
- `import_wallet_from_private_key()`: Imports wallet from hex private key
- `export_wallet()`: Exports encrypted wallet backup (AES-256-GCM)
- `get_wallet_info()`: Gets wallet address and balance from blockchain

**Blockchain Query**:
- Uses `reqwest` for HTTP REST API queries
- Queries Cosmos SDK bank module for balance
- Environment variable: `BLOCKCHAIN_REST_URL` (default: `http://localhost:1317`)

#### 6. Model Downloader

**Real HTTP Download**:
- **HuggingFace Download**: Real HTTP download using `reqwest` with streaming
- **IPFS Fallback**: Falls back to IPFS gateway if HuggingFace fails
- **Progress Tracking**: Real-time download progress with callback support
- **Resume Capability**: Supports resuming interrupted downloads
- **Disk Space Check**: Checks available disk space before download
- **SHA256 Verification**: Verifies downloaded file integrity

**Features**:
- Async download with `tokio` runtime
- Progress callbacks for UI updates
- Automatic retry on network errors
- File integrity verification
- Disk space validation

**Usage**:
```rust
let downloader = ModelDownloader::new()?;
let model_path = downloader.download_model(&config, Some(Box::new(|progress| {
    println!("Progress: {:.2}%", progress.percentage);
}))).await?;
```

**Wallet Manager**:
- "Create New Wallet" button
- "Import Existing Private Key/Mnemonic" option
- Private keys are stored in the operating system's secure vault:
  - **Windows**: Windows Credential Manager
  - **macOS**: Keychain
  - **Linux**: Secret Service (libsecret)

**Wallet Features**:
- Wallet address display
- Balance viewing (R3MES token)
- Transaction history (local)
- Export wallet (encrypted backup)

**Security**:
- Private keys are never shown as plain text in the GUI
- Keychain/SecretStorage usage is mandatory
- Mnemonic phrase display (only once, on secure screen)

#### 6. Mining Dashboard (Statistics)

The main screen displays the following widgets:

**Current Hashrate/Loss**:
- 🔥 Hashrate: 1,234 gradients/hour
- 📉 Current Loss: 0.1234 (decreasing: ✅)
- Training speed and loss trend graph

**Estimated Earnings**:
- 💰 Estimated Earnings: 12.5 R3MES/day
- 💵 Current Balance: 45.2 R3MES
- Reward calculation (based on gradient quality)

**GPU Temperature**:
- 🌡️ GPU Temp: 65°C (Normal)
- ⚠️ GPU Temp: 85°C (High - Warning)
- 🔴 GPU Temp: 95°C (Critical - Mining should be stopped)
- Fan speed and power draw information

**Additional Metrics**:
- VRAM Usage: 18GB / 24GB (75%)
- Training Epoch: 42
- Gradient Norm: 0.001234
- Uptime: 2d 5h 30m

#### 7. System Tray Integration

- System tray icon
- Context menu (Show/Hide/Quit)
- Background operation support
- Status indicator (mining aktif mi?)

#### 8. Status Dashboard

- Process status cards
- Real-time metrics
- Health indicators

### Installation

```bash
cd launcher
npm install
npm run tauri:dev  # Development
npm run tauri:build  # Production build
```

### Managed Sidecar: IPFS Integration

The Launcher starts and manages an isolated IPFS process internally:

- **No external installation required from the user**
- IPFS daemon is automatically started by the Launcher
- IPFS daemon stops when the Launcher closes
- IPFS data directory: `~/.r3mes/ipfs/` (isolated)

**Advantages**:
- Zero-configuration setup
- Isolated IPFS instance
- Automatic cleanup on exit

### Hardware Compatibility Layer

The following checks are performed at launcher startup:

1. **GPU Detection**
   - Is NVIDIA GPU present?
   - Is CUDA driver installed?
   - Is CUDA version compatible? (≥12.1)

2. **VRAM Check**
   - Minimum: 12GB (RTX 3060)
   - Recommended: 24GB+ (RTX 3090/4090)

3. **System Resources**
   - RAM: Minimum 16GB, Recommended 32GB+
   - Disk: Minimum 100GB SSD, Recommended 500GB NVMe

4. **Network Connectivity**
   - Is blockchain endpoint accessible?
   - Is IPFS gateway accessible?

### Silent Auto-Update

The golden rule of production: You cannot tell 1000 people to "download a new exe". If you find a bug in the system or update the model (e.g., BitNet v2 is released), automatic updates must be performed.

#### Update Strategy

Launcher her açılışta güncellemeyi kontrol etmeli ve kullanıcıya hissettirmeden indirip dosyaları değiştirmeli.

**Update Components**:
1. **Chain Binary Updates**: `remesd` binary güncellemeleri
2. **Model Weights Updates**: BitNet base model güncellemeleri (BitNet v1 → v2)
3. **Miner Engine Updates**: `miner.py` ve miner engine kod güncellemeleri
4. **Launcher Updates**: Tauri launcher güncellemeleri

#### Silent Auto-Update Implementation

```rust
// desktop-launcher-tauri/src-tauri/src/updater.rs
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::fs;

#[derive(Debug, Serialize, Deserialize)]
struct UpdateManifest {
    version: String,
    chain_binary: UpdateInfo,
    model_weights: UpdateInfo,
    miner_engine: UpdateInfo,
    launcher: UpdateInfo,
}

#[derive(Debug, Serialize, Deserialize)]
struct UpdateInfo {
    version: String,
    download_url: String,
    checksum: String,
    required: bool,  // If false, update is optional
}

pub struct SilentUpdater {
    manifest_url: String,
    update_dir: PathBuf,
}

impl SilentUpdater {
    pub fn new() -> Self {
        Self {
            manifest_url: "https://releases.r3mes.network/manifest.json".to_string(),
            update_dir: dirs::home_dir()
                .unwrap()
                .join(".r3mes")
                .join("updates"),
        }
    }
    
    pub async fn check_and_update(&self) -> Result<(), String> {
        // 1. Fetch manifest
        let manifest = self.fetch_manifest().await?;
        
        // 2. Check each component
        self.update_chain_binary(&manifest.chain_binary).await?;
        self.update_model_weights(&manifest.model_weights).await?;
        self.update_miner_engine(&manifest.miner_engine).await?;
        self.update_launcher(&manifest.launcher).await?;
        
        Ok(())
    }
    
    async fn update_miner_engine(&self, update_info: &UpdateInfo) -> Result<(), String> {
        let current_version = self.get_miner_engine_version()?;
        
        if current_version == update_info.version {
            return Ok(());  // Already up to date
        }
        
        println!("🔄 Updating miner engine: {} → {}", current_version, update_info.version);
        
        // Download new miner engine
        let download_path = self.download_file(&update_info.download_url).await?;
        
        // Verify checksum
        self.verify_checksum(&download_path, &update_info.checksum)?;
        
        // Replace old miner engine
        let miner_path = self.get_miner_engine_path()?;
        fs::copy(&download_path, &miner_path)
            .map_err(|e| format!("Failed to replace miner engine: {}", e))?;
        
        println!("✅ Miner engine updated to {}", update_info.version);
        Ok(())
    }
    
    async fn update_model_weights(&self, update_info: &UpdateInfo) -> Result<(), String> {
        let current_version = self.get_model_version()?;
        
        if current_version == update_info.version {
            return Ok(());  // Already up to date
        }
        
        println!("🔄 Updating model weights: {} → {}", current_version, update_info.version);
        
        // Download new model weights
        let download_path = self.download_file(&update_info.download_url).await?;
        
        // Verify checksum
        self.verify_checksum(&download_path, &update_info.checksum)?;
        
        // Extract model weights to model directory
        let model_dir = self.get_model_directory()?;
        self.extract_model_weights(&download_path, &model_dir)?;
        
        println!("✅ Model weights updated to {}", update_info.version);
        Ok(())
    }
    
    async fn fetch_manifest(&self) -> Result<UpdateManifest, String> {
        let response = reqwest::get(&self.manifest_url)
            .await
            .map_err(|e| format!("Failed to fetch manifest: {}", e))?;
        
        let manifest: UpdateManifest = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse manifest: {}", e))?;
        
        Ok(manifest)
    }
    
    fn verify_checksum(&self, file_path: &PathBuf, expected_checksum: &str) -> Result<(), String> {
        use sha2::{Sha256, Digest};
        use std::fs::File;
        use std::io::Read;
        
        let mut file = File::open(file_path)
            .map_err(|e| format!("Failed to open file: {}", e))?;
        
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 8192];
        
        loop {
            let bytes_read = file.read(&mut buffer)
                .map_err(|e| format!("Failed to read file: {}", e))?;
            
            if bytes_read == 0 {
                break;
            }
            
            hasher.update(&buffer[..bytes_read]);
        }
        
        let computed_hash = format!("{:x}", hasher.finalize());
        
        if computed_hash != expected_checksum {
            return Err(format!(
                "Checksum mismatch: expected {}, got {}",
                expected_checksum, computed_hash
            ));
        }
        
        Ok(())
    }
}
```

#### Update Manifest Format

```json
{
  "version": "1.2.0",
  "chain_binary": {
    "version": "v0.1.2",
    "download_url": "https://github.com/r3mes/remesd/releases/download/v0.1.2/remesd-linux-amd64",
    "checksum": "abc123...",
    "required": true
  },
  "model_weights": {
    "version": "bitnet-v2",
    "download_url": "https://ipfs.r3mes.network/ipfs/QmModelV2...",
    "checksum": "def456...",
    "required": false
  },
  "miner_engine": {
    "version": "1.2.0",
    "download_url": "https://github.com/r3mes/miner-engine/releases/download/v1.2.0/miner-engine.tar.gz",
    "checksum": "ghi789...",
    "required": true
  },
  "launcher": {
    "version": "1.2.0",
    "download_url": "https://github.com/r3mes/launcher/releases/download/v1.2.0/r3mes-launcher.AppImage",
    "checksum": "jkl012...",
    "required": false
  }
}
```

#### Update Flow

1. **Launcher Startup**: Manifest is checked on every launch
2. **Background Download**: New versions are downloaded in the background
3. **Checksum Verification**: Downloaded files are verified for integrity
4. **Silent Replacement**: Files are replaced without user notification
5. **Restart Prompt**: If launcher update exists, restart is recommended (optional)

**Key Features**:
- **Silent Updates**: No user prompt, automatic updates
- **Checksum Verification**: Downloaded files are verified for integrity (SHA256)
- **Component-Specific**: Each component (chain, model, miner, launcher) can be updated separately
- **Required vs Optional**: Some updates are mandatory, some are optional
- **Rollback Support**: Return to previous version in case of faulty update
- **Real HTTP Fetch**: Uses `reqwest` crate for actual HTTP downloads (no mock data)
- **Progress Tracking**: Real-time download progress with resume capability
- **Error Handling**: Proper error handling with retry mechanism

### OTA Update Strategy (Legacy - Detailed)

The Launcher provides automatic update support:

1. **Chain Binary Updates**: `remesd` binary updates
2. **Model Weights Updates**: BitNet base model updates
3. **Launcher Updates**: Tauri launcher updates

**Update Flow**:
1. Launcher checks version on startup
2. If new version exists, notification is shown to user
3. Automatic download starts with user approval
4. Restart is recommended after download completes

**Update Sources**:
- Chain: GitHub Releases
- Models: IPFS Gateway
- Launcher: GitHub Releases

**Note**: Silent Auto-Update (above) is the production-ready version of this strategy.

---

## ⚠️ Legacy: Control Panel (Tkinter)

**Status**: Legacy (Old) - **Do Not Use**

The old Tkinter-based Control Panel is now marked as legacy. New users should use the **Tauri Desktop Launcher** or **Web Dashboard**.

**Legacy File**: `legacy/r3mes_control_panel.py`

**Why Legacy?**
- Tkinter is an old GUI technology
- Modern Tauri launcher has better features
- Web Dashboard is more modern and accessible
- Tauri launcher provides cross-platform native performance
- Tauri launcher has modern features like system tray and auto-update

**Recommended Alternatives:**
1. **Web Dashboard** (Recommended): `cd web-dashboard && npm run dev`
2. **Tauri Desktop Launcher**: `cd desktop-launcher-tauri && npm run tauri:dev`

**If you still want to use it (Not Recommended):**
```bash
cd ~/R3MES/legacy
python3 r3mes_control_panel.py
```

---

## 🚀 Quick Start

### Desktop Launcher (Recommended)

```bash
cd ~/R3MES/launcher
npm install
npm run tauri:dev
```

### System Dependencies (Linux)

```bash
sudo apt update && sudo apt install -y \
    libwebkit2gtk-4.1-dev \
    libjavascriptcoregtk-4.1-dev \
    libsoup2.4-dev \
    build-essential \
    libgtk-3-dev \
    libayatana-appindicator3-dev \
    librsvg2-dev \
    libglib2.0-dev \
    libgdk-pixbuf2.0-dev \
    libpango1.0-dev \
    libatk1.0-dev \
    pkg-config
```

### Windows

Visual Studio Installer'dan **"C++ ile masaüstü geliştirme"** workload'ını seçin.

---

## 📊 Feature Comparison: Desktop Launcher vs. Web Dashboard

### When to Use Desktop Launcher

Use the Desktop Launcher if you:
- Want to run a blockchain node locally
- Want to mine and train AI models
- Need embedded IPFS management
- Require system tray integration for background operation
- Prefer native desktop application
- Want local process management and monitoring

### When to Use Web Dashboard

Use the Web Dashboard if you:
- Want to use AI inference (chat interface)
- Want to stake tokens or participate in governance
- Want to explore the network (3D globe, miners table)
- Prefer web-based interface (no installation)
- Want to access from any device/browser
- Don't need to run local infrastructure

### Feature Comparison Matrix

| Feature | Desktop Launcher | Web Dashboard | Notes |
|---------|-----------------|---------------|--------|
| **Network Explorer** | ❌ | ✅ | 3D globe visualization |
| **Staking** | ❌ | ✅ | No node required |
| **Governance** | ❌ | ✅ | Proposal voting |
| **Bridge** | ❌ | ✅ | Cross-chain transfers |
| **Mining** | ✅ | ❌ | Hardware required |
| **Model Training** | ✅ | ❌ | GPU/CPU required |
| **Local Node** | ✅ | ❌ | Run full node |
| **Wallet Signing** | ✅ | ✅ | Local signing (Launcher) / Browser wallet (Dashboard) |
| **IPFS Management** | ✅ | ❌ | Embedded IPFS daemon |
| **System Tray** | ✅ | ❌ | Background operation |
| **Process Monitoring** | ✅ | ❌ | Real-time local process monitoring |
| **Setup Wizard** | ✅ | ❌ | Hardware compatibility checks |
| **Model Downloader** | ✅ | ❌ | Progress tracking, resume capability |
| **Live Logs** | ✅ | ❌ | Real-time log streaming |
| **AI Chat** | ❌ | ✅ | Streaming chat interface |

### Integration

Both interfaces work together:
- Desktop Launcher manages local infrastructure (node, miner, IPFS)
- Web Dashboard accesses the network through the blockchain and provides user-facing features
- Role registration is done via Web Dashboard (`/roles` page) even if you use Desktop Launcher for local management

---

## 🔐 Security Notes

- Launcher only runs on localhost
- Private keys are not stored in the GUI
- Processes are stopped safely (graceful shutdown)
- IPFS data directory is isolated

---

**Son Güncelleme**: 2025-12-19

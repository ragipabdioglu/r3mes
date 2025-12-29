# R3MES Desktop Launcher (Tauri)

Native desktop launcher uygulaması - Tauri ile geliştirilmiş, hafif ve performanslı.

## 🎯 Özellikler

### Temel Özellikler
- ✅ **Process Management**: Node, Miner, IPFS başlatma/durdurma
- ✅ **Status Monitoring**: Real-time process durumu
- ✅ **System Tray**: Tray icon ve context menu
- ✅ **Log Viewer**: Process loglarını görüntüleme
- ✅ **Native UI**: Platform-native görünüm
- ✅ **Auto-start**: Sistem açılışında otomatik başlatma (opsiyonel)

### UI Bileşenleri
- **Dashboard**: Ana kontrol paneli
- **Process Cards**: Her process için kart (Node, Miner, IPFS)
- **Status Indicators**: Çalışma durumu göstergeleri
- **Log Viewer**: Scrollable log görüntüleyici
- **Settings**: Yapılandırma paneli

## 🛠️ Teknik Stack

- **Backend**: Rust (Tauri)
- **Frontend**: React + TypeScript
- **UI Framework**: Tauri native components
- **Process Management**: Rust std::process
- **System Tray**: Tauri tray API

## 📦 Kurulum

### Ön Gereksinimler

1. **Rust** (Cargo):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
cargo --version
```

2. **Node.js** (18+):
```bash
node --version
```

3. **System Dependencies**:

**Linux**:
```bash
sudo apt update
sudo apt install libwebkit2gtk-4.1-dev \
    build-essential \
    curl \
    wget \
    file \
    libssl-dev \
    libgtk-3-dev \
    libayatana-appindicator3-dev \
    librsvg2-dev
```

**macOS**:
```bash
xcode-select --install
```

**Windows**:
- Visual Studio Build Tools
- WebView2 (otomatik kurulur)

### Proje Kurulumu

```bash
cd ~/R3MES/desktop-launcher-tauri

# Tauri CLI kurulumu
cargo install create-tauri-app --locked

# Veya npm ile
npm install -g @tauri-apps/cli

# Proje oluştur (eğer henüz oluşturulmadıysa)
npm create tauri-app@latest

# Bağımlılıkları kur
npm install

# Development mode
npm run tauri dev

# Production build
npm run tauri build
```

## 🚀 Kullanım

### Development

```bash
npm run tauri dev
```

### Production Build

```bash
npm run tauri build
```

Build çıktısı: `src-tauri/target/release/` (Linux) veya `src-tauri/target/release/bundle/` (installer)

## 📁 Proje Yapısı

```
desktop-launcher-tauri/
├── src/                    # Frontend (React)
│   ├── components/         # UI components
│   ├── pages/              # Pages
│   ├── hooks/              # React hooks
│   └── main.tsx            # Entry point
├── src-tauri/              # Backend (Rust)
│   ├── src/
│   │   ├── main.rs         # Tauri entry point
│   │   ├── commands.rs     # Tauri commands
│   │   └── process_manager.rs # Process management
│   ├── Cargo.toml          # Rust dependencies
│   └── tauri.conf.json     # Tauri config
├── package.json
└── README.md
```

## 🔧 Yapılandırma

### Tauri Config (`src-tauri/tauri.conf.json`)

```json
{
  "build": {
    "beforeDevCommand": "npm run dev",
    "beforeBuildCommand": "npm run build",
    "devPath": "http://localhost:1420",
    "distDir": "../dist"
  },
  "package": {
    "productName": "R3MES Launcher",
    "version": "0.1.0"
  },
  "tauri": {
    "allowlist": {
      "all": false,
      "shell": {
        "all": false,
        "execute": true,
        "sidecar": true,
        "open": true
      },
      "process": {
        "all": false,
        "relaunch": true
      }
    },
    "systemTray": {
      "iconPath": "icons/icon.png",
      "iconAsTemplate": true
    },
    "windows": [
      {
        "title": "R3MES Launcher",
        "width": 900,
        "height": 700,
        "resizable": true,
        "fullscreen": false
      }
    ]
  }
}
```

## 🎨 UI Tasarım

### Ana Ekran

```
┌─────────────────────────────────────────┐
│  R3MES Launcher                        │
├─────────────────────────────────────────┤
│                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────┐ │
│  │  Node    │  │  Miner   │  │ IPFS │ │
│  │          │  │          │  │      │ │
│  │ [●] Running │ [○] Stopped │ [○]   │ │
│  │          │  │          │  │      │ │
│  │ [Stop]  │  │ [Start]  │  │[Start]│ │
│  └──────────┘  └──────────┘  └──────┘ │
│                                         │
│  ┌──────────────────────────────────┐ │
│  │  Logs                              │ │
│  │  ┌──────────────────────────────┐ │ │
│  │  │ [Node] Starting...            │ │ │
│  │  │ [Miner] Gradient computed...  │ │ │
│  │  └──────────────────────────────┘ │ │
│  └──────────────────────────────────┘ │
│                                         │
└─────────────────────────────────────────┘
```

## 🔌 Tauri Commands (Rust Backend)

### Process Management

```rust
// src-tauri/src/commands.rs
#[tauri::command]
async fn start_node() -> Result<ProcessResult, String> {
    // Start remesd process
}

#[tauri::command]
async fn stop_node() -> Result<(), String> {
    // Stop remesd process
}

#[tauri::command]
async fn start_miner() -> Result<ProcessResult, String> {
    // Start r3mes-miner process
}

#[tauri::command]
async fn stop_miner() -> Result<(), String> {
    // Stop r3mes-miner process
}

#[tauri::command]
async fn get_status() -> Result<ProcessStatus, String> {
    // Get all process statuses
}
```

## 📱 Frontend (React)

### Component Structure

```typescript
// src/components/ProcessCard.tsx
interface ProcessCardProps {
  name: string;
  status: 'running' | 'stopped';
  onStart: () => void;
  onStop: () => void;
}

// src/components/LogViewer.tsx
interface LogViewerProps {
  logs: LogEntry[];
}

// src/pages/Dashboard.tsx
// Ana dashboard sayfası
```

## 🔐 Güvenlik

- **Context Isolation**: Enabled
- **Node Integration**: Disabled
- **Shell Commands**: Whitelisted only
- **Process Management**: Secure process spawning

## 📊 Performans

- **Bundle Size**: ~5-10 MB (Electron: ~100+ MB)
- **Memory Usage**: ~50-100 MB (Electron: ~200+ MB)
- **Startup Time**: < 1 second

## 🐛 Sorun Giderme

### Rust/Cargo bulunamıyor

```bash
# Rust kurulumu
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

### System dependencies eksik (Linux)

```bash
sudo apt install libwebkit2gtk-4.1-dev build-essential
```

### Build hatası

```bash
# Rust toolchain güncelle
rustup update

# Clean build
cargo clean
npm run tauri build
```

## 📝 Notlar

- Tauri, native webview kullanır (Electron'dan daha hafif)
- Frontend React/TypeScript ile yazılır
- Backend Rust ile yazılır (güvenli ve performanslı)
- Process management Rust tarafında yapılır
- System tray native API'ler kullanılır


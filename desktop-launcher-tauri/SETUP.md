# R3MES Desktop Launcher - Tauri Setup Guide

## 🚀 Hızlı Kurulum

### 1. Rust Kurulumu

```bash
# Rust ve Cargo'yu kur
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Kurulumu doğrula
cargo --version
rustc --version
```

### 2. System Dependencies (Linux)

```bash
sudo apt update
sudo apt install -y \
    libwebkit2gtk-4.1-dev \
    build-essential \
    curl \
    wget \
    file \
    libssl-dev \
    libgtk-3-dev \
    libayatana-appindicator3-dev \
    librsvg2-dev
```

### 3. Tauri CLI Kurulumu

```bash
# Cargo ile (önerilen)
cargo install tauri-cli --locked

# Veya npm ile
npm install -g @tauri-apps/cli
```

### 4. Proje Kurulumu

```bash
cd ~/R3MES/desktop-launcher-tauri

# Bağımlılıkları kur
npm install

# Rust dependencies (ilk build'de otomatik)
cargo build --manifest-path src-tauri/Cargo.toml
```

## 🏃 Development

```bash
# Development mode (hot reload)
npm run tauri:dev

# Veya
cargo tauri dev
```

## 📦 Production Build

```bash
# Build
npm run tauri:build

# Çıktı:
# - Linux: src-tauri/target/release/r3mes-desktop-launcher
# - Windows: src-tauri/target/release/r3mes-desktop-launcher.exe
# - macOS: src-tauri/target/release/bundle/macos/R3MES Launcher.app
```

## 📁 Proje Yapısı

```
desktop-launcher-tauri/
├── src/                      # Frontend (React + TypeScript)
│   ├── components/           # UI components
│   │   ├── ProcessCard.tsx
│   │   └── LogViewer.tsx
│   ├── App.tsx              # Ana component
│   ├── main.tsx             # Entry point
│   └── styles.css
├── src-tauri/               # Backend (Rust)
│   ├── src/
│   │   ├── main.rs          # Tauri entry point
│   │   ├── commands.rs     # Tauri commands (IPC)
│   │   └── process_manager.rs # Process management
│   ├── Cargo.toml           # Rust dependencies
│   └── tauri.conf.json      # Tauri configuration
├── package.json
└── vite.config.ts
```

## 🔧 Yapılandırma

### Workspace Path

Varsayılan: `~/R3MES`

Değiştirmek için `src-tauri/src/process_manager.rs` içinde:
```rust
let workspace = PathBuf::from(home).join("R3MES");
```

### Window Size

`src-tauri/tauri.conf.json` içinde:
```json
"windows": [{
  "width": 900,
  "height": 700
}]
```

## 🎨 UI Özellikleri

- **Native Look**: Platform-native görünüm
- **Dark Theme**: Otomatik dark mode
- **Responsive**: Grid layout, mobile-friendly
- **Real-time Updates**: 2 saniyede bir status polling

## 🔌 IPC API

Frontend'den Rust backend'e çağrılar:

```typescript
import { invoke } from "@tauri-apps/api/core";

// Process control
await invoke("start_node");
await invoke("stop_node");
await invoke("start_miner");
await invoke("stop_miner");
await invoke("start_ipfs");
await invoke("stop_ipfs");

// Status
const status = await invoke<ProcessStatus>("get_status");

// Logs
const logs = await invoke<string[]>("get_logs");
```

## 🐛 Sorun Giderme

### "Cargo not found"

```bash
source ~/.cargo/env
# Veya ~/.bashrc'ye ekle:
echo 'source $HOME/.cargo/env' >> ~/.bashrc
```

### "libwebkit2gtk not found" (Linux)

```bash
sudo apt install libwebkit2gtk-4.1-dev
```

### Build hatası

```bash
# Clean build
cargo clean
rm -rf node_modules
npm install
npm run tauri:build
```

## 📝 Notlar

- Tauri, Electron'dan çok daha hafif (~5-10 MB vs ~100+ MB)
- Native webview kullanır (platform'un kendi webview'ı)
- Rust backend güvenli ve performanslı
- Frontend React/TypeScript ile modern UI


# Tauri Launcher - Kurulum Düzeltmeleri

## Sorun: pkg-config Permission Denied

Bu hata, system dependencies eksik olduğunda veya pkg-config yanlış yapılandırıldığında oluşur.

## Çözüm

### 1. System Dependencies Kurulumu (Linux)

**ÖNEMLİ**: Aşağıdaki komutu terminalde çalıştırın (sudo şifresi isteyecek):

```bash
sudo apt update && sudo apt install -y \
    libsoup2.4-dev \
    libwebkit2gtk-4.1-dev \
    build-essential \
    curl \
    wget \
    file \
    libssl-dev \
    libgtk-3-dev \
    libayatana-appindicator3-dev \
    librsvg2-dev \
    libglib2.0-dev \
    libgdk-pixbuf2.0-dev \
    libpango1.0-dev \
    libatk1.0-dev \
    pkg-config
```

**Not**: `libsoup2.4-dev` paketi Tauri için kritik öneme sahiptir. Eksikse build başarısız olur.

### 2. pkg-config Kontrolü

```bash
# pkg-config kurulu mu?
which pkg-config

# Eğer yoksa:
sudo apt install pkg-config

# Test:
pkg-config --version
```

### 3. Rust Toolchain Güncelleme

```bash
rustup update stable
rustup default stable
```

### 4. Cargo Clean ve Yeniden Build

```bash
cd ~/R3MES/desktop-launcher-tauri
cargo clean --manifest-path src-tauri/Cargo.toml
cargo build --manifest-path src-tauri/Cargo.toml
```

## Alternatif: Tauri CLI ile Test

Eğer hala sorun varsa, Tauri CLI ile direkt test edin:

```bash
cd ~/R3MES/desktop-launcher-tauri
npm run tauri:dev
```

Tauri CLI, gerekli dependencies'i otomatik kontrol eder ve hata mesajları verir.

## Notlar

- Tauri 1.5 kullanıyoruz (daha stabil)
- System dependencies Linux için GTK3 ve WebKit2GTK gerektirir
- Windows'ta farklı dependencies gerekir (Visual Studio Build Tools)


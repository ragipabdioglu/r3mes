# Tauri 1.5 - Tüm System Dependencies (Ubuntu/Debian)

## Eksik Paket Hatası

```
The system library `javascriptcoregtk-4.0` required by crate `javascriptcore-rs-sys` was not found.
```

## Kesin Çözüm: Tüm Dependencies Tek Seferde

Aşağıdaki komutu **tek seferde** çalıştırın (sudo şifresi isteyecek):

```bash
sudo apt update && sudo apt install -y \
    libwebkit2gtk-4.1-dev \
    libjavascriptcoregtk-4.1-dev \
    libsoup2.4-dev \
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
    libcairo2-dev \
    libgirepository1.0-dev \
    pkg-config
```

## Önemli Notlar

### WebKit2GTK Versiyonları

- **Tauri 1.5**: `libwebkit2gtk-4.1-dev` ve `libjavascriptcoregtk-4.1-dev` gerektirir
- **Eski versiyonlar**: `libwebkit2gtk-4.0-dev` kullanabilir ama Tauri 1.5 için 4.1 gerekli

### Paket İsimleri

- `libjavascriptcoregtk-4.1-dev` - JavaScriptCore GTK 4.1 development files
- `libwebkit2gtk-4.1-dev` - WebKit2 GTK 4.1 development files
- `libsoup2.4-dev` - Soup HTTP library development files

## Kurulum Sonrası

```bash
# 1. Cargo'yu yükle (eğer yoksa)
source ~/.cargo/env

# 2. Clean build
cd ~/R3MES/desktop-launcher-tauri
cargo clean --manifest-path src-tauri/Cargo.toml

# 3. Test
npm run tauri:dev
```

## Paket Kontrolü

Kurulu paketleri kontrol etmek için:

```bash
# WebKit2GTK kontrolü
pkg-config --exists webkit2gtk-4.1 && echo "✅ webkit2gtk-4.1 OK" || echo "❌ webkit2gtk-4.1 MISSING"

# JavaScriptCore kontrolü
pkg-config --exists javascriptcoregtk-4.1 && echo "✅ javascriptcoregtk-4.1 OK" || echo "❌ javascriptcoregtk-4.1 MISSING"

# Soup kontrolü
pkg-config --exists libsoup-2.4 && echo "✅ libsoup-2.4 OK" || echo "❌ libsoup-2.4 MISSING"
```

## Alternatif: Paket Arama

Eğer paket bulunamazsa:

```bash
# JavaScriptCore paketlerini ara
apt-cache search javascriptcoregtk

# WebKit paketlerini ara
apt-cache search webkit2gtk | grep dev

# Tüm Tauri dependencies'i ara
apt-cache search libwebkit2gtk libjavascriptcoregtk libsoup2.4
```

## Sorun Giderme

### "Package not found" Hatası

Eğer `libjavascriptcoregtk-4.1-dev` bulunamazsa:

1. **Ubuntu 22.04+**: Paket mevcut olmalı
2. **Eski Ubuntu**: `libwebkit2gtk-4.0-dev` kullanmayı deneyin (Tauri 1.4 için)
3. **PPA ekleyin** (gerekirse):
   ```bash
   sudo add-apt-repository ppa:webkit-team/ppa
   sudo apt update
   ```

### "Permission denied" Hatası

```bash
# Dosya izinlerini kontrol et
ls -la ~/.cargo/bin/

# Cargo'yu PATH'e ekle
source ~/.cargo/env
```

## Minimum Gereksinimler

**Kritik Paketler** (olmadan build başarısız):
- `libwebkit2gtk-4.1-dev`
- `libjavascriptcoregtk-4.1-dev`
- `libsoup2.4-dev`
- `build-essential`
- `pkg-config`

**Önerilen Paketler** (tam özellik desteği için):
- `libgtk-3-dev`
- `libayatana-appindicator3-dev` (system tray için)
- `librsvg2-dev` (icon desteği için)

## Test Komutu

Tüm dependencies kurulduktan sonra:

```bash
cd ~/R3MES/desktop-launcher-tauri
source ~/.cargo/env
cargo check --manifest-path src-tauri/Cargo.toml
```

Başarılı olursa `Finished` mesajı görmelisiniz.


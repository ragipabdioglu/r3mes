# Hızlı Çözüm: Permission Denied Hatası

## Sorun
```
Error failed to get cargo metadata: Permission denied (os error 13)
```

veya

```
The system library `libsoup-2.4` required by crate `soup2-sys` was not found.
```

## Çözüm

### Adım 1: System Dependencies Kurulumu

Terminalde şu komutu çalıştırın (sudo şifresi isteyecek):

```bash
sudo apt update && sudo apt install -y \
    libsoup2.4-dev \
    libwebkit2gtk-4.1-dev \
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

### Adım 2: Cargo Clean

```bash
cd ~/R3MES/desktop-launcher-tauri/src-tauri
cargo clean
```

### Adım 3: Test

```bash
cd ~/R3MES/desktop-launcher-tauri
npm run tauri:dev
```

## Alternatif: Eksik Paket Kontrolü

Hangi paketlerin eksik olduğunu görmek için:

```bash
pkg-config --list-all | grep -E "(soup|webkit|gtk)"
```

Eksik paketleri görmek için:

```bash
pkg-config --exists libsoup-2.4 && echo "OK" || echo "MISSING"
pkg-config --exists webkit2gtk-4.1 && echo "OK" || echo "MISSING"
pkg-config --exists gtk+-3.0 && echo "OK" || echo "MISSING"
```

## Notlar

- `libsoup2.4-dev` Tauri için kritik öneme sahiptir
- Tüm dependencies kurulmadan build başarısız olur
- İlk build uzun sürebilir (crate'ler indirilir)


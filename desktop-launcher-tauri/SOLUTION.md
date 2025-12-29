# Kesin Çözüm: "Permission denied" ve "cargo not found" Hatası

## Sorun

```
Command 'cargo' not found
Error failed to get cargo metadata: Permission denied (os error 13)
```

## Kök Neden

Cargo kurulu ama **mevcut shell session'ında PATH'e eklenmemiş**. `~/.cargo/env` dosyası var ama shell'de source edilmemiş.

## Kesin Çözüm

### Adım 1: Cargo'yu PATH'e Ekle (Mevcut Terminal İçin)

```bash
source ~/.cargo/env
```

### Adım 2: Doğrula

```bash
cargo --version
# cargo 1.92.0 (344c4567c 2025-10-21) gibi bir çıktı görmelisiniz
```

### Adım 3: Tauri'yi Çalıştır

```bash
cd ~/R3MES/desktop-launcher-tauri
npm run tauri:dev
```

## Kalıcı Çözüm (Her Terminal Açılışında Otomatik)

Eğer `~/.bashrc` veya `~/.zshrc` dosyanızda `source ~/.cargo/env` yoksa, ekleyin:

```bash
# ~/.bashrc veya ~/.zshrc dosyasına ekle
echo 'source "$HOME/.cargo/env"' >> ~/.bashrc

# Veya zsh kullanıyorsanız:
echo 'source "$HOME/.cargo/env"' >> ~/.zshrc
```

Sonra yeni bir terminal açın veya:

```bash
source ~/.bashrc  # veya source ~/.zshrc
```

## Alternatif: Doğrudan Cargo Yolu Kullanma

Eğer source etmek istemiyorsanız:

```bash
~/.cargo/bin/cargo --version
```

Ama Tauri CLI bunu otomatik bulamayabilir, bu yüzden PATH'e eklemek daha iyi.

## Test

```bash
# 1. Cargo'yu yükle
source ~/.cargo/env

# 2. Versiyonu kontrol et
cargo --version

# 3. Tauri projesini test et
cd ~/R3MES/desktop-launcher-tauri
npm run tauri:dev
```

## Notlar

- `~/.cargo/env` dosyası Rust kurulumu sırasında otomatik oluşturulur
- Bu dosya `PATH` ve `RUSTUP_HOME` gibi environment variable'ları ayarlar
- Her yeni terminal açılışında bu dosyayı source etmek gerekir (veya bashrc'ye eklemek)
- `which cargo` komutu cargo'nun nerede olduğunu gösterir: `/home/rabdi/.cargo/bin/cargo`

## Sistem Dependencies (Eğer Hala Hata Varsa)

Cargo çalışıyorsa ama build hatası alıyorsanız:

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

**ÖNEMLİ**: `libjavascriptcoregtk-4.1-dev` paketi Tauri 1.5 için kritik öneme sahiptir!


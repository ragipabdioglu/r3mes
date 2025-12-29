# R3MES Test Kılavuzu
## Projeyi Test Etme ve Başlatma Rehberi

Bu kılavuz, R3MES projesini test etmek için gerekli tüm adımları içerir.

---

## 📋 İçindekiler

1. [Ön Gereksinimler](#ön-gereksinimler)
2. [Hızlı Başlangıç](#hızlı-başlangıç)
3. [Adım Adım Test Süreci](#adım-adım-test-süreci)
4. [Bileşenler ve Portlar](#bileşenler-ve-portlar)
5. [Test Senaryoları](#test-senaryoları)
6. [Sorun Giderme](#sorun-giderme)

---

## 🔧 Ön Gereksinimler

### Gerekli Yazılımlar

1. **Python 3.8+**
   ```bash
   python3 --version
   # veya
   python --version
   ```

2. **Node.js 18+ ve npm**
   ```bash
   node --version
   npm --version
   ```

3. **Git**
   ```bash
   git --version
   ```

### Opsiyonel (Önerilir)

- **Docker** (containerized execution için)
- **CUDA** (GPU hızlandırma için)
- **Keplr Wallet Extension** (blockchain işlemleri için)

---

## 🚀 Hızlı Başlangıç

### Tek Komutla Başlatma (Önerilen)

```bash
# Proje kök dizininde
make start-all
```

Bu komut tüm servisleri sırayla başlatır.

### Manuel Başlatma

Eğer `make` komutu çalışmıyorsa, aşağıdaki adımları takip edin.

---

## 📝 Adım Adım Test Süreci

### 1. Bağımlılıkları Kontrol Et

```bash
cd backend
python3 check_dependencies.py
```

Bu script Docker ve CUDA kurulumunu kontrol eder.

### 2. Backend Bağımlılıklarını Kur

```bash
cd backend
pip install -r requirements.txt
```

**Not:** Virtual environment kullanmanız önerilir:

```bash
# Virtual environment oluştur
python3 -m venv venv

# Aktif et
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Bağımlılıkları kur
pip install -r requirements.txt
```

### 3. Backend'i Başlat

#### Yöntem 1: Python modülü olarak

```bash
cd backend
python3 -m app.main
```

#### Yöntem 2: run_backend.py script'i ile

```bash
# Proje kök dizininde
python3 run_backend.py
```

#### Yöntem 3: uvicorn ile direkt

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Beklenen Çıktı:**
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Backend Başarıyla Başladı mı Kontrol Et:**
```bash
curl http://localhost:8000/health
# veya tarayıcıda: http://localhost:8000/docs
```

### 4. Frontend Bağımlılıklarını Kur

```bash
cd web-dashboard
npm install
```

**Not:** İlk kurulum biraz zaman alabilir (5-10 dakika).

### 5. Frontend'i Başlat

```bash
cd web-dashboard
npm run dev
```

**Beklenen Çıktı:**
```
  ▲ Next.js 14.x.x
  - Local:        http://localhost:3000
  - ready started server on 0.0.0.0:3000
```

**Frontend Başarıyla Başladı mı Kontrol Et:**
Tarayıcıda `http://localhost:3000` adresini açın.

### 6. Blockchain Node'u Başlat (Opsiyonel)

Eğer blockchain işlemlerini test etmek istiyorsanız:

```bash
cd remes
# Blockchain node'u başlat (detaylar remes/README.md'de)
```

**Not:** Blockchain node başlatmak için Go ve Cosmos SDK kurulumu gerekir.

---

## 🔌 Bileşenler ve Portlar

### Backend (FastAPI)
- **Port:** `8000`
- **Health Check:** `http://localhost:8000/health`
- **API Docs:** `http://localhost:8000/docs`
- **Config:** Environment variables veya `~/.r3mes/config.json`

### Frontend (Next.js)
- **Port:** `3000`
- **URL:** `http://localhost:3000`
- **Dev Server:** Hot reload aktif

### Blockchain (Cosmos SDK)
- **RPC Port:** `26657` (varsayılan)
- **gRPC Port:** `9090` (varsayılan)
- **API Port:** `1317` (varsayılan)

### Miner Engine
- **Arrow Flight Port:** `9090` (varsayılan, environment variable ile değiştirilebilir)

---

## 🧪 Test Senaryoları

### Senaryo 1: Temel Backend Testi

1. Backend'i başlat
2. Health check yap:
   ```bash
   curl http://localhost:8000/health
   ```
3. API dokümantasyonunu kontrol et:
   - Tarayıcıda `http://localhost:8000/docs` aç
   - Swagger UI'da endpoint'leri görüntüle

### Senaryo 2: Frontend-Backend Entegrasyonu

1. Backend ve Frontend'i başlat
2. Tarayıcıda `http://localhost:3000` aç
3. Ana sayfayı kontrol et:
   - Network stats görünüyor mu?
   - Loading state çalışıyor mu?
   - Error handling çalışıyor mu?

### Senaryo 3: Chat Özelliği

1. Frontend'de `/chat` sayfasına git
2. Bir mesaj yaz ve gönder
3. Kontrol et:
   - Mesaj gönderildi mi?
   - Response geldi mi?
   - Streaming çalışıyor mu?
   - Credit sistemi çalışıyor mu?

### Senaryo 4: Wallet Entegrasyonu

1. Keplr Wallet extension'ı yükle
2. Frontend'de wallet bağla
3. Kontrol et:
   - Wallet adresi görünüyor mu?
   - Credits gösteriliyor mu?
   - Transaction history çalışıyor mu?

### Senaryo 5: Settings Yönetimi

1. Frontend'de `/settings` sayfasına git
2. Ayarları değiştir:
   - Model path
   - GPU memory limit
   - P2P port
3. Kaydet ve kontrol et:
   - Ayarlar kaydedildi mi?
   - Backend'de yansıdı mı?

### Senaryo 6: Onboarding Flow

1. Browser localStorage'ı temizle:
   ```javascript
   localStorage.clear()
   ```
2. Frontend'i yenile
3. Onboarding ekranı görünmeli
4. Adımları takip et veya "Atla" butonuna bas

### Senaryo 7: Dark/Light Mode

1. Navbar'da tema toggle butonuna bas
2. Kontrol et:
   - Tema değişti mi?
   - Tercih localStorage'da saklandı mı?
   - Sayfa yenilendiğinde tema korunuyor mu?

### Senaryo 8: Error Handling

1. Backend'i durdur
2. Frontend'de bir işlem yap
3. Kontrol et:
   - Kullanıcı dostu hata mesajı gösteriliyor mu?
   - Error boundary çalışıyor mu?

---

## 🔍 Detaylı Test Komutları

### Backend API Testleri

```bash
# Health check
curl http://localhost:8000/health

# Network stats
curl http://localhost:8000/network/stats

# User info (wallet address gerekli)
curl http://localhost:8000/user/info/YOUR_WALLET_ADDRESS

# Chat endpoint (POST)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "wallet_address": "YOUR_WALLET"}'

# Config get
curl http://localhost:8000/config

# Config update (PUT)
curl -X PUT http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{"mining_difficulty": 1500.0}'
```

### Frontend Testleri

```bash
# Build test
cd web-dashboard
npm run build

# Lint check
npm run lint

# Type check
npm run type-check  # Eğer varsa
```

### Database Testleri

```bash
# Backend dizininde
cd backend
python3 -m pytest tests/ -v
```

---

## 🐛 Sorun Giderme

### Backend Başlamıyor

**Sorun:** Port 8000 kullanımda
```bash
# Port'u kontrol et
lsof -i :8000  # Linux/macOS
netstat -ano | findstr :8000  # Windows

# Farklı port kullan
uvicorn app.main:app --port 8001
```

**Sorun:** Bağımlılık hatası
```bash
# Virtual environment kullan
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

**Sorun:** Database hatası
```bash
# Database dosyasını kontrol et
ls -la backend/database.db

# Database'i yeniden oluştur (DİKKAT: Veri kaybı olur)
rm backend/database.db
# Backend'i yeniden başlat, otomatik oluşturulur
```

### Frontend Başlamıyor

**Sorun:** Port 3000 kullanımda
```bash
# Farklı port kullan
PORT=3001 npm run dev
```

**Sorun:** npm install hatası
```bash
# Cache temizle
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

**Sorun:** Build hatası
```bash
# .next klasörünü temizle
rm -rf .next
npm run build
```

### CORS Hatası

**Sorun:** Frontend'den backend'e istek atılamıyor
```bash
# Backend'de CORS_ALLOWED_ORIGINS kontrol et
export CORS_ALLOWED_ORIGINS="http://localhost:3000"
# veya
export CORS_ALLOW_ALL="true"  # Sadece development için!
```

### API Key Hatası

**Sorun:** API key oluşturulamıyor
```bash
# Database'de api_keys tablosunu kontrol et
sqlite3 backend/database.db "SELECT * FROM api_keys;"
```

### Log Dosyaları

Log dosyaları şu konumda:
```bash
~/.r3mes/logs/
├── r3mes_backend.log
└── r3mes_backend_errors.log
```

Logları görüntüle:
```bash
tail -f ~/.r3mes/logs/r3mes_backend.log
```

---

## 📊 Test Checklist

### Backend
- [ ] Backend başlatıldı
- [ ] Health check başarılı
- [ ] API docs erişilebilir
- [ ] Database bağlantısı çalışıyor
- [ ] Logging çalışıyor
- [ ] Config yönetimi çalışıyor

### Frontend
- [ ] Frontend başlatıldı
- [ ] Ana sayfa yükleniyor
- [ ] Network stats görünüyor
- [ ] Onboarding çalışıyor
- [ ] Dark/Light mode çalışıyor
- [ ] Settings sayfası çalışıyor
- [ ] Help sayfası çalışıyor

### Entegrasyon
- [ ] Frontend-Backend iletişimi çalışıyor
- [ ] Chat özelliği çalışıyor
- [ ] Wallet entegrasyonu çalışıyor
- [ ] Error handling çalışıyor
- [ ] Loading states çalışıyor

### Güvenlik
- [ ] CORS doğru yapılandırılmış
- [ ] API key hashing çalışıyor
- [ ] Rate limiting çalışıyor
- [ ] Input validation çalışıyor

---

## 🎯 Hızlı Test Komutları

### Tüm Servisleri Başlat (Terminal 1)
```bash
cd backend && python3 -m app.main
```

### Tüm Servisleri Başlat (Terminal 2)
```bash
cd web-dashboard && npm run dev
```

### Health Check
```bash
curl http://localhost:8000/health && echo "✅ Backend OK"
curl http://localhost:3000 && echo "✅ Frontend OK"
```

### Logları İzle
```bash
# Backend logs
tail -f ~/.r3mes/logs/r3mes_backend.log

# Frontend logs (terminal çıktısı)
```

---

## 📝 Notlar

1. **İlk Çalıştırma:** İlk çalıştırmada model dosyaları indirilebilir, bu zaman alabilir.

2. **Database:** İlk çalıştırmada database otomatik oluşturulur.

3. **Config:** Config dosyası `~/.r3mes/config.json` konumunda oluşturulur.

4. **Portlar:** Portlar environment variable'larla değiştirilebilir.

5. **Development Mode:** Development modunda CORS daha esnek, production'da sıkı.

---

## 🆘 Yardım

Sorun yaşıyorsanız:
1. Log dosyalarını kontrol edin
2. Health check endpoint'lerini test edin
3. Portların kullanılabilir olduğundan emin olun
4. Bağımlılıkların kurulu olduğunu kontrol edin
5. [Help Sayfası](/help) veya GitHub Issues'a bakın

---

## 📚 İlgili Dokümantasyon

- [COMPREHENSIVE_DOCUMENTATION.md](./COMPREHENSIVE_DOCUMENTATION.md) - Tüm API endpoint'leri ve component'ler
- [PROJECT_STATUS_REPORT.md](./PROJECT_STATUS_REPORT.md) - Proje durum raporu
- [QUICK_START.md](./QUICK_START.md) - Hızlı başlangıç kılavuzu
- [README_INSTALLATION.md](./README_INSTALLATION.md) - Kurulum kılavuzu

---

**Son Güncelleme:** 2024  
**Test Versiyonu:** 1.0.0


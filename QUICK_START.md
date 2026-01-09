# R3MES Hızlı Başlangıç Kılavuzu
## 5 Dakikada Projeyi Test Etme

Bu kılavuz, projeyi en hızlı şekilde test etmeniz için gerekli adımları içerir.

---

## 🚀 En Hızlı Yöntem (Otomatik)

### Linux/macOS

```bash
# Proje kök dizininde
./start_test.sh
```

Bu script:
- ✅ Bağımlılıkları kontrol eder
- ✅ Eksik bağımlılıkları kurar
- ✅ Backend'i başlatır
- ✅ Frontend'i başlatır
- ✅ Health check yapar

### Durdurmak için:

```bash
./stop_test.sh
```

---

## 📝 Manuel Başlatma (Adım Adım)

### 1. Backend'i Başlat (Terminal 1)

```bash
cd backend
python3 -m venv venv          # İlk seferinde
source venv/bin/activate      # Linux/macOS
pip install -r requirements.txt
python3 -m app.main
```

**Beklenen Çıktı:**
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Kontrol Et:**
- Tarayıcıda: http://localhost:8000/docs
- veya: `curl http://localhost:8000/health`

### 2. Frontend'i Başlat (Terminal 2)

```bash
cd web-dashboard
npm install                   # İlk seferinde
npm run dev
```

**Beklenen Çıktı:**
```
  ▲ Next.js 14.x.x
  - Local:        http://localhost:3000
```

**Kontrol Et:**
- Tarayıcıda: http://localhost:3000

---

## ✅ Test Checklist

### Backend Kontrolleri

- [ ] Backend başladı (port 8000)
- [ ] Health check çalışıyor: http://localhost:8000/health
- [ ] API docs erişilebilir: http://localhost:8000/docs
- [ ] Network stats endpoint çalışıyor

### Frontend Kontrolleri

- [ ] Frontend başladı (port 3000)
- [ ] Ana sayfa yükleniyor
- [ ] Network stats görünüyor
- [ ] Onboarding çalışıyor (ilk açılışta)

### Entegrasyon Kontrolleri

- [ ] Frontend backend'e bağlanabiliyor
- [ ] API istekleri başarılı
- [ ] CORS hatası yok

---

## 🎯 İlk Test Senaryoları

### 1. Ana Sayfa Testi

1. Tarayıcıda `http://localhost:3000` aç
2. Network stats kartlarını kontrol et
3. Loading state'i gözlemle
4. Error handling'i test et (backend'i durdurup sayfayı yenile)

### 2. Chat Testi

1. `/chat` sayfasına git
2. Bir mesaj yaz ve gönder
3. Response'un geldiğini kontrol et
4. Streaming'in çalıştığını kontrol et

### 3. Settings Testi

1. `/settings` sayfasına git (wallet bağlı olmalı)
2. Bir ayar değiştir (örn: mining difficulty)
3. Kaydet
4. Backend'de değişikliğin yansıdığını kontrol et

### 4. Onboarding Testi

1. Browser console'da: `localStorage.clear()`
2. Sayfayı yenile
3. Onboarding ekranı görünmeli
4. Adımları takip et veya "Atla" butonuna bas

---

## 🔧 Sorun Giderme

### Port Kullanımda

```bash
# Port'u kontrol et
lsof -i :8000  # Linux/macOS
netstat -ano | findstr :8000  # Windows

# Farklı port kullan
# Backend için:
uvicorn app.main:app --port 8001

# Frontend için:
PORT=3001 npm run dev
```

### Bağımlılık Hatası

```bash
# Backend
cd backend
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Frontend
cd web-dashboard
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

### Database Hatası

```bash
# Database'i yeniden oluştur (DİKKAT: Veri kaybı olur)
rm backend/database.db
# Backend'i yeniden başlat
```

### CORS Hatası

```bash
# Backend'i environment variable ile başlat
export CORS_ALLOWED_ORIGINS="http://localhost:3000"
# veya development için:
export CORS_ALLOW_ALL="true"
python3 -m app.main
```

---

## 📊 Servis Durumu Kontrolü

### Backend Health Check

```bash
curl http://localhost:8000/health
```

**Beklenen Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "adapters_count": 0
}
```

### Frontend Kontrolü

```bash
curl http://localhost:3000
```

**Beklenen:** HTML response (200 OK)

### Network Stats Test

```bash
curl http://localhost:8000/network/stats
```

**Beklenen Response:**
```json
{
  "active_miners": 0,
  "total_users": 0,
  "total_credits": 0.0,
  "block_height": null
}
```

---

## 🎉 Başarılı Test İşaretleri

✅ **Backend:**
- Port 8000'de çalışıyor
- Health check başarılı
- API docs erişilebilir
- Log dosyaları oluşturuldu (`~/.r3mes/logs/`)

✅ **Frontend:**
- Port 3000'de çalışıyor
- Ana sayfa yükleniyor
- Network stats görünüyor
- Onboarding çalışıyor

✅ **Entegrasyon:**
- Frontend backend'e bağlanabiliyor
- API istekleri başarılı
- CORS hatası yok
- Error handling çalışıyor

---

## 📝 Notlar

1. **İlk Çalıştırma:** İlk çalıştırmada model dosyaları indirilebilir (zaman alabilir)

2. **Database:** İlk çalıştırmada otomatik oluşturulur

3. **Config:** Config dosyası `~/.r3mes/config.json` konumunda oluşturulur

4. **Logs:** Log dosyaları `~/.r3mes/logs/` konumunda

5. **Development Mode:** Development modunda CORS daha esnek

---

## 🆘 Yardım

Detaylı bilgi için:
- [TEST_GUIDE.md](./TEST_GUIDE.md) - Kapsamlı test kılavuzu
- [README_INSTALLATION.md](./README_INSTALLATION.md) - Kurulum kılavuzu
- [PROJECT_STATUS_REPORT.md](./PROJECT_STATUS_REPORT.md) - Proje durum raporu
- [COMPREHENSIVE_DOCUMENTATION.md](./COMPREHENSIVE_DOCUMENTATION.md) - Tüm API endpoint'leri ve component'ler

---

**Hızlı Başlangıç Versiyonu:** 1.0.0


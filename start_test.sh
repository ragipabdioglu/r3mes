#!/bin/bash

# R3MES Test Başlatma Script'i
# Tüm servisleri sırayla başlatır ve kontrol eder

set -e

# Renkler
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 R3MES Test Başlatma Script'i${NC}"
echo "================================"
echo ""

# Proje kök dizinini bul
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 1. Bağımlılıkları kontrol et
echo -e "${YELLOW}📦 Bağımlılıkları kontrol ediliyor...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 bulunamadı!${NC}"
    exit 1
fi

if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js bulunamadı!${NC}"
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo -e "${RED}❌ npm bulunamadı!${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Temel bağımlılıklar mevcut${NC}"
echo ""

# 2. Backend bağımlılıklarını kontrol et
echo -e "${YELLOW}📦 Backend bağımlılıkları kontrol ediliyor...${NC}"
cd backend

if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment bulunamadı, oluşturuluyor...${NC}"
    python3 -m venv venv
fi

source venv/bin/activate

if [ ! -f ".deps_installed" ]; then
    echo -e "${YELLOW}📥 Backend bağımlılıkları kuruluyor...${NC}"
    pip install -r requirements.txt
    touch .deps_installed
fi

echo -e "${GREEN}✅ Backend bağımlılıkları hazır${NC}"
echo ""

# 3. Frontend bağımlılıklarını kontrol et
echo -e "${YELLOW}📦 Frontend bağımlılıkları kontrol ediliyor...${NC}"
cd ../web-dashboard

if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}📥 Frontend bağımlılıkları kuruluyor...${NC}"
    npm install
fi

echo -e "${GREEN}✅ Frontend bağımlılıkları hazır${NC}"
echo ""

# 4. Portları kontrol et
echo -e "${YELLOW}🔌 Portlar kontrol ediliyor...${NC}"

check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo -e "${RED}❌ Port $1 kullanımda!${NC}"
        return 1
    else
        echo -e "${GREEN}✅ Port $1 kullanılabilir${NC}"
        return 0
    fi
}

PORT_8000_OK=true
PORT_3000_OK=true

if ! check_port 8000; then
    PORT_8000_OK=false
fi

if ! check_port 3000; then
    PORT_3000_OK=false
fi

if [ "$PORT_8000_OK" = false ] || [ "$PORT_3000_OK" = false ]; then
    echo -e "${YELLOW}⚠️  Bazı portlar kullanımda. Devam ediliyor...${NC}"
fi

echo ""

# 5. Servisleri başlat
echo -e "${GREEN}🚀 Servisler başlatılıyor...${NC}"
echo ""

# Backend'i arka planda başlat
echo -e "${YELLOW}📡 Backend başlatılıyor (port 8000)...${NC}"
cd ../backend
source venv/bin/activate
python3 -m app.main > ../backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Backend'in başlamasını bekle
sleep 5

# Health check
echo -e "${YELLOW}🔍 Backend health check...${NC}"
for i in {1..10}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Backend başarıyla başlatıldı!${NC}"
        break
    else
        if [ $i -eq 10 ]; then
            echo -e "${RED}❌ Backend başlatılamadı! Logları kontrol edin: backend.log${NC}"
            kill $BACKEND_PID 2>/dev/null || true
            exit 1
        fi
        echo "Bekleniyor... ($i/10)"
        sleep 2
    fi
done

echo ""

# Frontend'i arka planda başlat
echo -e "${YELLOW}🌐 Frontend başlatılıyor (port 3000)...${NC}"
cd ../web-dashboard
npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

# Frontend'in başlamasını bekle
sleep 10

# Frontend check
echo -e "${YELLOW}🔍 Frontend kontrol ediliyor...${NC}"
for i in {1..10}; do
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Frontend başarıyla başlatıldı!${NC}"
        break
    else
        if [ $i -eq 10 ]; then
            echo -e "${RED}❌ Frontend başlatılamadı! Logları kontrol edin: frontend.log${NC}"
            kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
            exit 1
        fi
        echo "Bekleniyor... ($i/10)"
        sleep 2
    fi
done

echo ""
echo -e "${GREEN}════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Tüm servisler başarıyla başlatıldı!${NC}"
echo -e "${GREEN}════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}📊 Servis Bilgileri:${NC}"
echo "  Backend:  http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo "  Frontend: http://localhost:3000"
echo ""
echo -e "${YELLOW}📝 Log Dosyaları:${NC}"
echo "  Backend:  backend.log"
echo "  Frontend: frontend.log"
echo ""
echo -e "${YELLOW}🛑 Durdurmak için:${NC}"
echo "  kill $BACKEND_PID $FRONTEND_PID"
echo "  veya: pkill -f 'app.main' && pkill -f 'next dev'"
echo ""
echo -e "${GREEN}🎉 Test etmeye hazırsınız!${NC}"


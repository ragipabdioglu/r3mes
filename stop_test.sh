#!/bin/bash

# R3MES Test Durdurma Script'i
# Tüm servisleri güvenli bir şekilde durdurur

set -e

# Renkler
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🛑 R3MES Servisleri Durduruluyor...${NC}"
echo ""

# Backend'i durdur
if pgrep -f "app.main" > /dev/null; then
    echo -e "${YELLOW}📡 Backend durduruluyor...${NC}"
    pkill -f "app.main" || true
    sleep 2
    echo -e "${GREEN}✅ Backend durduruldu${NC}"
else
    echo -e "${YELLOW}⚠️  Backend zaten çalışmıyor${NC}"
fi

# Frontend'i durdur
if pgrep -f "next dev" > /dev/null; then
    echo -e "${YELLOW}🌐 Frontend durduruluyor...${NC}"
    pkill -f "next dev" || true
    sleep 2
    echo -e "${GREEN}✅ Frontend durduruldu${NC}"
else
    echo -e "${YELLOW}⚠️  Frontend zaten çalışmıyor${NC}"
fi

# Portları kontrol et
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${RED}⚠️  Port 8000 hala kullanımda!${NC}"
    lsof -ti :8000 | xargs kill -9 2>/dev/null || true
fi

if lsof -Pi :3000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${RED}⚠️  Port 3000 hala kullanımda!${NC}"
    lsof -ti :3000 | xargs kill -9 2>/dev/null || true
fi

echo ""
echo -e "${GREEN}✅ Tüm servisler durduruldu!${NC}"


#!/bin/bash
# R3MES Node Control Script
# Node'u başlatma, durdurma ve reset etme komutları

set -e

NODE_BINARY="$HOME/R3MES/remes/build/remesd"
NODE_HOME="$HOME/.remes"

# Renkler
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Fonksiyonlar
check_node_running() {
    if pgrep -f "remesd start" > /dev/null; then
        return 0
    else
        return 1
    fi
}

stop_node() {
    echo -e "${YELLOW}Node durduruluyor...${NC}"
    
    # Graceful shutdown
    pkill -15 remesd 2>/dev/null || true
    sleep 2
    
    # Eğer hala çalışıyorsa force kill
    if check_node_running; then
        echo -e "${YELLOW}Force kill yapılıyor...${NC}"
        pkill -9 remesd 2>/dev/null || true
        sleep 1
    fi
    
    # Port kontrolü
    if lsof -i :26656 -i :26657 -i :9090 -i :1317 > /dev/null 2>&1; then
        echo -e "${RED}⚠️  Port'lar hala kullanımda!${NC}"
        lsof -i :26656 -i :26657 -i :9090 -i :1317
    else
        echo -e "${GREEN}✅ Node durduruldu${NC}"
    fi
}

start_node() {
    if check_node_running; then
        echo -e "${RED}⚠️  Node zaten çalışıyor!${NC}"
        ps aux | grep remesd | grep -v grep
        return 1
    fi
    
    echo -e "${GREEN}Node başlatılıyor...${NC}"
    cd "$HOME/R3MES/remes"
    nohup ./build/remesd start > /tmp/remesd.log 2>&1 &
    
    sleep 3
    
    if check_node_running; then
        echo -e "${GREEN}✅ Node başlatıldı${NC}"
        echo -e "Log dosyası: /tmp/remesd.log"
        echo -e "Node durumu: curl http://localhost:26657/status"
    else
        echo -e "${RED}❌ Node başlatılamadı!${NC}"
        echo -e "Log'u kontrol et: tail -f /tmp/remesd.log"
        return 1
    fi
}

reset_node() {
    echo -e "${RED}⚠️  DİKKAT: Bu işlem tüm blockchain state'ini siler!${NC}"
    read -p "Devam etmek istediğinize emin misiniz? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        echo "İşlem iptal edildi"
        return 1
    fi
    
    stop_node
    
    echo -e "${YELLOW}Node reset ediliyor...${NC}"
    cd "$HOME/R3MES/remes"
    ./build/remesd tendermint unsafe-reset-all --home "$NODE_HOME"
    
    echo -e "${GREEN}✅ Node reset edildi${NC}"
}

status_node() {
    if check_node_running; then
        echo -e "${GREEN}✅ Node çalışıyor${NC}"
        ps aux | grep remesd | grep -v grep
        
        echo -e "\n${YELLOW}Port durumu:${NC}"
        lsof -i :26656 -i :26657 -i :9090 -i :1317 2>/dev/null || echo "Port'lar kullanılmıyor"
        
        echo -e "\n${YELLOW}Block height:${NC}"
        curl -s http://localhost:26657/status | jq -r '.result.sync_info.latest_block_height' 2>/dev/null || echo "Node'a erişilemiyor"
    else
        echo -e "${RED}❌ Node çalışmıyor${NC}"
    fi
}

# Main
case "$1" in
    start)
        start_node
        ;;
    stop)
        stop_node
        ;;
    restart)
        stop_node
        sleep 2
        start_node
        ;;
    reset)
        reset_node
        ;;
    status)
        status_node
        ;;
    *)
        echo "Kullanım: $0 {start|stop|restart|reset|status}"
        echo ""
        echo "Komutlar:"
        echo "  start   - Node'u başlat"
        echo "  stop    - Node'u durdur"
        echo "  restart - Node'u yeniden başlat"
        echo "  reset   - Node'u reset et (TÜM STATE SİLİNİR!)"
        echo "  status  - Node durumunu göster"
        exit 1
        ;;
esac


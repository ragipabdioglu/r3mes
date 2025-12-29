#!/bin/bash
# R3MES Production Test Script
# Tüm özellikleri otomatik test eder

# set -e kaldırıldı - testlerin devam etmesi için
# Hatalar log_error ile kaydedilecek ama script durmayacak

PROJECT_ROOT="$HOME/R3MES"
cd "$PROJECT_ROOT"

# Renkler
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test sonuçları
PASSED=0
FAILED=0
SKIPPED=0

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}✅${NC} $1"
    ((PASSED++)) || true  # set -e için güvenlik
}

log_error() {
    echo -e "${RED}❌${NC} $1"
    ((FAILED++)) || true  # set -e için güvenlik
}

log_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
    ((SKIPPED++)) || true  # set -e için güvenlik
}

log_test() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}TEST:${NC} $1"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

# Test 1: Sistem Kontrolü
test_system_check() {
    log_test "1. Sistem Kontrolü"
    
    # IPFS kontrolü
    if command -v ipfs &> /dev/null; then
        IPFS_VERSION=$(ipfs version 2>/dev/null | head -1 || echo "unknown")
        log_success "IPFS kurulu: $IPFS_VERSION"
    else
        log_error "IPFS kurulu değil"
    fi
    
    # Node binary kontrolü
    if [ -f "$PROJECT_ROOT/remes/build/remesd" ]; then
        log_success "Node binary mevcut"
    else
        log_error "Node binary bulunamadı: $PROJECT_ROOT/remes/build/remesd"
    fi
    
    # Python venv kontrolü
    if [ -d "$PROJECT_ROOT/miner-engine/venv" ]; then
        log_success "Python venv mevcut"
    else
        log_error "Python venv bulunamadı: $PROJECT_ROOT/miner-engine/venv"
    fi
    
    # Dashboard kontrolü
    if [ -d "$PROJECT_ROOT/web-dashboard/node_modules" ]; then
        log_success "Dashboard dependencies kurulu"
    else
        log_warning "Dashboard dependencies kurulu değil (npm install gerekli)"
    fi
}

# Test 2: Port Kontrolü
test_port_check() {
    log_test "2. Port Kontrolü"
    
    PORTS=(5001 26656 26657 9090 1317 3000 8080)
    ALL_CLEAR=true
    
    for port in "${PORTS[@]}"; do
        if lsof -i :$port &> /dev/null; then
            log_warning "Port $port kullanımda"
            ALL_CLEAR=false
        fi
    done
    
    if [ "$ALL_CLEAR" = true ]; then
        log_success "Tüm port'lar boş"
    fi
}

# Test 3: IPFS Başlatma
test_ipfs_start() {
    log_test "3. IPFS Başlatma"
    
    # IPFS zaten çalışıyor mu?
    if curl -s http://localhost:5001/api/v0/version &> /dev/null; then
        log_success "IPFS zaten çalışıyor"
        return 0
    fi
    
    log_info "IPFS başlatılıyor..."
    # IPFS daemon'u background'da başlat
    ipfs daemon &
    IPFS_PID=$!
    
    # 5 saniye bekle
    sleep 5
    
    # Kontrol et
    if curl -s http://localhost:5001/api/v0/version &> /dev/null; then
        log_success "IPFS başarıyla başlatıldı"
        echo $IPFS_PID > /tmp/r3mes_ipfs.pid
    else
        log_error "IPFS başlatılamadı"
    fi
}

# Test 4: Node Başlatma
test_node_start() {
    log_test "4. Blockchain Node Başlatma"
    
    # Node zaten çalışıyor mu?
    if curl -s http://localhost:26657/status &> /dev/null; then
        log_success "Node zaten çalışıyor"
        return 0
    fi
    
    log_info "Node başlatılıyor..."
    "$PROJECT_ROOT/scripts/node_control.sh" start
    
    # 5 saniye bekle
    sleep 5
    
    # Kontrol et
    BLOCK_HEIGHT=$(curl -s http://localhost:26657/status | jq -r .result.sync_info.latest_block_height 2>/dev/null || echo "0")
    
    if [ "$BLOCK_HEIGHT" != "0" ] && [ "$BLOCK_HEIGHT" != "null" ]; then
        log_success "Node başarıyla başlatıldı (Block Height: $BLOCK_HEIGHT)"
    else
        log_error "Node başlatılamadı veya block height alınamadı"
    fi
}

# Test 5: REST API Testleri
test_rest_api() {
    log_test "5. REST API Testleri"
    
    # Status endpoint
    if curl -s http://localhost:1317/api/dashboard/status &> /dev/null; then
        log_success "Dashboard Status API çalışıyor"
    else
        log_error "Dashboard Status API çalışmıyor"
    fi
    
    # Miners endpoint
    if curl -s http://localhost:1317/api/dashboard/miners?limit=10 &> /dev/null; then
        log_success "Dashboard Miners API çalışıyor"
    else
        log_error "Dashboard Miners API çalışmıyor"
    fi
    
    # Blocks endpoint
    if curl -s http://localhost:1317/api/dashboard/blocks?limit=10 &> /dev/null; then
        log_success "Dashboard Blocks API çalışıyor"
    else
        log_error "Dashboard Blocks API çalışmıyor"
    fi
    
    # IPFS Health endpoint
    if curl -s http://localhost:1317/api/dashboard/ipfs/health &> /dev/null; then
        log_success "IPFS Health API çalışıyor"
    else
        log_error "IPFS Health API çalışmıyor"
    fi
}

# Test 6: Miner Stats Server Testi
test_miner_stats() {
    log_test "6. Miner Stats Server Testi"
    
    # Stats server çalışıyor mu?
    if curl -s http://localhost:8080/health &> /dev/null; then
        log_success "Miner Stats Server çalışıyor"
        
        # Stats endpoint testi
        if curl -s http://localhost:8080/stats &> /dev/null; then
            log_success "Miner Stats endpoint çalışıyor"
        else
            log_error "Miner Stats endpoint çalışmıyor"
        fi
    else
        log_warning "Miner Stats Server çalışmıyor (miner başlatılmamış olabilir)"
    fi
}

# Test 7: Dataset Kontrolü
test_dataset() {
    log_test "7. Dataset Kontrolü"
    
    if [ -f "$PROJECT_ROOT/dataset/haberler.jsonl" ]; then
        log_success "Dataset dosyası mevcut: haberler.jsonl"
        
        # Dosya boyutu kontrolü
        SIZE=$(stat -f%z "$PROJECT_ROOT/dataset/haberler.jsonl" 2>/dev/null || stat -c%s "$PROJECT_ROOT/dataset/haberler.jsonl" 2>/dev/null)
        log_info "Dataset boyutu: $(numfmt --to=iec-i --suffix=B $SIZE 2>/dev/null || echo "$SIZE bytes")"
    else
        log_warning "Dataset dosyası bulunamadı: haberler.jsonl"
    fi
}

# Test 8: gRPC Testi
test_grpc() {
    log_test "8. gRPC Testi"
    
    if command -v grpcurl &> /dev/null; then
        if grpcurl -plaintext localhost:9090 list &> /dev/null; then
            log_success "gRPC server çalışıyor"
        else
            log_error "gRPC server çalışmıyor"
        fi
    else
        log_warning "grpcurl kurulu değil (gRPC testi atlandı)"
    fi
}

# Test 9: Block Time Kontrolü
test_block_time() {
    log_test "9. Block Time Kontrolü"
    
    if ! curl -s http://localhost:26657/status &> /dev/null; then
        log_warning "Node çalışmıyor (block time testi atlandı)"
        return
    fi
    
    log_info "Block time ölçülüyor (10 saniye)..."
    
    HEIGHT1=$(curl -s http://localhost:26657/status | jq -r .result.sync_info.latest_block_height 2>/dev/null || echo "0")
    sleep 10
    HEIGHT2=$(curl -s http://localhost:26657/status | jq -r .result.sync_info.latest_block_height 2>/dev/null || echo "0")
    
    if [ "$HEIGHT1" != "0" ] && [ "$HEIGHT2" != "0" ] && [ "$HEIGHT1" != "$HEIGHT2" ]; then
        BLOCKS=$((HEIGHT2 - HEIGHT1))
        AVG_TIME=$((10 / BLOCKS))
        log_success "Block time: ~${AVG_TIME} saniye (10 saniyede $BLOCKS blok)"
    else
        log_warning "Block time ölçülemedi (node senkronize olmamış olabilir)"
    fi
}

# Ana test fonksiyonu
main() {
    echo -e "${GREEN}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║     R3MES Production Test Suite                            ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}\n"
    
    # Testleri çalıştır (her test bağımsız çalışmalı)
    test_system_check || true
    test_port_check || true
    test_ipfs_start || true
    test_node_start || true
    test_rest_api || true
    test_miner_stats || true
    test_dataset || true
    test_grpc || true
    test_block_time || true
    
    # Sonuçları göster
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}TEST SONUÇLARI${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    
    echo -e "${GREEN}✅ Başarılı:${NC} $PASSED"
    echo -e "${RED}❌ Başarısız:${NC} $FAILED"
    echo -e "${YELLOW}⚠️  Atlandı:${NC} $SKIPPED"
    echo ""
    
    TOTAL=$((PASSED + FAILED + SKIPPED))
    if [ $TOTAL -gt 0 ]; then
        SUCCESS_RATE=$((PASSED * 100 / TOTAL))
        echo -e "Başarı Oranı: ${SUCCESS_RATE}%"
    fi
    
    echo ""
    
    if [ $FAILED -eq 0 ]; then
        echo -e "${GREEN}🎉 Tüm testler başarılı!${NC}"
        exit 0
    else
        echo -e "${RED}⚠️  Bazı testler başarısız. Detaylar için yukarıya bakın.${NC}"
        exit 1
    fi
}

# Script çalıştır
main


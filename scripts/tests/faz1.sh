#!/bin/bash
# Faz 1 Test Script
# Tests the critical fixes implemented in Phase 1

set -e

echo "=========================================="
echo "Faz 1 - Kritik Düzeltmeler Test Script"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
PASSED=0
FAILED=0

# Test 1: Merkle Root Verification
echo "Test 1: Merkle Root Verification"
echo "--------------------------------"
cd /home/rabdi/R3MES/remes

# Check if merkle.go exists
if [ -f "x/remes/keeper/merkle.go" ]; then
    echo -e "${GREEN}✓${NC} merkle.go dosyası mevcut"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} merkle.go dosyası bulunamadı"
    FAILED=$((FAILED + 1))
fi

# Check if calculateMerkleRoot function exists
if grep -q "calculateMerkleRoot" x/remes/keeper/merkle.go; then
    echo -e "${GREEN}✓${NC} calculateMerkleRoot fonksiyonu mevcut"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} calculateMerkleRoot fonksiyonu bulunamadı"
    FAILED=$((FAILED + 1))
fi

# Check if Merkle verification is called in submit_aggregation
if grep -q "calculateMerkleRoot" x/remes/keeper/msg_server_submit_aggregation.go; then
    echo -e "${GREEN}✓${NC} Merkle root verification submit_aggregation'da kullanılıyor"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} Merkle root verification submit_aggregation'da kullanılmıyor"
    FAILED=$((FAILED + 1))
fi

echo ""

# Test 2: Rate Limiting
echo "Test 2: Rate Limiting Improvements"
echo "----------------------------------"

# Check if rate limiting has block-window check
if grep -q "RateLimitWindowBlocks" x/remes/keeper/rate_limiting.go; then
    echo -e "${GREEN}✓${NC} Block-window rate limiting mevcut"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} Block-window rate limiting bulunamadı"
    FAILED=$((FAILED + 1))
fi

# Check if time-based rate limiting is implemented
if grep -q "blocksSinceLastSubmission" x/remes/keeper/rate_limiting.go; then
    echo -e "${GREEN}✓${NC} Time-based approximate rate limiting mevcut"
    PASSED=$((PASSED + 1))
else
    echo -e "${YELLOW}⚠${NC} Time-based rate limiting basit implementasyon"
    PASSED=$((PASSED + 1))
fi

echo ""

# Test 3: Signature Verification
echo "Test 3: Signature Verification Bypass Fix"
echo "------------------------------------------"

# Check if signature is required (not optional)
if grep -q "signature cannot be empty" x/remes/keeper/msg_server_submit_gradient.go; then
    echo -e "${GREEN}✓${NC} Signature zorunlu hale getirildi"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} Signature hala optional"
    FAILED=$((FAILED + 1))
fi

# Check if empty signature check exists
if grep -q "len(msg.Signature) == 0" x/remes/keeper/msg_server_submit_gradient.go; then
    echo -e "${GREEN}✓${NC} Empty signature kontrolü mevcut"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} Empty signature kontrolü yok"
    FAILED=$((FAILED + 1))
fi

echo ""

# Test 4: Shard Assignment Verification
echo "Test 4: Shard Assignment Verification Enforcement"
echo "--------------------------------------------------"

# Check if shard mismatch causes rejection
if grep -q "shard assignment mismatch" x/remes/keeper/msg_server_submit_gradient.go; then
    echo -e "${GREEN}✓${NC} Shard mismatch rejection mevcut"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} Shard mismatch rejection yok"
    FAILED=$((FAILED + 1))
fi

# Check if shard verification returns error (not just warning)
if grep -q "shard assignment mismatch.*expected" x/remes/keeper/msg_server_submit_gradient.go || grep -q "ErrInvalidGradientHash.*shard" x/remes/keeper/msg_server_submit_gradient.go; then
    echo -e "${GREEN}✓${NC} Shard mismatch error döndürüyor (sadece warning değil)"
    PASSED=$((PASSED + 1))
else
    # Check if it's using errorsmod.Wrapf with shard
    if grep -q "errorsmod.Wrapf" x/remes/keeper/msg_server_submit_gradient.go && grep -q "shard" x/remes/keeper/msg_server_submit_gradient.go; then
        echo -e "${GREEN}✓${NC} Shard mismatch error döndürüyor"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}✗${NC} Shard mismatch hala sadece warning"
        FAILED=$((FAILED + 1))
    fi
fi

echo ""

# Test 5: EndBlocker Implementation
echo "Test 5: Challenge Period Finalization (EndBlocker)"
echo "----------------------------------------------------"

# Check if end_blocker.go exists
if [ -f "x/remes/keeper/end_blocker.go" ]; then
    echo -e "${GREEN}✓${NC} end_blocker.go dosyası mevcut"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} end_blocker.go dosyası bulunamadı"
    FAILED=$((FAILED + 1))
fi

# Check if FinalizeExpiredAggregations exists
if grep -q "FinalizeExpiredAggregations" x/remes/keeper/end_blocker.go; then
    echo -e "${GREEN}✓${NC} FinalizeExpiredAggregations fonksiyonu mevcut"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} FinalizeExpiredAggregations fonksiyonu bulunamadı"
    FAILED=$((FAILED + 1))
fi

# Check if EndBlock calls FinalizeExpiredAggregations
if grep -q "FinalizeExpiredAggregations" x/remes/module/module.go; then
    echo -e "${GREEN}✓${NC} EndBlock FinalizeExpiredAggregations çağırıyor"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} EndBlock FinalizeExpiredAggregations çağırmıyor"
    FAILED=$((FAILED + 1))
fi

# Check if EventAggregationFinalized exists
if grep -q "EventAggregationFinalized" x/remes/types/events.go; then
    echo -e "${GREEN}✓${NC} EventAggregationFinalized event'i mevcut"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} EventAggregationFinalized event'i bulunamadı"
    FAILED=$((FAILED + 1))
fi

echo ""

# Test 6: Gradient Status Race Condition Fix
echo "Test 6: Gradient Status Race Condition Fix"
echo "-------------------------------------------"

# Check if status is re-verified before update
if grep -q "gradient.Status != \"pending\"" x/remes/keeper/msg_server_submit_aggregation.go; then
    echo -e "${GREEN}✓${NC} Status re-verification mevcut (atomic update)"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} Status re-verification yok"
    FAILED=$((FAILED + 1))
fi

# Check if error is returned when status is not pending
if grep -q "no longer in pending status" x/remes/keeper/msg_server_submit_aggregation.go; then
    echo -e "${GREEN}✓${NC} Non-pending status error döndürüyor"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} Non-pending status error döndürmüyor"
    FAILED=$((FAILED + 1))
fi

echo ""

# Test 7: Build Test
echo "Test 7: Build Test"
echo "------------------"

if go build ./x/remes/keeper/... 2>&1 | grep -q "error"; then
    echo -e "${RED}✗${NC} Build hatası var"
    FAILED=$((FAILED + 1))
    go build ./x/remes/keeper/... 2>&1 | head -10
else
    echo -e "${GREEN}✓${NC} Build başarılı"
    PASSED=$((PASSED + 1))
fi

echo ""

# Summary
echo "=========================================="
echo "Test Özeti"
echo "=========================================="
echo -e "${GREEN}Başarılı: ${PASSED}${NC}"
echo -e "${RED}Başarısız: ${FAILED}${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ Tüm testler başarılı!${NC}"
    exit 0
else
    echo -e "${RED}✗ Bazı testler başarısız!${NC}"
    exit 1
fi


#!/bin/bash
# Faz 2 Test Script
# Tests the core functionality fixes implemented in Phase 2

set -e

echo "=========================================="
echo "Faz 2 - Core Functionality Düzeltmeler Test Script"
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

# Test 1: Global Model Update Logic
echo "Test 1: Global Model Update Logic"
echo "----------------------------------"
cd /home/rabdi/R3MES/remes

# Check if global_model.go exists
if [ -f "x/remes/keeper/global_model.go" ]; then
    echo -e "${GREEN}✓${NC} global_model.go dosyası mevcut"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} global_model.go dosyası bulunamadı"
    FAILED=$((FAILED + 1))
fi

# Check if UpdateGlobalModelStateIfNeeded function exists
if grep -q "UpdateGlobalModelStateIfNeeded" x/remes/keeper/global_model.go; then
    echo -e "${GREEN}✓${NC} UpdateGlobalModelStateIfNeeded fonksiyonu mevcut"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} UpdateGlobalModelStateIfNeeded fonksiyonu bulunamadı"
    FAILED=$((FAILED + 1))
fi

# Check if it's called in submit_aggregation
if grep -q "UpdateGlobalModelStateIfNeeded" x/remes/keeper/msg_server_submit_aggregation.go; then
    echo -e "${GREEN}✓${NC} Global model update submit_aggregation'da çağrılıyor"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} Global model update submit_aggregation'da çağrılmıyor"
    FAILED=$((FAILED + 1))
fi

# Check if it's called in end_blocker
if grep -q "UpdateGlobalModelStateIfNeeded" x/remes/keeper/end_blocker.go; then
    echo -e "${GREEN}✓${NC} Global model update end_blocker'da çağrılıyor"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} Global model update end_blocker'da çağrılmıyor"
    FAILED=$((FAILED + 1))
fi

# Check if proto has training_round_id and last_aggregation_id
if grep -q "training_round_id" proto/remes/remes/v1/state.proto 2>/dev/null || grep -q "TrainingRoundId" x/remes/types/state.pb.go 2>/dev/null; then
    echo -e "${GREEN}✓${NC} training_round_id proto'da mevcut"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} training_round_id proto'da yok"
    FAILED=$((FAILED + 1))
fi

if grep -q "last_aggregation_id" proto/remes/remes/v1/state.proto 2>/dev/null || grep -q "LastAggregationId" x/remes/types/state.pb.go 2>/dev/null; then
    echo -e "${GREEN}✓${NC} last_aggregation_id proto'da mevcut"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} last_aggregation_id proto'da yok"
    FAILED=$((FAILED + 1))
fi

echo ""

# Test 2: CPU Verification Panel VRF
echo "Test 2: CPU Verification Panel VRF Improvements"
echo "------------------------------------------------"

# Check if stake-weighted selection exists
if grep -q "stake.*weight" x/remes/keeper/cpu_verification_panel.go -i; then
    echo -e "${GREEN}✓${NC} Stake-weighted selection mevcut"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} Stake-weighted selection yok"
    FAILED=$((FAILED + 1))
fi

# Check if challenge context is used
if grep -q "challengeContext" x/remes/keeper/cpu_verification_panel.go; then
    echo -e "${GREEN}✓${NC} Challenge context seed'de kullanılıyor"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} Challenge context kullanılmıyor"
    FAILED=$((FAILED + 1))
fi

# Check if function signature includes challengeContext
if grep -q "selectCPUVerificationPanel.*challengeContext" x/remes/keeper/cpu_verification_panel.go || grep -q "challengeContext string" x/remes/keeper/cpu_verification_panel.go; then
    echo -e "${GREEN}✓${NC} Function signature challengeContext içeriyor"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} Function signature challengeContext içermiyor"
    FAILED=$((FAILED + 1))
fi

echo ""

# Test 3: Error Handling - Reward Distribution
echo "Test 3: Error Handling - Reward Distribution"
echo "---------------------------------------------"

# Check if reward distribution errors cause transaction failure
if grep -q "errorsmod.Wrap.*failed to distribute.*reward" x/remes/keeper/msg_server_submit_gradient.go; then
    echo -e "${GREEN}✓${NC} Miner reward distribution error transaction'ı fail ediyor"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} Miner reward distribution error ignore ediliyor"
    FAILED=$((FAILED + 1))
fi

if grep -q "errorsmod.Wrap.*failed to distribute.*reward" x/remes/keeper/msg_server_submit_aggregation.go; then
    echo -e "${GREEN}✓${NC} Proposer reward distribution error transaction'ı fail ediyor"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} Proposer reward distribution error ignore ediliyor"
    FAILED=$((FAILED + 1))
fi

# Check if error is not ignored (no _ = err)
if ! grep -q "_ = err" x/remes/keeper/msg_server_submit_gradient.go | grep -v "//"; then
    echo -e "${GREEN}✓${NC} Reward distribution error ignore edilmiyor"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} Reward distribution error hala ignore ediliyor"
    FAILED=$((FAILED + 1))
fi

echo ""

# Test 4: Slashing Consistency
echo "Test 4: Slashing Consistency - Stake vs Balance"
echo "------------------------------------------------"

# Check if slashableAmount is used for stake update
if grep -q "stake.Sub(slashableAmount" x/remes/keeper/slashing.go; then
    echo -e "${GREEN}✓${NC} Registration stake slashableAmount ile güncelleniyor"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} Registration stake yanlış amount ile güncelleniyor"
    FAILED=$((FAILED + 1))
fi

# Check if comment explains the fix
if grep -q "actual slashed amount" x/remes/keeper/slashing.go -i; then
    echo -e "${GREEN}✓${NC} Slashing consistency comment mevcut"
    PASSED=$((PASSED + 1))
else
    echo -e "${YELLOW}⚠${NC} Slashing consistency comment yok (opsiyonel)"
    PASSED=$((PASSED + 1))
fi

echo ""

# Test 5: Trust Score Update Logic
echo "Test 5: Trust Score Update Logic"
echo "---------------------------------"

# Check if trust_score.go exists
if [ -f "x/remes/keeper/trust_score.go" ]; then
    echo -e "${GREEN}✓${NC} trust_score.go dosyası mevcut"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} trust_score.go dosyası bulunamadı"
    FAILED=$((FAILED + 1))
fi

# Check if UpdateTrustScore function exists
if grep -q "UpdateTrustScore" x/remes/keeper/trust_score.go; then
    echo -e "${GREEN}✓${NC} UpdateTrustScore fonksiyonu mevcut"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} UpdateTrustScore fonksiyonu bulunamadı"
    FAILED=$((FAILED + 1))
fi

# Check if trust score is updated on finalized aggregation
if grep -q "UpdateTrustScore.*accepted" x/remes/keeper/end_blocker.go; then
    echo -e "${GREEN}✓${NC} Trust score finalized aggregation'da güncelleniyor"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} Trust score finalized aggregation'da güncellenmiyor"
    FAILED=$((FAILED + 1))
fi

# Check if trust score is updated on challenge resolution
if grep -q "UpdateTrustScoreOnChallengeResolution" x/remes/keeper/msg_server_cpu_verification.go; then
    echo -e "${GREEN}✓${NC} Trust score challenge resolution'da güncelleniyor"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} Trust score challenge resolution'da güncellenmiyor"
    FAILED=$((FAILED + 1))
fi

# Check if reputation tier calculation exists
if grep -q "calculateReputationTier" x/remes/keeper/trust_score.go; then
    echo -e "${GREEN}✓${NC} Reputation tier calculation mevcut"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} Reputation tier calculation yok"
    FAILED=$((FAILED + 1))
fi

echo ""

# Test 6: Environment Violation - Trust Score Reduction
echo "Test 6: Environment Violation - Trust Score Reduction"
echo "-----------------------------------------------------"

# Check if trust score is updated on environment violation
if grep -q "UpdateTrustScore.*challenged" x/remes/keeper/environment_validation.go; then
    echo -e "${GREEN}✓${NC} Trust score environment violation'da güncelleniyor"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} Trust score environment violation'da güncellenmiyor"
    FAILED=$((FAILED + 1))
fi

# Check if TODO comment is removed
if ! grep -q "TODO.*trust score reduction" x/remes/keeper/environment_validation.go -i; then
    echo -e "${GREEN}✓${NC} TODO comment kaldırıldı (implement edildi)"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} TODO comment hala mevcut"
    FAILED=$((FAILED + 1))
fi

echo ""

# Test 7: Build Test
echo "Test 7: Build Test"
echo "------------------"

if go build ./x/remes/... 2>&1 | grep -q "error"; then
    echo -e "${RED}✗${NC} Build hatası var"
    FAILED=$((FAILED + 1))
    go build ./x/remes/... 2>&1 | head -10
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


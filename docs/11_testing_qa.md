# R3MES Testing & Quality Assurance

## Genel Bakış

R3MES, kapsamlı test stratejisi ile production-ready kalite sağlar. Unit tests, integration tests, performance tests ve security tests içerir.

---

## 🧪 Test Kategorileri

### 1. Unit Tests

#### Go Tests

```bash
cd remes
go test ./...
go test -v ./x/remes/keeper/...
```

**Test Dosyaları**:
- `keeper/integration_test.go`: End-to-end gradient submission
- `keeper/loss_verification_test.go`: Loss verification logic
- `keeper/performance_test.go`: Performance under load
- `keeper/proof_of_replication_test.go`: PoRep verification
- `keeper/block_time_test.go`: Block time calculation
- `types/genesis_test.go`: Genesis state validation

**Beklenen Sonuç**: Tüm testler geçmeli (PASS)

#### Python Tests

```bash
cd miner-engine
source venv/bin/activate
pytest tests/ -v
```

**Test Dosyaları**:
- `tests/test_deterministic_execution.py`: Deterministic execution tests
- `tests/test_bitnet_properties.py`: BitNet property tests
- `test_miner.py`: Miner engine basic tests

**Durum**: Bazı BitNet property testleri implementasyon sorunları nedeniyle başarısız (düşük öncelik)

---

### 2. Integration Tests

#### End-to-End Protocol Flow

**Test Senaryosu**:
1. IPFS daemon başlat
2. Blockchain node başlat
3. Miner başlat
4. Gradient submission test et
5. Dashboard'da görünürlüğü kontrol et

**Test Script**:
```bash
./test_e2e.sh
```

**Beklenen**:
- ✅ IPFS connection: OK
- ✅ Blockchain connection: OK
- ✅ Gradient submission: OK
- ✅ Transaction hash: Alındı
- ✅ Dashboard'da görünür: OK

#### Multi-Miner Test

**Test Senaryosu**:
1. 1 Node + 1 IPFS + 3 Miner
2. Her miner farklı shard'a atanır
3. Aggregation oluşur

### 2.5. End-to-End (E2E) Tests

#### Playwright Test Suite

**Framework**: Playwright with TypeScript

**Test Files**:
- `tests/e2e/smoke.test.ts`: **Smoke tests for critical flows** (tagged with @smoke, runs in CD pipeline)
  - Homepage loads successfully
  - Health check endpoint responds
  - Dashboard page is accessible

**Planned Test Files** (to be implemented):
- `tests/e2e/protocol_flow.test.ts`: Complete protocol flow (node → miner → gradient → verify)
- `tests/e2e/chat_flow.test.ts`: Chat interface flow (wallet → message → response)
- `tests/e2e/dashboard_integration.test.ts`: Dashboard integration tests
- `tests/e2e/governance_voting.test.ts`: Governance voting flow
- `tests/e2e/wallet_connection.test.ts`: Wallet connection flow

**Durum**: 
- ✅ Smoke tests implemented and tagged with `@smoke` for CD pipeline
- ⏳ Full E2E test suite implementation in progress

**Test Utilities**:
- `tests/e2e/utils/test-helpers.ts`: Helper functions (waitForNetworkIdle, expectNoErrorMessages, etc.)
- `tests/e2e/utils/mock-server.ts`: Mock backend API responses

**Configuration**:
- `playwright.config.ts`: Playwright configuration with multiple browsers (Chromium, Firefox, WebKit)
- CI integration: GitHub Actions workflow for automated E2E testing

**Running Tests**:
```bash
cd web-dashboard
npx playwright test
npx playwright test --ui  # Interactive mode
npx playwright test --headed  # Run with browser visible
```

**CI Integration**:
- Automated E2E tests run on every push/PR
- Test results uploaded as artifacts
- Screenshots and videos on failure

---

### 3. Performance Tests

#### Load Testing (Locust)

**Framework**: Locust (Python)

**Test Scenarios**:
- Normal load: 100 concurrent users
- High load: 1000 concurrent users
- Stress test: 2000 concurrent users
- Miner load: 500 concurrent miners

**Usage**:
```bash
# Run specific scenario
./tests/performance/load_test_scenarios.sh http://localhost:8000 normal

# Run all scenarios
./tests/performance/load_test_scenarios.sh http://localhost:8000 all
```

**Test Files**:
- `tests/performance/locustfile.py`: Main Locust test file
- `tests/performance/load_test_scenarios.sh`: Scenario runner script

**Metrics**:
- Requests per second (RPS)
- Response time (p50, p95, p99)
- Error rate
- Concurrent users

#### Load Testing (k6)

**Framework**: k6 (JavaScript)

**Features**:
- Gradual ramp-up/ramp-down
- Custom metrics
- Threshold-based testing

**Usage**:
```bash
k6 run --out json=reports/k6_results.json tests/performance/k6_load_test.js
```

**Configuration**:
- Stages: Ramp up to 200 users over 14 minutes
- Thresholds: 95% of requests < 500ms, error rate < 1%

#### Memory Profiling

**Script**: `tests/performance/memory_profiling.py`

**Profiles**:
- Database memory usage
- Model manager memory usage
- Inference executor memory usage

**Usage**:
```bash
python tests/performance/memory_profiling.py
```

**Output**:
- Current memory usage
- Peak memory usage
- Memory per operation

#### Latency Testing

**Script**: `tests/performance/latency_test.py`

**Tests**:
- Health check latency
- Network stats latency
- User info latency
- Chat message latency
- Leaderboard latency
- Metrics endpoint latency

**Usage**:
```bash
python tests/performance/latency_test.py http://localhost:8000
```

**Metrics**:
- Min/Max/Avg latency
- Median latency
- P95/P99 latency
- Success rate

#### Miner Performance

```bash
# Miner engine performance test
cd miner-engine
python test_miner.py --performance
```

**Metrikler**:
- Gradient computation time
- IPFS upload time
- Blockchain submission time
- Total iteration time

#### Node Performance

```bash
# Load test: 200 gradient submission
cd remes
go test -v ./x/remes/keeper/performance_test.go
```

**Beklenen**: 200 gradient submission ~10ms

---

### 4. Security Tests

#### Authentication Tests

- Message signing verification
- Nonce replay attack prevention
- Rate limiting enforcement
- Staking requirement validation

#### Economic Attack Tests

- Slashing mechanism tests
- Reputation system tests
- Challenge/dispute resolution tests

---

## 📋 Test Senaryoları

### Senaryo 1: Tek Miner Test

**Adımlar**:
1. IPFS başlat
2. Blockchain node başlat
3. Miner başlat
4. Gradient submission test et
5. Dashboard'da görünürlüğü kontrol et

**Beklenen**:
- ✅ Active miners: 1
- ✅ Block height: Artıyor
- ✅ Network hash rate: > 0
- ✅ Miner locations: 3D globe'da görünür

### Senaryo 2: Web Dashboard Test (Gerçek Senaryo)

**Adımlar**:
1. Backend servisini başlat (`python run_backend.py`)
2. Frontend servisini başlat (`cd web-dashboard && npm run dev`)
3. Web Dashboard'u aç (`http://localhost:3000`)
4. Onboarding'i tamamla
5. Cüzdan bağla (Keplr)
6. Chat sayfasını test et:
   - Mesaj gönder
   - Streaming response kontrol et
   - Kredi düşüşünü kontrol et
7. Mine sayfasını test et:
   - Miner stats görüntüle
   - Earnings history kontrol et
   - Hashrate graph kontrol et
8. Network sayfasını test et:
   - 3D globe görüntüle
   - Miners table kontrol et
   - Recent blocks kontrol et
9. Settings sayfasını test et:
   - Config değerlerini görüntüle
   - Config değerlerini güncelle

**Beklenen**: Tüm adımlar başarılı

**Not**: Legacy Tkinter Control Panel artık kullanılmıyor, Web Dashboard kullanılmalı.

### Senaryo 3: Desktop Launcher Test

**Adımlar**:
1. Desktop launcher'ı başlat
2. Node'u başlat (menüden)
3. Miner'ı başlat (menüden)
4. Dashboard'da kontrol et
5. Node'u durdur
6. Miner'ı durdur

**Beklenen**: Tüm process'ler doğru başlatılıp durduruluyor

---

## 🔍 Test Checklist

### Go Tests
- [ ] Integration tests geçiyor
- [ ] Loss verification tests geçiyor
- [ ] Performance tests geçiyor
- [ ] PoRep tests geçiyor
- [ ] Block time tests geçiyor
- [ ] Genesis validation tests geçiyor

### Python Tests
- [ ] Deterministic execution tests geçiyor
- [ ] Miner engine basic tests geçiyor
- [ ] BitNet property tests (düşük öncelik)

### Integration Tests
- [ ] End-to-end protocol flow çalışıyor
- [ ] Multi-miner test çalışıyor
- [ ] Dashboard integration çalışıyor

### Performance Tests
- [ ] Miner performance acceptable
- [ ] Node performance acceptable
- [ ] Load testing başarılı

### Security Tests
- [ ] Authentication tests geçiyor
- [ ] Economic attack tests geçiyor
- [ ] Slashing mechanism tests geçiyor

---

## 🐛 Troubleshooting

### Test Başarısız Olursa

1. **Virtual Environment Kontrolü**:
   ```bash
   # Python tests için
   cd miner-engine
   source venv/bin/activate
   pip install -r requirements.txt
   pip install pytest hypothesis torch
   ```

2. **Go Dependencies Kontrolü**:
   ```bash
   # Go tests için
   cd remes
   go mod download
   go mod tidy
   ```

3. **Sequence Initialization**:
   ```go
   // Test ortamında sequence'ları manuel başlat
   k.StoredGradientID.Set(ctx, 1)
   k.AggregationID.Set(ctx, 1)
   ```

---

## 📊 Test Sonuçları

### Go Tests: ✅ %100 Başarılı

- Integration tests: ✅
- Loss verification: ✅
- Performance tests: ✅
- PoRep tests: ✅
- Block time tests: ✅

### Python Tests: ⚠️ Kısmen Başarılı

- Deterministic execution: ✅
- Miner engine basic: ✅
- BitNet property tests: ⚠️ (Implementasyon sorunları)

---

Bu testing rehberi, R3MES sisteminin kalitesini ve güvenilirliğini sağlamak için kapsamlı test stratejisi sağlar.


# Backend API Refactoring Roadmap

## Phase 1: Repository Pattern Implementation (Priority 1)

### Step 1.1: Create Repository Base Class
**File**: `backend/app/repositories/base_repository.py`
- Abstract base class with common CRUD operations
- Error handling wrapper
- Logging integration

### Step 1.2: Implement UserRepository
**File**: `backend/app/repositories/user_repository.py`
- `get_user_info(wallet_address)` - from main.py:942
- `get_network_stats()` - from main.py:961
- `create_user(wallet_address)`
- `update_credits(wallet_address, amount)`

### Step 1.3: Implement APIKeyRepository
**File**: `backend/app/repositories/api_key_repository.py`
- `create_api_key(wallet_address, name, expires_days)` - from main.py:1129
- `validate_api_key(api_key)` - from main.py:594
- `list_api_keys(wallet_address)` - from main.py:1163
- `revoke_api_key(api_key_id, wallet_address)` - from main.py:1178
- `delete_api_key(api_key_id, wallet_address)` - from main.py:1203

### Step 1.4: Implement MiningStatsRepository
**File**: `backend/app/repositories/mining_stats_repository.py`
- `get_miner_stats(wallet_address)` - from main.py:1020
- `get_earnings_history(wallet_address, days)` - from main.py:1049
- `get_hashrate_history(wallet_address, days)` - from main.py:1070
- `record_mining_stats(wallet_address, stats)`

### Step 1.5: Implement CreditRepository
**File**: `backend/app/repositories/credit_repository.py`
- `reserve_credit_atomic(wallet_address, amount)` - from main.py:750
- `confirm_credit_reservation(reservation_id)` - from main.py:800
- `rollback_credit_reservation(reservation_id)` - from main.py:810
- `deduct_credits(wallet_address, amount)`

---

## Phase 2: Main.py Refactoring (Priority 1)

### Step 2.1: Extract Lifespan Management
**File**: `backend/app/lifespan/manager.py`
- Move lifespan context manager (main.py:147-420)
- Create startup orchestrator
- Create shutdown orchestrator

### Step 2.2: Extract Request Models
**File**: `backend/app/models/requests.py`
- ChatRequest (main.py:623)
- CreateAPIKeyRequest (main.py:1076)
- RevokeAPIKeyRequest (main.py:1110)
- LoRARegisterRequest (main.py:1273)
- ServingNodeRegisterRequest (main.py:1308)
- ServingNodeHeartbeatRequest (main.py:1337)

### Step 2.3: Extract Response Models
**File**: `backend/app/models/responses.py`
- UserInfoResponse (main.py:659)
- NetworkStatsResponse (main.py:664)
- BlockResponse (main.py:670)
- BlocksResponse (main.py:676)
- MinerStatsResponse (main.py:681)
- EarningsHistoryResponse (main.py:690)
- HashrateHistoryResponse (main.py:693)

### Step 2.4: Extract Middleware Setup
**File**: `backend/app/middleware/setup.py`
- CORS configuration (main.py:500-570)
- Middleware ordering
- Error handler setup

### Step 2.5: Create Refactored main.py
- Reduce to ~100 lines
- Only app initialization
- Router inclusion

---

## Phase 3: Error Handling Standardization (Priority 2)

### Step 3.1: Create Error Handler Middleware
**File**: `backend/app/middleware/error_handler.py`
- Catch all exceptions
- Convert to appropriate HTTP responses
- Consistent error format

### Step 3.2: Standardize Endpoint Error Handling
- Replace HTTPException with R3MESException
- Use exception hierarchy
- Consistent logging

### Step 3.3: Fix Silent Failures
- Adapter loading: fail-fast or skip gracefully
- Blockchain indexer: optional component
- Cache warming: non-blocking
- System metrics: optional component

---

## Phase 4: Input Validation Hardening (Priority 2)

### Step 4.1: Enhance Pydantic Validators
**File**: `backend/app/models/validators.py`
- Centralize all validators
- Add XSS prevention
- Add SQL injection prevention
- Add SSRF prevention

### Step 4.2: Update All Request Models
- Use enhanced validators
- Add field-level validation
- Add cross-field validation

### Step 4.3: Add Wallet-Based Rate Limiting
**File**: `backend/app/middleware/wallet_rate_limiter.py`
- Track per-wallet rate limits
- Combine with IP-based limits
- Configurable thresholds

---

## Implementation Order

### Week 1:
1. Create repository base class
2. Implement UserRepository
3. Implement APIKeyRepository
4. Implement MiningStatsRepository
5. Implement CreditRepository

### Week 2:
1. Extract lifespan management
2. Extract request/response models
3. Extract middleware setup
4. Refactor main.py
5. Update all endpoints to use repositories

### Week 3:
1. Create error handler middleware
2. Standardize endpoint error handling
3. Fix silent failures
4. Enhance Pydantic validators
5. Add wallet-based rate limiting

---

## Testing Strategy

### Unit Tests:
- Repository methods
- Validators
- Error handlers

### Integration Tests:
- Endpoint with repository
- Error handling flow
- Rate limiting

### Load Tests:
- Repository performance
- Concurrent requests
- Rate limiting effectiveness

---

## Rollback Plan

1. Keep old main.py as main_old.py
2. Feature flag for new repositories
3. Gradual endpoint migration
4. Monitor error rates
5. Rollback if issues detected


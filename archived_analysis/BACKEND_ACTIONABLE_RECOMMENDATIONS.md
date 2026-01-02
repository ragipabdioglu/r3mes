# Backend API - Actionable Recommendations

## Priority 0: CRITICAL SECURITY FIXES

### 1. API Key Hashing (CRITICAL)

**Current Issue:**
```python
# database.py - API keys stored in plain text
cursor.execute("""
    INSERT INTO api_keys (api_key, wallet_address, name, expires_at)
    VALUES (?, ?, ?, ?)
""", (api_key, wallet_address, name, expires_at))
```

**Risk:** If database is compromised, all API keys are exposed

**Fix:**
```python
import hashlib

def create_api_key(self, wallet_address: str, name: Optional[str] = None):
    api_key = f"r3mes_{secrets.token_urlsafe(32)}"
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    
    # Store hash, not plain key
    cursor.execute("""
        INSERT INTO api_keys (api_key_hash, wallet_address, name, expires_at)
        VALUES (?, ?, ?, ?)
    """, (api_key_hash, wallet_address, name, expires_at))
    
    return api_key  # Return to user only once

def validate_api_key(self, api_key: str):
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    cursor.execute("""
        SELECT wallet_address, name, is_active, expires_at
        FROM api_keys
        WHERE api_key_hash = ?
    """, (api_key_hash,))
```

**Timeline:** 1-2 hours  
**Impact:** HIGH - Prevents API key compromise

---

### 2. Rate Limiting on Auth Endpoints

**Current Issue:**
```python
# No rate limiting on sensitive endpoints
@router.post("/api/keys")
async def create_api_key(...):
    pass
```

**Risk:** Brute force attacks on API key creation

**Fix:**
```python
from slowapi import Limiter

@router.post("/api/keys")
@limiter.limit("5/minute")  # 5 requests per minute
async def create_api_key(request: Request, ...):
    pass

@router.post("/chat")
@limiter.limit("10/minute")  # Already has this
async def chat(request: Request, ...):
    pass

# Add to all auth endpoints
@router.post("/login")
@limiter.limit("5/minute")
async def login(...):
    pass
```

**Timeline:** 30 minutes  
**Impact:** HIGH - Prevents brute force attacks

---

### 3. Input Validation Hardening

**Current Issue:**
```python
# Incomplete validation in some endpoints
@app.post("/chat")
async def chat(request: Request, chat_request: ChatRequest):
    # wallet_address might be None
    wallet_address = wallet_from_auth or chat_request.wallet_address
    if not wallet_address:
        # Missing error handling
        pass
```

**Fix:**
```python
from .exceptions import MissingCredentialsError

@app.post("/chat")
async def chat(request: Request, chat_request: ChatRequest):
    wallet_address = wallet_from_auth or chat_request.wallet_address
    
    if not wallet_address:
        raise MissingCredentialsError(
            "Either API key or wallet_address is required",
            credential_type="wallet_address"
        )
    
    # Validate wallet address format
    try:
        wallet_address = validate_wallet_address(wallet_address)
    except InvalidWalletAddressError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

**Timeline:** 1 hour  
**Impact:** MEDIUM - Prevents invalid input exploitation

---

## Priority 1: IMPORTANT REFACTORING

### 4. Split main.py (1316 lines)

**Current Structure:**
```
main.py (1316 lines)
├── Imports (100 lines)
├── Lazy loaders (50 lines)
├── Lifespan context manager (400 lines)
├── FastAPI app setup (100 lines)
├── Middleware setup (50 lines)
├── Router includes (50 lines)
├── Request models (100 lines)
├── Authentication (50 lines)
└── Endpoints (400 lines)
```

**Proposed Structure:**
```
backend/app/
├── main.py (100 lines) - Entry point only
├── app_factory.py (150 lines) - App creation
├── lifespan_manager.py (400 lines) - Startup/shutdown
├── request_models.py (100 lines) - Pydantic models
├── auth_middleware.py (100 lines) - Authentication
├── middleware_setup.py (100 lines) - Middleware config
└── endpoints/
    ├── __init__.py
    ├── chat.py (200 lines)
    ├── users.py (150 lines)
    └── ...
```

**New main.py:**
```python
from fastapi import FastAPI
from .app_factory import create_app
from .lifespan_manager import lifespan

app = create_app(lifespan=lifespan)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Timeline:** 4-6 hours  
**Impact:** HIGH - Improves maintainability

---

### 5. Implement Repository Pattern

**Current Issue:**
```python
# Database operations scattered in endpoints
@app.get("/users/{wallet}")
async def get_user(wallet: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE wallet_address = ?", (wallet,))
    # ...
```

**Proposed Solution:**
```python
# repositories/user_repository.py
class UserRepository:
    def __init__(self, database: AsyncDatabase):
        self.db = database
    
    async def get_by_wallet(self, wallet: str) -> Optional[User]:
        """Get user by wallet address."""
        result = await self.db.query(
            "SELECT * FROM users WHERE wallet_address = ?",
            (wallet,)
        )
        return User(**result) if result else None
    
    async def create(self, user: User) -> User:
        """Create new user."""
        await self.db.execute(
            "INSERT INTO users (wallet_address, credits) VALUES (?, ?)",
            (user.wallet_address, user.credits)
        )
        return user
    
    async def update_credits(self, wallet: str, amount: float) -> bool:
        """Update user credits."""
        return await self.db.execute(
            "UPDATE users SET credits = credits + ? WHERE wallet_address = ?",
            (amount, wallet)
        )

# endpoints/users.py
@router.get("/users/{wallet}")
async def get_user(
    wallet: str,
    repo: UserRepository = Depends(get_user_repository)
):
    user = await repo.get_by_wallet(wallet)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
```

**Timeline:** 6-8 hours  
**Impact:** HIGH - Improves testability and maintainability

---

### 6. Centralize Error Handling

**Current Issue:**
```python
# Inconsistent error handling
try:
    result = await operation()
except Exception as e:
    logger.error(f"Error: {e}")
    raise  # Sometimes

try:
    result = await operation()
except Exception as e:
    logger.warning(f"Error: {e}")
    return False  # Sometimes

try:
    result = await operation()
except Exception as e:
    logger.debug(f"Error: {e}")
    continue  # Sometimes
```

**Proposed Solution:**
```python
# middleware/error_handler.py
from fastapi import Request
from fastapi.responses import JSONResponse
from .exceptions import R3MESException

@app.exception_handler(R3MESException)
async def r3mes_exception_handler(request: Request, exc: R3MESException):
    return JSONResponse(
        status_code=get_status_code_for_exception(exc),
        content=exc.to_dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "request_id": request.headers.get("X-Request-ID")
        }
    )

# Usage in endpoints
@app.get("/users/{wallet}")
async def get_user(wallet: str):
    # No try/except needed - handled by middleware
    user = await user_repo.get_by_wallet(wallet)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
```

**Timeline:** 2-3 hours  
**Impact:** MEDIUM - Improves consistency

---

## Priority 2: PERFORMANCE IMPROVEMENTS

### 7. Add Database Indexes

**Current Issue:**
```python
# No indexes on frequently queried columns
SELECT * FROM users WHERE wallet_address = ?  # No index
SELECT * FROM api_keys WHERE api_key = ?  # No index
SELECT * FROM mining_stats WHERE wallet_address = ? ORDER BY recorded_at DESC  # No index
```

**Fix - Create Migration:**
```python
# alembic/versions/004_add_indexes.py
def upgrade():
    op.create_index('idx_users_wallet', 'users', ['wallet_address'])
    op.create_index('idx_api_keys_hash', 'api_keys', ['api_key_hash'])
    op.create_index('idx_mining_stats_wallet_time', 'mining_stats', 
                    ['wallet_address', 'recorded_at'], postgresql_using='btree')
    op.create_index('idx_earnings_history_wallet', 'earnings_history', ['wallet_address'])
    op.create_index('idx_hashrate_history_wallet', 'hashrate_history', ['wallet_address'])

def downgrade():
    op.drop_index('idx_users_wallet')
    op.drop_index('idx_api_keys_hash')
    op.drop_index('idx_mining_stats_wallet_time')
    op.drop_index('idx_earnings_history_wallet')
    op.drop_index('idx_hashrate_history_wallet')
```

**Timeline:** 1 hour  
**Impact:** HIGH - Improves query performance 10-100x

---

### 8. Implement Circuit Breaker

**Current Issue:**
```python
# No circuit breaker for external calls
response = requests.get(f"{self.rpc_endpoint}/status", timeout=2)
# If RPC is down, every request fails
```

**Fix:**
```python
# utils/circuit_breaker.py
from enum import Enum
import time

class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    async def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise

    def on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

# Usage
rpc_breaker = CircuitBreaker(failure_threshold=5, timeout=60)

async def get_blockchain_status():
    return await rpc_breaker.call(
        requests.get,
        f"{self.rpc_endpoint}/status",
        timeout=2
    )
```

**Timeline:** 2-3 hours  
**Impact:** MEDIUM - Improves resilience

---

### 9. Add Request/Response Caching

**Current Issue:**
```python
# Cache decorator exists but not widely used
@app.get("/network/stats")
async def get_network_stats():
    # No caching - recalculates every time
    return {
        "active_miners": count_miners(),
        "total_users": count_users(),
        "total_credits": sum_credits()
    }
```

**Fix:**
```python
from .cache import cache_response

@app.get("/network/stats")
@cache_response(ttl=300)  # Cache for 5 minutes
async def get_network_stats():
    return {
        "active_miners": await count_miners(),
        "total_users": await count_users(),
        "total_credits": await sum_credits()
    }

# Add cache invalidation
@app.post("/users")
async def create_user(user: User):
    result = await user_repo.create(user)
    # Invalidate cache
    await cache_manager.delete("network_stats")
    return result
```

**Timeline:** 1-2 hours  
**Impact:** MEDIUM - Improves response times

---

## Priority 3: MONITORING & OBSERVABILITY

### 10. Add Custom Metrics

**Current Issue:**
```python
# Prometheus metrics exist but limited
# Missing: business metrics, custom counters
```

**Fix:**
```python
# metrics/business_metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Custom metrics
chat_requests_total = Counter(
    'chat_requests_total',
    'Total chat requests',
    ['status', 'adapter']
)

chat_request_duration = Histogram(
    'chat_request_duration_seconds',
    'Chat request duration',
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0)
)

active_users = Gauge(
    'active_users',
    'Number of active users'
)

credits_distributed = Counter(
    'credits_distributed_total',
    'Total credits distributed',
    ['reason']
)

# Usage
@app.post("/chat")
async def chat(request: Request, chat_request: ChatRequest):
    start_time = time.time()
    try:
        result = await process_chat(chat_request)
        chat_requests_total.labels(status='success', adapter=adapter).inc()
        return result
    except Exception as e:
        chat_requests_total.labels(status='error', adapter='unknown').inc()
        raise
    finally:
        duration = time.time() - start_time
        chat_request_duration.observe(duration)
```

**Timeline:** 2-3 hours  
**Impact:** MEDIUM - Improves observability

---

## Implementation Timeline

### Week 1 (Priority 0 - CRITICAL)
- [ ] API Key Hashing (2 hours)
- [ ] Rate Limiting on Auth (1 hour)
- [ ] Input Validation Hardening (2 hours)
- [ ] Testing (2 hours)
- **Total: 7 hours**

### Week 2 (Priority 1 - IMPORTANT)
- [ ] Split main.py (6 hours)
- [ ] Repository Pattern (8 hours)
- [ ] Centralize Error Handling (3 hours)
- [ ] Testing (4 hours)
- **Total: 21 hours**

### Week 3 (Priority 2 - IMPROVEMENTS)
- [ ] Database Indexes (1 hour)
- [ ] Circuit Breaker (3 hours)
- [ ] Request Caching (2 hours)
- [ ] Custom Metrics (3 hours)
- [ ] Testing (2 hours)
- **Total: 11 hours**

**Grand Total: 39 hours (~1 week of development)**

---

## Testing Strategy

### Unit Tests
```python
# tests/test_repositories.py
@pytest.mark.asyncio
async def test_user_repository_get_by_wallet():
    repo = UserRepository(mock_database)
    user = await repo.get_by_wallet("remes1...")
    assert user.wallet_address == "remes1..."

# tests/test_api_keys.py
def test_api_key_hashing():
    api_key = "r3mes_test123"
    hash1 = hashlib.sha256(api_key.encode()).hexdigest()
    hash2 = hashlib.sha256(api_key.encode()).hexdigest()
    assert hash1 == hash2
```

### Integration Tests
```python
# tests/test_endpoints.py
@pytest.mark.asyncio
async def test_chat_endpoint_requires_wallet():
    response = await client.post("/chat", json={"message": "test"})
    assert response.status_code == 400

@pytest.mark.asyncio
async def test_rate_limiting():
    for i in range(6):
        response = await client.post("/api/keys", ...)
        if i < 5:
            assert response.status_code == 201
        else:
            assert response.status_code == 429  # Too Many Requests
```

---

## Monitoring & Alerts

### Key Metrics to Monitor
1. **API Performance:**
   - Response time (p50, p95, p99)
   - Error rate
   - Request rate

2. **Database:**
   - Query time
   - Connection pool usage
   - Slow queries

3. **Business:**
   - Chat requests per minute
   - Credits distributed
   - Active users

### Alert Thresholds
- Response time > 1s: WARNING
- Error rate > 1%: CRITICAL
- Database connection pool > 80%: WARNING
- Circuit breaker OPEN: CRITICAL


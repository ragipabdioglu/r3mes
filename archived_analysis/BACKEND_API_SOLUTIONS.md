# Backend API - Detaylı Çözüm Örnekleri

---

## ÇÖZÜM 1: Database Duplicate Code Birleştirme

### Sorun
- `database.py` (SQLite sync) ve `database_async.py` (async wrapper) aynı işi yapıyor
- 40%+ code duplication
- Bakım ve test zorluğu

### Çözüm Mimarisi

**database_base.py (Abstract Base Class)**
```python
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any

class DatabaseBase(ABC):
    """Abstract base class for database implementations."""
    
    @abstractmethod
    async def connect(self):
        """Establish database connection."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Close database connection."""
        pass
    
    @abstractmethod
    async def add_credits(self, wallet: str, amount: float) -> bool:
        """Add credits to user."""
        pass
    
    @abstractmethod
    async def deduct_credit(self, wallet: str, amount: float) -> bool:
        """Deduct credits from user."""
        pass
    
    @abstractmethod
    async def check_credits(self, wallet: str) -> float:
        """Check user credits."""
        pass
    
    @abstractmethod
    async def get_user_info(self, wallet: str) -> Optional[Dict]:
        """Get user information."""
        pass
    
    @abstractmethod
    async def create_api_key(self, wallet: str, name: str) -> str:
        """Create API key."""
        pass
    
    @abstractmethod
    async def validate_api_key(self, api_key: str) -> Optional[Dict]:
        """Validate API key."""
        pass
```

**database_sqlite.py (SQLite Implementation)**
```python
import aiosqlite
from .database_base import DatabaseBase

class SQLiteDatabase(DatabaseBase):
    """SQLite database implementation."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._connection: Optional[aiosqlite.Connection] = None
    
    async def connect(self):
        """Establish SQLite connection."""
        self._connection = await aiosqlite.connect(self.db_path)
        await self._init_tables()
    
    async def disconnect(self):
        """Close SQLite connection."""
        if self._connection:
            await self._connection.close()
    
    async def add_credits(self, wallet: str, amount: float) -> bool:
        """Add credits to user."""
        await self._connection.execute(
            "UPDATE users SET credits = credits + ? WHERE wallet_address = ?",
            (amount, wallet)
        )
        await self._connection.commit()
        return True
    
    async def deduct_credit(self, wallet: str, amount: float) -> bool:
        """Deduct credits from user."""
        cursor = await self._connection.execute(
            "SELECT credits FROM users WHERE wallet_address = ?",
            (wallet,)
        )
        row = await cursor.fetchone()
        
        if not row or row[0] < amount:
            return False
        
        await self._connection.execute(
            "UPDATE users SET credits = credits - ? WHERE wallet_address = ?",
            (amount, wallet)
        )
        await self._connection.commit()
        return True
    
    async def check_credits(self, wallet: str) -> float:
        """Check user credits."""
        cursor = await self._connection.execute(
            "SELECT credits FROM users WHERE wallet_address = ?",
            (wallet,)
        )
        row = await cursor.fetchone()
        return row[0] if row else 0.0
    
    async def _init_tables(self):
        """Initialize database tables."""
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                wallet_address TEXT PRIMARY KEY,
                credits REAL DEFAULT 0.0,
                is_miner BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await self._connection.commit()
```

**database_postgresql.py (PostgreSQL Implementation)**
```python
import asyncpg
from .database_base import DatabaseBase

class PostgreSQLDatabase(DatabaseBase):
    """PostgreSQL database implementation."""
    
    def __init__(self, connection_string: str, min_size: int = 10, max_size: int = 20):
        self.connection_string = connection_string
        self.min_size = min_size
        self.max_size = max_size
        self.pool: Optional[asyncpg.Pool] = None
    
    async def connect(self):
        """Establish PostgreSQL connection pool."""
        self.pool = await asyncpg.create_pool(
            self.connection_string,
            min_size=self.min_size,
            max_size=self.max_size
        )
        await self._init_tables()
    
    async def disconnect(self):
        """Close PostgreSQL connection pool."""
        if self.pool:
            await self.pool.close()
    
    async def add_credits(self, wallet: str, amount: float) -> bool:
        """Add credits to user."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE users SET credits = credits + $1 WHERE wallet_address = $2",
                amount, wallet
            )
        return True
    
    async def deduct_credit(self, wallet: str, amount: float) -> bool:
        """Deduct credits from user."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT credits FROM users WHERE wallet_address = $1",
                wallet
            )
            
            if not row or row['credits'] < amount:
                return False
            
            await conn.execute(
                "UPDATE users SET credits = credits - $1 WHERE wallet_address = $2",
                amount, wallet
            )
        return True
    
    async def check_credits(self, wallet: str) -> float:
        """Check user credits."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT credits FROM users WHERE wallet_address = $1",
                wallet
            )
            return row['credits'] if row else 0.0
    
    async def _init_tables(self):
        """Initialize database tables."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    wallet_address TEXT PRIMARY KEY,
                    credits REAL DEFAULT 0.0,
                    is_miner BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
```

**database_factory.py (Factory Pattern)**
```python
from .database_base import DatabaseBase
from .database_sqlite import SQLiteDatabase
from .database_postgresql import PostgreSQLDatabase
import os

def create_database() -> DatabaseBase:
    """Create appropriate database instance based on configuration."""
    db_type = os.getenv("DATABASE_TYPE", "sqlite")
    
    if db_type == "postgresql":
        connection_string = os.getenv("DATABASE_URL")
        if not connection_string:
            raise ValueError("DATABASE_URL must be set for PostgreSQL")
        
        return PostgreSQLDatabase(
            connection_string,
            min_size=int(os.getenv("DB_POOL_MIN_SIZE", "10")),
            max_size=int(os.getenv("DB_POOL_MAX_SIZE", "20"))
        )
    else:
        db_path = os.getenv("DATABASE_PATH", "r3mes.db")
        return SQLiteDatabase(db_path)
```

---

## ÇÖZÜM 2: N+1 Query Problem Düzeltme

### Sorun
- Serving endpoints: 100 nodes = 200 queries
- Advanced analytics: 1000 validators = 1000 queries
- Performance degradation 10x

### Çözüm

**Batch Query Implementation**
```python
# ✅ BEFORE: N+1 queries
async def get_serving_nodes_with_stats_slow(self, limit: int = 100):
    """❌ N+1 queries - slow"""
    nodes = await self.get_serving_nodes(limit)
    
    result = []
    for node in nodes:
        status = await self.get_node_status(node['address'])  # Query per node
        stats = await self.get_node_stats(node['address'])    # Another query per node
        result.append({**node, **status, **stats})
    
    return result

# ✅ AFTER: Single batch query
async def get_serving_nodes_with_stats_fast(self, limit: int = 100):
    """✅ Single query with JOIN - fast"""
    query = """
    SELECT 
        n.node_address,
        n.model_version,
        n.model_ipfs_hash,
        COUNT(r.request_id) as total_requests,
        SUM(CASE WHEN r.status = 'success' THEN 1 ELSE 0 END) as successful_requests,
        AVG(r.latency_ms) as average_latency_ms,
        MAX(n.last_heartbeat) as last_heartbeat
    FROM serving_nodes n
    LEFT JOIN inference_requests r ON n.node_address = r.serving_node
    GROUP BY n.node_address, n.model_version, n.model_ipfs_hash
    ORDER BY n.node_address
    LIMIT $1
    """
    
    async with self.pool.acquire() as conn:
        rows = await conn.fetch(query, limit)
    
    return [dict(row) for row in rows]
```

**Validator Query Optimization**
```python
# ✅ BEFORE: N+1 queries
async def get_validators_with_trust_scores_slow(self):
    """❌ N+1 queries"""
    validators = await self.get_validators()
    
    result = []
    for validator in validators:
        verification = await self.get_verification_records(validator['address'])  # Query per validator
        trust_score = calculate_trust_score(verification)
        result.append({**validator, 'trust_score': trust_score})
    
    return result

# ✅ AFTER: Single batch query
async def get_validators_with_trust_scores_fast(self):
    """✅ Single query with aggregation"""
    query = """
    SELECT 
        v.validator_address,
        v.stake,
        v.commission,
        COUNT(vr.id) as verification_count,
        SUM(CASE WHEN vr.status = 'verified' THEN 1 ELSE 0 END) as verified_count,
        CASE 
            WHEN COUNT(vr.id) = 0 THEN 0.5
            ELSE CAST(SUM(CASE WHEN vr.status = 'verified' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(vr.id)
        END as trust_score
    FROM validators v
    LEFT JOIN validator_verifications vr ON v.validator_address = vr.validator_address
    GROUP BY v.validator_address, v.stake, v.commission
    ORDER BY trust_score DESC
    """
    
    async with self.pool.acquire() as conn:
        rows = await conn.fetch(query)
    
    return [dict(row) for row in rows]
```

---

## ÇÖZÜM 3: Tight Coupling Kaldırma - Event-Driven Architecture

### Sorun
- Database → Cache direct calls
- API → Database hardcoded
- Services → Blockchain hardcoded

### Çözüm

**Event System**
```python
# events.py
from dataclasses import dataclass
from typing import Any, Dict
from datetime import datetime

@dataclass
class Event:
    """Base event class."""
    timestamp: datetime
    source: str
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now()

@dataclass
class UserCreditUpdatedEvent(Event):
    """Fired when user credits are updated."""
    wallet: str
    amount: float
    operation: str  # 'add' or 'deduct'
    reason: str = ""

@dataclass
class UserCreatedEvent(Event):
    """Fired when new user is created."""
    wallet: str
    is_miner: bool = False

@dataclass
class APIKeyCreatedEvent(Event):
    """Fired when API key is created."""
    wallet: str
    api_key_hash: str
    name: str
```

**Event Bus**
```python
# event_bus.py
from typing import Callable, Dict, List, Type
import asyncio
import logging

logger = logging.getLogger(__name__)

class EventBus:
    """Simple event bus for decoupling components."""
    
    def __init__(self):
        self._subscribers: Dict[Type[Event], List[Callable]] = {}
    
    def subscribe(self, event_type: Type[Event], handler: Callable):
        """Subscribe to an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
        logger.debug(f"Subscribed {handler.__name__} to {event_type.__name__}")
    
    async def publish(self, event: Event):
        """Publish an event to all subscribers."""
        handlers = self._subscribers.get(type(event), [])
        
        if not handlers:
            logger.debug(f"No handlers for {type(event).__name__}")
            return
        
        # Run handlers concurrently
        tasks = [handler(event) for handler in handlers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log any errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error in handler {handlers[i].__name__}: {result}")
```

**Refactored Database**
```python
# database_async.py (refactored)
from .event_bus import EventBus
from .events import UserCreditUpdatedEvent

class AsyncDatabase:
    """Database with event publishing."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
    
    async def add_credits(self, wallet: str, amount: float, reason: str = "") -> bool:
        """Add credits and publish event."""
        # Database operation
        await self._execute_add_credits(wallet, amount)
        
        # ✅ Publish event instead of direct cache call
        event = UserCreditUpdatedEvent(
            source="database",
            wallet=wallet,
            amount=amount,
            operation="add",
            reason=reason
        )
        await self.event_bus.publish(event)
        
        return True
    
    async def deduct_credit(self, wallet: str, amount: float, reason: str = "") -> bool:
        """Deduct credits and publish event."""
        # Database operation
        success = await self._execute_deduct_credit(wallet, amount)
        
        if success:
            # ✅ Publish event
            event = UserCreditUpdatedEvent(
                source="database",
                wallet=wallet,
                amount=amount,
                operation="deduct",
                reason=reason
            )
            await self.event_bus.publish(event)
        
        return success
```

**Event Handlers**
```python
# cache_invalidation.py (refactored)
from .events import UserCreditUpdatedEvent, UserCreatedEvent
from .cache import CacheManager
import logging

logger = logging.getLogger(__name__)

class CacheInvalidationHandler:
    """Handles cache invalidation events."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
    
    async def handle_credit_updated(self, event: UserCreditUpdatedEvent):
        """Handle credit update event."""
        logger.debug(f"Invalidating cache for {event.wallet}")
        await self.cache_manager.invalidate_user_cache(event.wallet)
    
    async def handle_user_created(self, event: UserCreatedEvent):
        """Handle user creation event."""
        logger.debug(f"Initializing cache for {event.wallet}")
        await self.cache_manager.initialize_user_cache(event.wallet)

# notification.py (refactored)
from .events import UserCreditUpdatedEvent
from .notifications import NotificationService
import logging

logger = logging.getLogger(__name__)

class NotificationHandler:
    """Handles notification events."""
    
    def __init__(self, notification_service: NotificationService):
        self.notification_service = notification_service
    
    async def handle_low_credits(self, event: UserCreditUpdatedEvent):
        """Send notification for low credits."""
        if event.operation == "deduct":
            # Check if credits are low
            credits = await self.database.check_credits(event.wallet)
            if credits < 10:
                await self.notification_service.send_notification(
                    wallet=event.wallet,
                    title="Low Credits Alert",
                    message=f"Your credits are running low: {credits}",
                    priority="high"
                )
```

**Application Setup**
```python
# main.py (refactored)
from .event_bus import EventBus
from .database_async import AsyncDatabase
from .cache_invalidation import CacheInvalidationHandler
from .notifications import NotificationHandler

# Create event bus
event_bus = EventBus()

# Create services
database = AsyncDatabase(event_bus)
cache_manager = CacheManager()
notification_service = NotificationService()

# Create handlers
cache_handler = CacheInvalidationHandler(cache_manager)
notification_handler = NotificationHandler(notification_service)

# Subscribe handlers to events
event_bus.subscribe(UserCreditUpdatedEvent, cache_handler.handle_credit_updated)
event_bus.subscribe(UserCreatedEvent, cache_handler.handle_user_created)
event_bus.subscribe(UserCreditUpdatedEvent, notification_handler.handle_low_credits)
```

---

## ÇÖZÜM 4: Silent Failures Düzeltme

### Sorun
- Kritik hatalar debug seviyesinde loglanıyor
- Production'da sorunlar fark edilmiyor
- Fallback logic sessiz şekilde çalışıyor

### Çözüm

**Structured Error Handling**
```python
# advanced_analytics.py (refactored)
import logging
from .notifications import NotificationService, NotificationPriority

logger = logging.getLogger(__name__)
notification_service = NotificationService()

async def _build_timeline_data(days: int, granularity: str, blockchain_client):
    """Build timeline data with proper error handling."""
    
    try:
        # Try to use indexed historical data first
        if database.config.is_postgresql():
            try:
                indexer = get_indexer(database)
                snapshots = await indexer.get_network_snapshots(
                    start_date=start_date,
                    end_date=end_date,
                    limit=1000
                )
                
                if snapshots:
                    timeline = _convert_snapshots_to_timeline(snapshots)
                    logger.info(f"Built timeline from {len(snapshots)} indexed snapshots")
                    return timeline
            
            except IndexError as e:
                # Specific error handling
                logger.warning(f"Timeline indexing failed: {e}")
                # Continue with fallback
            
            except Exception as e:
                # Unexpected errors
                logger.error(f"Unexpected error in timeline indexing: {e}", exc_info=True)
                
                # Send alert
                await notification_service.send_system_alert(
                    component="analytics",
                    alert_type="timeline_error",
                    message=f"Timeline processing failed: {e}",
                    priority=NotificationPriority.HIGH
                )
                # Continue with fallback
        
        # Fallback: use current data
        logger.info("Using fallback timeline data")
        timeline = await _get_current_timeline_data()
        return timeline
    
    except Exception as e:
        # Final fallback
        logger.error(f"Failed to build timeline data: {e}", exc_info=True)
        
        # Send critical alert
        await notification_service.send_system_alert(
            component="analytics",
            alert_type="timeline_critical",
            message=f"Timeline data unavailable: {e}",
            priority=NotificationPriority.CRITICAL
        )
        
        # Return empty timeline
        return []

async def get_blockchain_params():
    """Get blockchain parameters with proper error handling."""
    
    try:
        block_data = await client._query_rest("/remes/remes/v1/params")
        return block_data
    
    except ConnectionError as e:
        # Network error - retry
        logger.warning(f"Blockchain connection failed, retrying: {e}")
        
        try:
            await asyncio.sleep(1)
            block_data = await retry_blockchain_query()
            return block_data
        except Exception as retry_error:
            logger.error(f"Retry failed: {retry_error}", exc_info=True)
            # Continue with defaults
    
    except TimeoutError as e:
        # Timeout error
        logger.error(f"Blockchain query timeout: {e}", exc_info=True)
        
        await notification_service.send_system_alert(
            component="blockchain",
            alert_type="timeout",
            message=f"Blockchain query timeout: {e}",
            priority=NotificationPriority.HIGH
        )
    
    except Exception as e:
        # Unexpected errors
        logger.error(f"Failed to fetch blockchain data: {e}", exc_info=True)
        
        await notification_service.send_system_alert(
            component="blockchain",
            alert_type="query_error",
            message=f"Blockchain query failed: {e}",
            priority=NotificationPriority.HIGH
        )
    
    # Use defaults with warning
    logger.warning("Using default values for blockchain parameters")
    return {
        "miners_count": 0,
        "validators_count": 0,
        "total_stake": 0.0
    }
```

---

## ÇÖZÜM 5: Input Validation Consolidation

### Sorun
- Duplicate validation logic (input_validation.py vs input_validator.py)
- Weak wallet address validation
- Inconsistent validation patterns

### Çözüm

**Centralized Validators**
```python
# validators.py (single source of truth)
import re
from typing import Optional, List
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Input validation error."""
    pass

class WalletValidator:
    """Centralized wallet address validation."""
    
    # Strict regex pattern for R3MES wallet addresses
    PATTERN = re.compile(r'^remes1[a-z0-9]{38}

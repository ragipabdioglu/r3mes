"""
Batch Loader - N+1 Query Problem Solution

Implements DataLoader pattern to eliminate N+1 queries by batching
database operations and providing intelligent caching.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)

T = TypeVar('T')
K = TypeVar('K')


@dataclass
class BatchRequest:
    """Represents a batch request for data loading."""
    keys: List[Any]
    resolve_fn: Callable
    created_at: datetime = field(default_factory=datetime.now)
    timeout: float = 0.1  # 100ms batch window


class DataLoader(Generic[K, T]):
    """
    DataLoader implementation to solve N+1 query problems.
    
    Batches individual loads into a single batch operation,
    caches results, and provides efficient data access patterns.
    """
    
    def __init__(
        self,
        batch_load_fn: Callable[[List[K]], List[T]],
        batch_size: int = 100,
        cache_ttl: int = 300,  # 5 minutes
        max_batch_delay: float = 0.1  # 100ms
    ):
        """
        Initialize DataLoader.
        
        Args:
            batch_load_fn: Function that loads data for a list of keys
            batch_size: Maximum batch size
            cache_ttl: Cache time-to-live in seconds
            max_batch_delay: Maximum delay before executing batch
        """
        self.batch_load_fn = batch_load_fn
        self.batch_size = batch_size
        self.cache_ttl = cache_ttl
        self.max_batch_delay = max_batch_delay
        
        # Internal state
        self._cache: Dict[K, tuple[T, datetime]] = {}
        self._pending_keys: List[K] = []
        self._pending_futures: Dict[K, asyncio.Future] = {}
        self._batch_timer: Optional[asyncio.Task] = None
        self._stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'batch_loads': 0,
            'total_keys_loaded': 0
        }
    
    async def load(self, key: K) -> Optional[T]:
        """
        Load a single item by key.
        
        Args:
            key: Key to load
            
        Returns:
            Loaded item or None if not found
        """
        # Check cache first
        cached_result = self._get_from_cache(key)
        if cached_result is not None:
            self._stats['cache_hits'] += 1
            return cached_result
        
        self._stats['cache_misses'] += 1
        
        # Check if already pending
        if key in self._pending_futures:
            return await self._pending_futures[key]
        
        # Create future for this key
        future = asyncio.Future()
        self._pending_futures[key] = future
        self._pending_keys.append(key)
        
        # Start batch timer if not already running
        if self._batch_timer is None:
            self._batch_timer = asyncio.create_task(self._batch_timer_task())
        
        # If batch is full, execute immediately
        if len(self._pending_keys) >= self.batch_size:
            await self._execute_batch()
        
        return await future
    
    async def load_many(self, keys: List[K]) -> List[Optional[T]]:
        """
        Load multiple items by keys.
        
        Args:
            keys: List of keys to load
            
        Returns:
            List of loaded items (None for not found)
        """
        # Check cache for all keys
        results = []
        uncached_keys = []
        key_to_index = {}
        
        for i, key in enumerate(keys):
            cached_result = self._get_from_cache(key)
            if cached_result is not None:
                results.append(cached_result)
                self._stats['cache_hits'] += 1
            else:
                results.append(None)  # Placeholder
                uncached_keys.append(key)
                key_to_index[key] = i
                self._stats['cache_misses'] += 1
        
        # Load uncached keys
        if uncached_keys:
            loaded_results = await self._load_batch(uncached_keys)
            
            # Fill in results
            for key, result in zip(uncached_keys, loaded_results):
                if key in key_to_index:
                    results[key_to_index[key]] = result
        
        return results
    
    async def prime(self, key: K, value: T):
        """
        Prime the cache with a known value.
        
        Args:
            key: Key to prime
            value: Value to cache
        """
        self._cache[key] = (value, datetime.now())
    
    async def clear(self, key: Optional[K] = None):
        """
        Clear cache for a specific key or all keys.
        
        Args:
            key: Key to clear, or None to clear all
        """
        if key is None:
            self._cache.clear()
        else:
            self._cache.pop(key, None)
    
    def _get_from_cache(self, key: K) -> Optional[T]:
        """Get item from cache if not expired."""
        if key not in self._cache:
            return None
        
        value, cached_at = self._cache[key]
        
        # Check if expired
        if datetime.now() - cached_at > timedelta(seconds=self.cache_ttl):
            del self._cache[key]
            return None
        
        return value
    
    async def _batch_timer_task(self):
        """Timer task to execute batch after delay."""
        try:
            await asyncio.sleep(self.max_batch_delay)
            if self._pending_keys:
                await self._execute_batch()
        except asyncio.CancelledError:
            pass
        finally:
            self._batch_timer = None
    
    async def _execute_batch(self):
        """Execute pending batch load."""
        if not self._pending_keys:
            return
        
        # Cancel timer if running
        if self._batch_timer:
            self._batch_timer.cancel()
            self._batch_timer = None
        
        # Get current batch
        keys_to_load = self._pending_keys[:]
        futures_to_resolve = {k: f for k, f in self._pending_futures.items() if k in keys_to_load}
        
        # Clear pending state
        self._pending_keys.clear()
        self._pending_futures.clear()
        
        try:
            # Load batch
            results = await self._load_batch(keys_to_load)
            
            # Resolve futures and cache results
            for key, result in zip(keys_to_load, results):
                if key in futures_to_resolve:
                    futures_to_resolve[key].set_result(result)
                
                # Cache result
                if result is not None:
                    self._cache[key] = (result, datetime.now())
            
            self._stats['batch_loads'] += 1
            self._stats['total_keys_loaded'] += len(keys_to_load)
            
        except Exception as e:
            # Resolve all futures with exception
            for future in futures_to_resolve.values():
                if not future.done():
                    future.set_exception(e)
            
            logger.error(f"Batch load failed: {e}")
    
    async def _load_batch(self, keys: List[K]) -> List[Optional[T]]:
        """Load batch of keys using the batch load function."""
        try:
            return await self.batch_load_fn(keys)
        except Exception as e:
            logger.error(f"Batch load function failed: {e}")
            return [None] * len(keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics."""
        total_requests = self._stats['cache_hits'] + self._stats['cache_misses']
        cache_hit_ratio = (
            self._stats['cache_hits'] / total_requests 
            if total_requests > 0 else 0
        )
        
        return {
            **self._stats,
            'cache_hit_ratio': cache_hit_ratio,
            'cache_size': len(self._cache),
            'pending_keys': len(self._pending_keys)
        }


class DatabaseBatchLoader:
    """
    Database-specific batch loader implementation.
    
    Provides pre-configured data loaders for common database operations
    to eliminate N+1 queries.
    """
    
    def __init__(self, database_instance):
        """
        Initialize database batch loader.
        
        Args:
            database_instance: Database instance
        """
        self.db = database_instance
        self._loaders: Dict[str, DataLoader] = {}
        
        # Initialize common loaders
        self._setup_user_loaders()
        self._setup_api_key_loaders()
        self._setup_credit_loaders()
    
    def _setup_user_loaders(self):
        """Setup user-related data loaders."""
        
        async def batch_load_users(wallet_addresses: List[str]) -> List[Optional[Dict[str, Any]]]:
            """Batch load users by wallet addresses."""
            if not wallet_addresses:
                return []
            
            # Create placeholders for IN clause
            placeholders = ','.join(['?' for _ in wallet_addresses])
            query = f"""
                SELECT wallet_address, credits, is_miner, created_at, updated_at
                FROM users 
                WHERE wallet_address IN ({placeholders})
            """
            
            try:
                if hasattr(self.db, '_db') and self.db._db:
                    # PostgreSQL
                    async with self.db._db.pool.acquire() as conn:
                        rows = await conn.fetch(query.replace('?', '$1'), *wallet_addresses)
                else:
                    # SQLite
                    if not self.db._connection:
                        await self.db.connect()
                    cursor = await self.db._connection.cursor()
                    await cursor.execute(query, wallet_addresses)
                    rows = await cursor.fetchall()
                
                # Convert to dict and map to original order
                user_map = {row['wallet_address']: dict(row) for row in rows}
                return [user_map.get(addr) for addr in wallet_addresses]
                
            except Exception as e:
                logger.error(f"Failed to batch load users: {e}")
                return [None] * len(wallet_addresses)
        
        async def batch_load_user_credits(wallet_addresses: List[str]) -> List[Optional[float]]:
            """Batch load user credits by wallet addresses."""
            if not wallet_addresses:
                return []
            
            placeholders = ','.join(['?' for _ in wallet_addresses])
            query = f"""
                SELECT wallet_address, credits
                FROM users 
                WHERE wallet_address IN ({placeholders})
            """
            
            try:
                if hasattr(self.db, '_db') and self.db._db:
                    # PostgreSQL
                    async with self.db._db.pool.acquire() as conn:
                        rows = await conn.fetch(query.replace('?', '$1'), *wallet_addresses)
                else:
                    # SQLite
                    if not self.db._connection:
                        await self.db.connect()
                    cursor = await self.db._connection.cursor()
                    await cursor.execute(query, wallet_addresses)
                    rows = await cursor.fetchall()
                
                # Map credits to original order
                credit_map = {row['wallet_address']: row['credits'] for row in rows}
                return [credit_map.get(addr) for addr in wallet_addresses]
                
            except Exception as e:
                logger.error(f"Failed to batch load user credits: {e}")
                return [None] * len(wallet_addresses)
        
        # Create loaders
        self._loaders['users'] = DataLoader(batch_load_users)
        self._loaders['user_credits'] = DataLoader(batch_load_user_credits)
    
    def _setup_api_key_loaders(self):
        """Setup API key-related data loaders."""
        
        async def batch_load_api_keys_by_wallet(wallet_addresses: List[str]) -> List[List[Dict[str, Any]]]:
            """Batch load API keys by wallet addresses."""
            if not wallet_addresses:
                return []
            
            placeholders = ','.join(['?' for _ in wallet_addresses])
            query = f"""
                SELECT wallet_address, api_key_hash, name, is_active, created_at, expires_at
                FROM api_keys 
                WHERE wallet_address IN ({placeholders})
                ORDER BY wallet_address, created_at DESC
            """
            
            try:
                if hasattr(self.db, '_db') and self.db._db:
                    # PostgreSQL
                    async with self.db._db.pool.acquire() as conn:
                        rows = await conn.fetch(query.replace('?', '$1'), *wallet_addresses)
                else:
                    # SQLite
                    if not self.db._connection:
                        await self.db.connect()
                    cursor = await self.db._connection.cursor()
                    await cursor.execute(query, wallet_addresses)
                    rows = await cursor.fetchall()
                
                # Group by wallet address
                api_keys_by_wallet = defaultdict(list)
                for row in rows:
                    api_keys_by_wallet[row['wallet_address']].append(dict(row))
                
                return [api_keys_by_wallet.get(addr, []) for addr in wallet_addresses]
                
            except Exception as e:
                logger.error(f"Failed to batch load API keys: {e}")
                return [[] for _ in wallet_addresses]
        
        self._loaders['api_keys_by_wallet'] = DataLoader(batch_load_api_keys_by_wallet)
    
    def _setup_credit_loaders(self):
        """Setup credit-related data loaders."""
        
        async def batch_load_credit_reservations(wallet_addresses: List[str]) -> List[List[Dict[str, Any]]]:
            """Batch load credit reservations by wallet addresses."""
            if not wallet_addresses:
                return []
            
            placeholders = ','.join(['?' for _ in wallet_addresses])
            query = f"""
                SELECT wallet_address, reservation_id, amount, status, created_at, expires_at
                FROM credit_reservations 
                WHERE wallet_address IN ({placeholders})
                AND status = 'pending'
                ORDER BY wallet_address, created_at DESC
            """
            
            try:
                if hasattr(self.db, '_db') and self.db._db:
                    # PostgreSQL
                    async with self.db._db.pool.acquire() as conn:
                        rows = await conn.fetch(query.replace('?', '$1'), *wallet_addresses)
                else:
                    # SQLite
                    if not self.db._connection:
                        await self.db.connect()
                    cursor = await self.db._connection.cursor()
                    await cursor.execute(query, wallet_addresses)
                    rows = await cursor.fetchall()
                
                # Group by wallet address
                reservations_by_wallet = defaultdict(list)
                for row in rows:
                    reservations_by_wallet[row['wallet_address']].append(dict(row))
                
                return [reservations_by_wallet.get(addr, []) for addr in wallet_addresses]
                
            except Exception as e:
                logger.error(f"Failed to batch load credit reservations: {e}")
                return [[] for _ in wallet_addresses]
        
        self._loaders['credit_reservations'] = DataLoader(batch_load_credit_reservations)
    
    async def load_user(self, wallet_address: str) -> Optional[Dict[str, Any]]:
        """Load user by wallet address (batched)."""
        return await self._loaders['users'].load(wallet_address)
    
    async def load_users(self, wallet_addresses: List[str]) -> List[Optional[Dict[str, Any]]]:
        """Load multiple users by wallet addresses (batched)."""
        return await self._loaders['users'].load_many(wallet_addresses)
    
    async def load_user_credits(self, wallet_address: str) -> Optional[float]:
        """Load user credits by wallet address (batched)."""
        return await self._loaders['user_credits'].load(wallet_address)
    
    async def load_api_keys_for_wallet(self, wallet_address: str) -> List[Dict[str, Any]]:
        """Load API keys for wallet address (batched)."""
        result = await self._loaders['api_keys_by_wallet'].load(wallet_address)
        return result or []
    
    async def load_credit_reservations_for_wallet(self, wallet_address: str) -> List[Dict[str, Any]]:
        """Load credit reservations for wallet address (batched)."""
        result = await self._loaders['credit_reservations'].load(wallet_address)
        return result or []
    
    async def prime_user(self, wallet_address: str, user_data: Dict[str, Any]):
        """Prime user cache with known data."""
        await self._loaders['users'].prime(wallet_address, user_data)
        if 'credits' in user_data:
            await self._loaders['user_credits'].prime(wallet_address, user_data['credits'])
    
    async def invalidate_user(self, wallet_address: str):
        """Invalidate user-related caches."""
        await self._loaders['users'].clear(wallet_address)
        await self._loaders['user_credits'].clear(wallet_address)
        await self._loaders['api_keys_by_wallet'].clear(wallet_address)
        await self._loaders['credit_reservations'].clear(wallet_address)
    
    def get_loader_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all loaders."""
        return {
            name: loader.get_stats() 
            for name, loader in self._loaders.items()
        }


class BatchOperationManager:
    """
    Manager for batch database operations.
    
    Provides utilities for batching INSERT, UPDATE, and DELETE operations
    to improve performance.
    """
    
    def __init__(self, database_instance):
        """
        Initialize batch operation manager.
        
        Args:
            database_instance: Database instance
        """
        self.db = database_instance
        self.batch_size = 100
    
    async def batch_insert_users(self, users: List[Dict[str, Any]]) -> int:
        """
        Batch insert users.
        
        Args:
            users: List of user data dictionaries
            
        Returns:
            Number of users inserted
        """
        if not users:
            return 0
        
        # Process in batches
        total_inserted = 0
        for i in range(0, len(users), self.batch_size):
            batch = users[i:i + self.batch_size]
            inserted = await self._insert_user_batch(batch)
            total_inserted += inserted
        
        return total_inserted
    
    async def _insert_user_batch(self, users: List[Dict[str, Any]]) -> int:
        """Insert a batch of users."""
        if not users:
            return 0
        
        try:
            if hasattr(self.db, '_db') and self.db._db:
                # PostgreSQL - use COPY or batch insert
                async with self.db._db.pool.acquire() as conn:
                    query = """
                        INSERT INTO users (wallet_address, credits, is_miner)
                        VALUES ($1, $2, $3)
                        ON CONFLICT (wallet_address) DO NOTHING
                    """
                    
                    # Prepare data
                    data = [
                        (user['wallet_address'], user.get('credits', 0.0), user.get('is_miner', False))
                        for user in users
                    ]
                    
                    # Execute batch
                    result = await conn.executemany(query, data)
                    return len(data)
            else:
                # SQLite - use executemany
                if not self.db._connection:
                    await self.db.connect()
                
                cursor = await self.db._connection.cursor()
                query = """
                    INSERT OR IGNORE INTO users (wallet_address, credits, is_miner)
                    VALUES (?, ?, ?)
                """
                
                data = [
                    (user['wallet_address'], user.get('credits', 0.0), user.get('is_miner', False))
                    for user in users
                ]
                
                await cursor.executemany(query, data)
                await self.db._connection.commit()
                return cursor.rowcount
                
        except Exception as e:
            logger.error(f"Failed to batch insert users: {e}")
            return 0
    
    async def batch_update_credits(self, credit_updates: List[Dict[str, Any]]) -> int:
        """
        Batch update user credits.
        
        Args:
            credit_updates: List of {'wallet_address': str, 'credits': float}
            
        Returns:
            Number of users updated
        """
        if not credit_updates:
            return 0
        
        try:
            if hasattr(self.db, '_db') and self.db._db:
                # PostgreSQL
                async with self.db._db.pool.acquire() as conn:
                    query = """
                        UPDATE users SET credits = $2, updated_at = NOW()
                        WHERE wallet_address = $1
                    """
                    
                    data = [
                        (update['wallet_address'], update['credits'])
                        for update in credit_updates
                    ]
                    
                    await conn.executemany(query, data)
                    return len(data)
            else:
                # SQLite
                if not self.db._connection:
                    await self.db.connect()
                
                cursor = await self.db._connection.cursor()
                query = """
                    UPDATE users SET credits = ?, updated_at = datetime('now')
                    WHERE wallet_address = ?
                """
                
                data = [
                    (update['credits'], update['wallet_address'])
                    for update in credit_updates
                ]
                
                await cursor.executemany(query, data)
                await self.db._connection.commit()
                return cursor.rowcount
                
        except Exception as e:
            logger.error(f"Failed to batch update credits: {e}")
            return 0
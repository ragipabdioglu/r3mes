"""
Async Database - Unified database interface supporting both SQLite and PostgreSQL

Automatically selects database backend based on DATABASE_TYPE environment variable.
Supports both SQLite (development) and PostgreSQL (production).
"""

import aiosqlite
import json
import asyncio
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Union, Any
from pathlib import Path
import os
import logging

from .database_config import DatabaseConfig
from .database_postgres import AsyncPostgreSQL
from .exceptions import InvalidInputError

logger = logging.getLogger(__name__)


class AsyncDatabase:
    """
    Unified async database wrapper supporting both SQLite and PostgreSQL.
    
    Automatically selects backend based on DATABASE_TYPE environment variable:
    - "sqlite" (default): Uses aiosqlite
    - "postgresql": Uses asyncpg with connection pooling
    
    All database operations are async to prevent blocking the event loop.
    """
    
    def __init__(self, db_path: Optional[str] = None, chain_json_path: Optional[str] = None):
        """
        Initialize async database.
        
        Args:
            db_path: SQLite database file path (ignored for PostgreSQL)
            chain_json_path: Blockchain JSON file path
        """
        self.config = DatabaseConfig()
        
        # Normalize chain_json_path
        chain_json_path = chain_json_path or os.getenv("CHAIN_JSON_PATH", "chain.json")
        if not Path(chain_json_path).is_absolute():
            self.chain_json_path = str(Path.cwd() / chain_json_path)
        else:
            self.chain_json_path = str(Path(chain_json_path).resolve())
        
        # RPC endpoint for blockchain queries
        # In production, BLOCKCHAIN_RPC_URL must be set (no localhost fallback)
        is_production = os.getenv("R3MES_ENV", "development").lower() == "production"
        rpc_url = os.getenv("BLOCKCHAIN_RPC_URL")
        if not rpc_url:
            if is_production:
                raise ValueError(
                    "BLOCKCHAIN_RPC_URL environment variable must be set in production. "
                    "Do not use localhost in production."
                )
            # Development fallback
            self.rpc_endpoint = "http://localhost:26657"
            logger.warning("BLOCKCHAIN_RPC_URL not set, using localhost fallback (development only)")
        else:
            self.rpc_endpoint = rpc_url
            # Validate that production doesn't use localhost
            if is_production and ("localhost" in self.rpc_endpoint or "127.0.0.1" in self.rpc_endpoint):
                raise ValueError(
                    f"BLOCKCHAIN_RPC_URL cannot use localhost in production: {self.rpc_endpoint}"
                )
        
        # Initialize appropriate database backend
        if self.config.is_postgresql():
            self._db: Optional[AsyncPostgreSQL] = AsyncPostgreSQL(
                self.config.get_connection_string(),
                min_size=self.config.pool_min_size,
                max_size=self.config.pool_max_size
            )
            self._connection = None  # Not used for PostgreSQL
            logger.info("Using PostgreSQL database backend")
        else:
            # SQLite fallback - normalize path
            if db_path is None:
                db_path = self.config.get_connection_string()
            
            # Normalize SQLite database path (resolve relative paths)
            if not Path(db_path).is_absolute():
                db_path = str(Path.cwd() / db_path)
            else:
                db_path = str(Path(db_path).resolve())
            
            self.db_path = db_path
            self._connection: Optional[aiosqlite.Connection] = None
            self._db = None
            # Create database directory if needed
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using SQLite database backend: {self.db_path}")
    
    async def connect(self):
        """Create database connection."""
        try:
            if self.config.is_postgresql():
                await self._db.connect()
            else:
                if self._connection is None:
                    self._connection = await aiosqlite.connect(self.db_path)
                    # Enable WAL (Write-Ahead Logging) mode for better concurrency
                    await self._connection.execute("PRAGMA journal_mode=WAL")
                    await self._connection.execute("PRAGMA synchronous=NORMAL")  # Better performance with WAL
                    await self._connection.execute("PRAGMA cache_size=-64000")  # 64MB cache
                    await self._connection.execute("PRAGMA foreign_keys=ON")
                    await self._connection.commit()
                    await self._init_database()
        except Exception as e:
            logger.error(f"Database connection failed: {e}", exc_info=True)
            # Send notification for database connection failure
            try:
                from .notifications import get_notification_service, NotificationPriority
                notification_service = get_notification_service()
                await notification_service.send_system_alert(
                    component="database",
                    alert_type="connection_failure",
                    message=f"Failed to connect to database: {e}",
                    priority=NotificationPriority.CRITICAL
                )
            except Exception as notif_error:
                logger.warning(f"Failed to send database connection failure notification: {notif_error}")
            raise
    
    async def close(self):
        """Close database connection."""
        if self.config.is_postgresql():
            await self._db.close()
        else:
            if self._connection:
                await self._connection.close()
                self._connection = None
    
    async def _init_database(self):
        """Initialize database tables (SQLite only)."""
        if self.config.is_postgresql():
            # PostgreSQL initialization is handled by AsyncPostgreSQL
            return
        
        if not self._connection:
            await self.connect()
        
        cursor = await self._connection.cursor()
        
        # Users table
        await cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                wallet_address TEXT PRIMARY KEY,
                credits REAL DEFAULT 0.0,
                is_miner BOOLEAN DEFAULT 0,
                last_mining_time TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Mining stats table
        await cursor.execute("""
            CREATE TABLE IF NOT EXISTS mining_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wallet_address TEXT NOT NULL,
                hashrate REAL DEFAULT 0.0,
                gpu_temperature REAL DEFAULT 0.0,
                blocks_found INTEGER DEFAULT 0,
                uptime_percentage REAL DEFAULT 0.0,
                network_difficulty REAL DEFAULT 0.0,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (wallet_address) REFERENCES users(wallet_address)
            )
        """)
        
        # Earnings history table
        await cursor.execute("""
            CREATE TABLE IF NOT EXISTS earnings_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wallet_address TEXT NOT NULL,
                earnings REAL DEFAULT 0.0,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (wallet_address) REFERENCES users(wallet_address)
            )
        """)
        
        # Hashrate history table
        await cursor.execute("""
            CREATE TABLE IF NOT EXISTS hashrate_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wallet_address TEXT NOT NULL,
                hashrate REAL DEFAULT 0.0,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (wallet_address) REFERENCES users(wallet_address)
            )
        """)
        
        # API Keys table (with hashed keys)
        await cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                api_key_hash TEXT UNIQUE NOT NULL,
                wallet_address TEXT NOT NULL,
                name TEXT,
                is_active BOOLEAN DEFAULT 1,
                last_used TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                FOREIGN KEY (wallet_address) REFERENCES users(wallet_address)
            )
        """)
        
        # Create index on api_key_hash for faster lookups
        await cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_api_key_hash ON api_keys(api_key_hash)
        """)
        
        # LoRA Registry table (created by migration, but create here as fallback)
        await cursor.execute("""
            CREATE TABLE IF NOT EXISTS lora_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                ipfs_hash TEXT NOT NULL,
                description TEXT,
                category TEXT,
                version TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        """)
        
        # Serving Nodes table (created by migration, but create here as fallback)
        await cursor.execute("""
            CREATE TABLE IF NOT EXISTS serving_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wallet_address TEXT UNIQUE NOT NULL,
                endpoint_url TEXT NOT NULL,
                available_lora_list TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                last_heartbeat TIMESTAMP NOT NULL,
                current_load INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Credit Reservations table (for atomic credit operations)
        await cursor.execute("""
            CREATE TABLE IF NOT EXISTS credit_reservations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reservation_id TEXT UNIQUE NOT NULL,
                wallet_address TEXT NOT NULL,
                amount REAL NOT NULL,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                confirmed_at TIMESTAMP,
                FOREIGN KEY (wallet_address) REFERENCES users(wallet_address)
            )
        """)
        
        # Create indexes for LoRA and serving nodes
        await cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_lora_registry_name ON lora_registry(name)
        """)
        await cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_lora_registry_category ON lora_registry(category)
        """)
        await cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_lora_registry_is_active ON lora_registry(is_active)
        """)
        await cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_serving_nodes_wallet ON serving_nodes(wallet_address)
        """)
        await cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_serving_nodes_status ON serving_nodes(status)
        """)
        await cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_serving_nodes_last_heartbeat ON serving_nodes(last_heartbeat)
        """)
        
        # Create indexes for credit reservations
        await cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_credit_reservations_wallet ON credit_reservations(wallet_address)
        """)
        await cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_credit_reservations_status ON credit_reservations(status)
        """)
        await cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_credit_reservations_expires ON credit_reservations(expires_at)
        """)
        
        # Apply additional recommended indexes for SQLite
        from .database_optimization import IndexAuditor
        index_queries = IndexAuditor.get_sqlite_index_queries()
        for query in index_queries:
            try:
                await cursor.execute(query)
            except Exception as e:
                logger.warning(f"Failed to create index: {e}")
        
        await self._connection.commit()
        logger.info("Async database initialized")
    
    def _hash_api_key(self, api_key: str) -> str:
        """Hash API key using SHA256."""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    async def add_credits(self, wallet: str, amount: float) -> bool:
        """
        Add credits to user (async).
        
        Invalidates user cache after credit update.
        """
        if self.config.is_postgresql():
            return await self._db.add_credits(wallet, amount)
        
        if not self._connection:
            await self.connect()
        
        cursor = await self._connection.cursor()
        
        # Check if user exists
        await cursor.execute("SELECT credits FROM users WHERE wallet_address = ?", (wallet,))
        result = await cursor.fetchone()
        
        if result:
            # Update existing user
            new_credits = result[0] + amount
            await cursor.execute(
                "UPDATE users SET credits = ? WHERE wallet_address = ?",
                (new_credits, wallet)
            )
        else:
            # Create new user
            await cursor.execute(
                "INSERT INTO users (wallet_address, credits) VALUES (?, ?)",
                (wallet, amount)
            )
        
        await self._connection.commit()
        
        # Invalidate user cache
        try:
            from .cache_invalidation import get_cache_invalidator
            invalidator = get_cache_invalidator()
            await invalidator.invalidate_user_cache(wallet)
        except Exception as e:
            logger.warning(f"Failed to invalidate cache for user {wallet}: {e}")
        
        return True
    
    async def deduct_credit(self, wallet: str, amount: float) -> bool:
        """Deduct credits from user (async)."""
        if self.config.is_postgresql():
            return await self._db.deduct_credit(wallet, amount)
        
        if not self._connection:
            await self.connect()
        
        cursor = await self._connection.cursor()
        
        await cursor.execute("SELECT credits FROM users WHERE wallet_address = ?", (wallet,))
        result = await cursor.fetchone()
        
        if not result:
            return False
        
        current_credits = result[0]
        if current_credits < amount:
            return False
        
        new_credits = current_credits - amount
        await cursor.execute(
            "UPDATE users SET credits = ? WHERE wallet_address = ?",
            (new_credits, wallet)
        )
        
        await self._connection.commit()
        
        # Invalidate user cache
        try:
            from .cache_invalidation import get_cache_invalidator
            invalidator = get_cache_invalidator()
            await invalidator.invalidate_user_cache(wallet)
        except Exception as e:
            logger.warning(f"Failed to invalidate cache for user {wallet}: {e}")
        
        return True
    
    async def reserve_credit_atomic(self, wallet: str, amount: float) -> Dict:
        """
        Atomically reserve credits for a transaction.
        
        This method performs an atomic check-and-reserve operation to prevent
        race conditions where multiple concurrent requests could overdraw credits.
        
        The reservation is stored in a separate table and must be either:
        - Confirmed via confirm_credit_reservation() on success
        - Rolled back via rollback_credit_reservation() on failure
        
        Args:
            wallet: Wallet address
            amount: Amount to reserve
            
        Returns:
            Dict with:
                - success: bool
                - reservation_id: str (if successful)
                - error: str (if failed)
        """
        import uuid
        
        if self.config.is_postgresql():
            return await self._db.reserve_credit_atomic(wallet, amount)
        
        if not self._connection:
            await self.connect()
        
        reservation_id = str(uuid.uuid4())
        
        # Use a transaction with IMMEDIATE to get write lock
        try:
            cursor = await self._connection.cursor()
            
            # Begin immediate transaction for write lock
            await cursor.execute("BEGIN IMMEDIATE")
            
            # Check current credits
            await cursor.execute(
                "SELECT credits FROM users WHERE wallet_address = ?",
                (wallet,)
            )
            result = await cursor.fetchone()
            
            if not result:
                await cursor.execute("ROLLBACK")
                return {
                    "success": False,
                    "error": "User not found",
                    "reservation_id": None
                }
            
            current_credits = result[0]
            
            # Check if sufficient credits (including any pending reservations)
            await cursor.execute("""
                SELECT COALESCE(SUM(amount), 0) FROM credit_reservations 
                WHERE wallet_address = ? AND status = 'pending'
            """, (wallet,))
            pending_result = await cursor.fetchone()
            pending_amount = pending_result[0] if pending_result else 0.0
            
            available_credits = current_credits - pending_amount
            
            if available_credits < amount:
                await cursor.execute("ROLLBACK")
                return {
                    "success": False,
                    "error": f"Insufficient credits. Available: {available_credits}, Required: {amount}",
                    "reservation_id": None
                }
            
            # Create reservation
            await cursor.execute("""
                INSERT INTO credit_reservations 
                (reservation_id, wallet_address, amount, status, created_at, expires_at)
                VALUES (?, ?, ?, 'pending', datetime('now'), datetime('now', '+5 minutes'))
            """, (reservation_id, wallet, amount))
            
            await cursor.execute("COMMIT")
            
            logger.debug(f"Credit reservation created: {reservation_id} for {wallet}, amount: {amount}")
            
            return {
                "success": True,
                "reservation_id": reservation_id,
                "error": None
            }
            
        except Exception as e:
            try:
                await cursor.execute("ROLLBACK")
            except Exception as rollback_error:
                logger.warning(f"Failed to rollback transaction during credit reservation: {rollback_error}")
            logger.error(f"Failed to reserve credit for {wallet}: {e}")
            return {
                "success": False,
                "error": str(e),
                "reservation_id": None
            }
    
    async def confirm_credit_reservation(self, reservation_id: str) -> bool:
        """
        Confirm a credit reservation and deduct the credits.
        
        This should be called after the operation (e.g., inference) completes successfully.
        
        Args:
            reservation_id: The reservation ID from reserve_credit_atomic()
            
        Returns:
            True if confirmed successfully
        """
        if self.config.is_postgresql():
            return await self._db.confirm_credit_reservation(reservation_id)
        
        if not self._connection:
            await self.connect()
        
        try:
            cursor = await self._connection.cursor()
            
            await cursor.execute("BEGIN IMMEDIATE")
            
            # Get reservation details
            await cursor.execute("""
                SELECT wallet_address, amount, status FROM credit_reservations
                WHERE reservation_id = ?
            """, (reservation_id,))
            result = await cursor.fetchone()
            
            if not result:
                await cursor.execute("ROLLBACK")
                logger.warning(f"Reservation not found: {reservation_id}")
                return False
            
            wallet, amount, status = result
            
            if status != 'pending':
                await cursor.execute("ROLLBACK")
                logger.warning(f"Reservation {reservation_id} is not pending (status: {status})")
                return False
            
            # Deduct credits
            await cursor.execute("""
                UPDATE users SET credits = credits - ?
                WHERE wallet_address = ? AND credits >= ?
            """, (amount, wallet, amount))
            
            if cursor.rowcount == 0:
                # Insufficient credits (shouldn't happen if reservation was valid)
                await cursor.execute("""
                    UPDATE credit_reservations SET status = 'failed'
                    WHERE reservation_id = ?
                """, (reservation_id,))
                await cursor.execute("COMMIT")
                logger.error(f"Failed to deduct credits for reservation {reservation_id}")
                return False
            
            # Mark reservation as confirmed
            await cursor.execute("""
                UPDATE credit_reservations SET status = 'confirmed', confirmed_at = datetime('now')
                WHERE reservation_id = ?
            """, (reservation_id,))
            
            await cursor.execute("COMMIT")
            
            # Invalidate user cache
            try:
                from .cache_invalidation import get_cache_invalidator
                invalidator = get_cache_invalidator()
                await invalidator.invalidate_user_cache(wallet)
            except Exception as e:
                logger.warning(f"Failed to invalidate cache for user {wallet}: {e}")
            
            logger.debug(f"Credit reservation confirmed: {reservation_id}")
            return True
            
        except Exception as e:
            try:
                await cursor.execute("ROLLBACK")
            except Exception as rollback_error:
                logger.warning(f"Failed to rollback transaction during reservation confirmation: {rollback_error}")
            logger.error(f"Failed to confirm reservation {reservation_id}: {e}")
            return False
    
    async def rollback_credit_reservation(self, reservation_id: str) -> bool:
        """
        Rollback a credit reservation (release the reserved credits).
        
        This should be called if the operation fails before completion.
        
        Args:
            reservation_id: The reservation ID from reserve_credit_atomic()
            
        Returns:
            True if rolled back successfully
        """
        if self.config.is_postgresql():
            return await self._db.rollback_credit_reservation(reservation_id)
        
        if not self._connection:
            await self.connect()
        
        try:
            cursor = await self._connection.cursor()
            
            await cursor.execute("""
                UPDATE credit_reservations SET status = 'rolled_back', confirmed_at = datetime('now')
                WHERE reservation_id = ? AND status = 'pending'
            """, (reservation_id,))
            
            await self._connection.commit()
            
            logger.debug(f"Credit reservation rolled back: {reservation_id}")
            return cursor.rowcount > 0
            
        except Exception as e:
            logger.error(f"Failed to rollback reservation {reservation_id}: {e}")
            return False
    
    async def cleanup_expired_reservations(self) -> int:
        """
        Cleanup expired credit reservations.
        
        Reservations that are still pending after their expiry time are
        automatically rolled back.
        
        Returns:
            Number of reservations cleaned up
        """
        if self.config.is_postgresql():
            return await self._db.cleanup_expired_reservations()
        
        if not self._connection:
            await self.connect()
        
        try:
            cursor = await self._connection.cursor()
            
            await cursor.execute("""
                UPDATE credit_reservations 
                SET status = 'expired'
                WHERE status = 'pending' AND expires_at < datetime('now')
            """)
            
            count = cursor.rowcount
            await self._connection.commit()
            
            if count > 0:
                logger.info(f"Cleaned up {count} expired credit reservations")
            
            return count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired reservations: {e}")
            return 0
    
    async def check_credits(self, wallet: str) -> float:
        """Check user credits (async)."""
        if self.config.is_postgresql():
            return await self._db.check_credits(wallet)
        
        if not self._connection:
            await self.connect()
        
        cursor = await self._connection.cursor()
        
        await cursor.execute("SELECT credits FROM users WHERE wallet_address = ?", (wallet,))
        result = await cursor.fetchone()
        
        return result[0] if result else 0.0
    
    async def get_user_info(self, wallet: str) -> Optional[Dict]:
        """Get user information (async)."""
        if self.config.is_postgresql():
            return await self._db.get_user_info(wallet)
        
        if not self._connection:
            await self.connect()
        
        cursor = await self._connection.cursor()
        
        await cursor.execute(
            "SELECT wallet_address, credits, is_miner FROM users WHERE wallet_address = ?",
            (wallet,)
        )
        result = await cursor.fetchone()
        
        if not result:
            return None
        
        return {
            "wallet_address": result[0],
            "credits": result[1],
            "is_miner": bool(result[2]),
        }
    
    async def validate_api_key(self, api_key: str) -> Optional[Dict]:
        """Validate API key (async, using hash)."""
        if self.config.is_postgresql():
            return await self._db.validate_api_key(api_key)
        
        if not self._connection:
            await self.connect()
        
        # Hash the provided API key
        api_key_hash = self._hash_api_key(api_key)
        
        cursor = await self._connection.cursor()
        
        await cursor.execute("""
            SELECT id, wallet_address, name, is_active, expires_at
            FROM api_keys
            WHERE api_key_hash = ? AND is_active = 1
        """, (api_key_hash,))
        
        result = await cursor.fetchone()
        
        if not result:
            return None
        
        # Check expiration
        expires_at = result[4]
        if expires_at:
            expires_dt = datetime.fromisoformat(expires_at)
            if datetime.now() > expires_dt:
                return None
        
        # Update last_used
        await cursor.execute(
            "UPDATE api_keys SET last_used = ? WHERE id = ?",
            (datetime.now().isoformat(), result[0])
        )
        await self._connection.commit()
        
        return {
            "id": result[0],
            "wallet_address": result[1],
            "name": result[2],
            "is_active": bool(result[3]),
        }
    
    async def create_api_key(
        self,
        wallet: str,
        name: str = "Default",
        expires_in_days: Optional[int] = None
    ) -> str:
        """Create new API key (returns plaintext key, stores hash)."""
        if self.config.is_postgresql():
            return await self._db.create_api_key(wallet, name, expires_in_days)
        
        if not self._connection:
            await self.connect()
        
        # Generate API key
        api_key = f"r3mes_{secrets.token_urlsafe(32)}"
        api_key_hash = self._hash_api_key(api_key)
        
        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = (datetime.now() + timedelta(days=expires_in_days)).isoformat()
        
        cursor = await self._connection.cursor()
        
        await cursor.execute("""
            INSERT INTO api_keys (api_key_hash, wallet_address, name, expires_at)
            VALUES (?, ?, ?, ?)
        """, (api_key_hash, wallet, name, expires_at))
        
        await self._connection.commit()
        
        # Return plaintext key (only shown once)
        return api_key
    
    async def get_network_stats(self) -> Dict[str, Any]:
        """
        Get network statistics (async).
        
        Returns:
            Dictionary containing:
            - active_miners: Number of active miners
            - total_users: Total number of users
            - total_credits: Sum of all user credits
            - block_height: Current blockchain height
        """
        from .blockchain_rpc_client import get_blockchain_rpc_client
        
        if self.config.is_postgresql():
            stats = await self._db.get_network_stats()
        else:
            if not self._connection:
                await self.connect()
            
            cursor = await self._connection.cursor()
            
            # Count active miners
            await cursor.execute("SELECT COUNT(*) FROM users WHERE is_miner = 1")
            active_miners = (await cursor.fetchone())[0]
            
            # Count total users
            await cursor.execute("SELECT COUNT(*) FROM users")
            total_users = (await cursor.fetchone())[0]
            
            # Sum total credits
            await cursor.execute("SELECT SUM(credits) FROM users")
            total_credits_result = await cursor.fetchone()
            total_credits = total_credits_result[0] if total_credits_result[0] else 0.0
            
            stats = {
                "active_miners": active_miners,
                "total_users": total_users,
                "total_credits": total_credits,
            }
        
        # Get block height from blockchain RPC
        rpc_client = get_blockchain_rpc_client()
        block_height = rpc_client.get_latest_block_height()
        stats["block_height"] = block_height
        
        return stats
    
    async def get_recent_blocks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent blocks (async).
        
        Args:
            limit: Maximum number of blocks to return
            
        Returns:
            List of block dictionaries containing height, hash, timestamp, etc.
        """
        from .blockchain_rpc_client import get_blockchain_rpc_client
        
        # Query blockchain RPC for recent blocks
        rpc_client = get_blockchain_rpc_client()
        blocks = rpc_client.get_recent_blocks(limit=limit)
        
        return blocks
    
    async def get_miner_stats(self, wallet: str) -> Dict[str, Any]:
        """
        Get miner statistics (async).
        
        Args:
            wallet: Miner wallet address
            
        Returns:
            Dictionary containing:
            - wallet_address: Miner's wallet address
            - total_earnings: Total earnings in credits
            - hashrate: Current hash rate
            - gpu_temperature: GPU temperature in Celsius
            - blocks_found: Number of blocks found
            - uptime_percentage: Uptime percentage (0-100)
            - network_difficulty: Current network difficulty
        """
        if self.config.is_postgresql():
            return await self._db.get_miner_stats(wallet)
        
        if not self._connection:
            await self.connect()
        
        user_info = await self.get_user_info(wallet)
        if not user_info:
            # Try to get network difficulty from blockchain or use default
            network_difficulty = await self._get_network_difficulty()
            return {
                "wallet_address": wallet,
                "total_earnings": 0.0,
                "hashrate": 0.0,
                "gpu_temperature": 0.0,
                "blocks_found": 0,
                "uptime_percentage": 0.0,
                "network_difficulty": network_difficulty
            }
        
        cursor = await self._connection.cursor()
        
        # Get latest mining stats
        await cursor.execute("""
            SELECT hashrate, gpu_temperature, blocks_found, uptime_percentage, network_difficulty
            FROM mining_stats
            WHERE wallet_address = ?
            ORDER BY recorded_at DESC
            LIMIT 1
        """, (wallet,))
        stats = await cursor.fetchone()
        
        # Get network difficulty: prefer from stats, then blockchain, then default
        network_difficulty = 1234.0  # Default fallback
        if stats and stats[4]:
            network_difficulty = float(stats[4])
        else:
            # Try to fetch from blockchain
            network_difficulty = await self._get_network_difficulty()
        
        return {
            "wallet_address": wallet,
            "total_earnings": user_info['credits'],
            "hashrate": stats[0] if stats and stats[0] else 0.0,
            "gpu_temperature": stats[1] if stats and stats[1] else 0.0,
            "blocks_found": stats[2] if stats and stats[2] else 0,
            "uptime_percentage": stats[3] if stats and stats[3] else 0.0,
            "network_difficulty": network_difficulty
        }
    
    async def _get_network_difficulty(self) -> float:
        """
        Get network difficulty from blockchain or return default (async).
        
        Returns:
            Network difficulty value (default: 1234.0)
        """
        try:
            from .blockchain_query_client import BlockchainQueryClient
            client = BlockchainQueryClient()
            # Query params from blockchain
            params_data = client._query_rest("/remes/remes/v1/params")
            if "params" in params_data:
                params = params_data["params"]
                # Extract mining difficulty from params (if available)
                difficulty_str = params.get("mining_difficulty") or params.get("network_difficulty")
                if difficulty_str:
                    try:
                        difficulty = float(difficulty_str)
                        logger.debug(f"Fetched network difficulty from blockchain: {difficulty}")
                        return difficulty
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid difficulty value from blockchain: {difficulty_str}")
        except Exception as e:
            logger.debug(f"Could not fetch network difficulty from blockchain: {e}, using default")
        
        # Default fallback
        return 1234.0
    
    async def get_earnings_history(self, wallet: str, days: int = 7) -> List[Dict]:
        """Get earnings history (async)."""
        if self.config.is_postgresql():
            return await self._db.get_earnings_history(wallet, days)
        
        if not self._connection:
            await self.connect()
        
        cursor = await self._connection.cursor()
        
        await cursor.execute("""
            SELECT DATE(recorded_at) as date, SUM(earnings) as total_earnings
            FROM earnings_history
            WHERE wallet_address = ? AND recorded_at >= datetime('now', '-' || ? || ' days')
            GROUP BY DATE(recorded_at)
            ORDER BY date ASC
        """, (wallet, days))
        
        results = await cursor.fetchall()
        
        # Format results
        earnings_data = []
        for row in results:
            earnings_data.append({
                "date": row[0],
                "earnings": row[1] or 0.0
            })
        
        # Fill missing days with 0
        today = datetime.now().date()
        all_dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days-1, -1, -1)]
        
        earnings_dict = {item['date']: item['earnings'] for item in earnings_data}
        formatted_data = []
        for date_str in all_dates:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            formatted_data.append({
                "date": date_obj.strftime('%b %d'),
                "earnings": earnings_dict.get(date_str, 0.0)
            })
        
        return formatted_data
    
    async def get_hashrate_history(self, wallet: str, days: int = 7) -> List[Dict]:
        """Get hashrate history (async)."""
        if self.config.is_postgresql():
            return await self._db.get_hashrate_history(wallet, days)
        
        if not self._connection:
            await self.connect()
        
        cursor = await self._connection.cursor()
        
        await cursor.execute("""
            SELECT DATE(recorded_at) as date, AVG(hashrate) as avg_hashrate
            FROM hashrate_history
            WHERE wallet_address = ? AND recorded_at >= datetime('now', '-' || ? || ' days')
            GROUP BY DATE(recorded_at)
            ORDER BY date ASC
        """, (wallet, days))
        
        results = await cursor.fetchall()
        
        # Format results
        hashrate_data = []
        for row in results:
            hashrate_data.append({
                "date": row[0],
                "hashrate": row[1] or 0.0
            })
        
        # Fill missing days with 0
        today = datetime.now().date()
        all_dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days-1, -1, -1)]
        
        hashrate_dict = {item['date']: item['hashrate'] for item in hashrate_data}
        formatted_data = []
        for date_str in all_dates:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            formatted_data.append({
                "date": date_obj.strftime('%b %d'),
                "hashrate": hashrate_dict.get(date_str, 0.0)
            })
        
        return formatted_data
    
    async def list_api_keys(self, wallet_address: str) -> List[Dict]:
        """List API keys for a wallet (async)."""
        if self.config.is_postgresql():
            return await self._db.list_api_keys(wallet_address)
        
        if not self._connection:
            await self.connect()
        
        cursor = await self._connection.cursor()
        
        await cursor.execute("""
            SELECT id, name, is_active, created_at, expires_at, last_used
            FROM api_keys
            WHERE wallet_address = ?
            ORDER BY created_at DESC
        """, (wallet_address,))
        
        results = await cursor.fetchall()
        
        return [
            {
                "id": row[0],
                "name": row[1],
                "is_active": bool(row[2]),
                "created_at": row[3],
                "expires_at": row[4],
                "last_used": row[5],
            }
            for row in results
        ]
    
    async def revoke_api_key(self, api_key_id: int, wallet_address: str) -> bool:
        """Revoke API key (async)."""
        if self.config.is_postgresql():
            return await self._db.revoke_api_key(api_key_id, wallet_address)
        
        if not self._connection:
            await self.connect()
        
        cursor = await self._connection.cursor()
        
        await cursor.execute("""
            UPDATE api_keys
            SET is_active = 0
            WHERE id = ? AND wallet_address = ?
        """, (api_key_id, wallet_address))
        
        await self._connection.commit()
        return cursor.rowcount > 0
    
    async def delete_api_key(self, api_key_id: int, wallet_address: str) -> bool:
        """Delete API key (async)."""
        if self.config.is_postgresql():
            return await self._db.delete_api_key(api_key_id, wallet_address)
        
        if not self._connection:
            await self.connect()
        
        cursor = await self._connection.cursor()
        
        await cursor.execute("""
            DELETE FROM api_keys
            WHERE id = ? AND wallet_address = ?
        """, (api_key_id, wallet_address))
        
        await self._connection.commit()
        return cursor.rowcount > 0
    
    async def sync_with_blockchain(self) -> int:
        """
        Async blockchain sync: Read chain.json and process new blocks.
        Awards credits to miners for blocks they found (1 block = 100 credits).
        
        This is an async version of the sync operation, replacing the blocking
        sync_loop that used time.sleep(60) polling. This method should be called
        periodically via async task or triggered by blockchain events.
        
        Returns:
            Number of blocks processed
            
        Raises:
            InvalidInputError: If chain JSON file is invalid
            RuntimeError: If sync fails critically
        """
        chain_json_path = getattr(self, 'chain_json_path', None)
        if not chain_json_path:
            chain_json_path = os.getenv("CHAIN_JSON_PATH", "chain.json")
            if not Path(chain_json_path).is_absolute():
                chain_json_path = str(Path.cwd() / chain_json_path)
            else:
                chain_json_path = str(Path(chain_json_path).resolve())
        
        if not Path(chain_json_path).exists():
            logger.warning(f"Chain JSON file not found: {chain_json_path}")
            return 0
        
        try:
            # Read file asynchronously using asyncio.to_thread to avoid blocking
            loop = asyncio.get_event_loop()
            chain_data = await loop.run_in_executor(
                None,
                lambda: json.load(open(chain_json_path, 'r'))
            )
            
            # Process new blocks
            blocks = chain_data.get('blocks', [])
            
            if not isinstance(blocks, list):
                raise InvalidInputError(f"Invalid blocks format: expected list, got {type(blocks)}")
            
            processed_count = 0
            for block in blocks:
                if not isinstance(block, dict):
                    logger.warning(f"Invalid block format: {type(block)}, skipping")
                    continue
                
                # Extract miner address from block
                miner_address = block.get('miner', '')
                if miner_address:
                    try:
                        # Award credits for mining (1 block = 100 credits)
                        success = await self.add_credits(miner_address, 100.0)
                        if success:
                            processed_count += 1
                            logger.info(f"Awarded 100 credits to {miner_address} for block {block.get('height', 'unknown')}")
                        else:
                            logger.warning(f"Failed to award credits to {miner_address} (non-critical, continuing)")
                    except Exception as e:
                        # Log but continue with other blocks (non-critical)
                        logger.warning(f"Error awarding credits to {miner_address}: {e}")
                        continue
            
            if processed_count > 0:
                logger.info(f"Blockchain sync completed: {processed_count} blocks processed")
            
            return processed_count
        
        except FileNotFoundError:
            logger.warning(f"Chain JSON file not found: {chain_json_path}")
            return 0
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in chain file: {e}", exc_info=True)
            raise InvalidInputError(f"Chain JSON file is corrupted: {e}") from e
        except ValueError as e:
            logger.error(f"Invalid data format in chain file: {e}", exc_info=True)
            raise InvalidInputError(f"Invalid data format in chain file: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during blockchain sync: {e}", exc_info=True)
            raise RuntimeError(f"Blockchain sync failed: {e}") from e


    # =========================================================================
    # INFERENCE REQUESTS, AGGREGATIONS, AND COMMITS QUERY METHODS
    # =========================================================================
    
    async def get_inference_requests_by_serving_node(
        self,
        serving_node: str,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None
    ) -> List[Dict]:
        """
        Get inference requests for a specific serving node.
        
        This method queries the local database index for inference requests.
        The database is populated by the blockchain indexer.
        
        Args:
            serving_node: Serving node address
            limit: Maximum number of requests to return
            offset: Pagination offset
            status: Filter by status (optional)
            
        Returns:
            List of inference request dictionaries
        """
        if self.config.is_postgresql():
            return await self._db.get_inference_requests_by_serving_node(
                serving_node, limit, offset, status
            )
        
        if not self._connection:
            await self.connect()
        
        cursor = await self._connection.cursor()
        
        # Check if table exists
        await cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='inference_requests'
        """)
        if not await cursor.fetchone():
            # Table doesn't exist, return empty list
            return []
        
        # Build query with optional status filter
        query = """
            SELECT request_id, requester, serving_node, model_version,
                   input_data_ipfs_hash, fee, status, request_time,
                   result_ipfs_hash, latency_ms, tx_hash, block_height
            FROM inference_requests
            WHERE serving_node = ?
        """
        params = [serving_node]
        
        if status:
            query += " AND LOWER(status) = LOWER(?)"
            params.append(status)
        
        query += " ORDER BY block_height DESC, request_time DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        await cursor.execute(query, params)
        results = await cursor.fetchall()
        
        return [
            {
                "request_id": row[0],
                "requester": row[1],
                "serving_node": row[2],
                "model_version": row[3],
                "input_data_ipfs_hash": row[4],
                "fee": row[5],
                "status": row[6],
                "request_time": row[7],
                "result_ipfs_hash": row[8],
                "latency_ms": row[9],
                "tx_hash": row[10],
                "block_height": row[11],
            }
            for row in results
        ]
    
    async def get_aggregations(
        self,
        limit: int = 50,
        offset: int = 0,
        proposer: Optional[str] = None,
        training_round_id: Optional[int] = None
    ) -> List[Dict]:
        """
        Get aggregation records from local database index.
        
        Args:
            limit: Maximum number of aggregations to return
            offset: Pagination offset
            proposer: Filter by proposer address (optional)
            training_round_id: Filter by training round ID (optional)
            
        Returns:
            List of aggregation dictionaries
        """
        if self.config.is_postgresql():
            return await self._db.get_aggregations(
                limit, offset, proposer, training_round_id
            )
        
        if not self._connection:
            await self.connect()
        
        cursor = await self._connection.cursor()
        
        # Check if table exists
        await cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='aggregations'
        """)
        if not await cursor.fetchone():
            return []
        
        # Build query with optional filters
        query = """
            SELECT aggregation_id, proposer, aggregated_gradient_ipfs_hash,
                   merkle_root, participant_count, training_round_id,
                   block_height, timestamp, tx_hash
            FROM aggregations
            WHERE 1=1
        """
        params = []
        
        if proposer:
            query += " AND proposer = ?"
            params.append(proposer)
        
        if training_round_id is not None:
            query += " AND training_round_id = ?"
            params.append(training_round_id)
        
        query += " ORDER BY block_height DESC, aggregation_id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        await cursor.execute(query, params)
        results = await cursor.fetchall()
        
        return [
            {
                "aggregation_id": row[0],
                "proposer": row[1],
                "aggregated_gradient_ipfs_hash": row[2],
                "merkle_root": row[3],
                "participant_count": row[4],
                "training_round_id": row[5],
                "block_height": row[6],
                "timestamp": row[7],
                "tx_hash": row[8],
            }
            for row in results
        ]
    
    async def get_aggregation_commits(
        self,
        limit: int = 50,
        offset: int = 0,
        proposer: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict]:
        """
        Get aggregation commitment records from local database index.
        
        Args:
            limit: Maximum number of commits to return
            offset: Pagination offset
            proposer: Filter by proposer address (optional)
            status: Filter by status (optional)
            
        Returns:
            List of commitment dictionaries
        """
        if self.config.is_postgresql():
            return await self._db.get_aggregation_commits(
                limit, offset, proposer, status
            )
        
        if not self._connection:
            await self.connect()
        
        cursor = await self._connection.cursor()
        
        # Check if table exists
        await cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='aggregation_commits'
        """)
        if not await cursor.fetchone():
            return []
        
        # Build query with optional filters
        query = """
            SELECT commitment_id, proposer, commitment_hash, training_round_id,
                   gradient_count, status, commit_height, commit_time,
                   reveal_deadline, tx_hash
            FROM aggregation_commits
            WHERE 1=1
        """
        params = []
        
        if proposer:
            query += " AND proposer = ?"
            params.append(proposer)
        
        if status:
            query += " AND LOWER(status) = LOWER(?)"
            params.append(status)
        
        query += " ORDER BY commit_height DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        await cursor.execute(query, params)
        results = await cursor.fetchall()
        
        return [
            {
                "commitment_id": row[0],
                "proposer": row[1],
                "commitment_hash": row[2],
                "training_round_id": row[3],
                "gradient_count": row[4],
                "status": row[5],
                "commit_height": row[6],
                "commit_time": row[7],
                "reveal_deadline": row[8],
                "tx_hash": row[9],
            }
            for row in results
        ]
    
    async def _init_indexer_tables(self):
        """
        Initialize database tables for blockchain indexer.
        
        Creates tables for storing indexed blockchain data:
        - inference_requests
        - aggregations
        - aggregation_commits
        """
        if self.config.is_postgresql():
            # PostgreSQL tables are created by migrations
            return
        
        if not self._connection:
            await self.connect()
        
        cursor = await self._connection.cursor()
        
        # Inference requests table
        await cursor.execute("""
            CREATE TABLE IF NOT EXISTS inference_requests (
                request_id TEXT PRIMARY KEY,
                requester TEXT NOT NULL,
                serving_node TEXT NOT NULL,
                model_version TEXT,
                input_data_ipfs_hash TEXT,
                fee TEXT,
                status TEXT DEFAULT 'pending',
                request_time TEXT,
                result_ipfs_hash TEXT,
                latency_ms INTEGER,
                tx_hash TEXT,
                block_height INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index for serving_node queries
        await cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_inference_requests_serving_node 
            ON inference_requests(serving_node)
        """)
        
        # Aggregations table
        await cursor.execute("""
            CREATE TABLE IF NOT EXISTS aggregations (
                aggregation_id INTEGER PRIMARY KEY,
                proposer TEXT NOT NULL,
                aggregated_gradient_ipfs_hash TEXT,
                merkle_root TEXT,
                participant_count INTEGER DEFAULT 0,
                training_round_id INTEGER,
                block_height INTEGER,
                timestamp TEXT,
                tx_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index for proposer queries
        await cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_aggregations_proposer 
            ON aggregations(proposer)
        """)
        
        # Aggregation commits table
        await cursor.execute("""
            CREATE TABLE IF NOT EXISTS aggregation_commits (
                commitment_id TEXT PRIMARY KEY,
                proposer TEXT NOT NULL,
                commitment_hash TEXT,
                training_round_id INTEGER,
                gradient_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                commit_height INTEGER,
                commit_time TEXT,
                reveal_deadline TEXT,
                tx_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index for proposer queries
        await cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_aggregation_commits_proposer 
            ON aggregation_commits(proposer)
        """)
        
        await self._connection.commit()
        logger.info("Indexer tables initialized")

    # =========================================================================
    # BLOCKCHAIN CREDIT SYNCHRONIZATION
    # =========================================================================
    
    async def sync_credits_from_blockchain(self, wallet_address: str) -> Optional[float]:
        """
        Synchronize local credits with on-chain token balance.
        
        This method queries the blockchain for the actual R3MES token balance
        and updates the local database to match. This ensures consistency
        between off-chain credits and on-chain tokens.
        
        Args:
            wallet_address: The wallet address to sync
            
        Returns:
            The synchronized credit balance, or None if sync failed
            
        Note:
            This is a one-way sync from blockchain to local database.
            On-chain balance is the source of truth.
        """
        try:
            from .blockchain_query_client import BlockchainQueryClient
            
            client = BlockchainQueryClient()
            
            # Query on-chain balance
            balance_data = client._query_rest(f"/cosmos/bank/v1beta1/balances/{wallet_address}")
            
            if not balance_data or "balances" not in balance_data:
                logger.warning(f"Could not fetch blockchain balance for {wallet_address}")
                return None
            
            # Find R3MES token balance
            remes_balance = 0.0
            for balance in balance_data.get("balances", []):
                if balance.get("denom") == "remes":
                    try:
                        # Convert from smallest unit (uremes) to display unit
                        remes_balance = float(balance.get("amount", 0)) / 1_000_000
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid balance amount for {wallet_address}")
                        continue
            
            # Update local database
            if self.config.is_postgresql():
                await self._db.set_credits(wallet_address, remes_balance)
            else:
                if not self._connection:
                    await self.connect()
                
                cursor = await self._connection.cursor()
                
                # Upsert user with synced balance
                await cursor.execute("""
                    INSERT INTO users (wallet_address, credits)
                    VALUES (?, ?)
                    ON CONFLICT(wallet_address) DO UPDATE SET credits = ?
                """, (wallet_address, remes_balance, remes_balance))
                
                await self._connection.commit()
            
            logger.info(f"Synced credits for {wallet_address}: {remes_balance} R3MES")
            return remes_balance
            
        except Exception as e:
            logger.error(f"Failed to sync credits from blockchain for {wallet_address}: {e}")
            return None
    
    async def sync_all_credits_from_blockchain(self, batch_size: int = 100) -> Dict[str, int]:
        """
        Synchronize credits for all users from blockchain.
        
        This method iterates through all users in the local database
        and syncs their credits with on-chain balances.
        
        Args:
            batch_size: Number of users to process in each batch
            
        Returns:
            Dictionary with sync statistics:
            - total: Total users processed
            - success: Successfully synced
            - failed: Failed to sync
        """
        stats = {"total": 0, "success": 0, "failed": 0}
        
        try:
            # Get all wallet addresses
            if self.config.is_postgresql():
                wallets = await self._db.get_all_wallet_addresses()
            else:
                if not self._connection:
                    await self.connect()
                
                cursor = await self._connection.cursor()
                await cursor.execute("SELECT wallet_address FROM users")
                results = await cursor.fetchall()
                wallets = [row[0] for row in results]
            
            stats["total"] = len(wallets)
            
            # Process in batches to avoid overwhelming the blockchain node
            for i in range(0, len(wallets), batch_size):
                batch = wallets[i:i + batch_size]
                
                for wallet in batch:
                    result = await self.sync_credits_from_blockchain(wallet)
                    if result is not None:
                        stats["success"] += 1
                    else:
                        stats["failed"] += 1
                
                # Small delay between batches to be nice to the blockchain node
                if i + batch_size < len(wallets):
                    await asyncio.sleep(0.5)
            
            logger.info(f"Blockchain credit sync completed: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to sync all credits from blockchain: {e}")
            return stats
    
    async def get_blockchain_balance(self, wallet_address: str) -> Optional[Dict[str, float]]:
        """
        Get all token balances for a wallet from blockchain.
        
        Args:
            wallet_address: The wallet address to query
            
        Returns:
            Dictionary of denom -> balance, or None if query failed
        """
        try:
            from .blockchain_query_client import BlockchainQueryClient
            
            client = BlockchainQueryClient()
            balance_data = client._query_rest(f"/cosmos/bank/v1beta1/balances/{wallet_address}")
            
            if not balance_data or "balances" not in balance_data:
                return None
            
            balances = {}
            for balance in balance_data.get("balances", []):
                denom = balance.get("denom", "unknown")
                try:
                    # Convert from smallest unit to display unit
                    amount = float(balance.get("amount", 0)) / 1_000_000
                    balances[denom] = amount
                except (ValueError, TypeError):
                    continue
            
            return balances
            
        except Exception as e:
            logger.error(f"Failed to get blockchain balance for {wallet_address}: {e}")
            return None

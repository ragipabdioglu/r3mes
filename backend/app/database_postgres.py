"""
PostgreSQL Database - Async PostgreSQL wrapper using asyncpg

Production-ready database implementation with connection pooling.
"""

import asyncpg
from asyncpg import Pool
from typing import Optional, Dict, List
import json
import hashlib
import secrets
from datetime import datetime, timedelta
import logging
import os

logger = logging.getLogger(__name__)


class AsyncPostgreSQL:
    """
    Async PostgreSQL Database wrapper with connection pooling.
    
    All database operations are async to prevent blocking the event loop.
    """
    
    def __init__(self, connection_string: str, min_size: int = 5, max_size: int = 20):
        """
        Initialize PostgreSQL database.
        
        Args:
            connection_string: PostgreSQL connection string
            min_size: Minimum connection pool size
            max_size: Maximum connection pool size
        """
        self.connection_string = connection_string
        self.min_size = min_size
        self.max_size = max_size
        self.pool: Optional[Pool] = None
    
    async def connect(self):
        """Create connection pool with production-optimized settings."""
        if self.pool is None:
            # Production connection pool settings
            is_production = os.getenv("R3MES_ENV", "development").lower() == "production"
            
            pool_kwargs = {
                "min_size": self.min_size,
                "max_size": self.max_size,
                "command_timeout": 60,  # Query timeout
            }
            
            if is_production:
                # Production-specific pool settings
                pool_kwargs.update({
                    "max_queries": 50000,  # Max queries per connection before recycling
                    "max_inactive_connection_lifetime": 300,  # 5 minutes idle timeout
                    "max_connection_lifetime": 3600,  # 1 hour max lifetime
                })
            
            try:
                self.pool = await asyncpg.create_pool(
                    self.connection_string,
                    **pool_kwargs
                )
                await self._init_database()
                logger.info(
                    f"PostgreSQL connection pool created "
                    f"(min={self.min_size}, max={self.max_size}, "
                    f"production={'yes' if is_production else 'no'})"
                )
            except Exception as e:
                logger.error(f"PostgreSQL connection pool creation failed: {e}", exc_info=True)
                # Send notification for database connection failure
                try:
                    from .notifications import get_notification_service, NotificationPriority
                    notification_service = get_notification_service()
                    await notification_service.send_system_alert(
                        component="database",
                        alert_type="connection_failure",
                        message=f"Failed to create PostgreSQL connection pool: {e}",
                        priority=NotificationPriority.CRITICAL
                    )
                except Exception as notif_error:
                    logger.warning(f"Failed to send database connection failure notification: {notif_error}")
                raise
    
    async def close(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("PostgreSQL connection pool closed")
    
    async def execute(self, query: str, *args):
        """Execute a query."""
        if not self.pool:
            await self.connect()
        
        try:
            async with self.pool.acquire() as conn:
                return await conn.execute(query, *args)
        except Exception as e:
            logger.error(f"Database query execution failed: {e}", exc_info=True)
            # Check if it's a connection error
            if "connection" in str(e).lower() or "pool" in str(e).lower():
                try:
                    from .notifications import get_notification_service, NotificationPriority
                    notification_service = get_notification_service()
                    await notification_service.send_system_alert(
                        component="database",
                        alert_type="query_failure",
                        message=f"Database query failed due to connection issue: {e}",
                        priority=NotificationPriority.HIGH
                    )
                except Exception as notif_error:
                    logger.warning(f"Failed to send database query failure notification: {notif_error}")
            raise
    
    async def fetch(self, query: str, *args):
        """Fetch rows from a query."""
        if not self.pool:
            await self.connect()
        
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)
    
    async def fetchrow(self, query: str, *args):
        """Fetch a single row from a query."""
        if not self.pool:
            await self.connect()
        
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, *args)
    
    async def fetchval(self, query: str, *args):
        """Fetch a single value from a query."""
        if not self.pool:
            await self.connect()
        
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, *args)
    
    async def _init_database(self):
        """Initialize database tables."""
        if not self.pool:
            await self.connect()
        
        async with self.pool.acquire() as conn:
            # Users table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    wallet_address VARCHAR(255) PRIMARY KEY,
                    credits REAL DEFAULT 0.0,
                    is_miner BOOLEAN DEFAULT FALSE,
                    last_mining_time TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Mining stats table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS mining_stats (
                    id SERIAL PRIMARY KEY,
                    wallet_address VARCHAR(255) NOT NULL,
                    hashrate REAL DEFAULT 0.0,
                    gpu_temperature REAL DEFAULT 0.0,
                    blocks_found INTEGER DEFAULT 0,
                    uptime_percentage REAL DEFAULT 0.0,
                    network_difficulty REAL DEFAULT 0.0,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (wallet_address) REFERENCES users(wallet_address) ON DELETE CASCADE
                )
            """)
            
            # Earnings history table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS earnings_history (
                    id SERIAL PRIMARY KEY,
                    wallet_address VARCHAR(255) NOT NULL,
                    earnings REAL DEFAULT 0.0,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (wallet_address) REFERENCES users(wallet_address) ON DELETE CASCADE
                )
            """)
            
            # Hashrate history table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS hashrate_history (
                    id SERIAL PRIMARY KEY,
                    wallet_address VARCHAR(255) NOT NULL,
                    hashrate REAL DEFAULT 0.0,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (wallet_address) REFERENCES users(wallet_address) ON DELETE CASCADE
                )
            """)
            
            # API Keys table (with hashed keys)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id SERIAL PRIMARY KEY,
                    api_key_hash VARCHAR(64) UNIQUE NOT NULL,
                    wallet_address VARCHAR(255) NOT NULL,
                    name VARCHAR(255),
                    is_active BOOLEAN DEFAULT TRUE,
                    last_used TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    FOREIGN KEY (wallet_address) REFERENCES users(wallet_address) ON DELETE CASCADE
                )
            """)
            
            # LoRA Registry table (created by migration, but create here as fallback)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS lora_registry (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) UNIQUE NOT NULL,
                    ipfs_hash VARCHAR(128) NOT NULL,
                    description TEXT,
                    category VARCHAR(100),
                    version VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)
            
            # Serving Nodes table (created by migration, but create here as fallback)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS serving_nodes (
                    id SERIAL PRIMARY KEY,
                    wallet_address VARCHAR(255) UNIQUE NOT NULL,
                    endpoint_url VARCHAR(500) NOT NULL,
                    available_lora_list JSONB NOT NULL,
                    status VARCHAR(50) DEFAULT 'active',
                    last_heartbeat TIMESTAMP NOT NULL,
                    current_load INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_key_hash ON api_keys(api_key_hash)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_lora_registry_name ON lora_registry(name)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_lora_registry_category ON lora_registry(category)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_lora_registry_is_active ON lora_registry(is_active)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_serving_nodes_wallet ON serving_nodes(wallet_address)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_serving_nodes_status ON serving_nodes(status)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_serving_nodes_last_heartbeat ON serving_nodes(last_heartbeat)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_serving_nodes_status_heartbeat ON serving_nodes(status, last_heartbeat)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_mining_stats_wallet ON mining_stats(wallet_address)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_earnings_history_wallet ON earnings_history(wallet_address, recorded_at)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_hashrate_history_wallet ON hashrate_history(wallet_address, recorded_at)
            """)
            
            # Apply additional recommended indexes for PostgreSQL
            from .database_optimization import apply_indexes_postgresql
            await apply_indexes_postgresql(conn)
            
            logger.info("PostgreSQL database tables initialized")
    
    def _hash_api_key(self, api_key: str) -> str:
        """Hash API key using SHA256."""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    async def add_credits(self, wallet: str, amount: float) -> bool:
        """Add credits to user."""
        if not self.pool:
            await self.connect()
        
        async with self.pool.acquire() as conn:
            # Check if user exists
            result = await conn.fetchrow(
                "SELECT credits FROM users WHERE wallet_address = $1",
                wallet
            )
            
            if result:
                # Update existing user
                new_credits = result['credits'] + amount
                await conn.execute(
                    "UPDATE users SET credits = $1 WHERE wallet_address = $2",
                    new_credits, wallet
                )
            else:
                # Create new user
                await conn.execute(
                    "INSERT INTO users (wallet_address, credits) VALUES ($1, $2)",
                    wallet, amount
                )
            
            return True
    
    async def deduct_credit(self, wallet: str, amount: float) -> bool:
        """Deduct credits from user."""
        if not self.pool:
            await self.connect()
        
        async with self.pool.acquire() as conn:
            result = await conn.fetchrow(
                "SELECT credits FROM users WHERE wallet_address = $1",
                wallet
            )
            
            if not result:
                return False
            
            current_credits = result['credits']
            if current_credits < amount:
                return False
            
            new_credits = current_credits - amount
            await conn.execute(
                "UPDATE users SET credits = $1 WHERE wallet_address = $2",
                new_credits, wallet
            )
            
            return True
    
    async def check_credits(self, wallet: str) -> float:
        """Check user credits."""
        if not self.pool:
            await self.connect()
        
        result = await self.fetchval(
            "SELECT credits FROM users WHERE wallet_address = $1",
            wallet
        )
        
        return float(result) if result else 0.0
    
    async def get_user_info(self, wallet: str) -> Optional[Dict]:
        """Get user information."""
        if not self.pool:
            await self.connect()
        
        result = await self.fetchrow(
            "SELECT wallet_address, credits, is_miner FROM users WHERE wallet_address = $1",
            wallet
        )
        
        if not result:
            return None
        
        return {
            "wallet_address": result['wallet_address'],
            "credits": float(result['credits']),
            "is_miner": bool(result['is_miner']),
        }
    
    async def validate_api_key(self, api_key: str) -> Optional[Dict]:
        """Validate API key."""
        if not self.pool:
            await self.connect()
        
        api_key_hash = self._hash_api_key(api_key)
        
        result = await self.fetchrow("""
            SELECT id, wallet_address, name, is_active, expires_at
            FROM api_keys
            WHERE api_key_hash = $1 AND is_active = TRUE
        """, api_key_hash)
        
        if not result:
            return None
        
        # Check expiration
        if result['expires_at']:
            expires_dt = result['expires_at']
            if datetime.now() > expires_dt:
                return None
        
        # Update last_used
        await self.execute(
            "UPDATE api_keys SET last_used = $1 WHERE id = $2",
            datetime.now(), result['id']
        )
        
        return {
            "id": result['id'],
            "wallet_address": result['wallet_address'],
            "name": result['name'],
            "is_active": bool(result['is_active']),
        }
    
    async def create_api_key(
        self,
        wallet: str,
        name: str = "Default",
        expires_in_days: Optional[int] = None
    ) -> str:
        """Create new API key."""
        if not self.pool:
            await self.connect()
        
        api_key = f"r3mes_{secrets.token_urlsafe(32)}"
        api_key_hash = self._hash_api_key(api_key)
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)
        
        await self.execute("""
            INSERT INTO api_keys (api_key_hash, wallet_address, name, expires_at)
            VALUES ($1, $2, $3, $4)
        """, api_key_hash, wallet, name, expires_at)
        
        return api_key
    
    async def get_network_stats(self) -> Dict:
        """Get network statistics."""
        from ..app.blockchain_rpc_client import get_blockchain_rpc_client
        
        if not self.pool:
            await self.connect()
        
        active_miners = await self.fetchval(
            "SELECT COUNT(*) FROM users WHERE is_miner = TRUE"
        )
        
        total_users = await self.fetchval("SELECT COUNT(*) FROM users")
        
        total_credits = await self.fetchval("SELECT SUM(credits) FROM users")
        
        # Get block height from blockchain RPC
        try:
            rpc_client = get_blockchain_rpc_client()
            block_height = rpc_client.get_latest_block_height()
        except Exception as e:
            logger.warning(f"Failed to get block height from blockchain RPC: {e}")
            block_height = None
        
        return {
            "active_miners": active_miners or 0,
            "total_users": total_users or 0,
            "total_credits": float(total_credits) if total_credits else 0.0,
            "block_height": block_height
        }
    
    async def get_recent_blocks(self, limit: int = 10) -> List[Dict]:
        """Get recent blocks from blockchain RPC."""
        from ..app.blockchain_rpc_client import get_blockchain_rpc_client
        
        try:
            rpc_client = get_blockchain_rpc_client()
            blocks = rpc_client.get_recent_blocks(limit=limit)
            return blocks
        except Exception as e:
            logger.warning(f"Failed to get recent blocks from blockchain RPC: {e}")
            return []
    
    async def get_miner_stats(self, wallet: str) -> Dict:
        """Get miner statistics."""
        if not self.pool:
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
        
        result = await self.fetchrow("""
            SELECT hashrate, gpu_temperature, blocks_found, uptime_percentage, network_difficulty
            FROM mining_stats
            WHERE wallet_address = $1
            ORDER BY recorded_at DESC
            LIMIT 1
        """, wallet)
        
        # Get network difficulty: prefer from result, then blockchain, then default
        network_difficulty = 1234.0  # Default fallback
        if result and result.get('network_difficulty'):
            network_difficulty = float(result['network_difficulty'])
        else:
            # Try to fetch from blockchain
            network_difficulty = await self._get_network_difficulty()
        
        return {
            "wallet_address": wallet,
            "total_earnings": user_info['credits'],
            "hashrate": float(result['hashrate']) if result and result.get('hashrate') else 0.0,
            "gpu_temperature": float(result['gpu_temperature']) if result and result.get('gpu_temperature') else 0.0,
            "blocks_found": result['blocks_found'] if result and result.get('blocks_found') else 0,
            "uptime_percentage": float(result['uptime_percentage']) if result and result.get('uptime_percentage') else 0.0,
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
        """Get earnings history."""
        if not self.pool:
            await self.connect()
        
        results = await self.fetch("""
            SELECT DATE(recorded_at) as date, SUM(earnings) as total_earnings
            FROM earnings_history
            WHERE wallet_address = $1 AND recorded_at >= NOW() - INTERVAL '1 day' * $2
            GROUP BY DATE(recorded_at)
            ORDER BY date ASC
        """, wallet, days)
        
        earnings_data = []
        for row in results:
            earnings_data.append({
                "date": row['date'].strftime('%Y-%m-%d'),
                "earnings": float(row['total_earnings']) if row['total_earnings'] else 0.0
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
        """Get hashrate history."""
        if not self.pool:
            await self.connect()
        
        results = await self.fetch("""
            SELECT DATE(recorded_at) as date, AVG(hashrate) as avg_hashrate
            FROM hashrate_history
            WHERE wallet_address = $1 AND recorded_at >= NOW() - INTERVAL '1 day' * $2
            GROUP BY DATE(recorded_at)
            ORDER BY date ASC
        """, wallet, days)
        
        hashrate_data = []
        for row in results:
            hashrate_data.append({
                "date": row['date'].strftime('%Y-%m-%d'),
                "hashrate": float(row['avg_hashrate']) if row['avg_hashrate'] else 0.0
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
        """List API keys for a wallet."""
        if not self.pool:
            await self.connect()
        
        results = await self.fetch("""
            SELECT id, name, is_active, created_at, expires_at, last_used
            FROM api_keys
            WHERE wallet_address = $1
            ORDER BY created_at DESC
        """, wallet_address)
        
        return [
            {
                "id": row['id'],
                "name": row['name'],
                "is_active": bool(row['is_active']),
                "created_at": row['created_at'].isoformat() if row['created_at'] else None,
                "expires_at": row['expires_at'].isoformat() if row['expires_at'] else None,
                "last_used": row['last_used'].isoformat() if row['last_used'] else None,
            }
            for row in results
        ]
    
    async def revoke_api_key(self, api_key_id: int, wallet_address: str) -> bool:
        """Revoke API key."""
        if not self.pool:
            await self.connect()
        
        result = await self.execute("""
            UPDATE api_keys
            SET is_active = FALSE
            WHERE id = $1 AND wallet_address = $2
        """, api_key_id, wallet_address)
        
        return "UPDATE" in result
    
    async def delete_api_key(self, api_key_id: int, wallet_address: str) -> bool:
        """Delete API key."""
        if not self.pool:
            await self.connect()
        
        result = await self.execute("""
            DELETE FROM api_keys
            WHERE id = $1 AND wallet_address = $2
        """, api_key_id, wallet_address)
        
        return "DELETE" in result


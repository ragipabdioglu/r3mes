"""
Blockchain Indexer Service for R3MES

Listens to blockchain blocks and events, stores them in PostgreSQL for historical analytics.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
import os

from .database_async import AsyncDatabase
from .blockchain_rpc_client import BlockchainRPCClient
from .config_manager import get_config_manager

logger = logging.getLogger(__name__)


class BlockchainIndexer:
    """
    Indexer service that listens to blockchain blocks and stores events in PostgreSQL.
    
    This enables historical analytics without repeatedly querying the blockchain.
    """
    
    def __init__(self, database: AsyncDatabase, rpc_client: Optional[BlockchainRPCClient] = None):
        """
        Initialize blockchain indexer.
        
        Args:
            database: AsyncDatabase instance for storing indexed data
            rpc_client: BlockchainRPCClient instance (optional, creates new if None)
        """
        self.database = database
        self.rpc_client = rpc_client or BlockchainRPCClient()
        self.running = False
        self._indexed_height = 0
        self._task: Optional[asyncio.Task] = None
        self._batch_size = int(os.getenv("INDEXER_BATCH_SIZE", "10"))  # Process blocks in batches
        self._indexing_lag = 0  # Track indexing lag (blocks behind)
        self._last_indexed_time = None
    
    async def _init_indexer_tables(self):
        """Initialize indexer-specific database tables."""
        # Check if using PostgreSQL
        if not self.database.config.is_postgresql():
            logger.warning("Indexer requires PostgreSQL. Skipping table initialization for SQLite.")
            return
        
        # Get PostgreSQL connection
        if not hasattr(self.database, '_db') or not self.database._db:
            await self.database.connect()
        
        # Create blockchain_events table
        async with self.database._db.pool.acquire() as conn:
            # Blockchain events table (stores all blockchain events)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS blockchain_events (
                    id SERIAL PRIMARY KEY,
                    event_type VARCHAR(100) NOT NULL,
                    block_height BIGINT NOT NULL,
                    block_hash VARCHAR(64),
                    tx_hash VARCHAR(64),
                    miner_address VARCHAR(255),
                    validator_address VARCHAR(255),
                    pool_id BIGINT,
                    chunk_id BIGINT,
                    gradient_hash VARCHAR(128),
                    gradient_ipfs_hash VARCHAR(128),
                    amount DECIMAL(20, 8),
                    event_data JSONB,
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                
                -- Create indexes separately
                CREATE INDEX IF NOT EXISTS idx_event_type ON blockchain_events(event_type);
                CREATE INDEX IF NOT EXISTS idx_block_height ON blockchain_events(block_height);
                CREATE INDEX IF NOT EXISTS idx_miner_address ON blockchain_events(miner_address);
                CREATE INDEX IF NOT EXISTS idx_indexed_at ON blockchain_events(indexed_at);
            """)
            
            # Network snapshots table (stores periodic network state snapshots)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS network_snapshots (
                    id SERIAL PRIMARY KEY,
                    block_height BIGINT NOT NULL,
                    snapshot_date DATE NOT NULL,
                    total_miners INTEGER DEFAULT 0,
                    total_validators INTEGER DEFAULT 0,
                    total_stake DECIMAL(20, 8) DEFAULT 0.0,
                    total_gradients BIGINT DEFAULT 0,
                    total_aggregations BIGINT DEFAULT 0,
                    network_hashrate DECIMAL(20, 2) DEFAULT 0.0,
                    snapshot_data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(block_height, snapshot_date)
                )
                
                -- Create indexes separately
                CREATE INDEX IF NOT EXISTS idx_snapshot_date ON network_snapshots(snapshot_date);
                CREATE INDEX IF NOT EXISTS idx_snapshots_block_height ON network_snapshots(block_height);
            """)
            
            # Indexer state table (tracks last indexed block)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS indexer_state (
                    id INTEGER PRIMARY KEY DEFAULT 1,
                    last_indexed_height BIGINT DEFAULT 0,
                    last_indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT single_row CHECK (id = 1)
                )
            """)
            
            # Initialize indexer state if not exists
            await conn.execute("""
                INSERT INTO indexer_state (id, last_indexed_height, last_indexed_at)
                VALUES (1, 0, CURRENT_TIMESTAMP)
                ON CONFLICT (id) DO NOTHING
            """)
        
        logger.info("Indexer tables initialized")
    
    async def _get_last_indexed_height(self) -> int:
        """Get the last indexed block height from database."""
        if not self.database.config.is_postgresql():
            return self._indexed_height
        
        try:
            async with self.database._db.pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT last_indexed_height FROM indexer_state WHERE id = 1"
                )
                if row:
                    return row['last_indexed_height'] or 0
                return 0
        except Exception as e:
            logger.warning(f"Failed to get last indexed height: {e}, starting from 0")
            return 0
    
    async def _update_last_indexed_height(self, height: int):
        """Update the last indexed block height in database."""
        if not self.database.config.is_postgresql():
            self._indexed_height = height
            return
        
        try:
            async with self.database._db.pool.acquire() as conn:
                await conn.execute("""
                    UPDATE indexer_state
                    SET last_indexed_height = $1, last_indexed_at = CURRENT_TIMESTAMP
                    WHERE id = 1
                """, height)
        except Exception as e:
            logger.warning(f"Failed to update last indexed height: {e}")
    
    async def _index_block(self, height: int) -> bool:
        """
        Index a single block: extract events and store them in database.
        Also broadcasts updates via WebSocket for real-time UI updates.
        
        Args:
            height: Block height to index
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get block from RPC
            block = self.rpc_client.get_block(height)
            if not block:
                logger.debug(f"Block {height} not found, skipping")
                return False
            
            block_height = block.get("height", 0)
            block_hash = block.get("hash", "")
            timestamp = block.get("timestamp")
            proposer = block.get("proposer", "")
            
            # Broadcast block update via WebSocket
            await self._broadcast_block_update(block)
            
            if not self.database.config.is_postgresql():
                logger.debug(f"Skipping block indexing (SQLite mode), block {height}")
                return True
            
            # Parse block and transaction events
            async with self.database._db.pool.acquire() as conn:
                # Check if block already indexed (avoid duplicates)
                existing = await conn.fetchval(
                    "SELECT id FROM blockchain_events WHERE block_height = $1 AND event_type = $2",
                    block_height, "block"
                )
                
                if not existing:
                    # Store block as an event
                    await conn.execute("""
                        INSERT INTO blockchain_events (
                            event_type, block_height, block_hash, event_data
                        ) VALUES ($1, $2, $3, $4)
                    """, "block", block_height, block_hash, {
                        "timestamp": timestamp,
                        "tx_count": block.get("tx_count", 0),
                        "proposer": proposer,
                    })
                
                # Parse transaction events (if available)
                txs = block.get("txs", [])
                if txs:
                    await self._parse_transaction_events(conn, block_height, block_hash, txs)
                
                # Create network snapshot every 100 blocks (or configurable interval)
                if block_height % 100 == 0:
                    await self._create_network_snapshot(conn, block_height, timestamp)
                    # Broadcast network status update
                    await self._broadcast_network_status()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to index block {height}: {e}", exc_info=True)
            return False
    
    async def _broadcast_block_update(self, block: Dict[str, Any]):
        """
        Broadcast new block to WebSocket clients.
        
        Args:
            block: Block data dictionary
        """
        try:
            from .websocket_manager import broadcast_block_update
            
            block_data = {
                "height": block.get("height", 0),
                "hash": block.get("hash", ""),
                "proposer": block.get("proposer", ""),
                "timestamp": block.get("timestamp", ""),
                "tx_count": block.get("tx_count", 0),
            }
            
            await broadcast_block_update(block_data)
            logger.debug(f"Broadcasted block update for height {block_data['height']}")
        except Exception as e:
            logger.debug(f"Failed to broadcast block update: {e}")
    
    async def _broadcast_network_status(self):
        """
        Broadcast network status update to WebSocket clients.
        
        Queries current network statistics and broadcasts to all connected clients.
        """
        try:
            from .websocket_manager import broadcast_network_status
            from .blockchain_query_client import get_blockchain_client
            
            blockchain_client = get_blockchain_client()
            
            # Get current network statistics
            current_height = self.rpc_client.get_latest_block_height() or 0
            
            # Query miners and validators count
            miners_result = blockchain_client.get_all_miners(limit=1, offset=0)
            total_miners = miners_result.get("total", 0)
            
            validators_result = blockchain_client.get_all_validators(limit=1, offset=0)
            total_validators = validators_result.get("total", 0)
            
            # Get staking info
            staking_info = blockchain_client.get_staking_info()
            total_stake = staking_info.get("total_stake", 0.0) if staking_info else 0.0
            
            # Get network statistics
            blockchain_stats = blockchain_client.get_network_statistics()
            total_gradients = blockchain_stats.get("total_gradients", 0) if blockchain_stats else 0
            total_aggregations = blockchain_stats.get("total_aggregations", 0) if blockchain_stats else 0
            
            network_status = {
                "block_height": current_height,
                "total_miners": total_miners,
                "total_validators": total_validators,
                "total_stake": str(total_stake),
                "total_gradients": total_gradients,
                "total_aggregations": total_aggregations,
                "indexing_lag": self._indexing_lag,
                "last_indexed_time": self._last_indexed_time.isoformat() if self._last_indexed_time else None,
            }
            
            await broadcast_network_status(network_status)
            logger.debug(f"Broadcasted network status update")
        except Exception as e:
            logger.debug(f"Failed to broadcast network status: {e}")
    
    async def _parse_transaction_events(self, conn, block_height: int, block_hash: str, txs: List[Dict]):
        """
        Parse transaction events from block transactions.
        
        Args:
            conn: Database connection
            block_height: Block height
            block_hash: Block hash
            txs: List of transaction data
        """
        try:
            for tx in txs:
                tx_hash = tx.get("hash", "")
                tx_result = tx.get("tx_result", {})
                events = tx_result.get("events", [])
                
                # Parse Cosmos SDK events
                for event in events:
                    event_type = event.get("type", "")
                    attributes = event.get("attributes", [])
                    
                    # Build event data from attributes
                    event_data = {}
                    miner_address = None
                    validator_address = None
                    pool_id = None
                    chunk_id = None
                    gradient_hash = None
                    gradient_ipfs_hash = None
                    amount = None
                    
                    for attr in attributes:
                        key = attr.get("key", "")
                        value = attr.get("value", "")
                        
                        # Decode base64 if needed
                        if isinstance(value, str) and value.startswith("base64:"):
                            try:
                                import base64
                                value = base64.b64decode(value.split(":", 1)[1]).decode("utf-8")
                            except Exception:
                                pass
                        
                        event_data[key] = value
                        
                        # Extract common fields
                        if key == "miner" or key == "sender":
                            miner_address = value
                        elif key == "validator":
                            validator_address = value
                        elif key == "pool_id":
                            try:
                                pool_id = int(value)
                            except (ValueError, TypeError):
                                pass
                        elif key == "chunk_id":
                            try:
                                chunk_id = int(value)
                            except (ValueError, TypeError):
                                pass
                        elif key == "gradient_hash":
                            gradient_hash = value
                        elif key == "ipfs_hash" or key == "gradient_ipfs_hash":
                            gradient_ipfs_hash = value
                        elif key == "amount":
                            amount = value
                    
                    # Store transaction event
                    await conn.execute("""
                        INSERT INTO blockchain_events (
                            event_type, block_height, block_hash, tx_hash,
                            miner_address, validator_address, pool_id, chunk_id,
                            gradient_hash, gradient_ipfs_hash, amount, event_data
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    """, event_type, block_height, block_hash, tx_hash,
                        miner_address, validator_address, pool_id, chunk_id,
                        gradient_hash, gradient_ipfs_hash, amount, event_data)
        except Exception as e:
            logger.warning(f"Failed to parse transaction events for block {block_height}: {e}")
    
    async def _create_network_snapshot(
        self, 
        conn, 
        block_height: int, 
        timestamp: Optional[str]
    ):
        """
        Create a network state snapshot for analytics.
        
        Args:
            conn: Database connection
            block_height: Current block height
            timestamp: Block timestamp
        """
        try:
            from .blockchain_query_client import get_blockchain_client
            
            # Get current network state from blockchain
            blockchain_client = get_blockchain_client()
            
            # Query current network statistics
            indexer_miners_limit = int(os.getenv("BACKEND_INDEXER_MINERS_LIMIT", "1000"))
            miners_result = blockchain_client.get_all_miners(limit=indexer_miners_limit, offset=0)
            total_miners = miners_result.get("total", 0)
            
            indexer_validators_limit = int(os.getenv("BACKEND_INDEXER_VALIDATORS_LIMIT", "1000"))
            validators_result = blockchain_client.get_all_validators(limit=indexer_validators_limit, offset=0)
            total_validators = validators_result.get("total", 0)
            
            staking_info = blockchain_client.get_staking_info()
            total_stake = staking_info.get("total_stake", 0.0) if staking_info else 0.0
            
            blockchain_stats = blockchain_client.get_network_statistics()
            total_gradients = blockchain_stats.get("total_gradients", 0) if blockchain_stats else 0
            total_aggregations = blockchain_stats.get("total_aggregations", 0) if blockchain_stats else 0
            
            # Estimate network hashrate (miners * estimated hashrate per miner)
            network_hashrate = float(total_miners * 10.0)  # Estimate
            
            # Parse date from timestamp
            snapshot_date = datetime.now().date()
            if timestamp:
                try:
                    from datetime import datetime as dt
                    snapshot_date = dt.fromisoformat(timestamp.replace("Z", "+00:00")).date()
                except Exception:
                    pass
            
            # Store snapshot
            await conn.execute("""
                INSERT INTO network_snapshots (
                    block_height, snapshot_date, total_miners, total_validators,
                    total_stake, total_gradients, total_aggregations, network_hashrate,
                    snapshot_data
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (block_height, snapshot_date) DO UPDATE SET
                    total_miners = EXCLUDED.total_miners,
                    total_validators = EXCLUDED.total_validators,
                    total_stake = EXCLUDED.total_stake,
                    total_gradients = EXCLUDED.total_gradients,
                    total_aggregations = EXCLUDED.total_aggregations,
                    network_hashrate = EXCLUDED.network_hashrate,
                    snapshot_data = EXCLUDED.snapshot_data
            """, block_height, snapshot_date, total_miners, total_validators,
                float(total_stake), total_gradients, total_aggregations, network_hashrate,
                {}
            )
            
            logger.debug(f"Created network snapshot for block {block_height}")
            
        except Exception as e:
            logger.warning(f"Failed to create network snapshot: {e}")
    
    async def _indexing_loop(self):
        """Main indexing loop: continuously index new blocks."""
        logger.info("Indexer loop started")
        
        # Get last indexed height
        last_height = await self._get_last_indexed_height()
        
        while self.running:
            try:
                # Get current blockchain height
                current_height = self.rpc_client.get_latest_block_height()
                if current_height is None:
                    logger.warning("Failed to get current block height, retrying...")
                    await asyncio.sleep(10)
                    continue
                
                # Index blocks from last_height + 1 to current_height
                if current_height > last_height:
                    # Calculate indexing lag
                    self._indexing_lag = current_height - last_height
                    logger.info(f"Indexing blocks {last_height + 1} to {current_height} (lag: {self._indexing_lag} blocks)")
                    
                    # Process blocks in batches for better performance
                    blocks_to_index = list(range(last_height + 1, min(current_height + 1, last_height + 1 + self._batch_size)))
                    
                    # Index blocks in batch (parallel processing)
                    tasks = [self._index_block(height) for height in blocks_to_index]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Update last indexed height for successfully indexed blocks
                    for i, (height, result) in enumerate(zip(blocks_to_index, results)):
                        if not self.running:
                            break
                        if result is True:
                            last_height = height
                            await self._update_last_indexed_height(height)
                        elif isinstance(result, Exception):
                            logger.warning(f"Failed to index block {height}: {result}")
                    
                    self._last_indexed_time = datetime.now()
                    
                    # Small delay to avoid overwhelming the RPC
                    await asyncio.sleep(0.5)
                else:
                    # No new blocks, wait a bit
                    await asyncio.sleep(5)
                    
            except Exception as e:
                logger.error(f"Error in indexing loop: {e}", exc_info=True)
                await asyncio.sleep(10)  # Wait before retrying
        
        logger.info("Indexer loop stopped")
    
    async def start(self):
        """Start the indexer service."""
        if self.running:
            logger.warning("Indexer already running")
            return
        
        # Initialize tables
        await self._init_indexer_tables()
        
        # Start indexing loop
        self.running = True
        self._task = asyncio.create_task(self._indexing_loop())
        logger.info("Blockchain indexer started")
    
    async def stop(self):
        """Stop the indexer service."""
        if not self.running:
            return
        
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info("Blockchain indexer stopped")
    
    async def get_network_snapshots(
        self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get network snapshots for analytics.
        
        Args:
            start_date: Start date for snapshots
            end_date: End date for snapshots
            limit: Maximum number of snapshots to return
            
        Returns:
            List of network snapshot dictionaries
        """
        if not self.database.config.is_postgresql():
            return []
        
        try:
            async with self.database._db.pool.acquire() as conn:
                query = "SELECT * FROM network_snapshots WHERE 1=1"
                params = []
                param_count = 0
                
                if start_date:
                    param_count += 1
                    query += f" AND snapshot_date >= ${param_count}"
                    params.append(start_date.date())
                
                if end_date:
                    param_count += 1
                    query += f" AND snapshot_date <= ${param_count}"
                    params.append(end_date.date())
                
                query += " ORDER BY snapshot_date DESC LIMIT $" + str(param_count + 1)
                params.append(limit)
                
                rows = await conn.fetch(query, *params)
                
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get network snapshots: {e}", exc_info=True)
            return []


# Global indexer instance
_indexer: Optional[BlockchainIndexer] = None


def get_indexer(database: Optional[AsyncDatabase] = None) -> BlockchainIndexer:
    """
    Get or create the global indexer instance.
    
    Args:
        database: Optional database instance (uses default if None)
        
    Returns:
        BlockchainIndexer instance
    """
    global _indexer
    
    if _indexer is None:
        if database is None:
            config_manager = get_config_manager()
            config = config_manager.load()
            database = AsyncDatabase(
                db_path=config.database_path, 
                chain_json_path=config.chain_json_path
            )
        
        _indexer = BlockchainIndexer(database)
    
    return _indexer


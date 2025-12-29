"""
Cache Warming Strategy

Pre-loads frequently accessed data into cache on startup or periodically.
"""

import asyncio
import logging
from typing import List, Dict, Optional
from datetime import datetime

from .cache import get_cache_manager
from .cache_keys import (
    user_info_key,
    miner_stats_key,
    network_stats_key,
    block_height_key,
    recent_blocks_key,
)
from .database_async import AsyncDatabase
from .blockchain_rpc_client import BlockchainRPCClient
from .blockchain_query_client import BlockchainQueryClient

logger = logging.getLogger(__name__)


class CacheWarmer:
    """
    Warms cache by pre-loading frequently accessed data.
    
    Strategies:
    - Startup warming: Load critical data on application startup
    - Periodic warming: Refresh cache periodically
    - Event-driven warming: Warm cache on specific events
    """
    
    def __init__(
        self,
        database: Optional[AsyncDatabase] = None,
        rpc_client: Optional[BlockchainRPCClient] = None,
        query_client: Optional[BlockchainQueryClient] = None
    ):
        """
        Initialize cache warmer.
        
        Args:
            database: Database instance for querying data
            rpc_client: Blockchain RPC client
            query_client: Blockchain query client
        """
        self.cache = get_cache_manager()
        self.database = database
        self.rpc_client = rpc_client
        self.query_client = query_client
        self._warming_in_progress = False
    
    async def warm_on_startup(self):
        """
        Warm cache on application startup.
        
        Loads critical data that is frequently accessed:
        - Network statistics
        - Recent blocks
        - Block height
        """
        if self._warming_in_progress:
            logger.warning("Cache warming already in progress")
            return
        
        self._warming_in_progress = True
        logger.info("Starting cache warming on startup...")
        
        try:
            # Warm network stats
            await self._warm_network_stats()
            
            # Warm recent blocks
            await self._warm_recent_blocks()
            
            # Warm block height
            await self._warm_block_height()
            
            logger.info("Cache warming completed successfully")
        except Exception as e:
            logger.error(f"Error during cache warming: {e}", exc_info=True)
        finally:
            self._warming_in_progress = False
    
    async def _warm_network_stats(self):
        """Warm network statistics cache."""
        try:
            if self.database:
                stats = await self.database.get_network_stats()
                cache_key = network_stats_key()
                await self.cache.set(cache_key, stats, ttl=60)  # 1 minute TTL
                logger.info("Warmed network stats cache")
        except Exception as e:
            logger.warning(f"Failed to warm network stats: {e}")
    
    async def _warm_recent_blocks(self):
        """Warm recent blocks cache."""
        try:
            if self.database:
                blocks = await self.database.get_recent_blocks(limit=10)
                cache_key = recent_blocks_key()
                await self.cache.set(cache_key, {"blocks": blocks}, ttl=30)  # 30 seconds TTL
                logger.info("Warmed recent blocks cache")
        except Exception as e:
            logger.warning(f"Failed to warm recent blocks: {e}")
    
    async def _warm_block_height(self):
        """Warm block height cache."""
        try:
            if self.rpc_client:
                height = self.rpc_client.get_latest_block_height(use_cache=False)
                if height:
                    cache_key = block_height_key()
                    await self.cache.set(cache_key, {"height": height}, ttl=10)  # 10 seconds TTL
                    logger.info(f"Warmed block height cache: {height}")
        except Exception as e:
            logger.warning(f"Failed to warm block height: {e}")
    
    async def warm_user_data(self, wallet_address: str):
        """
        Warm cache for a specific user.
        
        Args:
            wallet_address: User wallet address
        """
        try:
            if not self.database:
                return
            
            # Warm user info
            user_info = await self.database.get_user_info(wallet_address)
            if user_info:
                cache_key = user_info_key(wallet_address)
                await self.cache.set(cache_key, user_info, ttl=300)  # 5 minutes TTL
            
            # Warm miner stats if user is a miner
            if user_info and user_info.get("is_miner"):
                miner_stats = await self.database.get_miner_stats(wallet_address)
                if miner_stats:
                    from .cache_keys import miner_stats_key
                    cache_key = miner_stats_key(wallet_address)
                    await self.cache.set(cache_key, miner_stats, ttl=60)  # 1 minute TTL
            
            logger.debug(f"Warmed cache for user: {wallet_address}")
        except Exception as e:
            logger.warning(f"Failed to warm user data for {wallet_address}: {e}")
    
    async def periodic_warm(self, interval: int = 300):
        """
        Periodically warm cache.
        
        Args:
            interval: Interval in seconds between warming cycles
        """
        logger.info(f"Starting periodic cache warming (interval: {interval}s)")
        
        while True:
            try:
                await asyncio.sleep(interval)
                logger.info("Running periodic cache warming...")
                
                # Warm critical data
                await self._warm_network_stats()
                await self._warm_recent_blocks()
                await self._warm_block_height()
                
                logger.info("Periodic cache warming completed")
            except Exception as e:
                logger.error(f"Error in periodic cache warming: {e}", exc_info=True)


# Global cache warmer instance
_cache_warmer: Optional[CacheWarmer] = None


def get_cache_warmer(
    database: Optional[AsyncDatabase] = None,
    rpc_client: Optional[BlockchainRPCClient] = None,
    query_client: Optional[BlockchainQueryClient] = None
) -> CacheWarmer:
    """Get or create global cache warmer instance."""
    global _cache_warmer
    if _cache_warmer is None:
        _cache_warmer = CacheWarmer(database, rpc_client, query_client)
    return _cache_warmer


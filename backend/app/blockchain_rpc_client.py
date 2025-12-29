"""
Blockchain RPC Client for R3MES

Provides Tendermint RPC interface for querying blocks and blockchain state.
Includes Redis caching for performance optimization.
"""

import requests
import os
import logging
import time
from typing import List, Dict, Optional, Any
from datetime import datetime

from .network_resilience import (
    retry_with_backoff,
    BLOCKCHAIN_RETRY_CONFIG,
    get_blockchain_circuit_breaker,
    with_timeout,
)
from .exceptions import (
    MissingEnvironmentVariableError,
    ProductionConfigurationError,
    BlockchainConnectionError,
    BlockchainQueryError,
    InvalidBlockchainResponseError,
)

logger = logging.getLogger(__name__)

# Cache TTLs (in seconds) - can be overridden via environment variables
BLOCK_CACHE_TTL = int(os.getenv("BACKEND_BLOCK_CACHE_TTL", "300"))  # 5 minutes (blocks don't change once finalized)
RECENT_BLOCKS_CACHE_TTL = int(os.getenv("BACKEND_RECENT_BLOCKS_CACHE_TTL", "30"))  # 30 seconds (recent blocks list changes more frequently)
BLOCK_HEIGHT_CACHE_TTL = int(os.getenv("BACKEND_BLOCK_HEIGHT_CACHE_TTL", "10"))  # 10 seconds (block height updates frequently)


class BlockchainRPCClient:
    """
    Client for querying R3MES blockchain via Tendermint RPC.
    
    Uses HTTP REST API for block queries.
    """
    
    def __init__(self, rpc_url: Optional[str] = None):
        """
        Initialize blockchain RPC client.
        
        Args:
            rpc_url: Tendermint RPC endpoint URL (default: from BLOCKCHAIN_RPC_URL env var)
        """
        # Get RPC URL from parameter or environment variable
        # In production, BLOCKCHAIN_RPC_URL must be set (no localhost fallback)
        is_production = os.getenv("R3MES_ENV", "development").lower() == "production"
        
        if rpc_url:
            self.rpc_url = rpc_url
        else:
            rpc_url_env = os.getenv("BLOCKCHAIN_RPC_URL")
            if not rpc_url_env:
                if is_production:
                    raise MissingEnvironmentVariableError(
                        "BLOCKCHAIN_RPC_URL environment variable must be set in production. "
                        "Do not use localhost in production."
                    )
                # Development fallback
                self.rpc_url = "http://localhost:26657"
                logger.warning("BLOCKCHAIN_RPC_URL not set, using localhost fallback (development only)")
            else:
                self.rpc_url = rpc_url_env
                # Validate that production doesn't use localhost
                if is_production and ("localhost" in self.rpc_url or "127.0.0.1" in self.rpc_url):
                    raise ProductionConfigurationError(
                        f"BLOCKCHAIN_RPC_URL cannot use localhost in production: {self.rpc_url}"
                    )
        
        logger.info(f"Blockchain RPC client initialized: {self.rpc_url}")
    
    def _rpc_request(self, method: str, params: Optional[Dict] = None) -> Dict:
        """Make RPC request to Tendermint with retry mechanism and circuit breaker."""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or {}
        }
        
        # Use circuit breaker for blockchain requests
        circuit_breaker = get_blockchain_circuit_breaker()
        
        def make_request():
            rpc_timeout = int(os.getenv("BACKEND_BLOCKCHAIN_RPC_TIMEOUT", "10"))
            response = requests.post(
                self.rpc_url,
                json=payload,
                timeout=rpc_timeout
            )
            response.raise_for_status()
            data = response.json()
            
            # Check for RPC error in response (invalid method, params, etc.)
            if "error" in data:
                error_code = data["error"].get("code", 0)
                error_message = data["error"].get("message", "")
                
                # RPC errors (invalid method, params) should not be retried
                # Only network/connection errors should be retried
                raise BlockchainQueryError(f"RPC error: {error_message} (code: {error_code})")
            
            return data.get("result", {})
        
        # Execute with circuit breaker
        try:
            return circuit_breaker.call(make_request)
        except Exception as e:
            # Convert to appropriate exception type
            if isinstance(e, requests.exceptions.ConnectionError):
                error_msg = f"Failed to connect to blockchain RPC: {e}"
                # Send notification for blockchain connection failure
                try:
                    from .notifications import get_notification_service, NotificationPriority
                    notification_service = get_notification_service()
                    # Use asyncio to run async notification in sync context
                    # FIXED: Properly handle running event loop case
                    import asyncio
                    
                    async def send_notification():
                        await notification_service.send_system_alert(
                            component="blockchain",
                            alert_type="connection_failure",
                            message=error_msg,
                            priority=NotificationPriority.CRITICAL
                        )
                    
                    try:
                        loop = asyncio.get_running_loop()
                        # Event loop is running - schedule as task instead of blocking
                        asyncio.create_task(send_notification())
                    except RuntimeError:
                        # No running loop - safe to use run_until_complete
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            loop.run_until_complete(send_notification())
                        finally:
                            loop.close()
                except Exception as notif_error:
                    logger.warning(f"Failed to send blockchain connection failure notification: {notif_error}")
                raise BlockchainConnectionError(error_msg) from e
            elif isinstance(e, requests.exceptions.Timeout):
                error_msg = f"Blockchain RPC request timed out: {e}"
                # Send notification for blockchain timeout
                try:
                    from .notifications import get_notification_service, NotificationPriority
                    notification_service = get_notification_service()
                    # FIXED: Properly handle running event loop case
                    import asyncio
                    
                    async def send_notification():
                        await notification_service.send_system_alert(
                            component="blockchain",
                            alert_type="timeout",
                            message=error_msg,
                            priority=NotificationPriority.HIGH
                        )
                    
                    try:
                        loop = asyncio.get_running_loop()
                        # Event loop is running - schedule as task instead of blocking
                        asyncio.create_task(send_notification())
                    except RuntimeError:
                        # No running loop - safe to use run_until_complete
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            loop.run_until_complete(send_notification())
                        finally:
                            loop.close()
                except Exception as notif_error:
                    logger.warning(f"Failed to send blockchain timeout notification: {notif_error}")
                raise BlockchainConnectionError(error_msg) from e
            raise
    
    def get_latest_block_height(self, use_cache: bool = True) -> Optional[int]:
        """
        Get latest block height (with optional caching).
        
        Args:
            use_cache: Whether to use cache (default: True)
        
        Returns:
            Block height or None if query fails
        """
        # Try cache first
        if use_cache:
            try:
                from .cache import get_cache_manager
                import asyncio
                
                cache = get_cache_manager()
                cache_key = "blockchain:latest_height"
                
                # Use asyncio to run async cache operation
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                if loop.is_running():
                    # If event loop is running, we can't use sync await
                    # Fall through to RPC call
                    pass
                else:
                    cached = loop.run_until_complete(cache.get(cache_key))
                    if cached and isinstance(cached, dict) and "height" in cached:
                        logger.debug(f"Cache hit for latest block height: {cached['height']}")
                        return cached["height"]
            except Exception as e:
                logger.debug(f"Cache check failed, using RPC: {e}")
        
        try:
            result = self._rpc_request("status")
            sync_info = result.get("sync_info", {})
            latest_block_height = sync_info.get("latest_block_height")
            height = int(latest_block_height) if latest_block_height else None
            
            # Cache the result
            if use_cache and height is not None:
                try:
                    from .cache import get_cache_manager
                    import asyncio
                    
                    cache = get_cache_manager()
                    cache_key = "blockchain:latest_height"
                    
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    if not loop.is_running():
                        loop.run_until_complete(
                            cache.set(cache_key, {"height": height}, BLOCK_HEIGHT_CACHE_TTL)
                        )
                except Exception as e:
                    logger.debug(f"Failed to cache block height: {e}")
            
            return height
        except Exception as e:
            logger.error(f"Failed to get latest block height: {e}")
            return None
    
    def get_block(self, height: Optional[int] = None, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get block by height (or latest if height is None) with optional caching.
        
        Args:
            height: Block height (None for latest)
            use_cache: Whether to use cache (default: True)
            
        Returns:
            Block data or None if query fails
        """
        # Try cache first (only for specific heights, not latest)
        if use_cache and height is not None:
            try:
                from .cache import get_cache_manager
                import asyncio
                
                cache = get_cache_manager()
                cache_key = f"blockchain:block:{height}"
                
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                if not loop.is_running():
                    cached = loop.run_until_complete(cache.get(cache_key))
                    if cached and isinstance(cached, dict):
                        logger.debug(f"Cache hit for block {height}")
                        return cached
            except Exception as e:
                logger.debug(f"Cache check failed for block {height}, using RPC: {e}")
        
        try:
            height_str = str(height) if height else "0"
            result = self._rpc_request("block", {"height": height_str})
            
            block = result.get("block", {})
            if not block:
                return None
            
            header = block.get("header", {})
            block_height = int(header.get("height", 0))
            block_hash = header.get("hash", "")
            
            # Parse timestamp
            timestamp_str = header.get("time", "")
            timestamp = None
            if timestamp_str:
                try:
                    # Parse RFC3339 timestamp
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                except Exception:
                    pass
            
            # Count transactions
            txs = block.get("data", {}).get("txs", [])
            tx_count = len(txs) if txs else 0
            
            block_data = {
                "height": block_height,
                "hash": block_hash,
                "timestamp": timestamp.isoformat() if timestamp else None,
                "tx_count": tx_count,
            }
            
            # Cache the result (only for specific heights, not latest)
            if use_cache and height is not None:
                try:
                    from .cache import get_cache_manager
                    import asyncio
                    
                    cache = get_cache_manager()
                    cache_key = f"blockchain:block:{height}"
                    
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    if not loop.is_running():
                        loop.run_until_complete(
                            cache.set(cache_key, block_data, BLOCK_CACHE_TTL)
                        )
                except Exception as e:
                    logger.debug(f"Failed to cache block {height}: {e}")
            
            return block_data
        except Exception as e:
            logger.error(f"Failed to get block {height}: {e}")
            return None
    
    def get_recent_blocks(self, limit: int = 10, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Get recent blocks with caching optimization.
        
        Uses cache for individual blocks to avoid redundant RPC calls.
        The list itself is also cached for a short period.
        
        Args:
            limit: Maximum number of blocks to return
            use_cache: Whether to use cache (default: True)
            
        Returns:
            List of block data
        """
        # Try cache for the full list first
        if use_cache:
            try:
                from .cache import get_cache_manager
                from .cache_keys import recent_blocks_key
                import asyncio
                
                cache = get_cache_manager()
                cache_key = recent_blocks_key(limit)
                
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                if not loop.is_running():
                    cached = loop.run_until_complete(cache.get(cache_key))
                    if cached and isinstance(cached, list):
                        logger.debug(f"Cache hit for recent blocks (limit={limit})")
                        return cached
            except Exception as e:
                logger.debug(f"Cache check failed for recent blocks, using RPC: {e}")
        
        try:
            latest_height = self.get_latest_block_height(use_cache=use_cache)
            if latest_height is None:
                return []
            
            blocks = []
            # Get blocks from latest down to (latest - limit + 1)
            # Individual blocks are cached, so repeated calls are fast
            for i in range(min(limit, latest_height)):
                height = latest_height - i
                block = self.get_block(height, use_cache=use_cache)
                if block:
                    blocks.append(block)
                else:
                    break  # Stop if we can't get a block
            
            # Cache the full list
            if use_cache and blocks:
                try:
                    from .cache import get_cache_manager
                    from .cache_keys import recent_blocks_key
                    import asyncio
                    
                    cache = get_cache_manager()
                    cache_key = recent_blocks_key(limit)
                    
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    if not loop.is_running():
                        loop.run_until_complete(
                            cache.set(cache_key, blocks, RECENT_BLOCKS_CACHE_TTL)
                        )
                except Exception as e:
                    logger.debug(f"Failed to cache recent blocks list: {e}")
            
            return blocks
        except Exception as e:
            logger.error(f"Failed to get recent blocks: {e}")
            return []


# Global client instance
_rpc_client: Optional[BlockchainRPCClient] = None


def get_blockchain_rpc_client() -> BlockchainRPCClient:
    """Get global blockchain RPC client instance."""
    global _rpc_client
    if _rpc_client is None:
        _rpc_client = BlockchainRPCClient()
    return _rpc_client


"""
WebSocket Manager

Manages WebSocket connections for real-time updates.
Includes authentication for protected channels.
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List, Set, Callable, Any, Optional
import json
import logging
import asyncio
import os
import secrets
import hashlib
from datetime import datetime, timedelta
from .graceful_shutdown import get_graceful_shutdown

logger = logging.getLogger(__name__)


# Channel security configuration
PUBLIC_CHANNELS = {"network_status", "block_updates"}
PROTECTED_CHANNELS = {"miner_stats", "training_metrics", "serving", "admin"}


class WebSocketAuthenticator:
    """
    WebSocket authentication handler.
    
    Validates tokens for protected channels to prevent unauthorized access
    to sensitive data like miner statistics and training metrics.
    """
    
    def __init__(self):
        # Token cache: token_hash -> (wallet_address, expires_at)
        self._token_cache: Dict[str, tuple] = {}
        # Cleanup interval
        self._cleanup_interval = 300  # 5 minutes
    
    def generate_ws_token(self, wallet_address: str, expires_in_seconds: int = 3600) -> str:
        """
        Generate a WebSocket authentication token.
        
        Args:
            wallet_address: Wallet address to associate with token
            expires_in_seconds: Token validity period (default: 1 hour)
            
        Returns:
            Token string
        """
        token = f"ws_{secrets.token_urlsafe(32)}"
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        expires_at = datetime.now() + timedelta(seconds=expires_in_seconds)
        
        self._token_cache[token_hash] = (wallet_address, expires_at)
        
        logger.debug(f"Generated WebSocket token for {wallet_address}, expires at {expires_at}")
        return token
    
    async def validate_token(self, token: str) -> Optional[str]:
        """
        Validate a WebSocket token.
        
        Args:
            token: Token to validate
            
        Returns:
            Wallet address if valid, None otherwise
        """
        if not token:
            return None
        
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        
        if token_hash not in self._token_cache:
            # Try to validate against API key
            return await self._validate_api_key_as_token(token)
        
        wallet_address, expires_at = self._token_cache[token_hash]
        
        if datetime.now() > expires_at:
            # Token expired
            del self._token_cache[token_hash]
            return None
        
        return wallet_address
    
    async def _validate_api_key_as_token(self, token: str) -> Optional[str]:
        """
        Validate an API key as WebSocket token.
        
        This allows users to use their API key for WebSocket authentication.
        """
        try:
            from .database_async import AsyncDatabase
            from .config_manager import get_config_manager
            
            config = get_config_manager().load()
            db = AsyncDatabase(db_path=config.database_path)
            await db.connect()
            
            api_key_info = await db.validate_api_key(token)
            await db.close()
            
            if api_key_info and api_key_info.get("is_active"):
                return api_key_info.get("wallet_address")
            
            return None
        except Exception as e:
            logger.warning(f"Failed to validate API key as WebSocket token: {e}")
            return None
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a WebSocket token."""
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        if token_hash in self._token_cache:
            del self._token_cache[token_hash]
            return True
        return False
    
    def cleanup_expired_tokens(self):
        """Remove expired tokens from cache."""
        now = datetime.now()
        expired = [
            token_hash for token_hash, (_, expires_at) in self._token_cache.items()
            if now > expires_at
        ]
        for token_hash in expired:
            del self._token_cache[token_hash]
        
        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired WebSocket tokens")


# Global authenticator instance
_ws_authenticator: Optional[WebSocketAuthenticator] = None


def get_ws_authenticator() -> WebSocketAuthenticator:
    """Get the global WebSocket authenticator instance."""
    global _ws_authenticator
    if _ws_authenticator is None:
        _ws_authenticator = WebSocketAuthenticator()
    return _ws_authenticator


class ConnectionManager:
    """Manages WebSocket connections with authentication support."""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.disconnect_callbacks: List[Callable[[str], Any]] = []
        # Track authenticated connections: websocket -> wallet_address
        self._authenticated_connections: Dict[WebSocket, str] = {}
    
    def register_disconnect_callback(self, callback: Callable[[str], Any]):
        """Register a callback to be called when a WebSocket disconnects.
        
        Args:
            callback: Async or sync function that takes channel name as parameter
        """
        self.disconnect_callbacks.append(callback)
    
    async def connect(
        self,
        websocket: WebSocket,
        channel: str,
        token: Optional[str] = None
    ) -> bool:
        """
        Connect a WebSocket to a channel with authentication.
        
        Args:
            websocket: WebSocket connection
            channel: Channel to connect to
            token: Authentication token (required for protected channels)
            
        Returns:
            True if connection successful, False otherwise
        """
        # Check if channel requires authentication
        if channel in PROTECTED_CHANNELS:
            if not token:
                logger.warning(f"WebSocket connection to protected channel '{channel}' rejected: no token provided")
                await websocket.close(code=4001, reason="Authentication required for this channel")
                return False
            
            # Validate token
            authenticator = get_ws_authenticator()
            wallet_address = await authenticator.validate_token(token)
            
            if not wallet_address:
                logger.warning(f"WebSocket connection to protected channel '{channel}' rejected: invalid token")
                await websocket.close(code=4003, reason="Invalid or expired token")
                return False
            
            # Store authenticated connection
            self._authenticated_connections[websocket] = wallet_address
            logger.info(f"Authenticated WebSocket connection to '{channel}' for wallet {wallet_address}")
        
        await websocket.accept()
        
        if channel not in self.active_connections:
            self.active_connections[channel] = set()
        
        self.active_connections[channel].add(websocket)
        logger.info(f"WebSocket connected to channel: {channel}")
        return True
    
    def disconnect(self, websocket: WebSocket, channel: str):
        """Disconnect a WebSocket from a channel."""
        if channel in self.active_connections:
            self.active_connections[channel].discard(websocket)
            if not self.active_connections[channel]:
                del self.active_connections[channel]
        
        # Remove from authenticated connections
        if websocket in self._authenticated_connections:
            del self._authenticated_connections[websocket]
        
        logger.info(f"WebSocket disconnected from channel: {channel}")
        
        # Trigger disconnect callbacks (event-driven cleanup)
        for callback in self.disconnect_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(channel))
                else:
                    callback(channel)
            except Exception as e:
                logger.warning(f"Error in disconnect callback: {e}")
    
    def get_connection_wallet(self, websocket: WebSocket) -> Optional[str]:
        """Get the wallet address associated with an authenticated connection."""
        return self._authenticated_connections.get(websocket)
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to a specific WebSocket."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
    
    async def broadcast(self, message: dict, channel: str):
        """Broadcast message to all connections in a channel."""
        if channel not in self.active_connections:
            return
        
        disconnected = set()
        for connection in self.active_connections[channel]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.add(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection, channel)
    
    async def broadcast_to_wallet(self, message: dict, channel: str, wallet_address: str):
        """Broadcast message only to connections authenticated with specific wallet."""
        if channel not in self.active_connections:
            return
        
        for connection in self.active_connections[channel]:
            if self._authenticated_connections.get(connection) == wallet_address:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error sending to wallet {wallet_address}: {e}")


# Global connection manager
connection_manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket, channel: str, token: Optional[str] = None):
    """WebSocket endpoint handler with authentication, graceful shutdown support and heartbeat."""
    graceful_shutdown = get_graceful_shutdown()
    
    # Connect with authentication
    connected = await connection_manager.connect(websocket, channel, token)
    if not connected:
        return  # Connection was rejected (authentication failed)
    
    # Heartbeat/ping-pong mechanism
    last_pong = asyncio.get_event_loop().time()
    heartbeat_interval = int(os.getenv("BACKEND_WS_HEARTBEAT_INTERVAL", "30"))  # Send ping every N seconds
    pong_timeout = int(os.getenv("BACKEND_WS_PONG_TIMEOUT", "60"))  # Close connection if no pong received in N seconds
    
    async def send_heartbeat():
        """Send periodic heartbeat (ping) to keep connection alive."""
        while True:
            await asyncio.sleep(heartbeat_interval)
            try:
                await websocket.send_json({"type": "ping", "timestamp": asyncio.get_event_loop().time()})
            except Exception as e:
                logger.debug(f"Heartbeat send error on {channel}: {e}")
                break
    
    # Start heartbeat task
    heartbeat_task = asyncio.create_task(send_heartbeat())
    
    try:
        while True:
            # Check if shutdown has been requested
            if graceful_shutdown.is_shutdown_requested():
                logger.info(f"Shutdown requested, closing WebSocket connection on {channel}")
                await websocket.close(code=1001, reason="Server shutting down")
                break
            
            # Check pong timeout
            current_time = asyncio.get_event_loop().time()
            if current_time - last_pong > pong_timeout:
                logger.warning(f"No pong received for {pong_timeout}s on {channel}, closing connection")
                await websocket.close(code=1001, reason="Pong timeout")
                break
            
            # Keep connection alive and handle incoming messages
            # Use asyncio.wait_for to allow checking shutdown status
            try:
                shutdown_check_interval = float(os.getenv("BACKEND_WS_SHUTDOWN_CHECK_INTERVAL", "1.0"))
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=shutdown_check_interval  # Check shutdown status every N seconds
                )
                try:
                    message = json.loads(data)
                    # Handle pong messages
                    if message.get("type") == "pong":
                        last_pong = current_time
                        logger.debug(f"Received pong on {channel}")
                    # Handle authentication refresh
                    elif message.get("type") == "auth_refresh":
                        new_token = message.get("token")
                        if new_token and channel in PROTECTED_CHANNELS:
                            authenticator = get_ws_authenticator()
                            wallet = await authenticator.validate_token(new_token)
                            if wallet:
                                connection_manager._authenticated_connections[websocket] = wallet
                                await websocket.send_json({"type": "auth_refreshed", "success": True})
                            else:
                                await websocket.send_json({"type": "auth_refreshed", "success": False})
                    else:
                        # Handle other client messages if needed
                        logger.debug(f"Received message on {channel}: {message}")
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received on {channel}")
            except asyncio.TimeoutError:
                # Timeout is expected - continue loop to check shutdown status
                continue
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected from {channel}")
        connection_manager.disconnect(websocket, channel)
    except Exception as e:
        logger.error(f"WebSocket error on {channel}: {e}")
        connection_manager.disconnect(websocket, channel)
    finally:
        # Cancel heartbeat task
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        # Ensure connection is cleaned up
        connection_manager.disconnect(websocket, channel)


async def broadcast_miner_stats(stats: dict):
    """Broadcast miner statistics to all connected clients."""
    await connection_manager.broadcast({
        "type": "miner_stats",
        "data": stats,
    }, "miner_stats")


async def broadcast_training_metrics(metrics: dict):
    """Broadcast training metrics to all connected clients."""
    await connection_manager.broadcast({
        "type": "training_metrics",
        "data": metrics,
    }, "training_metrics")


async def broadcast_network_status(status: dict):
    """Broadcast network status to all connected clients."""
    await connection_manager.broadcast({
        "type": "network_status",
        "data": status,
    }, "network_status")


async def broadcast_block_update(block: dict):
    """Broadcast new block to all connected clients."""
    await connection_manager.broadcast({
        "type": "block_update",
        "data": block,
    }, "block_updates")



async def broadcast_notification(notification: dict, wallet_address: Optional[str] = None):
    """
    Broadcast notification to connected clients.
    
    Args:
        notification: Notification data
        wallet_address: If provided, only send to this wallet's connections
    """
    message = {
        "type": "notification",
        "data": notification,
    }
    
    if wallet_address:
        # Send to specific wallet
        await connection_manager.broadcast_to_wallet(message, "notifications", wallet_address)
    else:
        # Broadcast to all
        await connection_manager.broadcast(message, "notifications")


async def broadcast_governance_update(update: dict):
    """Broadcast governance update (new proposal, vote, etc.)."""
    await connection_manager.broadcast({
        "type": "governance_update",
        "data": update,
    }, "governance")


async def broadcast_staking_update(update: dict, wallet_address: Optional[str] = None):
    """Broadcast staking update (delegation, rewards, etc.)."""
    message = {
        "type": "staking_update",
        "data": update,
    }
    
    if wallet_address:
        await connection_manager.broadcast_to_wallet(message, "staking", wallet_address)
    else:
        await connection_manager.broadcast(message, "staking")


async def broadcast_validator_update(update: dict):
    """Broadcast validator update (status change, slashing, etc.)."""
    await connection_manager.broadcast({
        "type": "validator_update",
        "data": update,
    }, "validators")

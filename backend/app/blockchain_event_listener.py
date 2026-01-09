"""
Blockchain Event Listener for R3MES Backend

Listens to blockchain events and triggers appropriate actions:
- Adapter approval → Trigger adapter sync
- Model upgrade → Notify frontend
- Dataset approval → Update registry
"""

import asyncio
import logging
from typing import Optional, Callable, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class BlockchainEventListener:
    """
    Listens to blockchain events via WebSocket and triggers callbacks.
    """
    
    def __init__(self, websocket_url: str):
        self.websocket_url = websocket_url
        self.running = False
        self.callbacks: Dict[str, list[Callable]] = {}
        self._task: Optional[asyncio.Task] = None
        
    def register_callback(self, event_type: str, callback: Callable):
        """Register a callback for a specific event type."""
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)
        logger.info(f"Registered callback for event type: {event_type}")
    
    async def start(self):
        """Start listening to blockchain events."""
        if self.running:
            logger.warning("Event listener already running")
            return
        
        self.running = True
        self._task = asyncio.create_task(self._listen_loop())
        logger.info(f"Blockchain event listener started (WebSocket: {self.websocket_url})")
    
    async def stop(self):
        """Stop listening to blockchain events."""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Blockchain event listener stopped")
    
    async def _listen_loop(self):
        """Main event listening loop."""
        import websockets
        
        while self.running:
            try:
                async with websockets.connect(self.websocket_url) as websocket:
                    logger.info("Connected to blockchain WebSocket")
                    
                    # Subscribe to relevant events
                    await self._subscribe_to_events(websocket)
                    
                    # Listen for events
                    async for message in websocket:
                        if not self.running:
                            break
                        
                        try:
                            await self._handle_message(message)
                        except Exception as e:
                            logger.error(f"Error handling blockchain event: {e}", exc_info=True)
                
            except asyncio.CancelledError:
                logger.info("Event listener cancelled")
                break
            except Exception as e:
                logger.error(f"WebSocket connection error: {e}")
                if self.running:
                    logger.info("Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)
    
    async def _subscribe_to_events(self, websocket):
        """Subscribe to blockchain events."""
        import json
        
        # Subscribe to adapter approval events
        subscribe_msg = json.dumps({
            "jsonrpc": "2.0",
            "method": "subscribe",
            "id": "1",
            "params": {
                "query": "tm.event='Tx' AND remes.adapter.approved EXISTS"
            }
        })
        await websocket.send(subscribe_msg)
        
        # Subscribe to model upgrade events
        subscribe_msg = json.dumps({
            "jsonrpc": "2.0",
            "method": "subscribe",
            "id": "2",
            "params": {
                "query": "tm.event='Tx' AND remes.model.upgraded EXISTS"
            }
        })
        await websocket.send(subscribe_msg)
        
        # Subscribe to dataset approval events
        subscribe_msg = json.dumps({
            "jsonrpc": "2.0",
            "method": "subscribe",
            "id": "3",
            "params": {
                "query": "tm.event='Tx' AND remes.dataset.approved EXISTS"
            }
        })
        await websocket.send(subscribe_msg)
        
        logger.info("Subscribed to blockchain events")
    
    async def _handle_message(self, message: str):
        """Handle incoming blockchain event message."""
        import json
        
        try:
            data = json.loads(message)
            
            # Extract event type and data
            if "result" in data and "events" in data["result"]:
                events = data["result"]["events"]
                
                for event_key, event_values in events.items():
                    if event_key.startswith("remes.adapter.approved"):
                        await self._trigger_callbacks("adapter_approved", {
                            "adapter_id": event_values.get("adapter_id"),
                            "ipfs_hash": event_values.get("ipfs_hash"),
                            "checksum": event_values.get("checksum"),
                            "timestamp": datetime.utcnow().isoformat()
                        })
                    
                    elif event_key.startswith("remes.model.upgraded"):
                        await self._trigger_callbacks("model_upgraded", {
                            "model_name": event_values.get("model_name"),
                            "version": event_values.get("version"),
                            "ipfs_hash": event_values.get("ipfs_hash"),
                            "timestamp": datetime.utcnow().isoformat()
                        })
                    
                    elif event_key.startswith("remes.dataset.approved"):
                        await self._trigger_callbacks("dataset_approved", {
                            "dataset_id": event_values.get("dataset_id"),
                            "ipfs_hash": event_values.get("ipfs_hash"),
                            "timestamp": datetime.utcnow().isoformat()
                        })
        
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse blockchain event message: {message[:100]}")
        except Exception as e:
            logger.error(f"Error processing blockchain event: {e}", exc_info=True)
    
    async def _trigger_callbacks(self, event_type: str, event_data: Dict[str, Any]):
        """Trigger all registered callbacks for an event type."""
        if event_type in self.callbacks:
            logger.info(f"Triggering {len(self.callbacks[event_type])} callbacks for event: {event_type}")
            
            for callback in self.callbacks[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event_data)
                    else:
                        callback(event_data)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}", exc_info=True)


# Singleton instance
_event_listener: Optional[BlockchainEventListener] = None


def get_event_listener(websocket_url: Optional[str] = None) -> BlockchainEventListener:
    """Get or create the blockchain event listener singleton."""
    global _event_listener
    
    if _event_listener is None:
        if websocket_url is None:
            import os
            from .config_manager import get_config_manager
            config = get_config_manager().load()
            
            # Get WebSocket URL from config or environment
            websocket_url = os.getenv(
                "BLOCKCHAIN_WEBSOCKET_URL",
                config.blockchain_config.get("websocket_endpoint", "ws://localhost:26657/websocket")
            )
        
        _event_listener = BlockchainEventListener(websocket_url)
    
    return _event_listener


async def init_event_listener_with_callbacks(adapter_sync_service=None):
    """
    Initialize event listener and register callbacks.
    
    Args:
        adapter_sync_service: AdapterSyncService instance for hot-reload
    """
    event_listener = get_event_listener()
    
    # Register adapter approval callback
    if adapter_sync_service:
        async def on_adapter_approved(event_data: Dict[str, Any]):
            logger.info(f"Adapter approved event received: {event_data}")
            try:
                # Trigger immediate sync for the new adapter
                await adapter_sync_service.sync_single_adapter(
                    adapter_id=event_data.get("adapter_id"),
                    ipfs_hash=event_data.get("ipfs_hash"),
                    checksum=event_data.get("checksum")
                )
                logger.info(f"✅ Adapter {event_data.get('adapter_id')} synced successfully")
            except Exception as e:
                logger.error(f"Failed to sync adapter on approval event: {e}")
        
        event_listener.register_callback("adapter_approved", on_adapter_approved)
    
    # Register model upgrade callback
    async def on_model_upgraded(event_data: Dict[str, Any]):
        logger.info(f"Model upgraded event received: {event_data}")
        # Notify frontend via WebSocket
        try:
            from .websocket_manager import connection_manager
            await connection_manager.broadcast_to_channel(
                "system",
                {
                    "type": "model_upgraded",
                    "data": event_data
                }
            )
        except Exception as e:
            logger.error(f"Failed to broadcast model upgrade event: {e}")
    
    event_listener.register_callback("model_upgraded", on_model_upgraded)
    
    # Register dataset approval callback
    async def on_dataset_approved(event_data: Dict[str, Any]):
        logger.info(f"Dataset approved event received: {event_data}")
        # Notify frontend via WebSocket
        try:
            from .websocket_manager import connection_manager
            await connection_manager.broadcast_to_channel(
                "system",
                {
                    "type": "dataset_approved",
                    "data": event_data
                }
            )
        except Exception as e:
            logger.error(f"Failed to broadcast dataset approval event: {e}")
    
    event_listener.register_callback("dataset_approved", on_dataset_approved)
    
    # Start listening
    await event_listener.start()
    
    return event_listener

"""
Unit tests for BlockchainEventListener

Tests WebSocket event listening, callback execution, and reconnection logic.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from backend.app.blockchain_event_listener import (
    BlockchainEventListener,
    get_event_listener,
    init_event_listener_with_callbacks,
)


@pytest.fixture
def event_listener():
    """Create BlockchainEventListener for testing."""
    return BlockchainEventListener(websocket_url="ws://localhost:26657/websocket")


@pytest.fixture
def mock_adapter_sync_service():
    """Mock AdapterSyncService for testing."""
    service = Mock()
    service.sync_single_adapter = AsyncMock(return_value=True)
    return service


class TestBlockchainEventListener:
    """Test suite for BlockchainEventListener."""
    
    def test_initialization(self, event_listener):
        """Test event listener initialization."""
        assert event_listener.websocket_url == "ws://localhost:26657/websocket"
        assert event_listener.running is False
        assert len(event_listener.callbacks) == 0
    
    def test_register_callback(self, event_listener):
        """Test callback registration."""
        callback = Mock()
        event_listener.register_callback("test_event", callback)
        
        assert "test_event" in event_listener.callbacks
        assert callback in event_listener.callbacks["test_event"]
    
    def test_register_multiple_callbacks(self, event_listener):
        """Test registering multiple callbacks for same event."""
        callback1 = Mock()
        callback2 = Mock()
        
        event_listener.register_callback("test_event", callback1)
        event_listener.register_callback("test_event", callback2)
        
        assert len(event_listener.callbacks["test_event"]) == 2
    
    @pytest.mark.asyncio
    async def test_trigger_callbacks_sync(self, event_listener):
        """Test triggering synchronous callbacks."""
        callback = Mock()
        event_listener.register_callback("test_event", callback)
        
        event_data = {"key": "value"}
        await event_listener._trigger_callbacks("test_event", event_data)
        
        callback.assert_called_once_with(event_data)
    
    @pytest.mark.asyncio
    async def test_trigger_callbacks_async(self, event_listener):
        """Test triggering asynchronous callbacks."""
        callback = AsyncMock()
        event_listener.register_callback("test_event", callback)
        
        event_data = {"key": "value"}
        await event_listener._trigger_callbacks("test_event", event_data)
        
        callback.assert_called_once_with(event_data)
    
    @pytest.mark.asyncio
    async def test_trigger_callbacks_error_handling(self, event_listener):
        """Test callback error handling."""
        def failing_callback(data):
            raise Exception("Callback error")
        
        callback_success = Mock()
        
        event_listener.register_callback("test_event", failing_callback)
        event_listener.register_callback("test_event", callback_success)
        
        event_data = {"key": "value"}
        # Should not raise exception
        await event_listener._trigger_callbacks("test_event", event_data)
        
        # Second callback should still be called
        callback_success.assert_called_once_with(event_data)
    
    @pytest.mark.asyncio
    async def test_handle_adapter_approved_event(self, event_listener):
        """Test handling adapter approved event."""
        callback = AsyncMock()
        event_listener.register_callback("adapter_approved", callback)
        
        message = json.dumps({
            "result": {
                "events": {
                    "remes.adapter.approved": {
                        "adapter_id": "test-adapter",
                        "ipfs_hash": "QmTest123",
                        "checksum": "abc123",
                    }
                }
            }
        })
        
        await event_listener._handle_message(message)
        
        callback.assert_called_once()
        call_args = callback.call_args[0][0]
        assert call_args["adapter_id"] == "test-adapter"
        assert call_args["ipfs_hash"] == "QmTest123"
    
    @pytest.mark.asyncio
    async def test_handle_model_upgraded_event(self, event_listener):
        """Test handling model upgraded event."""
        callback = AsyncMock()
        event_listener.register_callback("model_upgraded", callback)
        
        message = json.dumps({
            "result": {
                "events": {
                    "remes.model.upgraded": {
                        "model_name": "bitnet-b1.58",
                        "version": "2.0.0",
                        "ipfs_hash": "QmModel456",
                    }
                }
            }
        })
        
        await event_listener._handle_message(message)
        
        callback.assert_called_once()
        call_args = callback.call_args[0][0]
        assert call_args["model_name"] == "bitnet-b1.58"
        assert call_args["version"] == "2.0.0"
    
    @pytest.mark.asyncio
    async def test_handle_dataset_approved_event(self, event_listener):
        """Test handling dataset approved event."""
        callback = AsyncMock()
        event_listener.register_callback("dataset_approved", callback)
        
        message = json.dumps({
            "result": {
                "events": {
                    "remes.dataset.approved": {
                        "dataset_id": "test-dataset",
                        "ipfs_hash": "QmDataset789",
                    }
                }
            }
        })
        
        await event_listener._handle_message(message)
        
        callback.assert_called_once()
        call_args = callback.call_args[0][0]
        assert call_args["dataset_id"] == "test-dataset"
    
    @pytest.mark.asyncio
    async def test_handle_invalid_json(self, event_listener):
        """Test handling invalid JSON message."""
        callback = Mock()
        event_listener.register_callback("test_event", callback)
        
        # Should not raise exception
        await event_listener._handle_message("invalid json {")
        
        # Callback should not be called
        callback.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_start_stop(self, event_listener):
        """Test starting and stopping event listener."""
        with patch("websockets.connect") as mock_connect:
            mock_ws = AsyncMock()
            mock_ws.__aiter__.return_value = []
            mock_connect.return_value.__aenter__.return_value = mock_ws
            
            await event_listener.start()
            assert event_listener.running is True
            assert event_listener._task is not None
            
            await event_listener.stop()
            assert event_listener.running is False
    
    @pytest.mark.asyncio
    async def test_subscribe_to_events(self, event_listener):
        """Test subscribing to blockchain events."""
        mock_ws = AsyncMock()
        
        await event_listener._subscribe_to_events(mock_ws)
        
        # Should send 3 subscription messages
        assert mock_ws.send.call_count == 3
        
        # Verify subscription messages
        calls = mock_ws.send.call_args_list
        for call in calls:
            message = json.loads(call[0][0])
            assert message["method"] == "subscribe"
            assert "query" in message["params"]


@pytest.mark.asyncio
async def test_init_event_listener_with_callbacks(mock_adapter_sync_service):
    """Test initializing event listener with callbacks."""
    with patch("backend.app.blockchain_event_listener.get_event_listener") as mock_get:
        mock_listener = Mock()
        mock_listener.register_callback = Mock()
        mock_listener.start = AsyncMock()
        mock_get.return_value = mock_listener
        
        listener = await init_event_listener_with_callbacks(
            adapter_sync_service=mock_adapter_sync_service
        )
        
        # Should register 3 callbacks
        assert mock_listener.register_callback.call_count == 3
        
        # Should start listening
        mock_listener.start.assert_called_once()


@pytest.mark.asyncio
async def test_adapter_approved_callback_integration(mock_adapter_sync_service):
    """Test adapter approved callback integration."""
    with patch("backend.app.blockchain_event_listener.get_event_listener") as mock_get:
        mock_listener = Mock()
        registered_callbacks = {}
        
        def register_callback(event_type, callback):
            registered_callbacks[event_type] = callback
        
        mock_listener.register_callback = register_callback
        mock_listener.start = AsyncMock()
        mock_get.return_value = mock_listener
        
        await init_event_listener_with_callbacks(
            adapter_sync_service=mock_adapter_sync_service
        )
        
        # Trigger adapter approved callback
        event_data = {
            "adapter_id": "test-adapter",
            "ipfs_hash": "QmTest123",
            "checksum": "abc123",
        }
        
        await registered_callbacks["adapter_approved"](event_data)
        
        # Should trigger sync
        mock_adapter_sync_service.sync_single_adapter.assert_called_once_with(
            adapter_id="test-adapter",
            ipfs_hash="QmTest123",
            checksum="abc123",
        )


def test_get_event_listener_singleton():
    """Test event listener singleton pattern."""
    listener1 = get_event_listener("ws://test1")
    listener2 = get_event_listener("ws://test2")
    
    # Should return same instance
    assert listener1 is listener2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

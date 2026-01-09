"""
Unit tests for AdapterSyncService

Tests blockchain adapter synchronization, IPFS download, and hot-reload functionality.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import hashlib
import json

from backend.app.adapter_sync_service import (
    AdapterSyncService,
    ApprovedAdapter,
    SyncResult,
)


@pytest.fixture
def mock_model_manager():
    """Mock AIModelManager for testing."""
    manager = Mock()
    manager.load_adapter = Mock(return_value=True)
    return manager


@pytest.fixture
def adapter_sync_service(tmp_path, mock_model_manager):
    """Create AdapterSyncService with temporary directory."""
    service = AdapterSyncService(
        blockchain_rest_url="http://localhost:1317",
        ipfs_gateway="http://localhost:8080",
        adapters_dir=str(tmp_path / "adapters"),
        model_manager=mock_model_manager,
    )
    return service


@pytest.fixture
def sample_adapter():
    """Sample approved adapter for testing."""
    return ApprovedAdapter(
        adapter_id="test-adapter-1",
        name="Test Adapter",
        adapter_type="dora",
        version="1.0.0",
        ipfs_hash="QmTest123",
        checksum="a" * 64,
        size_bytes=1024 * 1024,
        domain="coding",
        description="Test adapter for coding domain",
        approved_at=1234567890,
        approval_tx_hash="0xtest",
    )


class TestAdapterSyncService:
    """Test suite for AdapterSyncService."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, adapter_sync_service):
        """Test service initialization."""
        assert adapter_sync_service.blockchain_rest_url == "http://localhost:1317"
        assert adapter_sync_service.ipfs_gateway == "http://localhost:8080"
        assert adapter_sync_service.adapters_dir.exists()
        assert adapter_sync_service._running is False
    
    @pytest.mark.asyncio
    async def test_query_approved_adapters_success(self, adapter_sync_service, sample_adapter):
        """Test successful blockchain query for approved adapters."""
        mock_response = {
            "adapters": [
                {
                    "adapter_id": sample_adapter.adapter_id,
                    "name": sample_adapter.name,
                    "adapter_type": sample_adapter.adapter_type,
                    "version": sample_adapter.version,
                    "ipfs_hash": sample_adapter.ipfs_hash,
                    "checksum": sample_adapter.checksum,
                    "size_bytes": sample_adapter.size_bytes,
                    "domain": sample_adapter.domain,
                    "description": sample_adapter.description,
                    "approved_at": sample_adapter.approved_at,
                    "approval_tx_hash": sample_adapter.approval_tx_hash,
                }
            ]
        }
        
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aenter__.return_value = mock_resp
            
            adapters = await adapter_sync_service.query_approved_adapters()
            
            assert len(adapters) == 1
            assert adapters[0].adapter_id == sample_adapter.adapter_id
            assert adapters[0].name == sample_adapter.name
    
    @pytest.mark.asyncio
    async def test_query_approved_adapters_failure(self, adapter_sync_service):
        """Test blockchain query failure handling."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_resp = AsyncMock()
            mock_resp.status = 500
            mock_get.return_value.__aenter__.return_value = mock_resp
            
            adapters = await adapter_sync_service.query_approved_adapters()
            
            assert adapters == []
    
    @pytest.mark.asyncio
    async def test_download_adapter_success(self, adapter_sync_service, sample_adapter):
        """Test successful adapter download from IPFS."""
        # Create mock content with matching checksum
        content = b"test adapter content"
        sample_adapter.checksum = hashlib.sha256(content).hexdigest()
        
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.read = AsyncMock(return_value=content)
            mock_get.return_value.__aenter__.return_value = mock_resp
            
            adapter_path = await adapter_sync_service.download_adapter_from_ipfs(sample_adapter)
            
            assert adapter_path is not None
            assert adapter_path.exists()
            assert (adapter_path / "adapter_model.bin").exists()
            assert (adapter_path / "adapter_config.json").exists()
            
            # Verify metadata
            with open(adapter_path / "adapter_config.json") as f:
                metadata = json.load(f)
            assert metadata["adapter_id"] == sample_adapter.adapter_id
            assert metadata["checksum"] == sample_adapter.checksum
    
    @pytest.mark.asyncio
    async def test_download_adapter_checksum_mismatch(self, adapter_sync_service, sample_adapter):
        """Test adapter download with checksum mismatch."""
        content = b"test adapter content"
        sample_adapter.checksum = "wrong_checksum_" + "a" * 48
        
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.read = AsyncMock(return_value=content)
            mock_get.return_value.__aenter__.return_value = mock_resp
            
            adapter_path = await adapter_sync_service.download_adapter_from_ipfs(sample_adapter)
            
            assert adapter_path is None
    
    @pytest.mark.asyncio
    async def test_hot_reload_adapter(self, adapter_sync_service, mock_model_manager, tmp_path):
        """Test adapter hot-reload."""
        adapter_id = "test-adapter"
        adapter_path = tmp_path / "test-adapter"
        adapter_path.mkdir()
        
        success = await adapter_sync_service.hot_reload_adapter(adapter_id, adapter_path)
        
        assert success is True
        mock_model_manager.load_adapter.assert_called_once_with(adapter_id, str(adapter_path))
    
    @pytest.mark.asyncio
    async def test_sync_single_adapter(self, adapter_sync_service, sample_adapter):
        """Test single adapter sync (event-driven)."""
        content = b"test adapter content"
        sample_adapter.checksum = hashlib.sha256(content).hexdigest()
        
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.read = AsyncMock(return_value=content)
            mock_get.return_value.__aenter__.return_value = mock_resp
            
            success = await adapter_sync_service.sync_single_adapter(
                adapter_id=sample_adapter.adapter_id,
                ipfs_hash=sample_adapter.ipfs_hash,
                checksum=sample_adapter.checksum,
            )
            
            assert success is True
            assert sample_adapter.adapter_id in adapter_sync_service._synced_adapters
    
    @pytest.mark.asyncio
    async def test_sync_adapters_new_adapter(self, adapter_sync_service, sample_adapter):
        """Test full sync with new adapter."""
        content = b"test adapter content"
        sample_adapter.checksum = hashlib.sha256(content).hexdigest()
        
        with patch.object(adapter_sync_service, "query_approved_adapters", return_value=[sample_adapter]):
            with patch("aiohttp.ClientSession.get") as mock_get:
                mock_resp = AsyncMock()
                mock_resp.status = 200
                mock_resp.read = AsyncMock(return_value=content)
                mock_get.return_value.__aenter__.return_value = mock_resp
                
                result = await adapter_sync_service.sync_adapters()
                
                assert result.success is True
                assert len(result.new_adapters) == 1
                assert result.new_adapters[0] == sample_adapter.adapter_id
                assert len(result.failed_adapters) == 0
    
    @pytest.mark.asyncio
    async def test_sync_adapters_already_synced(self, adapter_sync_service, sample_adapter):
        """Test sync with already synced adapter."""
        # Pre-populate synced adapters
        adapter_sync_service._synced_adapters[sample_adapter.adapter_id] = sample_adapter
        
        with patch.object(adapter_sync_service, "query_approved_adapters", return_value=[sample_adapter]):
            result = await adapter_sync_service.sync_adapters()
            
            assert result.success is True
            assert len(result.new_adapters) == 0
            assert len(result.updated_adapters) == 0
    
    @pytest.mark.asyncio
    async def test_periodic_sync_start_stop(self, adapter_sync_service):
        """Test periodic sync start and stop."""
        with patch.object(adapter_sync_service, "sync_adapters", return_value=SyncResult(True, [], [], [])):
            await adapter_sync_service.start_periodic_sync(interval_seconds=1)
            
            assert adapter_sync_service._running is True
            assert adapter_sync_service._sync_task is not None
            
            # Wait a bit for at least one sync
            await asyncio.sleep(1.5)
            
            await adapter_sync_service.stop_periodic_sync()
            
            assert adapter_sync_service._running is False
    
    @pytest.mark.asyncio
    async def test_get_synced_adapters(self, adapter_sync_service, sample_adapter):
        """Test getting synced adapters."""
        adapter_sync_service._synced_adapters[sample_adapter.adapter_id] = sample_adapter
        
        synced = adapter_sync_service.get_synced_adapters()
        
        assert len(synced) == 1
        assert sample_adapter.adapter_id in synced
    
    @pytest.mark.asyncio
    async def test_get_adapter_path(self, adapter_sync_service, tmp_path):
        """Test getting adapter path."""
        adapter_id = "test-adapter"
        adapter_path = adapter_sync_service.adapters_dir / adapter_id
        adapter_path.mkdir(parents=True)
        
        path = adapter_sync_service.get_adapter_path(adapter_id)
        
        assert path is not None
        assert path.exists()
        
        # Non-existent adapter
        path = adapter_sync_service.get_adapter_path("non-existent")
        assert path is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

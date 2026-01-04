"""
Integration Tests for ServingEngine ↔ InferencePipeline

Tests the production integration between ServingEngine and InferencePipeline:
- Pipeline initialization
- Health/readiness probes
- Inference request processing
- Graceful shutdown
- Metrics collection

Note: These tests require external dependencies (ipfshttpclient, pycryptodome, ecdsa)
Tests are skipped if dependencies are not available.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import torch
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Skip tests if dependencies are not available
def check_dependencies():
    """Check if required dependencies are available."""
    try:
        import ipfshttpclient
        import Crypto
        import ecdsa
        return True
    except ImportError:
        return False


DEPS_AVAILABLE = check_dependencies()
skip_if_no_deps = pytest.mark.skipif(
    not DEPS_AVAILABLE,
    reason="Required dependencies (ipfshttpclient, pycryptodome, ecdsa) not available"
)


class TestServingEngineInitialization:
    """Tests for ServingEngine initialization."""
    
    @pytest.fixture
    def mock_blockchain_client(self):
        """Create mock blockchain client."""
        client = Mock()
        client.update_serving_node_status.return_value = {"success": True}
        client.get_pending_inference_requests.return_value = []
        return client
    
    @pytest.fixture
    def mock_ipfs_client(self):
        """Create mock IPFS client."""
        client = Mock()
        client.get.return_value = "/tmp/test_model"
        client.add_bytes.return_value = "QmTestHash123"
        return client

    @skip_if_no_deps
    def test_engine_initialization(self, mock_blockchain_client, mock_ipfs_client):
        """Test ServingEngine basic initialization."""
        from r3mes.serving import engine as engine_module
        
        with patch.object(engine_module, 'BlockchainClient', return_value=mock_blockchain_client), \
             patch.object(engine_module, 'IPFSClient', return_value=mock_ipfs_client):
            
            from r3mes.serving.engine import ServingEngine, EngineState
            
            engine = ServingEngine(
                private_key="a" * 64,
                blockchain_url="localhost:9090",
                chain_id="test-chain",
            )
            
            assert engine._health.state == EngineState.INITIALIZING
            assert engine._health.is_healthy is True
            assert engine._health.is_ready is False
    
    @skip_if_no_deps
    def test_engine_with_pipeline_config(self, mock_blockchain_client, mock_ipfs_client):
        """Test ServingEngine with custom pipeline configuration."""
        from r3mes.serving import engine as engine_module
        
        with patch.object(engine_module, 'BlockchainClient', return_value=mock_blockchain_client), \
             patch.object(engine_module, 'IPFSClient', return_value=mock_ipfs_client):
            
            from r3mes.serving.engine import ServingEngine
            
            engine = ServingEngine(
                private_key="b" * 64,
                enable_rag=False,
                vram_capacity_mb=1024,
                ram_capacity_mb=4096,
            )
            
            assert engine._pipeline_config.enable_rag is False
            assert engine._pipeline_config.vram_capacity_mb == 1024


class TestServingEngineHealth:
    """Tests for health and readiness probes."""
    
    @pytest.fixture
    def engine_with_mocks(self):
        """Create engine with mocked dependencies."""
        if not DEPS_AVAILABLE:
            pytest.skip("Required dependencies not available")
        
        from r3mes.serving import engine as engine_module
        
        with patch.object(engine_module, 'BlockchainClient') as mock_bc, \
             patch.object(engine_module, 'IPFSClient') as mock_ipfs:
            
            mock_bc.return_value = Mock()
            mock_ipfs.return_value = Mock()
            
            from r3mes.serving.engine import ServingEngine
            engine = ServingEngine(private_key="c" * 64, blockchain_url="localhost:9090")
            yield engine
    
    @skip_if_no_deps
    def test_health_check_initial(self, engine_with_mocks):
        """Test initial health check."""
        health = engine_with_mocks.get_health()
        
        assert health["is_healthy"] is True
        assert health["is_ready"] is False
        assert health["pipeline_initialized"] is False
    
    @skip_if_no_deps
    def test_is_healthy(self, engine_with_mocks):
        """Test is_healthy method."""
        assert engine_with_mocks.is_healthy() is True
        engine_with_mocks._shutdown_requested = True
        assert engine_with_mocks.is_healthy() is False
    
    @skip_if_no_deps
    def test_is_ready(self, engine_with_mocks):
        """Test is_ready method."""
        assert engine_with_mocks.is_ready() is False
        engine_with_mocks._health.is_ready = True
        engine_with_mocks._health.pipeline_initialized = True
        assert engine_with_mocks.is_ready() is True
    
    @skip_if_no_deps
    def test_metrics_collection(self, engine_with_mocks):
        """Test metrics collection."""
        engine_with_mocks._health.total_requests = 100
        engine_with_mocks._health.successful_requests = 95
        engine_with_mocks._health.failed_requests = 5
        engine_with_mocks._health.avg_latency_ms = 150.5
        
        metrics = engine_with_mocks.get_metrics()
        
        assert metrics["serving_engine_requests_total"] == 100
        assert metrics["serving_engine_requests_success"] == 95
        assert metrics["serving_engine_latency_avg_ms"] == 150.5


class TestServingEnginePipeline:
    """Tests for pipeline integration."""
    
    @pytest.fixture
    def mock_pipeline(self):
        """Create mock pipeline."""
        pipeline = Mock()
        pipeline.initialize = AsyncMock(return_value=True)
        pipeline.load_model = AsyncMock(return_value=True)
        pipeline.warmup = AsyncMock()
        pipeline.shutdown = AsyncMock()
        pipeline.run = AsyncMock(return_value=Mock(
            success=True,
            output=torch.randn(1, 10),
            text="Test response",
            metrics=Mock(to_dict=lambda: {"total_time_ms": 100}),
            experts_used=[("medical_dora", 0.7), ("general_dora", 0.3)],
            rag_context="Test context",
        ))
        pipeline.get_stats.return_value = {
            "total_requests": 10,
            "error_rate": 0.1,
            "cache": {"vram": {"used_mb": 100}, "ram": {"used_mb": 500}, "stats": {"hits": 8, "misses": 2}},
        }
        return pipeline
    
    @skip_if_no_deps
    @pytest.mark.asyncio
    async def test_initialize_pipeline(self, mock_pipeline):
        """Test pipeline initialization."""
        from r3mes.serving import engine as engine_module
        
        with patch.object(engine_module, 'BlockchainClient'), \
             patch.object(engine_module, 'IPFSClient'), \
             patch.object(engine_module, 'create_pipeline', return_value=mock_pipeline):
            
            from r3mes.serving.engine import ServingEngine
            engine = ServingEngine(private_key="d" * 64)
            
            success = await engine.initialize_pipeline()
            
            assert success is True
            assert engine._health.pipeline_initialized is True
            mock_pipeline.initialize.assert_called_once()
    
    @skip_if_no_deps
    @pytest.mark.asyncio
    async def test_load_model_with_pipeline(self, mock_pipeline):
        """Test model loading through pipeline."""
        from r3mes.serving import engine as engine_module
        
        mock_ipfs = Mock()
        mock_ipfs.get.return_value = "/tmp/model_path"
        
        with patch.object(engine_module, 'BlockchainClient'), \
             patch.object(engine_module, 'IPFSClient', return_value=mock_ipfs), \
             patch.object(engine_module, 'create_pipeline', return_value=mock_pipeline):
            
            from r3mes.serving.engine import ServingEngine, EngineState
            engine = ServingEngine(private_key="e" * 64, model_ipfs_hash="QmTestModelHash")
            
            success = await engine.load_model()
            
            assert success is True
            assert engine._health.model_loaded is True
            assert engine._health.state == EngineState.READY


class TestServingEngineInference:
    """Tests for inference processing."""
    
    @pytest.fixture
    def mock_pipeline_for_inference(self):
        """Create mock pipeline for inference tests."""
        pipeline = Mock()
        pipeline.initialize = AsyncMock(return_value=True)
        pipeline.run = AsyncMock(return_value=Mock(
            success=True,
            output=torch.randn(1, 10),
            text="Generated response text",
            metrics=Mock(to_dict=lambda: {"total_time_ms": 150.0}),
            experts_used=[("medical_dora", 0.8), ("turkish_dora", 0.2)],
            rag_context="Retrieved context",
        ))
        pipeline.get_stats.return_value = {"total_requests": 1}
        pipeline.shutdown = AsyncMock()
        return pipeline
    
    @skip_if_no_deps
    @pytest.mark.asyncio
    async def test_direct_inference_api(self, mock_pipeline_for_inference):
        """Test direct inference API."""
        from r3mes.serving import engine as engine_module
        
        with patch.object(engine_module, 'BlockchainClient'), \
             patch.object(engine_module, 'IPFSClient'), \
             patch.object(engine_module, 'create_pipeline', return_value=mock_pipeline_for_inference):
            
            from r3mes.serving.engine import ServingEngine
            engine = ServingEngine(private_key="g" * 64)
            await engine.initialize_pipeline()
            engine._health.pipeline_initialized = True
            
            result = await engine.infer(query="What is diabetes treatment?", skip_rag=False)
            
            assert result.success is True
            assert result.text == "Generated response text"
            assert len(result.experts_used) == 2


class TestServingEngineGracefulShutdown:
    """Tests for graceful shutdown."""
    
    @skip_if_no_deps
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self):
        """Test graceful shutdown sequence."""
        from r3mes.serving import engine as engine_module
        
        mock_pipeline = Mock()
        mock_pipeline.initialize = AsyncMock(return_value=True)
        mock_pipeline.shutdown = AsyncMock()
        mock_pipeline.get_stats.return_value = {}
        
        mock_bc = Mock()
        mock_bc.update_serving_node_status.return_value = {"success": True}
        
        with patch.object(engine_module, 'BlockchainClient', return_value=mock_bc), \
             patch.object(engine_module, 'IPFSClient'), \
             patch.object(engine_module, 'create_pipeline', return_value=mock_pipeline):
            
            from r3mes.serving.engine import ServingEngine
            engine = ServingEngine(private_key="i" * 64)
            await engine.initialize_pipeline()
            
            await engine._cleanup_async()
            
            assert engine._health.is_healthy is False
            assert engine._health.is_ready is False
            mock_pipeline.shutdown.assert_called_once()
    
    @skip_if_no_deps
    @pytest.mark.asyncio
    async def test_shutdown_event(self):
        """Test shutdown event handling."""
        from r3mes.serving import engine as engine_module
        
        with patch.object(engine_module, 'BlockchainClient'), \
             patch.object(engine_module, 'IPFSClient'):
            
            from r3mes.serving.engine import ServingEngine, EngineState
            engine = ServingEngine(private_key="j" * 64)
            
            engine._signal_handler(15, None)  # SIGTERM
            
            assert engine._shutdown_requested is True
            assert engine._health.state == EngineState.SHUTTING_DOWN
            assert engine._shutdown_event.is_set()


class TestServingEngineMetrics:
    """Tests for metrics and statistics."""
    
    @skip_if_no_deps
    @pytest.mark.asyncio
    async def test_latency_tracking(self):
        """Test latency statistics tracking."""
        from r3mes.serving import engine as engine_module
        
        with patch.object(engine_module, 'BlockchainClient'), \
             patch.object(engine_module, 'IPFSClient'):
            
            from r3mes.serving.engine import ServingEngine
            engine = ServingEngine(private_key="k" * 64)
            
            for latency in [100, 150, 200, 120, 180]:
                engine._update_latency_stats(latency)
            
            assert len(engine._latency_samples) == 5
            assert engine._health.avg_latency_ms == 150.0
    
    @skip_if_no_deps
    @pytest.mark.asyncio
    async def test_metrics_with_pipeline(self):
        """Test metrics collection with pipeline stats."""
        from r3mes.serving import engine as engine_module
        
        mock_pipeline = Mock()
        mock_pipeline.initialize = AsyncMock(return_value=True)
        mock_pipeline.get_stats.return_value = {
            "total_requests": 50,
            "error_rate": 0.02,
            "cache": {"vram": {"used_mb": 512}, "ram": {"used_mb": 2048}, "stats": {"hits": 45, "misses": 5}},
        }
        
        with patch.object(engine_module, 'BlockchainClient'), \
             patch.object(engine_module, 'IPFSClient'), \
             patch.object(engine_module, 'create_pipeline', return_value=mock_pipeline):
            
            from r3mes.serving.engine import ServingEngine
            engine = ServingEngine(private_key="l" * 64)
            await engine.initialize_pipeline()
            
            metrics = engine.get_metrics()
            
            assert metrics["pipeline_total_requests"] == 50
            assert metrics["cache_vram_used_mb"] == 512
            assert metrics["cache_hits"] == 45


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

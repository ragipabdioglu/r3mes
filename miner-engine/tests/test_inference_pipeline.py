"""
Tests for Inference Pipeline

Tests the full BitNet + DoRA + RAG inference pipeline.
"""

import pytest
import asyncio
import torch
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from r3mes.serving.inference_pipeline import (
    InferencePipeline,
    StreamingInferencePipeline,
    PipelineConfig,
    PipelineResult,
    PipelineMetrics,
    PipelineStage,
    create_pipeline,
    create_streaming_pipeline,
)


# Fixtures
@pytest.fixture
def pipeline_config():
    """Create test pipeline configuration."""
    return PipelineConfig(
        enable_rag=False,  # Disable RAG for basic tests
        router_strategy="keyword",  # Use keyword-only for speed
        vram_capacity_mb=1024,
        ram_capacity_mb=2048,
        disk_cache_dir=tempfile.mkdtemp(),
        max_batch_size=1,
        max_seq_length=512,
    )


@pytest.fixture
def mock_router():
    """Create mock router."""
    router = Mock()
    router.route.return_value = (
        [("general_dora", 1.0)],
        Mock(
            used_fast_path=True,
            keyword_confidence=0.9,
            total_time_ms=1.0,
        )
    )
    router.get_stats.return_value = {"total_routes": 1}
    router.reset_stats = Mock()
    return router


@pytest.fixture
def mock_cache():
    """Create mock cache."""
    cache = Mock()
    cache.get = AsyncMock(return_value=None)
    cache.put = AsyncMock()
    cache.clear = Mock()
    cache.get_usage.return_value = {"vram": {"used_mb": 0}}
    return cache


@pytest.fixture
def mock_backend():
    """Create mock backend."""
    backend = Mock()
    backend.load_model.return_value = True
    backend.load_adapter.return_value = True
    backend.unload_adapter.return_value = True
    backend.adapters = {}
    
    # Mock inference result
    backend.inference.return_value = Mock(
        output=torch.randn(1, 10, 100),
        latency_ms=50.0,
        backend_used="pytorch",
    )
    
    backend.get_capabilities.return_value = Mock(
        to_dict=lambda: {"name": "mock_backend"}
    )
    backend.get_vram_usage.return_value = {"used_mb": 0, "total_mb": 8192}
    
    return backend


@pytest.fixture
def mock_retriever():
    """Create mock retriever."""
    retriever = Mock()
    retriever.search.return_value = [
        Mock(doc_id="doc1", content="Test context", score=0.9, metadata={}),
    ]
    retriever.add_document = Mock()
    retriever.add_documents.return_value = 1
    retriever.get_stats.return_value = {"num_documents": 1}
    return retriever


class TestPipelineConfig:
    """Tests for PipelineConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PipelineConfig()
        
        assert config.enable_rag is True
        assert config.rag_top_k == 3
        assert config.router_strategy == "hybrid"
        assert config.keyword_weight == 0.3
        assert config.semantic_weight == 0.7
        assert config.fast_path_threshold == 0.85
        assert config.vram_capacity_mb == 2048
        assert config.fallback_expert == "general_dora"
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = PipelineConfig(
            enable_rag=False,
            rag_top_k=5,
            router_strategy="keyword",
            vram_capacity_mb=4096,
        )
        
        assert config.enable_rag is False
        assert config.rag_top_k == 5
        assert config.router_strategy == "keyword"
        assert config.vram_capacity_mb == 4096


class TestPipelineMetrics:
    """Tests for PipelineMetrics."""
    
    def test_default_metrics(self):
        """Test default metrics values."""
        metrics = PipelineMetrics()
        
        assert metrics.total_time_ms == 0.0
        assert metrics.rag_time_ms == 0.0
        assert metrics.routing_time_ms == 0.0
        assert metrics.adapters_loaded == []
        assert metrics.cache_hits == 0
    
    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        metrics = PipelineMetrics(
            total_time_ms=100.0,
            rag_time_ms=10.0,
            routing_time_ms=5.0,
            adapters_loaded=["expert1", "expert2"],
            cache_hits=2,
        )
        
        d = metrics.to_dict()
        
        assert d["total_time_ms"] == 100.0
        assert d["rag_time_ms"] == 10.0
        assert d["adapters_loaded"] == ["expert1", "expert2"]
        assert d["cache_hits"] == 2


class TestPipelineResult:
    """Tests for PipelineResult."""
    
    def test_successful_result(self):
        """Test successful result creation."""
        result = PipelineResult(
            output=torch.randn(1, 10),
            success=True,
            experts_used=[("expert1", 0.7), ("expert2", 0.3)],
        )
        
        assert result.success is True
        assert result.error is None
        assert len(result.experts_used) == 2
    
    def test_failed_result(self):
        """Test failed result creation."""
        result = PipelineResult(
            output=torch.tensor([]),
            success=False,
            error="Test error",
        )
        
        assert result.success is False
        assert result.error == "Test error"


class TestInferencePipeline:
    """Tests for InferencePipeline."""
    
    @pytest.mark.asyncio
    async def test_pipeline_creation(self, pipeline_config):
        """Test pipeline creation."""
        pipeline = InferencePipeline(config=pipeline_config)
        
        assert pipeline.config == pipeline_config
        assert pipeline._initialized is False
    
    @pytest.mark.asyncio
    async def test_pipeline_with_mocks(
        self, pipeline_config, mock_router, mock_cache, mock_backend
    ):
        """Test pipeline with mock components."""
        pipeline = InferencePipeline(
            config=pipeline_config,
            router=mock_router,
            cache=mock_cache,
            backend=mock_backend,
        )
        
        # Mark as initialized
        pipeline._initialized = True
        pipeline._model_loaded = True
        
        # Run inference
        result = await pipeline.run("Test query")
        
        assert result.success is True
        assert mock_router.route.called
        assert mock_backend.inference.called
    
    @pytest.mark.asyncio
    async def test_pipeline_with_rag(
        self, mock_router, mock_cache, mock_backend, mock_retriever
    ):
        """Test pipeline with RAG enabled."""
        config = PipelineConfig(enable_rag=True)
        
        pipeline = InferencePipeline(
            config=config,
            router=mock_router,
            cache=mock_cache,
            backend=mock_backend,
            retriever=mock_retriever,
        )
        
        pipeline._initialized = True
        pipeline._model_loaded = True
        
        result = await pipeline.run("Test query")
        
        assert result.success is True
        assert mock_retriever.search.called
    
    @pytest.mark.asyncio
    async def test_pipeline_skip_rag(
        self, mock_router, mock_cache, mock_backend, mock_retriever
    ):
        """Test pipeline with RAG skipped."""
        config = PipelineConfig(enable_rag=True)
        
        pipeline = InferencePipeline(
            config=config,
            router=mock_router,
            cache=mock_cache,
            backend=mock_backend,
            retriever=mock_retriever,
        )
        
        pipeline._initialized = True
        
        result = await pipeline.run("Test query", skip_rag=True)
        
        assert result.success is True
        assert not mock_retriever.search.called
    
    @pytest.mark.asyncio
    async def test_pipeline_force_experts(
        self, mock_router, mock_cache, mock_backend
    ):
        """Test pipeline with forced experts."""
        config = PipelineConfig(enable_rag=False)
        
        pipeline = InferencePipeline(
            config=config,
            router=mock_router,
            cache=mock_cache,
            backend=mock_backend,
        )
        
        pipeline._initialized = True
        
        result = await pipeline.run(
            "Test query",
            force_experts=["medical_dora", "coding_dora"]
        )
        
        assert result.success is True
        assert not mock_router.route.called  # Router should be bypassed
        assert len(result.experts_used) == 2
    
    @pytest.mark.asyncio
    async def test_pipeline_error_handling(
        self, mock_router, mock_cache, mock_backend
    ):
        """Test pipeline error handling."""
        config = PipelineConfig(enable_rag=False)
        
        # Make backend raise an error
        mock_backend.inference.side_effect = RuntimeError("Test error")
        
        pipeline = InferencePipeline(
            config=config,
            router=mock_router,
            cache=mock_cache,
            backend=mock_backend,
        )
        
        pipeline._initialized = True
        
        result = await pipeline.run("Test query")
        
        assert result.success is False
        assert "Test error" in result.error
    
    @pytest.mark.asyncio
    async def test_pipeline_metrics(
        self, mock_router, mock_cache, mock_backend
    ):
        """Test pipeline metrics collection."""
        config = PipelineConfig(enable_rag=False)
        
        pipeline = InferencePipeline(
            config=config,
            router=mock_router,
            cache=mock_cache,
            backend=mock_backend,
        )
        
        pipeline._initialized = True
        
        result = await pipeline.run("Test query")
        
        assert result.metrics.total_time_ms > 0
        assert result.metrics.routing_time_ms >= 0
        assert result.metrics.backend_used == "pytorch"
    
    @pytest.mark.asyncio
    async def test_pipeline_batch(
        self, mock_router, mock_cache, mock_backend
    ):
        """Test batch inference."""
        config = PipelineConfig(enable_rag=False)
        
        pipeline = InferencePipeline(
            config=config,
            router=mock_router,
            cache=mock_cache,
            backend=mock_backend,
        )
        
        pipeline._initialized = True
        
        queries = ["Query 1", "Query 2", "Query 3"]
        results = await pipeline.run_batch(queries)
        
        assert len(results) == 3
        assert all(r.success for r in results)
    
    @pytest.mark.asyncio
    async def test_pipeline_stats(
        self, mock_router, mock_cache, mock_backend
    ):
        """Test pipeline statistics."""
        config = PipelineConfig(enable_rag=False)
        
        pipeline = InferencePipeline(
            config=config,
            router=mock_router,
            cache=mock_cache,
            backend=mock_backend,
        )
        
        pipeline._initialized = True
        
        # Run some queries
        await pipeline.run("Query 1")
        await pipeline.run("Query 2")
        
        stats = pipeline.get_stats()
        
        assert stats["initialized"] is True
        assert stats["total_requests"] == 2
        assert "router" in stats
        assert "cache" in stats
        assert "backend" in stats
    
    @pytest.mark.asyncio
    async def test_pipeline_shutdown(
        self, mock_router, mock_cache, mock_backend
    ):
        """Test pipeline shutdown."""
        config = PipelineConfig(enable_rag=False)
        
        pipeline = InferencePipeline(
            config=config,
            router=mock_router,
            cache=mock_cache,
            backend=mock_backend,
        )
        
        pipeline._initialized = True
        
        await pipeline.shutdown()
        
        assert pipeline._initialized is False
        assert mock_cache.clear.called


class TestStreamingPipeline:
    """Tests for StreamingInferencePipeline."""
    
    @pytest.mark.asyncio
    async def test_streaming_creation(self, pipeline_config):
        """Test streaming pipeline creation."""
        pipeline = StreamingInferencePipeline(config=pipeline_config)
        
        assert isinstance(pipeline, InferencePipeline)
    
    @pytest.mark.asyncio
    async def test_streaming_fallback(
        self, mock_router, mock_cache, mock_backend
    ):
        """Test streaming fallback to non-streaming."""
        config = PipelineConfig(enable_rag=False)
        
        pipeline = StreamingInferencePipeline(
            config=config,
            router=mock_router,
            cache=mock_cache,
            backend=mock_backend,
        )
        
        pipeline._initialized = True
        
        # Collect streaming output
        outputs = []
        async for output in pipeline.run_streaming("Test query"):
            outputs.append(output)
        
        assert len(outputs) > 0


class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_create_pipeline(self):
        """Test create_pipeline factory."""
        pipeline = create_pipeline(enable_rag=False)
        
        assert isinstance(pipeline, InferencePipeline)
        assert pipeline.config.enable_rag is False
    
    def test_create_streaming_pipeline(self):
        """Test create_streaming_pipeline factory."""
        pipeline = create_streaming_pipeline(enable_rag=False)
        
        assert isinstance(pipeline, StreamingInferencePipeline)
    
    def test_create_pipeline_with_config(self):
        """Test create_pipeline with config object."""
        config = PipelineConfig(
            enable_rag=True,
            rag_top_k=10,
        )
        
        pipeline = create_pipeline(config=config)
        
        assert pipeline.config.enable_rag is True
        assert pipeline.config.rag_top_k == 10


class TestPipelineIntegration:
    """Integration tests for pipeline components."""
    
    @pytest.mark.asyncio
    async def test_cache_hit_tracking(
        self, mock_router, mock_backend
    ):
        """Test cache hit/miss tracking."""
        cache = Mock()
        cache.get = AsyncMock(side_effect=[
            Mock(),  # First call: cache hit
            None,    # Second call: cache miss
        ])
        cache.put = AsyncMock()
        cache.clear = Mock()
        cache.get_usage.return_value = {}
        
        config = PipelineConfig(enable_rag=False)
        
        # Mock router to return 2 experts
        mock_router.route.return_value = (
            [("expert1", 0.6), ("expert2", 0.4)],
            Mock(used_fast_path=False, keyword_confidence=0.7)
        )
        
        pipeline = InferencePipeline(
            config=config,
            router=mock_router,
            cache=cache,
            backend=mock_backend,
        )
        
        pipeline._initialized = True
        
        result = await pipeline.run("Test query")
        
        assert result.metrics.cache_hits == 1
        assert result.metrics.cache_misses == 1
    
    @pytest.mark.asyncio
    async def test_rag_context_augmentation(
        self, mock_router, mock_cache, mock_backend
    ):
        """Test RAG context augmentation."""
        retriever = Mock()
        retriever.search.return_value = [
            Mock(doc_id="doc1", content="Context 1", score=0.9, metadata={}),
            Mock(doc_id="doc2", content="Context 2", score=0.8, metadata={}),
        ]
        
        config = PipelineConfig(
            enable_rag=True,
            rag_top_k=2,
            rag_context_template="Context:\n{context}\n\nQuery: {query}",
        )
        
        pipeline = InferencePipeline(
            config=config,
            router=mock_router,
            cache=mock_cache,
            backend=mock_backend,
            retriever=retriever,
        )
        
        pipeline._initialized = True
        
        result = await pipeline.run("Test query")
        
        assert result.success is True
        assert result.rag_context is not None
        assert "Context 1" in result.rag_context
        assert "Context 2" in result.rag_context
    
    @pytest.mark.asyncio
    async def test_expert_weight_normalization(
        self, mock_cache, mock_backend
    ):
        """Test expert weight normalization."""
        router = Mock()
        router.route.return_value = (
            [("expert1", 0.5), ("expert2", 0.3), ("expert3", 0.2)],
            Mock(used_fast_path=False, keyword_confidence=0.6)
        )
        
        config = PipelineConfig(enable_rag=False)
        
        pipeline = InferencePipeline(
            config=config,
            router=router,
            cache=mock_cache,
            backend=mock_backend,
        )
        
        pipeline._initialized = True
        
        result = await pipeline.run("Test query")
        
        # Check that backend received the experts
        call_args = mock_backend.inference.call_args
        adapter_ids = call_args.kwargs.get("adapter_ids") or call_args[1].get("adapter_ids")
        adapter_weights = call_args.kwargs.get("adapter_weights") or call_args[1].get("adapter_weights")
        
        assert len(adapter_ids) == 3
        assert len(adapter_weights) == 3


class TestPipelineEdgeCases:
    """Edge case tests for pipeline."""
    
    @pytest.mark.asyncio
    async def test_empty_query(
        self, mock_router, mock_cache, mock_backend
    ):
        """Test handling of empty query."""
        config = PipelineConfig(enable_rag=False)
        
        pipeline = InferencePipeline(
            config=config,
            router=mock_router,
            cache=mock_cache,
            backend=mock_backend,
        )
        
        pipeline._initialized = True
        
        result = await pipeline.run("")
        
        # Should still work (router handles empty queries)
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_very_long_query(
        self, mock_router, mock_cache, mock_backend
    ):
        """Test handling of very long query."""
        config = PipelineConfig(
            enable_rag=False,
            max_seq_length=100,
        )
        
        pipeline = InferencePipeline(
            config=config,
            router=mock_router,
            cache=mock_cache,
            backend=mock_backend,
        )
        
        pipeline._initialized = True
        
        long_query = "word " * 1000  # Very long query
        result = await pipeline.run(long_query)
        
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_rag_retrieval_failure(
        self, mock_router, mock_cache, mock_backend
    ):
        """Test handling of RAG retrieval failure."""
        retriever = Mock()
        retriever.search.side_effect = Exception("RAG error")
        
        config = PipelineConfig(enable_rag=True)
        
        pipeline = InferencePipeline(
            config=config,
            router=mock_router,
            cache=mock_cache,
            backend=mock_backend,
            retriever=retriever,
        )
        
        pipeline._initialized = True
        
        # Should continue without RAG context
        result = await pipeline.run("Test query")
        
        assert result.success is True
        assert result.rag_context is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

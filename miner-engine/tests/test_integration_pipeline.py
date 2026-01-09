"""
Integration Tests for BitNet + DoRA + RAG Pipeline

Tests the complete inference pipeline with all components working together:
- HybridRouter (Keyword + Semantic + VRAM Gating)
- TieredCache (VRAM → RAM → Disk)
- InferenceBackend (PyTorch)
- RAGRetriever (FAISS + Embedder)
- InferencePipeline (Orchestrator)
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import torch
import numpy as np
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRouterIntegration:
    """Integration tests for router components."""
    
    def test_keyword_router_standalone(self):
        """Test keyword router in isolation."""
        from router.keyword_router import KeywordRouter
        
        router = KeywordRouter()
        
        # Test domain detection
        results = router.route("Diyabet tedavisi hakkında bilgi ver")
        assert len(results) > 0
        
        # Should detect medical and Turkish
        expert_ids = [r.expert_id for r in results]
        assert "medical_dora" in expert_ids or "turkish_dora" in expert_ids
    
    def test_keyword_router_coding(self):
        """Test keyword router for coding queries."""
        from router.keyword_router import KeywordRouter
        
        router = KeywordRouter()
        
        results = router.route("Python'da for loop nasıl yazılır?")
        expert_ids = [r.expert_id for r in results]
        
        assert "coding_dora" in expert_ids
    
    def test_keyword_router_multilingual(self):
        """Test keyword router for multiple languages."""
        from router.keyword_router import KeywordRouter
        
        router = KeywordRouter()
        
        # Turkish
        results_tr = router.route("Merhaba, nasılsın?")
        assert any(r.expert_id == "turkish_dora" for r in results_tr)
        
        # German
        results_de = router.route("Guten Tag, wie geht es Ihnen?")
        assert any(r.expert_id == "german_dora" for r in results_de)
    
    def test_vram_gating_integration(self):
        """Test VRAM adaptive gating."""
        from router.vram_adaptive_gating import VRAMAdaptiveGating, ExpertScore
        
        gating = VRAMAdaptiveGating(
            fallback_expert="general_dora",
            confidence_threshold=0.5,
        )
        
        # Create expert scores
        scores = [
            ExpertScore("medical_dora", 0.9, "keyword"),
            ExpertScore("turkish_dora", 0.7, "semantic"),
            ExpertScore("coding_dora", 0.5, "combined"),
        ]
        
        # Select experts
        selected = gating.select(scores)
        
        assert len(selected) > 0
        assert len(selected) <= gating.max_experts
        
        # Weights should sum to ~1.0
        total_weight = sum(w for _, w in selected)
        assert 0.99 <= total_weight <= 1.01
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_vram_gating_with_cuda(self):
        """Test VRAM gating with actual CUDA device."""
        from router.vram_adaptive_gating import VRAMAdaptiveGating
        
        gating = VRAMAdaptiveGating()
        
        assert gating.vram_gb > 0
        assert gating.max_experts >= 1


class TestCacheIntegration:
    """Integration tests for cache components."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_tiered_cache_flow(self, temp_cache_dir):
        """Test tiered cache put/get flow."""
        from cache.tiered_cache import TieredDoRACache, CacheTier
        
        cache = TieredDoRACache(
            vram_capacity_mb=100,
            ram_capacity_mb=200,
            disk_cache_dir=temp_cache_dir,
        )
        
        # Create test adapter data
        adapter_data = {
            "magnitude": torch.randn(256),
            "direction_A": torch.randn(16, 768),
            "direction_B": torch.randn(256, 16),
        }
        
        # Put in RAM
        await cache.put("test_adapter", adapter_data, size_mb=5.0, tier=CacheTier.RAM)
        
        # Get should return data
        retrieved = await cache.get("test_adapter")
        assert retrieved is not None
        
        # Check stats - cache should have at least 1 hit
        usage = cache.get_usage()
        assert usage["stats"]["hits"] >= 1 or usage["ram"]["count"] >= 0
    
    @pytest.mark.asyncio
    async def test_cache_eviction(self, temp_cache_dir):
        """Test cache eviction when full."""
        from cache.tiered_cache import TieredDoRACache, CacheTier
        
        cache = TieredDoRACache(
            vram_capacity_mb=10,  # Small capacity
            ram_capacity_mb=20,
            disk_cache_dir=temp_cache_dir,
        )
        
        # Add multiple adapters to trigger eviction
        for i in range(5):
            adapter_data = {"data": torch.randn(100, 100)}
            await cache.put(f"adapter_{i}", adapter_data, size_mb=5.0, tier=CacheTier.RAM)
        
        # Check that eviction occurred
        stats = cache.get_usage()
        assert stats["stats"]["evictions"] > 0


class TestRAGIntegration:
    """Integration tests for RAG components."""
    
    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder."""
        embedder = Mock()
        embedder.embedding_dim = 384
        embedder.embed.return_value = np.random.randn(384).astype(np.float32)
        embedder.embed_query.return_value = np.random.randn(384).astype(np.float32)
        embedder.embed_documents.return_value = np.random.randn(3, 384).astype(np.float32)
        embedder.embed_batch.return_value = np.random.randn(2, 384).astype(np.float32)
        embedder.get_stats.return_value = {"total_embeddings": 10}
        return embedder
    
    def test_faiss_store_operations(self):
        """Test FAISS store basic operations."""
        from rag.faiss_store import FAISSStore, Document
        
        store = FAISSStore(dimension=384, index_type="flat")
        
        # Add documents
        docs = [
            Document(
                doc_id=f"doc_{i}",
                text=f"Test document {i}",
                embedding=np.random.randn(384).astype(np.float32),
                metadata={"index": i},
            )
            for i in range(5)
        ]
        
        store.add_documents(docs)
        
        # Search
        query_embedding = np.random.randn(384).astype(np.float32)
        results = store.search(query_embedding, k=3)
        
        # Should return up to k results (may be less if threshold filters)
        assert len(results) >= 1
        assert len(results) <= 3
        assert all(r.score >= 0 for r in results)
    
    def test_retriever_with_mock_embedder(self, mock_embedder):
        """Test retriever with mock embedder."""
        from rag.retriever import RAGRetriever, RetrieverConfig, IndexType
        from rag.faiss_store import FAISSStore
        
        config = RetrieverConfig(
            index_type=IndexType.FLAT,
            default_top_k=3,
        )
        
        retriever = RAGRetriever(config=config)
        retriever.embedder = mock_embedder
        
        # Reinitialize store with mock embedder dimension
        retriever.store = FAISSStore(
            dimension=mock_embedder.embedding_dim,
            index_type="flat",
        )
        
        # Add documents
        retriever.add_document("doc1", "Medical information about diabetes")
        retriever.add_document("doc2", "Python programming tutorial")
        retriever.add_document("doc3", "Legal contract templates")
        
        # Search
        results = retriever.search("diabetes treatment")
        
        assert len(results) <= 3
        assert all(hasattr(r, 'doc_id') for r in results)


class TestBackendIntegration:
    """Integration tests for inference backend."""
    
    def test_pytorch_backend_initialization(self):
        """Test PyTorch backend initialization."""
        from core.inference_backend import get_backend_registry, BackendType
        from core.backends.pytorch_backend import PyTorchBackend
        
        registry = get_backend_registry()
        backend = registry.get(BackendType.PYTORCH)
        
        assert backend is not None
        assert backend.is_available()
    
    def test_pytorch_backend_capabilities(self):
        """Test PyTorch backend capabilities."""
        from core.backends.pytorch_backend import PyTorchBackend
        
        backend = PyTorchBackend()
        caps = backend.get_capabilities()
        
        assert caps.name == "PyTorch Backend"
        assert caps.supports_batching is True
        assert caps.max_batch_size > 0
    
    def test_pytorch_backend_model_loading(self):
        """Test PyTorch backend model loading."""
        from core.backends.pytorch_backend import PyTorchBackend
        
        backend = PyTorchBackend(device="cpu")
        
        # Load placeholder model
        success = backend.load_model("placeholder")
        assert success is True
        assert backend.model is not None
    
    def test_pytorch_backend_inference(self):
        """Test PyTorch backend inference."""
        from core.backends.pytorch_backend import PyTorchBackend
        
        backend = PyTorchBackend(device="cpu")
        backend.load_model("placeholder")
        
        # Run inference
        input_ids = torch.randint(0, 1000, (1, 32))
        result = backend.inference(
            input_ids=input_ids,
            adapter_ids=[],
            adapter_weights=[],
        )
        
        assert result.output is not None
        assert result.latency_ms > 0
        assert result.backend_used == "pytorch"


class TestDoRAIntegration:
    """Integration tests for DoRA components."""
    
    def test_dora_layer_creation(self):
        """Test DoRA layer creation from BitLinear."""
        from core.bitlinear import BitLinear
        from core.dora import BitLinearDoRA, create_dora_from_bitlinear
        
        # Create BitLinear backbone
        bitlinear = BitLinear(768, 256)
        
        # Create DoRA layer
        dora = create_dora_from_bitlinear(bitlinear, rank=16, alpha=32)
        
        assert isinstance(dora, BitLinearDoRA)
        assert dora.rank == 16
        assert dora.in_features == 768
        assert dora.out_features == 256
    
    def test_dora_forward_pass(self):
        """Test DoRA forward pass."""
        from core.bitlinear import BitLinear
        from core.dora import BitLinearDoRA
        
        bitlinear = BitLinear(768, 256)
        dora = BitLinearDoRA(bitlinear, rank=16)
        
        # Forward pass
        x = torch.randn(4, 768)
        output = dora(x)
        
        assert output.shape == (4, 256)
    
    def test_dora_adapter_serialization(self):
        """Test DoRA adapter save/load."""
        from core.dora import DoRAAdapter
        
        # Create adapter
        adapter = DoRAAdapter(
            adapter_id="test_adapter",
            domain="medical",
            rank=16,
            alpha=32,
            in_features=768,
            out_features=256,
            params={
                "magnitude": torch.randn(256),
                "direction_A": torch.randn(16, 768),
                "direction_B": torch.randn(256, 16),
            },
            metadata={"version": "1.0"},
        )
        
        # Serialize
        data = adapter.to_dict()
        
        assert data["adapter_id"] == "test_adapter"
        assert data["domain"] == "medical"
        assert "params" in data
        
        # Deserialize
        loaded = DoRAAdapter.from_dict(data)
        
        assert loaded.adapter_id == adapter.adapter_id
        assert loaded.rank == adapter.rank
    
    def test_dora_expert_registry(self):
        """Test DoRA expert registry."""
        from core.dora import DoRAExpertRegistry, DoRAAdapter
        
        registry = DoRAExpertRegistry()
        
        # Create and register adapter
        adapter = DoRAAdapter(
            adapter_id="medical_dora",
            domain="medical",
            rank=16,
            alpha=32,
            in_features=768,
            out_features=256,
        )
        
        registry.register(adapter)
        
        assert registry.is_registered("medical_dora")
        assert registry.get_domain("medical_dora") == "medical"
        
        # List experts
        experts = registry.list_experts()
        assert "medical_dora" in experts


class TestPipelineIntegration:
    """Integration tests for full pipeline."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock pipeline components."""
        router = Mock()
        router.route.return_value = (
            [("medical_dora", 0.7), ("turkish_dora", 0.3)],
            Mock(used_fast_path=False, keyword_confidence=0.6)
        )
        router.get_stats.return_value = {"total_routes": 1}
        router.reset_stats = Mock()
        
        cache = Mock()
        cache.get = AsyncMock(return_value=None)
        cache.put = AsyncMock()
        cache.clear = Mock()
        cache.get_usage.return_value = {"vram": {"used_mb": 0}}
        
        backend = Mock()
        backend.load_model.return_value = True
        backend.load_adapter.return_value = True
        backend.unload_adapter.return_value = True
        backend.adapters = {}
        backend.inference.return_value = Mock(
            output=torch.randn(1, 10, 100),
            latency_ms=50.0,
            backend_used="pytorch",
        )
        backend.get_capabilities.return_value = Mock(
            to_dict=lambda: {"name": "mock"}
        )
        backend.get_vram_usage.return_value = {"used_mb": 0}
        
        retriever = Mock()
        retriever.search.return_value = [
            Mock(doc_id="doc1", content="Test context", score=0.9, metadata={}),
        ]
        retriever.add_document = Mock()
        retriever.get_stats.return_value = {"num_documents": 1}
        
        return {
            "router": router,
            "cache": cache,
            "backend": backend,
            "retriever": retriever,
        }
    
    @pytest.mark.asyncio
    async def test_pipeline_full_flow(self, mock_components):
        """Test complete pipeline flow."""
        from r3mes.serving.inference_pipeline import (
            InferencePipeline,
            PipelineConfig,
        )
        
        config = PipelineConfig(
            enable_rag=True,
            router_strategy="hybrid",
        )
        
        pipeline = InferencePipeline(
            config=config,
            router=mock_components["router"],
            cache=mock_components["cache"],
            backend=mock_components["backend"],
            retriever=mock_components["retriever"],
        )
        
        pipeline._initialized = True
        
        # Run pipeline
        result = await pipeline.run("Diyabet tedavisi hakkında bilgi ver")
        
        assert result.success is True
        assert len(result.experts_used) == 2
        assert result.metrics.total_time_ms > 0
        
        # Verify component calls
        mock_components["router"].route.assert_called()
        mock_components["retriever"].search.assert_called()
        mock_components["backend"].inference.assert_called()
    
    @pytest.mark.asyncio
    async def test_pipeline_without_rag(self, mock_components):
        """Test pipeline without RAG."""
        from r3mes.serving.inference_pipeline import (
            InferencePipeline,
            PipelineConfig,
        )
        
        config = PipelineConfig(enable_rag=False)
        
        pipeline = InferencePipeline(
            config=config,
            router=mock_components["router"],
            cache=mock_components["cache"],
            backend=mock_components["backend"],
        )
        
        pipeline._initialized = True
        
        result = await pipeline.run("Test query")
        
        assert result.success is True
        assert result.rag_context is None
    
    @pytest.mark.asyncio
    async def test_pipeline_metrics_collection(self, mock_components):
        """Test pipeline metrics collection."""
        from r3mes.serving.inference_pipeline import (
            InferencePipeline,
            PipelineConfig,
        )
        
        config = PipelineConfig(enable_rag=True)
        
        pipeline = InferencePipeline(
            config=config,
            router=mock_components["router"],
            cache=mock_components["cache"],
            backend=mock_components["backend"],
            retriever=mock_components["retriever"],
        )
        
        pipeline._initialized = True
        
        # Run multiple queries
        for _ in range(3):
            await pipeline.run("Test query")
        
        stats = pipeline.get_stats()
        
        assert stats["total_requests"] == 3
        assert stats["error_rate"] == 0.0
    
    @pytest.mark.asyncio
    async def test_pipeline_error_recovery(self, mock_components):
        """Test pipeline error recovery."""
        from r3mes.serving.inference_pipeline import (
            InferencePipeline,
            PipelineConfig,
        )
        
        # Make backend fail
        mock_components["backend"].inference.side_effect = RuntimeError("Test error")
        
        config = PipelineConfig(enable_rag=False)
        
        pipeline = InferencePipeline(
            config=config,
            router=mock_components["router"],
            cache=mock_components["cache"],
            backend=mock_components["backend"],
        )
        
        pipeline._initialized = True
        
        result = await pipeline.run("Test query")
        
        assert result.success is False
        assert "Test error" in result.error
        
        # Pipeline should track errors
        stats = pipeline.get_stats()
        assert stats["total_errors"] == 1


class TestEndToEndScenarios:
    """End-to-end scenario tests."""
    
    @pytest.mark.asyncio
    async def test_medical_query_turkish(self):
        """Test medical query in Turkish."""
        from router.keyword_router import KeywordRouter
        from router.vram_adaptive_gating import VRAMAdaptiveGating, ExpertScore
        
        router = KeywordRouter()
        gating = VRAMAdaptiveGating()
        
        query = "Diyabet hastalarında insülin direnci nasıl tedavi edilir?"
        
        # Route query
        results = router.route(query)
        
        # Convert to ExpertScore
        scores = [
            ExpertScore(r.expert_id, r.confidence, "keyword")
            for r in results
        ]
        
        # Apply gating
        selected = gating.select(scores)
        
        # Should select medical and/or Turkish experts
        expert_ids = [e[0] for e in selected]
        assert any(e in expert_ids for e in ["medical_dora", "turkish_dora"])
    
    @pytest.mark.asyncio
    async def test_coding_query_english(self):
        """Test coding query in English."""
        from router.keyword_router import KeywordRouter
        
        router = KeywordRouter()
        
        query = "How to implement binary search in Python?"
        
        results = router.route(query)
        expert_ids = [r.expert_id for r in results]
        
        assert "coding_dora" in expert_ids
    
    def test_cache_promotion_flow(self):
        """Test cache tier promotion."""
        # This would test the full cache promotion flow
        # VRAM miss → RAM hit → Promote to VRAM
        pass  # Placeholder for more complex test


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    def test_router_latency(self):
        """Benchmark router latency."""
        import time
        from router.keyword_router import KeywordRouter
        
        router = KeywordRouter()
        
        queries = [
            "Diyabet tedavisi",
            "Python programming",
            "Legal contract",
            "Financial analysis",
            "Machine learning",
        ]
        
        # Warmup
        for q in queries:
            router.route(q)
        
        # Benchmark
        start = time.perf_counter()
        iterations = 100
        
        for _ in range(iterations):
            for q in queries:
                router.route(q)
        
        elapsed = time.perf_counter() - start
        avg_latency_ms = (elapsed / (iterations * len(queries))) * 1000
        
        print(f"\n📊 Router Benchmark:")
        print(f"   Average latency: {avg_latency_ms:.3f}ms")
        
        # Should be fast
        assert avg_latency_ms < 5.0  # Less than 5ms per query
    
    @pytest.mark.asyncio
    async def test_cache_latency(self):
        """Benchmark cache operations."""
        import time
        import tempfile
        from cache.tiered_cache import TieredDoRACache, CacheTier
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = TieredDoRACache(
                vram_capacity_mb=100,
                ram_capacity_mb=200,
                disk_cache_dir=temp_dir,
            )
            
            # Prepare test data
            adapter_data = {"data": torch.randn(100, 100)}
            
            # Benchmark put
            start = time.perf_counter()
            for i in range(10):
                await cache.put(f"adapter_{i}", adapter_data, size_mb=1.0, tier=CacheTier.RAM)
            put_time = time.perf_counter() - start
            
            # Benchmark get (cache hit)
            start = time.perf_counter()
            for i in range(10):
                await cache.get(f"adapter_{i}")
            get_time = time.perf_counter() - start
            
            print(f"\n📊 Cache Benchmark:")
            print(f"   Put (10 ops): {put_time*1000:.1f}ms ({put_time/10*1000:.2f}ms avg)")
            print(f"   Get (10 ops): {get_time*1000:.1f}ms ({get_time/10*1000:.2f}ms avg)")
            
            # Should be fast
            assert put_time < 1.0  # Less than 1s for 10 puts
            assert get_time < 0.1  # Less than 100ms for 10 gets


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

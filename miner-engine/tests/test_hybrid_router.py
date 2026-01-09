"""
Tests for Hybrid Router Module.

Tests the 4-stage routing pipeline:
1. Keyword Router (fast pre-filter)
2. Semantic Router (deep understanding)
3. Score Fusion (weighted combination)
4. VRAM-Adaptive Gating (resource-aware selection)
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from router.hybrid_router import (
    HybridRouter,
    HybridRouterConfig,
    RoutingMetrics,
)
from router.keyword_router import KeywordRouter, RouterResult
from router.semantic_router import SemanticRouter, SemanticResult
from router.vram_adaptive_gating import VRAMAdaptiveGating, ExpertScore


class TestHybridRouterConfig:
    """Tests for HybridRouterConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = HybridRouterConfig()
        
        assert config.keyword_weight == 0.3
        assert config.semantic_weight == 0.7
        assert config.fast_path_threshold == 0.85
        assert config.fallback_threshold == 0.5
        assert config.enable_semantic is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = HybridRouterConfig(
            keyword_weight=0.5,
            semantic_weight=0.5,
            fast_path_threshold=0.9,
        )
        
        assert config.keyword_weight == 0.5
        assert config.semantic_weight == 0.5
        assert config.fast_path_threshold == 0.9


class TestRoutingMetrics:
    """Tests for RoutingMetrics."""
    
    def test_metrics_creation(self):
        """Test creating routing metrics."""
        metrics = RoutingMetrics(
            total_time_ms=15.5,
            keyword_time_ms=0.5,
            semantic_time_ms=10.0,
            used_fast_path=False,
            keyword_confidence=0.6,
            top_expert="medical_dora",
            num_experts_selected=2,
        )
        
        assert metrics.total_time_ms == 15.5
        assert metrics.keyword_time_ms == 0.5
        assert metrics.semantic_time_ms == 10.0
        assert metrics.used_fast_path is False
        assert metrics.keyword_confidence == 0.6
        assert metrics.top_expert == "medical_dora"
        assert metrics.num_experts_selected == 2


class TestHybridRouterWithMocks:
    """Tests for HybridRouter using mocked components."""
    
    @pytest.fixture
    def mock_keyword_router(self):
        """Create mock keyword router."""
        router = Mock(spec=KeywordRouter)
        router.route.return_value = [
            RouterResult("coding_dora", 0.8, ["python", "code"], "domain"),
            RouterResult("turkish_dora", 0.5, ["nasıl"], "language"),
        ]
        return router
    
    @pytest.fixture
    def mock_semantic_router(self):
        """Create mock semantic router."""
        router = Mock(spec=SemanticRouter)
        router.route.return_value = [
            SemanticResult("coding_dora", 0.85, "combined"),
            SemanticResult("education_dora", 0.6, "combined"),
        ]
        return router
    
    @pytest.fixture
    def mock_gating(self):
        """Create mock VRAM gating."""
        gating = Mock(spec=VRAMAdaptiveGating)
        gating.select.return_value = [
            ("coding_dora", 0.7),
            ("turkish_dora", 0.3),
        ]
        gating.get_status.return_value = {
            'vram_gb': 8.0,
            'max_experts': 2,
            'tier': 'medium',
        }
        return gating
    
    @pytest.fixture
    def router(self, mock_keyword_router, mock_semantic_router, mock_gating):
        """Create HybridRouter with mocked components."""
        config = HybridRouterConfig(enable_semantic=True)
        return HybridRouter(
            config=config,
            keyword_router=mock_keyword_router,
            semantic_router=mock_semantic_router,
            gating=mock_gating,
        )
    
    def test_initialization(self, router):
        """Test router initialization."""
        assert router.keyword_router is not None
        assert router.semantic_router is not None
        assert router.gating is not None
    
    def test_route_returns_experts_and_metrics(self, router):
        """Test routing returns experts and metrics."""
        selected, metrics = router.route("Python'da for loop nasıl yazılır?")
        
        assert len(selected) > 0
        assert isinstance(metrics, RoutingMetrics)
        assert metrics.total_time_ms > 0
    
    def test_route_simple(self, router):
        """Test simple routing without metrics."""
        selected = router.route_simple("Test query")
        
        assert len(selected) > 0
        assert all(isinstance(s, tuple) for s in selected)
        assert all(len(s) == 2 for s in selected)
    
    def test_fast_path_on_high_confidence(self, mock_keyword_router, mock_semantic_router, mock_gating):
        """Test fast path is used when keyword confidence is high."""
        # Set high confidence
        mock_keyword_router.route.return_value = [
            RouterResult("coding_dora", 0.95, ["python"], "domain"),
        ]
        
        config = HybridRouterConfig(fast_path_threshold=0.85)
        router = HybridRouter(
            config=config,
            keyword_router=mock_keyword_router,
            semantic_router=mock_semantic_router,
            gating=mock_gating,
        )
        
        _, metrics = router.route("Python code")
        
        assert metrics.used_fast_path is True
        assert metrics.semantic_time_ms == 0.0
        mock_semantic_router.route.assert_not_called()
    
    def test_semantic_path_on_low_confidence(self, mock_keyword_router, mock_semantic_router, mock_gating):
        """Test semantic router is used when keyword confidence is low."""
        # Set low confidence
        mock_keyword_router.route.return_value = [
            RouterResult("general_dora", 0.4, [], "domain"),
        ]
        
        config = HybridRouterConfig(fast_path_threshold=0.85)
        router = HybridRouter(
            config=config,
            keyword_router=mock_keyword_router,
            semantic_router=mock_semantic_router,
            gating=mock_gating,
        )
        
        _, metrics = router.route("Bu konuyu açıklar mısın?")
        
        assert metrics.used_fast_path is False
        mock_semantic_router.route.assert_called_once()
    
    def test_force_semantic(self, mock_keyword_router, mock_semantic_router, mock_gating):
        """Test forcing semantic routing."""
        # Set high confidence (would normally trigger fast path)
        mock_keyword_router.route.return_value = [
            RouterResult("coding_dora", 0.95, ["python"], "domain"),
        ]
        
        router = HybridRouter(
            keyword_router=mock_keyword_router,
            semantic_router=mock_semantic_router,
            gating=mock_gating,
        )
        
        _, metrics = router.route("Python code", force_semantic=True)
        
        # Should still use semantic despite high keyword confidence
        mock_semantic_router.route.assert_called_once()
    
    def test_get_stats(self, router):
        """Test getting router statistics."""
        # Do some routing
        router.route("Query 1")
        router.route("Query 2")
        
        stats = router.get_stats()
        
        assert stats['total_routes'] == 2
        assert 'fast_path_count' in stats
        assert 'semantic_path_count' in stats
        assert 'config' in stats
        assert 'gating' in stats
    
    def test_reset_stats(self, router):
        """Test resetting statistics."""
        router.route("Query 1")
        router.reset_stats()
        
        stats = router.get_stats()
        assert stats['total_routes'] == 0
        assert stats['fast_path_count'] == 0


class TestScoreFusion:
    """Tests for score fusion logic."""
    
    @pytest.fixture
    def router(self):
        """Create router with disabled semantic for fusion testing."""
        config = HybridRouterConfig(
            keyword_weight=0.3,
            semantic_weight=0.7,
            enable_semantic=False,
        )
        return HybridRouter(config=config)
    
    def test_fuse_scores_combined(self, router):
        """Test fusing scores from both routers."""
        keyword_results = [
            RouterResult("medical_dora", 0.8, [], "domain"),
            RouterResult("turkish_dora", 0.6, [], "language"),
        ]
        
        semantic_results = [
            SemanticResult("medical_dora", 0.9, "combined"),
            SemanticResult("coding_dora", 0.5, "combined"),
        ]
        
        fused = router._fuse_scores(keyword_results, semantic_results)
        
        # medical_dora should have highest score (combined)
        assert fused[0].expert_id == "medical_dora"
        assert fused[0].source == "combined"
        
        # Check score calculation: 0.3 * 0.8 + 0.7 * 0.9 = 0.24 + 0.63 = 0.87
        expected_medical = 0.3 * 0.8 + 0.7 * 0.9
        assert abs(fused[0].score - expected_medical) < 0.01
    
    def test_fuse_scores_keyword_only(self, router):
        """Test fusion with keyword-only expert."""
        keyword_results = [
            RouterResult("turkish_dora", 0.8, [], "language"),
        ]
        
        semantic_results = [
            SemanticResult("coding_dora", 0.7, "combined"),
        ]
        
        fused = router._fuse_scores(keyword_results, semantic_results)
        
        # Find turkish_dora
        turkish = next(f for f in fused if f.expert_id == "turkish_dora")
        assert turkish.source == "keyword"
        assert abs(turkish.score - 0.3 * 0.8) < 0.01
    
    def test_fuse_scores_semantic_only(self, router):
        """Test fusion with semantic-only expert."""
        keyword_results = []
        
        semantic_results = [
            SemanticResult("science_dora", 0.8, "combined"),
        ]
        
        fused = router._fuse_scores(keyword_results, semantic_results)
        
        assert len(fused) == 1
        assert fused[0].expert_id == "science_dora"
        assert fused[0].source == "semantic"
        assert abs(fused[0].score - 0.7 * 0.8) < 0.01


class TestHybridRouterIntegration:
    """Integration tests with real components (mocked sentence-transformers)."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock sentence transformer model."""
        model = Mock()
        
        def mock_encode(text, **kwargs):
            if isinstance(text, str):
                vec = np.random.randn(384)
                return vec / np.linalg.norm(vec)
            else:
                vecs = np.random.randn(len(text), 384)
                return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        
        model.encode = mock_encode
        return model
    
    def test_full_pipeline(self, mock_model):
        """Test full routing pipeline."""
        with patch('router.semantic_router._get_sentence_transformer') as mock_st:
            mock_st.return_value = lambda *args, **kwargs: mock_model
            
            config = HybridRouterConfig(
                fast_path_threshold=0.85,
                enable_semantic=True,
            )
            
            router = HybridRouter(config=config)
            router.semantic_router._model = mock_model
            router.initialize()
            
            # Test with coding query (should have high keyword confidence)
            selected, metrics = router.route("Python'da for loop nasıl yazılır?")
            
            assert len(selected) > 0
            assert metrics.keyword_time_ms > 0
    
    def test_semantic_disabled(self):
        """Test routing with semantic disabled."""
        config = HybridRouterConfig(enable_semantic=False)
        router = HybridRouter(config=config)
        
        selected, metrics = router.route("Test query")
        
        assert len(selected) > 0
        assert metrics.semantic_time_ms == 0.0
        assert router.semantic_router is None


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_query(self):
        """Test routing with empty query."""
        config = HybridRouterConfig(enable_semantic=False)
        router = HybridRouter(config=config)
        
        selected, metrics = router.route("")
        
        # Should return fallback
        assert len(selected) > 0
    
    def test_very_long_query(self):
        """Test routing with very long query."""
        config = HybridRouterConfig(enable_semantic=False)
        router = HybridRouter(config=config)
        
        long_query = "Python " * 1000
        selected, metrics = router.route(long_query)
        
        assert len(selected) > 0
    
    def test_special_characters(self):
        """Test routing with special characters."""
        config = HybridRouterConfig(enable_semantic=False)
        router = HybridRouter(config=config)
        
        selected, metrics = router.route("@#$%^&*()!?")
        
        # Should return fallback
        assert len(selected) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

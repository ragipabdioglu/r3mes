"""
Tests for Semantic Router Module.

Tests embedding-based routing using sentence-transformers.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path


class TestSemanticRouterWithMock:
    """Tests for SemanticRouter using mocked sentence-transformers."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock sentence transformer model."""
        model = Mock()
        
        # Mock encode to return normalized random vectors
        def mock_encode(text, **kwargs):
            if isinstance(text, str):
                vec = np.random.randn(384)
                vec = vec / np.linalg.norm(vec)
                return vec
            else:
                vecs = np.random.randn(len(text), 384)
                vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
                return vecs
        
        model.encode = mock_encode
        return model
    
    @pytest.fixture
    def router(self, mock_model):
        """Create SemanticRouter with mocked model."""
        with patch('router.semantic_router._get_sentence_transformer') as mock_st:
            mock_st.return_value = lambda *args, **kwargs: mock_model
            
            from router.semantic_router import SemanticRouter
            router = SemanticRouter(device="cpu")
            router._model = mock_model
            return router
    
    def test_initialization(self, router):
        """Test router initialization."""
        assert router.model_name == "all-MiniLM-L6-v2"
        assert router.device == "cpu"
        assert len(router._experts) > 0
    
    def test_default_experts_loaded(self, router):
        """Test default experts are loaded."""
        expert_ids = router.get_expert_ids()
        
        assert "medical_dora" in expert_ids
        assert "coding_dora" in expert_ids
        assert "general_dora" in expert_ids
        assert "turkish_dora" in expert_ids
    
    def test_initialize_computes_embeddings(self, router):
        """Test initialization computes embeddings."""
        router.initialize()
        
        for expert_id, expert in router._experts.items():
            assert expert.combined_embedding is not None
            assert expert.combined_embedding.shape == (384,)
    
    def test_route_returns_results(self, router):
        """Test routing returns results."""
        router.initialize()
        
        results = router.route("What are the symptoms of diabetes?")
        
        assert len(results) > 0
        assert all(hasattr(r, 'expert_id') for r in results)
        assert all(hasattr(r, 'similarity') for r in results)
    
    def test_route_results_sorted_by_similarity(self, router):
        """Test results are sorted by similarity descending."""
        router.initialize()
        
        results = router.route("How to write Python code?")
        
        if len(results) > 1:
            similarities = [r.similarity for r in results]
            assert similarities == sorted(similarities, reverse=True)
    
    def test_route_top_k(self, router):
        """Test top_k parameter limits results."""
        router.initialize()
        
        results = router.route("Test query", top_k=3)
        
        assert len(results) <= 3
    
    def test_route_batch(self, router):
        """Test batch routing."""
        router.initialize()
        
        queries = [
            "Medical question about health",
            "Programming question about Python",
            "Legal question about contracts",
        ]
        
        results = router.route_batch(queries, top_k=3)
        
        assert len(results) == 3
        assert all(len(r) <= 3 for r in results)
    
    def test_get_similarity(self, router):
        """Test getting similarity for specific expert."""
        router.initialize()
        
        similarity = router.get_similarity("Test query", "medical_dora")
        
        assert isinstance(similarity, float)
        assert -1.0 <= similarity <= 1.0
    
    def test_get_similarity_unknown_expert(self, router):
        """Test similarity for unknown expert returns 0."""
        router.initialize()
        
        similarity = router.get_similarity("Test query", "unknown_expert")
        
        assert similarity == 0.0
    
    def test_add_expert(self, router):
        """Test adding new expert."""
        router.initialize()
        
        router.add_expert(
            expert_id="custom_dora",
            description="Custom expert for testing",
            examples=["Example query 1", "Example query 2"],
        )
        
        assert "custom_dora" in router.get_expert_ids()
        assert router._experts["custom_dora"].combined_embedding is not None
    
    def test_remove_expert(self, router):
        """Test removing expert."""
        router.initialize()
        
        initial_count = len(router.get_expert_ids())
        result = router.remove_expert("medical_dora")
        
        assert result is True
        assert "medical_dora" not in router.get_expert_ids()
        assert len(router.get_expert_ids()) == initial_count - 1
    
    def test_remove_nonexistent_expert(self, router):
        """Test removing non-existent expert."""
        result = router.remove_expert("nonexistent_expert")
        
        assert result is False
    
    def test_get_status(self, router):
        """Test status reporting."""
        status = router.get_status()
        
        assert status['model_name'] == "all-MiniLM-L6-v2"
        assert status['device'] == "cpu"
        assert 'num_experts' in status
        assert 'expert_ids' in status
        assert status['embedding_dim'] == 384


class TestSemanticRouterCache:
    """Tests for SemanticRouter caching functionality."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock sentence transformer model."""
        model = Mock()
        
        def mock_encode(text, **kwargs):
            if isinstance(text, str):
                vec = np.random.randn(384)
                vec = vec / np.linalg.norm(vec)
                return vec
            else:
                vecs = np.random.randn(len(text), 384)
                vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
                return vecs
        
        model.encode = mock_encode
        return model
    
    def test_cache_save_and_load(self, mock_model):
        """Test saving and loading cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('router.semantic_router._get_sentence_transformer') as mock_st:
                mock_st.return_value = lambda *args, **kwargs: mock_model
                
                from router.semantic_router import SemanticRouter
                
                # Create router and initialize
                router1 = SemanticRouter(cache_dir=tmpdir, device="cpu")
                router1._model = mock_model
                router1.initialize()
                
                # Get embeddings
                original_embedding = router1._experts["medical_dora"].combined_embedding.copy()
                
                # Create new router and load from cache
                router2 = SemanticRouter(cache_dir=tmpdir, device="cpu")
                router2._model = mock_model
                
                # Should load from cache
                loaded = router2._load_cache()
                
                assert loaded is True
                assert router2._experts["medical_dora"].combined_embedding is not None


class TestExpertEmbedding:
    """Tests for ExpertEmbedding dataclass."""
    
    def test_expert_embedding_creation(self):
        """Test creating ExpertEmbedding."""
        from router.semantic_router import ExpertEmbedding
        
        expert = ExpertEmbedding(
            expert_id="test_dora",
            description="Test expert",
            examples=["Example 1", "Example 2"],
        )
        
        assert expert.expert_id == "test_dora"
        assert expert.description == "Test expert"
        assert len(expert.examples) == 2
        assert expert.description_embedding is None
        assert expert.combined_embedding is None


class TestSemanticResult:
    """Tests for SemanticResult dataclass."""
    
    def test_semantic_result_creation(self):
        """Test creating SemanticResult."""
        from router.semantic_router import SemanticResult
        
        result = SemanticResult(
            expert_id="medical_dora",
            similarity=0.85,
            embedding_source="combined",
        )
        
        assert result.expert_id == "medical_dora"
        assert result.similarity == 0.85
        assert result.embedding_source == "combined"


class TestMaxPooling:
    """Tests for max pooling functionality."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock()
        
        def mock_encode(text, **kwargs):
            if isinstance(text, str):
                vec = np.array([0.1, 0.2, 0.3] + [0.0] * 381)
                return vec / np.linalg.norm(vec)
            else:
                vecs = []
                for i, t in enumerate(text):
                    vec = np.array([0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1)] + [0.0] * 381)
                    vecs.append(vec / np.linalg.norm(vec))
                return np.array(vecs)
        
        model.encode = mock_encode
        return model
    
    def test_max_pool_single_embedding(self, mock_model):
        """Test max pooling with single embedding."""
        with patch('router.semantic_router._get_sentence_transformer') as mock_st:
            mock_st.return_value = lambda *args, **kwargs: mock_model
            
            from router.semantic_router import SemanticRouter, ExpertEmbedding
            
            router = SemanticRouter(device="cpu")
            router._model = mock_model
            
            expert = ExpertEmbedding(
                expert_id="test",
                description="Test description",
                examples=[],
            )
            
            expert.description_embedding = np.array([0.5, 0.5, 0.0] + [0.0] * 381)
            
            pooled = router._max_pool_embeddings(expert)
            
            assert pooled.shape == (384,)
    
    def test_max_pool_multiple_embeddings(self, mock_model):
        """Test max pooling with multiple embeddings."""
        with patch('router.semantic_router._get_sentence_transformer') as mock_st:
            mock_st.return_value = lambda *args, **kwargs: mock_model
            
            from router.semantic_router import SemanticRouter, ExpertEmbedding
            
            router = SemanticRouter(device="cpu")
            router._model = mock_model
            
            expert = ExpertEmbedding(
                expert_id="test",
                description="Test description",
                examples=["Ex1", "Ex2"],
            )
            
            expert.description_embedding = np.array([0.1, 0.2, 0.3] + [0.0] * 381)
            expert.example_embeddings = np.array([
                [0.2, 0.1, 0.4] + [0.0] * 381,
                [0.3, 0.3, 0.2] + [0.0] * 381,
            ])
            
            pooled = router._max_pool_embeddings(expert)
            
            assert pooled.shape == (384,)
            # Max pooling should take max values
            assert pooled[0] >= 0.1  # Max of 0.1, 0.2, 0.3
            assert pooled[1] >= 0.1  # Max of 0.2, 0.1, 0.3


class TestIntegrationWithKeywordRouter:
    """Integration tests with KeywordRouter."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock()
        
        # Create deterministic embeddings based on content
        def mock_encode(text, **kwargs):
            if isinstance(text, str):
                # Simple hash-based embedding for determinism
                hash_val = hash(text) % 1000 / 1000.0
                vec = np.array([hash_val, 1 - hash_val, 0.5] + [0.0] * 381)
                return vec / np.linalg.norm(vec)
            else:
                vecs = []
                for t in text:
                    hash_val = hash(t) % 1000 / 1000.0
                    vec = np.array([hash_val, 1 - hash_val, 0.5] + [0.0] * 381)
                    vecs.append(vec / np.linalg.norm(vec))
                return np.array(vecs)
        
        model.encode = mock_encode
        return model
    
    def test_keyword_and_semantic_routing(self, mock_model):
        """Test using both keyword and semantic routing."""
        from router.keyword_router import KeywordRouter
        
        with patch('router.semantic_router._get_sentence_transformer') as mock_st:
            mock_st.return_value = lambda *args, **kwargs: mock_model
            
            from router.semantic_router import SemanticRouter
            
            keyword_router = KeywordRouter()
            semantic_router = SemanticRouter(device="cpu")
            semantic_router._model = mock_model
            semantic_router.initialize()
            
            query = "Python programlama kodu"
            
            # Get keyword results
            keyword_results = keyword_router.route(query)
            
            # Get semantic results
            semantic_results = semantic_router.route(query)
            
            # Both should return results
            assert len(keyword_results) > 0
            assert len(semantic_results) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

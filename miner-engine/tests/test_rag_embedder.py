"""
Tests for RAG Embedder Module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch


class TestEmbedderConfig:
    """Tests for EmbedderConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        from rag.embedder import EmbedderConfig
        
        config = EmbedderConfig()
        
        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.device == "cpu"
        assert config.batch_size == 32
        assert config.normalize is True
        assert config.cache_enabled is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        from rag.embedder import EmbedderConfig
        
        config = EmbedderConfig(
            model_name="multilingual-e5-small",
            device="cuda",
            batch_size=64,
        )
        
        assert config.model_name == "multilingual-e5-small"
        assert config.device == "cuda"
        assert config.batch_size == 64


class TestRAGEmbedderWithMock:
    """Tests for RAGEmbedder using mocked sentence-transformers."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock sentence transformer model."""
        model = Mock()
        
        def mock_encode(text, **kwargs):
            normalize = kwargs.get('normalize_embeddings', True)
            if isinstance(text, str):
                vec = np.random.randn(384)
                if normalize:
                    vec = vec / np.linalg.norm(vec)
                return vec
            else:
                vecs = np.random.randn(len(text), 384)
                if normalize:
                    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
                return vecs
        
        model.encode = mock_encode
        return model
    
    @pytest.fixture
    def embedder(self, mock_model):
        """Create RAGEmbedder with mocked model."""
        with patch('rag.embedder._get_sentence_transformer') as mock_st:
            mock_st.return_value = lambda *args, **kwargs: mock_model
            
            from rag.embedder import RAGEmbedder, EmbedderConfig
            
            config = EmbedderConfig(cache_enabled=True)
            embedder = RAGEmbedder(config)
            embedder._model = mock_model
            return embedder
    
    def test_initialization(self, embedder):
        """Test embedder initialization."""
        assert embedder.embedding_dim == 384
        assert embedder.config.model_name == "all-MiniLM-L6-v2"
    
    def test_embed_single(self, embedder):
        """Test embedding single text."""
        embedding = embedder.embed("Test text")
        
        assert embedding.shape == (384,)
        assert np.isclose(np.linalg.norm(embedding), 1.0, atol=0.01)
    
    def test_embed_batch(self, embedder):
        """Test batch embedding."""
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = embedder.embed_batch(texts)
        
        assert embeddings.shape == (3, 384)
    
    def test_embed_query(self, embedder):
        """Test query embedding."""
        embedding = embedder.embed_query("What is Python?")
        
        assert embedding.shape == (384,)
    
    def test_embed_documents(self, embedder):
        """Test document embedding."""
        docs = ["Doc 1", "Doc 2"]
        embeddings = embedder.embed_documents(docs, show_progress=False)
        
        assert embeddings.shape == (2, 384)
    
    def test_cache_hit(self, embedder):
        """Test cache hit."""
        text = "Cached text"
        
        # First call - cache miss
        embedder.embed(text)
        assert embedder._cache_misses == 1
        
        # Second call - cache hit
        embedder.embed(text)
        assert embedder._cache_hits == 1
    
    def test_cache_disabled(self, mock_model):
        """Test with cache disabled."""
        with patch('rag.embedder._get_sentence_transformer') as mock_st:
            mock_st.return_value = lambda *args, **kwargs: mock_model
            
            from rag.embedder import RAGEmbedder, EmbedderConfig
            
            config = EmbedderConfig(cache_enabled=False)
            embedder = RAGEmbedder(config)
            embedder._model = mock_model
            
            embedder.embed("Text 1")
            embedder.embed("Text 1")
            
            # No cache tracking
            assert embedder._cache_hits == 0
    
    def test_similarity(self, embedder):
        """Test similarity computation."""
        emb1 = embedder.embed("Hello world")
        emb2 = embedder.embed("Hi there")
        
        sim = embedder.similarity(emb1, emb2)
        
        assert -1.0 <= sim <= 1.0
    
    def test_similarity_batch(self, embedder):
        """Test batch similarity."""
        query_emb = embedder.embed("Query")
        doc_embs = embedder.embed_batch(["Doc 1", "Doc 2", "Doc 3"])
        
        sims = embedder.similarity_batch(query_emb, doc_embs)
        
        assert sims.shape == (3,)
        assert all(-1.0 <= s <= 1.0 for s in sims)
    
    def test_clear_cache(self, embedder):
        """Test clearing cache."""
        embedder.embed("Text 1")
        embedder.embed("Text 2")
        
        embedder.clear_cache()
        
        assert len(embedder._cache) == 0
        assert embedder._cache_hits == 0
        assert embedder._cache_misses == 0
    
    def test_get_stats(self, embedder):
        """Test getting statistics."""
        embedder.embed("Text 1")
        embedder.embed("Text 1")  # Cache hit
        
        stats = embedder.get_stats()
        
        assert stats['model_name'] == "all-MiniLM-L6-v2"
        assert stats['embedding_dim'] == 384
        assert stats['cache_hits'] == 1
        assert stats['cache_misses'] == 1
        assert stats['cache_hit_rate'] == 0.5
    
    def test_empty_batch(self, embedder):
        """Test empty batch embedding."""
        embeddings = embedder.embed_batch([])
        
        assert embeddings.shape == (0, 384)


class TestCreateEmbedder:
    """Tests for create_embedder convenience function."""
    
    def test_create_embedder(self):
        """Test creating embedder with convenience function."""
        with patch('rag.embedder._get_sentence_transformer') as mock_st:
            mock_st.return_value = lambda *args, **kwargs: Mock()
            
            from rag.embedder import create_embedder
            
            embedder = create_embedder(
                model_name="all-MiniLM-L6-v2",
                device="cpu",
                cache_enabled=True,
            )
            
            assert embedder.config.model_name == "all-MiniLM-L6-v2"
            assert embedder.config.device == "cpu"
            assert embedder.config.cache_enabled is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

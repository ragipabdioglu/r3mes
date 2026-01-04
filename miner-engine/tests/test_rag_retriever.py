"""
Tests for RAG Retriever Module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path


class TestRetrieverConfig:
    """Tests for RetrieverConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        from rag.retriever import RetrieverConfig, IndexType
        
        config = RetrieverConfig()
        
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.embedding_device == "cpu"
        assert config.index_type == IndexType.FLAT
        assert config.default_top_k == 5


class TestRetrievalResult:
    """Tests for RetrievalResult."""
    
    def test_result_creation(self):
        """Test creating retrieval result."""
        from rag.retriever import RetrievalResult
        
        result = RetrievalResult(
            doc_id="doc1",
            content="Test content",
            score=0.85,
            metadata={"source": "test"},
        )
        
        assert result.doc_id == "doc1"
        assert result.content == "Test content"
        assert result.score == 0.85
        assert result.metadata["source"] == "test"


class TestRAGRetrieverWithMock:
    """Tests for RAGRetriever using mocked components."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock sentence transformer model."""
        model = Mock()
        
        def mock_encode(text, **kwargs):
            if isinstance(text, str):
                # Deterministic embedding based on text hash
                np.random.seed(hash(text) % 2**32)
                vec = np.random.randn(384)
                vec = vec / np.linalg.norm(vec)
                return vec
            else:
                vecs = []
                for t in text:
                    np.random.seed(hash(t) % 2**32)
                    vec = np.random.randn(384)
                    vecs.append(vec / np.linalg.norm(vec))
                return np.array(vecs)
        
        model.encode = mock_encode
        return model
    
    @pytest.fixture
    def retriever(self, mock_model):
        """Create RAGRetriever with mocked embedder."""
        with patch('rag.embedder._get_sentence_transformer') as mock_st:
            mock_st.return_value = lambda *args, **kwargs: mock_model
            
            from rag.retriever import RAGRetriever, RetrieverConfig
            
            config = RetrieverConfig()
            retriever = RAGRetriever(config)
            retriever.embedder._model = mock_model
            return retriever
    
    def test_initialization(self, retriever):
        """Test retriever initialization."""
        assert retriever.embedder is not None
        assert retriever.store is not None
        assert len(retriever._documents) == 0
    
    def test_add_document(self, retriever):
        """Test adding single document."""
        retriever.add_document(
            doc_id="doc1",
            content="Python is a programming language",
            metadata={"category": "programming"},
        )
        
        assert len(retriever._documents) == 1
        assert "doc1" in retriever._documents
    
    def test_add_documents_batch(self, retriever):
        """Test adding multiple documents."""
        documents = [
            ("doc1", "Python programming", {"lang": "en"}),
            ("doc2", "Java programming", {"lang": "en"}),
            ("doc3", "Türkçe metin", {"lang": "tr"}),
        ]
        
        count = retriever.add_documents(documents, show_progress=False)
        
        assert count == 3
        assert len(retriever._documents) == 3
    
    def test_search(self, retriever):
        """Test searching documents."""
        # Add documents
        retriever.add_document("doc1", "Python is great for data science")
        retriever.add_document("doc2", "Java is used for enterprise apps")
        retriever.add_document("doc3", "JavaScript runs in browsers")
        
        # Search
        results = retriever.search("Python programming", top_k=2)
        
        assert len(results) <= 2
        assert all(hasattr(r, 'doc_id') for r in results)
        assert all(hasattr(r, 'score') for r in results)
    
    def test_search_with_threshold(self, retriever):
        """Test search with similarity threshold."""
        retriever.add_document("doc1", "Python programming")
        retriever.add_document("doc2", "Cooking recipes")
        
        # High threshold - may filter out results
        results = retriever.search("Python", threshold=0.9)
        
        # All results should meet threshold
        assert all(r.score >= 0.9 for r in results)
    
    def test_search_with_metadata_filter(self, retriever):
        """Test search with metadata filter."""
        retriever.add_document("doc1", "Python code", {"lang": "en"})
        retriever.add_document("doc2", "Python kodu", {"lang": "tr"})
        retriever.add_document("doc3", "Java code", {"lang": "en"})
        
        results = retriever.search(
            "Python",
            filter_metadata={"lang": "tr"},
        )
        
        # Only Turkish documents
        assert all(r.metadata.get("lang") == "tr" for r in results)
    
    def test_search_batch(self, retriever):
        """Test batch search."""
        retriever.add_document("doc1", "Python programming")
        retriever.add_document("doc2", "Java programming")
        
        queries = ["Python", "Java"]
        results = retriever.search_batch(queries, top_k=1)
        
        assert len(results) == 2
        assert all(isinstance(r, list) for r in results)
    
    def test_get_document(self, retriever):
        """Test getting document by ID."""
        retriever.add_document("doc1", "Test content")
        
        doc = retriever.get_document("doc1")
        
        assert doc is not None
        assert doc.text == "Test content"
    
    def test_get_nonexistent_document(self, retriever):
        """Test getting non-existent document."""
        doc = retriever.get_document("nonexistent")
        
        assert doc is None
    
    def test_remove_document(self, retriever):
        """Test removing document."""
        retriever.add_document("doc1", "Test content")
        
        result = retriever.remove_document("doc1")
        
        assert result is True
        assert "doc1" not in retriever._documents
    
    def test_remove_nonexistent_document(self, retriever):
        """Test removing non-existent document."""
        result = retriever.remove_document("nonexistent")
        
        assert result is False
    
    def test_clear(self, retriever):
        """Test clearing all documents."""
        retriever.add_document("doc1", "Content 1")
        retriever.add_document("doc2", "Content 2")
        
        retriever.clear()
        
        assert len(retriever._documents) == 0
    
    def test_get_stats(self, retriever):
        """Test getting statistics."""
        retriever.add_document("doc1", "Test")
        
        stats = retriever.get_stats()
        
        assert stats['num_documents'] == 1
        assert 'embedding_model' in stats
        assert 'embedder_stats' in stats
        assert 'store_stats' in stats


class TestRetrieverPersistence:
    """Tests for retriever save/load functionality."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        
        def mock_encode(text, **kwargs):
            if isinstance(text, str):
                np.random.seed(42)
                vec = np.random.randn(384)
                return vec / np.linalg.norm(vec)
            else:
                vecs = []
                for i, t in enumerate(text):
                    np.random.seed(42 + i)
                    vec = np.random.randn(384)
                    vecs.append(vec / np.linalg.norm(vec))
                return np.array(vecs)
        
        model.encode = mock_encode
        return model
    
    def test_save_and_load(self, mock_model):
        """Test saving and loading retriever."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('rag.embedder._get_sentence_transformer') as mock_st:
                mock_st.return_value = lambda *args, **kwargs: mock_model
                
                from rag.retriever import RAGRetriever, RetrieverConfig
                
                # Create and populate retriever
                config = RetrieverConfig(persist_dir=tmpdir)
                retriever1 = RAGRetriever(config)
                retriever1.embedder._model = mock_model
                
                retriever1.add_document("doc1", "Test content", {"key": "value"})
                retriever1.save()
                
                # Create new retriever and load
                retriever2 = RAGRetriever(config)
                retriever2.embedder._model = mock_model
                retriever2.load()
                
                # Verify
                assert len(retriever2._documents) == 1
                doc = retriever2.get_document("doc1")
                assert doc.text == "Test content"
                assert doc.metadata["key"] == "value"


class TestCreateRetriever:
    """Tests for create_retriever convenience function."""
    
    def test_create_retriever(self):
        """Test creating retriever with convenience function."""
        with patch('rag.embedder._get_sentence_transformer') as mock_st:
            mock_st.return_value = lambda *args, **kwargs: Mock()
            
            from rag.retriever import create_retriever
            
            retriever = create_retriever(
                embedding_model="all-MiniLM-L6-v2",
                index_type="flat",
            )
            
            assert retriever.config.embedding_model == "all-MiniLM-L6-v2"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""
Tests for RAG Module.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

# Check if FAISS is available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

if FAISS_AVAILABLE:
    from rag.faiss_store import FAISSStore, SearchResult, Document


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not installed")
class TestFAISSStore:
    """Tests for FAISSStore."""
    
    @pytest.fixture
    def store(self):
        """Create a test store."""
        return FAISSStore(dimension=64, index_type="flat")
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents with embeddings."""
        docs = []
        embeddings = []
        
        for i in range(10):
            doc = Document(
                doc_id=f"doc_{i}",
                text=f"This is document number {i}",
                metadata={"index": i},
            )
            docs.append(doc)
            
            # Create random embedding
            emb = np.random.randn(64).astype(np.float32)
            embeddings.append(emb)
        
        return docs, np.array(embeddings)
    
    def test_initialization(self, store):
        """Test store initialization."""
        assert store.dimension == 64
        assert store.index_type == "flat"
        assert len(store) == 0
    
    def test_add_documents(self, store, sample_documents):
        """Test adding documents."""
        docs, embeddings = sample_documents
        store.add_documents(docs, embeddings)
        
        assert len(store) == 10
    
    def test_search(self, store, sample_documents):
        """Test similarity search."""
        docs, embeddings = sample_documents
        store.add_documents(docs, embeddings)
        
        # Search with first document's embedding
        query = embeddings[0]
        results = store.search(query, k=3)
        
        assert len(results) <= 3
        assert all(isinstance(r, SearchResult) for r in results)
        
        # First result should be the query document itself
        assert results[0].doc_id == "doc_0"
        assert results[0].score > 0.99  # Should be very similar
    
    def test_search_empty_store(self, store):
        """Test search on empty store."""
        query = np.random.randn(64).astype(np.float32)
        results = store.search(query, k=5)
        
        assert results == []
    
    def test_search_with_threshold(self, store, sample_documents):
        """Test search with score threshold."""
        docs, embeddings = sample_documents
        store.add_documents(docs, embeddings)
        
        query = np.random.randn(64).astype(np.float32)
        results = store.search(query, k=10, threshold=0.9)
        
        # All results should be above threshold
        assert all(r.score >= 0.9 for r in results)
    
    def test_search_batch(self, store, sample_documents):
        """Test batch search."""
        docs, embeddings = sample_documents
        store.add_documents(docs, embeddings)
        
        # Search with multiple queries
        queries = embeddings[:3]
        results = store.search_batch(queries, k=2)
        
        assert len(results) == 3
        assert all(len(r) <= 2 for r in results)
    
    def test_get_document(self, store, sample_documents):
        """Test getting document by ID."""
        docs, embeddings = sample_documents
        store.add_documents(docs, embeddings)
        
        doc = store.get("doc_5")
        assert doc is not None
        assert doc.doc_id == "doc_5"
        assert doc.metadata["index"] == 5
    
    def test_get_nonexistent(self, store):
        """Test getting nonexistent document."""
        doc = store.get("nonexistent")
        assert doc is None
    
    def test_remove_document(self, store, sample_documents):
        """Test removing document."""
        docs, embeddings = sample_documents
        store.add_documents(docs, embeddings)
        
        result = store.remove("doc_3")
        assert result is True
        
        doc = store.get("doc_3")
        assert doc is None
    
    def test_remove_nonexistent(self, store):
        """Test removing nonexistent document."""
        result = store.remove("nonexistent")
        assert result is False
    
    def test_save_and_load(self, store, sample_documents):
        """Test saving and loading store."""
        docs, embeddings = sample_documents
        store.add_documents(docs, embeddings)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            store.save(tmpdir)
            
            # Load
            loaded = FAISSStore.load(tmpdir)
            
            assert len(loaded) == len(store)
            assert loaded.dimension == store.dimension
            
            # Verify search works
            query = embeddings[0]
            results = loaded.search(query, k=1)
            assert results[0].doc_id == "doc_0"
    
    def test_get_stats(self, store, sample_documents):
        """Test getting statistics."""
        docs, embeddings = sample_documents
        store.add_documents(docs, embeddings)
        
        stats = store.get_stats()
        
        assert stats['num_documents'] == 10
        assert stats['dimension'] == 64
        assert stats['index_type'] == "flat"


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not installed")
class TestSearchResult:
    """Tests for SearchResult dataclass."""
    
    def test_creation(self):
        """Test SearchResult creation."""
        result = SearchResult(
            doc_id="test",
            score=0.95,
            text="Test document",
            metadata={"key": "value"},
        )
        
        assert result.doc_id == "test"
        assert result.score == 0.95
        assert result.text == "Test document"
        assert result.metadata["key"] == "value"


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not installed")
class TestDocument:
    """Tests for Document dataclass."""
    
    def test_creation(self):
        """Test Document creation."""
        doc = Document(
            doc_id="test",
            text="Test content",
            metadata={"source": "test"},
        )
        
        assert doc.doc_id == "test"
        assert doc.text == "Test content"
        assert doc.embedding is None


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not installed")
class TestIndexTypes:
    """Tests for different index types."""
    
    def test_flat_index(self):
        """Test flat index."""
        store = FAISSStore(dimension=32, index_type="flat")
        assert store.index_type == "flat"
    
    def test_ivf_index(self):
        """Test IVF index."""
        store = FAISSStore(dimension=32, index_type="ivf", nlist=10)
        
        # Add enough documents to train
        docs = [Document(f"doc_{i}", f"text {i}") for i in range(100)]
        embeddings = np.random.randn(100, 32).astype(np.float32)
        
        store.add_documents(docs, embeddings)
        assert len(store) == 100
    
    def test_hnsw_index(self):
        """Test HNSW index."""
        store = FAISSStore(dimension=32, index_type="hnsw")
        assert store.index_type == "hnsw"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

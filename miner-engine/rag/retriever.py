"""
RAG Retriever - FAISS + Embedder Integration

Provides high-level retrieval interface combining:
- RAGEmbedder for query/document embedding
- FAISSStore for vector similarity search

Features:
- Add documents with automatic embedding
- Search with query embedding
- Hybrid search (vector + keyword)
- Document management
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

from .embedder import RAGEmbedder, EmbedderConfig, create_embedder
from .faiss_store import FAISSStore, Document, SearchResult

logger = logging.getLogger(__name__)


class IndexType(Enum):
    """FAISS index types."""
    FLAT = "flat"
    IVF = "ivf"
    HNSW = "hnsw"

logger = logging.getLogger(__name__)


@dataclass
class RetrieverConfig:
    """Configuration for RAG Retriever."""
    # Embedder settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    
    # FAISS settings
    index_type: IndexType = IndexType.FLAT
    
    # Search settings
    default_top_k: int = 5
    similarity_threshold: float = 0.0
    
    # Storage
    persist_dir: Optional[str] = None


@dataclass
class RetrievalResult:
    """Result from retrieval operation."""
    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class RAGRetriever:
    """
    High-level RAG retriever combining embedding and vector search.
    
    Provides a simple interface for:
    - Adding documents (auto-embedded)
    - Searching by query (auto-embedded)
    - Managing document collection
    """
    
    def __init__(self, config: Optional[RetrieverConfig] = None):
        """
        Initialize RAG retriever.
        
        Args:
            config: Retriever configuration
        """
        self.config = config or RetrieverConfig()
        
        # Initialize embedder
        embedder_config = EmbedderConfig(
            model_name=self.config.embedding_model,
            device=self.config.embedding_device,
        )
        self.embedder = RAGEmbedder(embedder_config)
        
        # Initialize FAISS store
        self.store = FAISSStore(
            dimension=self.embedder.embedding_dim,
            index_type=self.config.index_type.value,  # Convert enum to string
        )
        
        # Document content storage (FAISS only stores vectors)
        self._documents: Dict[str, Document] = {}
        
        logger.info(
            f"RAGRetriever initialized: "
            f"model={self.config.embedding_model}, "
            f"index_type={self.config.index_type.value}"
        )
    
    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a single document.
        
        Args:
            doc_id: Unique document ID
            content: Document text content
            metadata: Optional metadata
        """
        # Embed document
        embedding = self.embedder.embed(content)
        
        # Create document (using faiss_store.Document fields)
        doc = Document(
            doc_id=doc_id,
            text=content,
            embedding=embedding,
            metadata=metadata or {},
        )
        
        # Add to store
        self.store.add_documents([doc])
        self._documents[doc_id] = doc
        
        logger.debug(f"Added document: {doc_id}")
    
    def add_documents(
        self,
        documents: List[Tuple[str, str, Optional[Dict[str, Any]]]],
        show_progress: bool = True,
    ) -> int:
        """
        Add multiple documents in batch.
        
        Args:
            documents: List of (doc_id, content, metadata) tuples
            show_progress: Show progress bar
            
        Returns:
            Number of documents added
        """
        if not documents:
            return 0
        
        # Extract contents for batch embedding
        doc_ids = [d[0] for d in documents]
        contents = [d[1] for d in documents]
        metadatas = [d[2] or {} for d in documents]
        
        # Batch embed
        embeddings = self.embedder.embed_documents(contents, show_progress=show_progress)
        
        # Create documents
        docs = []
        for i, (doc_id, content, metadata) in enumerate(documents):
            doc = Document(
                doc_id=doc_id,
                text=content,
                embedding=embeddings[i],
                metadata=metadata or {},
            )
            docs.append(doc)
            self._documents[doc_id] = doc
        
        # Add to store
        self.store.add_documents(docs)
        
        logger.info(f"Added {len(docs)} documents")
        return len(docs)
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            filter_metadata: Filter by metadata (exact match)
            
        Returns:
            List of RetrievalResult sorted by relevance
        """
        top_k = top_k or self.config.default_top_k
        threshold = threshold if threshold is not None else self.config.similarity_threshold
        
        # Embed query
        query_embedding = self.embedder.embed_query(query)
        
        # Search FAISS
        results = self.store.search(
            query_embedding,
            k=top_k * 2 if filter_metadata else top_k,  # Over-fetch if filtering
            threshold=threshold,
        )
        
        # Convert to RetrievalResult and apply metadata filter
        retrieval_results = []
        for r in results:
            # Get full document
            doc = self._documents.get(r.doc_id)
            if not doc:
                continue
            
            # Apply metadata filter
            if filter_metadata:
                if not self._matches_filter(doc.metadata, filter_metadata):
                    continue
            
            retrieval_results.append(RetrievalResult(
                doc_id=r.doc_id,
                content=doc.text,
                score=r.score,
                metadata=doc.metadata,
            ))
            
            if len(retrieval_results) >= top_k:
                break
        
        return retrieval_results
    
    def search_batch(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
    ) -> List[List[RetrievalResult]]:
        """
        Search for multiple queries in batch.
        
        Args:
            queries: List of search queries
            top_k: Number of results per query
            
        Returns:
            List of result lists
        """
        top_k = top_k or self.config.default_top_k
        
        # Batch embed queries
        query_embeddings = self.embedder.embed_batch(queries)
        
        # Batch search
        all_results = self.store.search_batch(query_embeddings, k=top_k)
        
        # Convert to RetrievalResult
        retrieval_results = []
        for results in all_results:
            query_results = []
            for r in results:
                doc = self._documents.get(r.doc_id)
                if doc:
                    query_results.append(RetrievalResult(
                        doc_id=r.doc_id,
                        content=doc.text,
                        score=r.score,
                        metadata=doc.metadata,
                    ))
            retrieval_results.append(query_results)
        
        return retrieval_results
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Get a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document or None if not found
        """
        return self._documents.get(doc_id)
    
    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document.
        
        Args:
            doc_id: Document ID to remove
            
        Returns:
            True if removed, False if not found
        """
        if doc_id not in self._documents:
            return False
        
        # Remove from store
        self.store.remove(doc_id)
        
        # Remove from local storage
        del self._documents[doc_id]
        
        logger.debug(f"Removed document: {doc_id}")
        return True
    
    def clear(self) -> None:
        """Clear all documents."""
        # Remove all documents from store
        for doc_id in list(self._documents.keys()):
            self.store.remove(doc_id)
        self._documents.clear()
        logger.info("Cleared all documents")
    
    def _matches_filter(
        self,
        metadata: Dict[str, Any],
        filter_dict: Dict[str, Any],
    ) -> bool:
        """Check if metadata matches filter."""
        for key, value in filter_dict.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save retriever state to disk.
        
        Args:
            path: Directory path (uses config.persist_dir if not provided)
        """
        save_path = Path(path or self.config.persist_dir)
        if not save_path:
            raise ValueError("No save path provided")
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        self.store.save(str(save_path / "faiss_index"))
        
        # Save documents
        import json
        docs_data = {
            doc_id: {
                'text': doc.text,
                'metadata': doc.metadata,
            }
            for doc_id, doc in self._documents.items()
        }
        
        with open(save_path / "documents.json", 'w', encoding='utf-8') as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved retriever to {save_path}")
    
    def load(self, path: Optional[str] = None) -> None:
        """
        Load retriever state from disk.
        
        Args:
            path: Directory path (uses config.persist_dir if not provided)
        """
        load_path = Path(path or self.config.persist_dir)
        if not load_path:
            raise ValueError("No load path provided")
        
        # Load FAISS index
        self.store.load(str(load_path / "faiss_index"))
        
        # Load documents
        import json
        with open(load_path / "documents.json", 'r', encoding='utf-8') as f:
            docs_data = json.load(f)
        
        # Reconstruct documents (embeddings are in FAISS)
        self._documents.clear()
        for doc_id, data in docs_data.items():
            # Get embedding from store
            store_doc = self.store.get(doc_id)
            embedding = store_doc.embedding if store_doc else np.zeros(self.embedder.embedding_dim)
            
            self._documents[doc_id] = Document(
                doc_id=doc_id,
                text=data.get('text', data.get('content', '')),  # Support both old and new format
                embedding=embedding,
                metadata=data.get('metadata', {}),
            )
        
        logger.info(f"Loaded retriever from {load_path}")
    
    def get_stats(self) -> Dict:
        """Get retriever statistics."""
        return {
            'num_documents': len(self._documents),
            'embedding_model': self.config.embedding_model,
            'index_type': self.config.index_type.value,
            'embedder_stats': self.embedder.get_stats(),
            'store_stats': self.store.get_stats(),
        }


# Convenience function
def create_retriever(
    embedding_model: str = "all-MiniLM-L6-v2",
    index_type: str = "flat",
    persist_dir: Optional[str] = None,
) -> RAGRetriever:
    """
    Create a RAG retriever with common settings.
    
    Args:
        embedding_model: Sentence transformer model name
        index_type: FAISS index type (flat, ivf, hnsw)
        persist_dir: Directory for persistence
        
    Returns:
        Configured RAGRetriever
    """
    index_type_enum = IndexType(index_type.lower())
    
    config = RetrieverConfig(
        embedding_model=embedding_model,
        index_type=index_type_enum,
        persist_dir=persist_dir,
    )
    
    return RAGRetriever(config)

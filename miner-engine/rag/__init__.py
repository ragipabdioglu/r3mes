"""
RAG (Retrieval-Augmented Generation) Module

Provides hybrid RAG capabilities:
- FAISS vector store for fast similarity search
- RAG Embedder for document/query embedding
- RAG Retriever for high-level retrieval interface
- Document processing and chunking
"""

from .faiss_store import FAISSStore, SearchResult, Document
from .embedder import RAGEmbedder, EmbedderConfig, create_embedder
from .retriever import RAGRetriever, RetrieverConfig, RetrievalResult, create_retriever, IndexType

__all__ = [
    # FAISS Store
    'FAISSStore',
    'SearchResult',
    'Document',
    # Embedder
    'RAGEmbedder',
    'EmbedderConfig',
    'create_embedder',
    # Retriever
    'RAGRetriever',
    'RetrieverConfig',
    'RetrievalResult',
    'create_retriever',
    'IndexType',
]

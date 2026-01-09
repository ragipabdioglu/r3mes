"""
RAG Embedder for Document and Query Embedding

Provides embedding functionality for RAG system using sentence-transformers.
Supports batch embedding, caching, and multiple embedding models.

Features:
- Lazy model loading
- Batch embedding with configurable batch size
- Embedding cache for repeated queries
- Support for multiple models (all-MiniLM-L6-v2, multilingual-e5-small)
"""

import numpy as np
from typing import List, Optional, Dict, Union
from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Lazy import for sentence-transformers
_sentence_transformer = None


def _get_sentence_transformer():
    """Lazy load sentence-transformers."""
    global _sentence_transformer
    if _sentence_transformer is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sentence_transformer = SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers required. "
                "Install with: pip install sentence-transformers"
            )
    return _sentence_transformer


@dataclass
class EmbedderConfig:
    """Configuration for RAG Embedder."""
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
    normalize: bool = True
    cache_enabled: bool = True
    cache_max_size: int = 10000


class RAGEmbedder:
    """
    Embedder for RAG document and query embedding.
    
    Uses sentence-transformers for high-quality embeddings.
    Supports caching for repeated queries.
    """
    
    # Known models and their dimensions
    MODEL_DIMS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "multilingual-e5-small": 384,
        "multilingual-e5-base": 768,
        "paraphrase-multilingual-MiniLM-L12-v2": 384,
    }
    
    def __init__(self, config: Optional[EmbedderConfig] = None):
        """
        Initialize RAG embedder.
        
        Args:
            config: Embedder configuration
        """
        self.config = config or EmbedderConfig()
        
        # Lazy model loading
        self._model = None
        
        # Embedding cache
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Get embedding dimension
        self._dim = self.MODEL_DIMS.get(self.config.model_name, 384)
        
        logger.info(
            f"RAGEmbedder created: model={self.config.model_name}, "
            f"dim={self._dim}, device={self.config.device}"
        )
    
    @property
    def model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            SentenceTransformer = _get_sentence_transformer()
            self._model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device,
            )
            logger.info(f"Loaded embedding model: {self.config.model_name}")
        return self._model
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self._dim
    
    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector (normalized if configured)
        """
        # Check cache
        if self.config.cache_enabled:
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                self._cache_hits += 1
                return self._cache[cache_key].copy()
            self._cache_misses += 1
        
        # Compute embedding
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
        )
        
        # Cache result
        if self.config.cache_enabled:
            self._add_to_cache(cache_key, embedding)
        
        return embedding
    
    def embed_batch(
        self,
        texts: List[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            show_progress: Show progress bar
            
        Returns:
            Array of embeddings, shape (n_texts, dim)
        """
        if not texts:
            return np.array([]).reshape(0, self._dim)
        
        # Check cache for all texts
        if self.config.cache_enabled:
            cached_indices = []
            uncached_indices = []
            uncached_texts = []
            
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self._cache:
                    cached_indices.append((i, cache_key))
                    self._cache_hits += 1
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
                    self._cache_misses += 1
            
            # If all cached, return from cache
            if not uncached_texts:
                embeddings = np.zeros((len(texts), self._dim))
                for i, cache_key in cached_indices:
                    embeddings[i] = self._cache[cache_key]
                return embeddings
            
            # Compute uncached embeddings
            new_embeddings = self.model.encode(
                uncached_texts,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize,
                batch_size=self.config.batch_size,
                show_progress_bar=show_progress,
            )
            
            # Combine cached and new embeddings
            embeddings = np.zeros((len(texts), self._dim))
            
            for i, cache_key in cached_indices:
                embeddings[i] = self._cache[cache_key]
            
            for idx, (i, text) in enumerate(zip(uncached_indices, uncached_texts)):
                embeddings[i] = new_embeddings[idx]
                cache_key = self._get_cache_key(text)
                self._add_to_cache(cache_key, new_embeddings[idx])
            
            return embeddings
        
        # No caching, direct batch encode
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
            batch_size=self.config.batch_size,
            show_progress_bar=show_progress,
        )
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query (alias for embed with query-specific handling).
        
        Args:
            query: Query text
            
        Returns:
            Query embedding
        """
        return self.embed(query)
    
    def embed_documents(
        self,
        documents: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed multiple documents.
        
        Args:
            documents: List of document texts
            show_progress: Show progress bar
            
        Returns:
            Array of document embeddings
        """
        return self.embed_batch(documents, show_progress=show_progress)
    
    def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        if self.config.normalize:
            # Already normalized, just dot product
            return float(np.dot(embedding1, embedding2))
        else:
            # Compute cosine similarity
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
    
    def similarity_batch(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute similarity between query and multiple documents.
        
        Args:
            query_embedding: Query embedding (1D)
            document_embeddings: Document embeddings (2D)
            
        Returns:
            Array of similarity scores
        """
        if self.config.normalize:
            return document_embeddings @ query_embedding
        else:
            query_norm = np.linalg.norm(query_embedding)
            doc_norms = np.linalg.norm(document_embeddings, axis=1)
            
            # Avoid division by zero
            doc_norms = np.where(doc_norms == 0, 1, doc_norms)
            
            if query_norm == 0:
                return np.zeros(len(document_embeddings))
            
            return (document_embeddings @ query_embedding) / (doc_norms * query_norm)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _add_to_cache(self, key: str, embedding: np.ndarray) -> None:
        """Add embedding to cache with size limit."""
        if len(self._cache) >= self.config.cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[key] = embedding.copy()
    
    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Embedding cache cleared")
    
    def get_stats(self) -> Dict:
        """Get embedder statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (
            self._cache_hits / total_requests
            if total_requests > 0 else 0.0
        )
        
        return {
            'model_name': self.config.model_name,
            'embedding_dim': self._dim,
            'device': self.config.device,
            'cache_enabled': self.config.cache_enabled,
            'cache_size': len(self._cache),
            'cache_max_size': self.config.cache_max_size,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_hit_rate': hit_rate,
        }


# Convenience function
def create_embedder(
    model_name: str = "all-MiniLM-L6-v2",
    device: str = "cpu",
    cache_enabled: bool = True,
) -> RAGEmbedder:
    """
    Create a RAG embedder with common settings.
    
    Args:
        model_name: Sentence transformer model name
        device: Device for inference
        cache_enabled: Enable embedding cache
        
    Returns:
        Configured RAGEmbedder
    """
    config = EmbedderConfig(
        model_name=model_name,
        device=device,
        cache_enabled=cache_enabled,
    )
    return RAGEmbedder(config)

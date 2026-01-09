"""
FAISS Vector Store for RAG

Provides fast similarity search using FAISS:
- Dense vector indexing
- GPU acceleration (optional)
- Serialization for IPFS storage
- Multiple index types (Flat, IVF, HNSW)

Reference: https://github.com/facebookresearch/faiss
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not installed. Install with: pip install faiss-cpu or faiss-gpu")


@dataclass
class SearchResult:
    """Result from similarity search."""
    doc_id: str
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """Document with embedding."""
    doc_id: str
    text: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class FAISSStore:
    """
    FAISS-based vector store for RAG.
    
    Supports multiple index types:
    - Flat: Exact search, best for small datasets
    - IVF: Approximate search, good for medium datasets
    - HNSW: Graph-based, good for large datasets
    
    Usage:
        store = FAISSStore(dimension=768)
        store.add_documents(docs, embeddings)
        results = store.search(query_embedding, k=5)
    """
    
    def __init__(
        self,
        dimension: int = 768,
        index_type: str = "flat",
        use_gpu: bool = False,
        nlist: int = 100,  # For IVF
        nprobe: int = 10,  # For IVF search
    ):
        """
        Initialize FAISS store.
        
        Args:
            dimension: Embedding dimension
            index_type: Index type ("flat", "ivf", "hnsw")
            use_gpu: Use GPU acceleration
            nlist: Number of clusters for IVF
            nprobe: Number of clusters to search for IVF
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is required. Install with: pip install faiss-cpu")
        
        self.dimension = dimension
        self.index_type = index_type
        self.use_gpu = use_gpu
        self.nlist = nlist
        self.nprobe = nprobe
        
        # Document storage
        self._documents: Dict[int, Document] = {}
        self._doc_id_to_idx: Dict[str, int] = {}
        self._idx_to_doc_id: Dict[int, str] = {}
        self._next_idx: int = 0
        
        # Create index
        self._index = self._create_index()
        
        logger.info(
            f"FAISSStore initialized: dim={dimension}, type={index_type}, "
            f"gpu={use_gpu}"
        )
    
    def _create_index(self) -> 'faiss.Index':
        """Create FAISS index based on type."""
        if self.index_type == "flat":
            index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine after normalization)
        
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
            index.nprobe = self.nprobe
        
        elif self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(self.dimension, 32)  # 32 neighbors
        
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Move to GPU if requested
        if self.use_gpu and faiss.get_num_gpus() > 0:
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
            logger.info("FAISS index moved to GPU")
        
        return index
    
    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[np.ndarray] = None,
    ):
        """
        Add documents to the store.
        
        Args:
            documents: List of Document objects
            embeddings: Optional pre-computed embeddings [n_docs, dimension]
        """
        if not documents:
            return
        
        # Get or use provided embeddings
        if embeddings is None:
            embeddings = np.array([doc.embedding for doc in documents])
        
        # Normalize for cosine similarity
        embeddings = self._normalize(embeddings)
        
        # Train IVF index if needed
        if self.index_type == "ivf" and not self._index.is_trained:
            logger.info("Training IVF index...")
            self._index.train(embeddings)
        
        # Add to index
        self._index.add(embeddings)
        
        # Store documents
        for i, doc in enumerate(documents):
            idx = self._next_idx + i
            self._documents[idx] = doc
            self._doc_id_to_idx[doc.doc_id] = idx
            self._idx_to_doc_id[idx] = doc.doc_id
        
        self._next_idx += len(documents)
        
        logger.info(f"Added {len(documents)} documents. Total: {self._next_idx}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        threshold: float = 0.0,
    ) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding [dimension] or [1, dimension]
            k: Number of results to return
            threshold: Minimum similarity score
            
        Returns:
            List of SearchResult sorted by score (descending)
        """
        if self._index.ntotal == 0:
            return []
        
        # Reshape if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize
        query_embedding = self._normalize(query_embedding)
        
        # Search
        scores, indices = self._index.search(query_embedding, k)
        
        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for missing results
                continue
            
            if score < threshold:
                continue
            
            doc_id = self._idx_to_doc_id.get(idx)
            if doc_id is None:
                continue
            
            doc = self._documents.get(idx)
            if doc is None:
                continue
            
            results.append(SearchResult(
                doc_id=doc_id,
                score=float(score),
                text=doc.text,
                metadata=doc.metadata,
            ))
        
        return results
    
    def search_batch(
        self,
        query_embeddings: np.ndarray,
        k: int = 5,
    ) -> List[List[SearchResult]]:
        """
        Batch search for similar documents.
        
        Args:
            query_embeddings: Query embeddings [n_queries, dimension]
            k: Number of results per query
            
        Returns:
            List of result lists
        """
        if self._index.ntotal == 0:
            return [[] for _ in range(len(query_embeddings))]
        
        # Normalize
        query_embeddings = self._normalize(query_embeddings)
        
        # Search
        scores, indices = self._index.search(query_embeddings, k)
        
        # Build results
        all_results = []
        for query_scores, query_indices in zip(scores, indices):
            results = []
            for score, idx in zip(query_scores, query_indices):
                if idx < 0:
                    continue
                
                doc_id = self._idx_to_doc_id.get(idx)
                if doc_id is None:
                    continue
                
                doc = self._documents.get(idx)
                if doc is None:
                    continue
                
                results.append(SearchResult(
                    doc_id=doc_id,
                    score=float(score),
                    text=doc.text,
                    metadata=doc.metadata,
                ))
            
            all_results.append(results)
        
        return all_results
    
    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings for cosine similarity."""
        embeddings = embeddings.astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        return embeddings / norms
    
    def remove(self, doc_id: str) -> bool:
        """
        Remove document from store.
        
        Note: FAISS doesn't support efficient removal.
        This marks the document as removed but doesn't free index space.
        
        Args:
            doc_id: Document ID to remove
            
        Returns:
            True if removed
        """
        if doc_id not in self._doc_id_to_idx:
            return False
        
        idx = self._doc_id_to_idx[doc_id]
        del self._documents[idx]
        del self._doc_id_to_idx[doc_id]
        del self._idx_to_doc_id[idx]
        
        logger.debug(f"Removed document: {doc_id}")
        return True
    
    def get(self, doc_id: str) -> Optional[Document]:
        """Get document by ID."""
        idx = self._doc_id_to_idx.get(doc_id)
        if idx is None:
            return None
        return self._documents.get(idx)
    
    def save(self, path: str):
        """
        Save store to disk.
        
        Args:
            path: Directory path to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save index
        index_path = path / "index.faiss"
        if self.use_gpu:
            # Move to CPU for saving
            cpu_index = faiss.index_gpu_to_cpu(self._index)
            faiss.write_index(cpu_index, str(index_path))
        else:
            faiss.write_index(self._index, str(index_path))
        
        # Save documents and mappings
        docs_data = {
            'documents': {
                str(idx): {
                    'doc_id': doc.doc_id,
                    'text': doc.text,
                    'metadata': doc.metadata,
                }
                for idx, doc in self._documents.items()
            },
            'doc_id_to_idx': self._doc_id_to_idx,
            'idx_to_doc_id': {str(k): v for k, v in self._idx_to_doc_id.items()},
            'next_idx': self._next_idx,
            'config': {
                'dimension': self.dimension,
                'index_type': self.index_type,
                'nlist': self.nlist,
                'nprobe': self.nprobe,
            }
        }
        
        docs_path = path / "documents.json"
        with open(docs_path, 'w', encoding='utf-8') as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved FAISSStore to {path}")
    
    @classmethod
    def load(cls, path: str, use_gpu: bool = False) -> 'FAISSStore':
        """
        Load store from disk.
        
        Args:
            path: Directory path to load from
            use_gpu: Use GPU acceleration
            
        Returns:
            Loaded FAISSStore
        """
        path = Path(path)
        
        # Load documents and config
        docs_path = path / "documents.json"
        with open(docs_path, 'r', encoding='utf-8') as f:
            docs_data = json.load(f)
        
        config = docs_data['config']
        
        # Create store
        store = cls(
            dimension=config['dimension'],
            index_type=config['index_type'],
            use_gpu=use_gpu,
            nlist=config.get('nlist', 100),
            nprobe=config.get('nprobe', 10),
        )
        
        # Load index
        index_path = path / "index.faiss"
        store._index = faiss.read_index(str(index_path))
        
        if use_gpu and faiss.get_num_gpus() > 0:
            store._index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, store._index
            )
        
        # Restore documents
        for idx_str, doc_data in docs_data['documents'].items():
            idx = int(idx_str)
            store._documents[idx] = Document(
                doc_id=doc_data['doc_id'],
                text=doc_data['text'],
                metadata=doc_data.get('metadata', {}),
            )
        
        store._doc_id_to_idx = docs_data['doc_id_to_idx']
        store._idx_to_doc_id = {int(k): v for k, v in docs_data['idx_to_doc_id'].items()}
        store._next_idx = docs_data['next_idx']
        
        logger.info(f"Loaded FAISSStore from {path} ({store._next_idx} documents)")
        return store
    
    def __len__(self) -> int:
        """Get number of documents."""
        return len(self._documents)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        return {
            'num_documents': len(self._documents),
            'index_size': self._index.ntotal,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'is_trained': getattr(self._index, 'is_trained', True),
        }

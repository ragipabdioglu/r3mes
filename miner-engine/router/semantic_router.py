"""
Semantic Router for DoRA Expert Selection

Embedding-based routing using sentence-transformers.
Stage 2 of the multi-stage routing pipeline.

Features:
- all-MiniLM-L6-v2 model (22M params, 384 dim)
- Max pooling over expert embeddings
- Cosine similarity scoring
- ~10ms latency per query
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging
import json

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
class SemanticResult:
    """Result from semantic routing."""
    expert_id: str
    similarity: float
    embedding_source: str = "description"  # description, examples, combined


@dataclass
class ExpertEmbedding:
    """Embedding data for a DoRA expert."""
    expert_id: str
    description: str
    examples: List[str] = field(default_factory=list)
    description_embedding: Optional[np.ndarray] = None
    example_embeddings: Optional[np.ndarray] = None  # Shape: (n_examples, dim)
    combined_embedding: Optional[np.ndarray] = None  # Max-pooled


class SemanticRouter:
    """
    Embedding-based semantic router for expert selection.
    
    Uses all-MiniLM-L6-v2 for fast, accurate semantic matching.
    Computes cosine similarity between query and expert embeddings.
    """
    
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    
    # Default expert definitions
    DEFAULT_EXPERTS: Dict[str, Dict] = {
        "medical_dora": {
            "description": "Medical and healthcare expert for diseases, treatments, symptoms, and health advice",
            "examples": [
                "What are the symptoms of diabetes?",
                "How to treat high blood pressure?",
                "Explain the side effects of aspirin",
                "Diyabet belirtileri nelerdir?",
                "Yüksek tansiyon nasıl tedavi edilir?",
            ],
        },
        "legal_dora": {
            "description": "Legal expert for laws, contracts, court procedures, and legal advice",
            "examples": [
                "How to write a contract?",
                "What are my rights as a tenant?",
                "Explain copyright law",
                "Sözleşme nasıl yazılır?",
                "Kiracı hakları nelerdir?",
            ],
        },
        "coding_dora": {
            "description": "Programming and software development expert for code, debugging, and technical questions",
            "examples": [
                "How to implement a binary search in Python?",
                "Debug this JavaScript code",
                "Explain async/await in TypeScript",
                "Python'da liste nasıl sıralanır?",
                "React component lifecycle nedir?",
            ],
        },
        "finance_dora": {
            "description": "Finance and investment expert for stocks, crypto, banking, and financial planning",
            "examples": [
                "How to diversify my portfolio?",
                "Explain compound interest",
                "What is DeFi?",
                "Hisse senedi nasıl alınır?",
                "Kripto para yatırımı güvenli mi?",
            ],
        },
        "science_dora": {
            "description": "Science expert for physics, chemistry, biology, and scientific concepts",
            "examples": [
                "Explain quantum entanglement",
                "How does photosynthesis work?",
                "What is the theory of relativity?",
                "Kuantum dolanıklık nedir?",
                "Fotosentez nasıl çalışır?",
            ],
        },
        "history_dora": {
            "description": "History expert for historical events, civilizations, and historical analysis",
            "examples": [
                "What caused World War I?",
                "Explain the Roman Empire's fall",
                "Who was Genghis Khan?",
                "Osmanlı İmparatorluğu nasıl kuruldu?",
                "Fransız Devrimi neden oldu?",
            ],
        },
        "education_dora": {
            "description": "Education expert for learning methods, teaching strategies, and academic guidance",
            "examples": [
                "How to study effectively?",
                "Best teaching methods for math",
                "How to prepare for exams?",
                "Etkili ders çalışma yöntemleri",
                "Sınava nasıl hazırlanılır?",
            ],
        },
        "turkish_dora": {
            "description": "Turkish language expert for Turkish grammar, vocabulary, and language questions",
            "examples": [
                "Türkçe dilbilgisi kuralları",
                "Bu cümlenin özne ve yüklemi nedir?",
                "Türkçe'de zarf nedir?",
            ],
        },
        "general_dora": {
            "description": "General purpose assistant for everyday questions and conversations",
            "examples": [
                "What's the weather like?",
                "Tell me a joke",
                "How are you?",
                "Bugün hava nasıl?",
                "Bana bir fıkra anlat",
            ],
        },
        "summarization_dora": {
            "description": "Summarization expert for condensing long texts into brief summaries",
            "examples": [
                "Summarize this article",
                "Give me the key points",
                "TL;DR of this document",
                "Bu metni özetle",
                "Ana fikirleri çıkar",
            ],
        },
        "translation_dora": {
            "description": "Translation expert for translating between languages",
            "examples": [
                "Translate this to English",
                "How do you say this in Turkish?",
                "Convert to German",
                "Bunu İngilizce'ye çevir",
                "Türkçe'ye tercüme et",
            ],
        },
        "creative_dora": {
            "description": "Creative writing expert for stories, poems, and creative content",
            "examples": [
                "Write a short story about space",
                "Compose a poem about love",
                "Create a dialogue for a movie scene",
                "Uzay hakkında bir hikaye yaz",
                "Aşk şiiri yaz",
            ],
        },
        "analysis_dora": {
            "description": "Analysis expert for evaluating, comparing, and examining topics in depth",
            "examples": [
                "Analyze this business strategy",
                "Compare these two approaches",
                "Evaluate the pros and cons",
                "Bu stratejiyi analiz et",
                "İki yaklaşımı karşılaştır",
            ],
        },
    }
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        experts: Optional[Dict[str, Dict]] = None,
        cache_dir: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Initialize semantic router.
        
        Args:
            model_name: Sentence transformer model name
            experts: Expert definitions (id -> {description, examples})
            cache_dir: Directory to cache embeddings
            device: Device for inference (cpu/cuda)
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Initialize model (lazy)
        self._model = None
        
        # Expert embeddings
        self._experts: Dict[str, ExpertEmbedding] = {}
        
        # Load expert definitions
        expert_defs = experts or self.DEFAULT_EXPERTS
        for expert_id, data in expert_defs.items():
            self._experts[expert_id] = ExpertEmbedding(
                expert_id=expert_id,
                description=data.get("description", ""),
                examples=data.get("examples", []),
            )
        
        self._initialized = False
        logger.info(f"SemanticRouter created with {len(self._experts)} experts")
    
    @property
    def model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            SentenceTransformer = _get_sentence_transformer()
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Loaded model: {self.model_name}")
        return self._model
    
    def initialize(self, force: bool = False) -> None:
        """
        Initialize expert embeddings.
        
        Args:
            force: Force recomputation even if cached
        """
        if self._initialized and not force:
            return
        
        # Try to load from cache
        if self.cache_dir and not force:
            if self._load_cache():
                self._initialized = True
                return
        
        # Compute embeddings
        self._compute_embeddings()
        
        # Save to cache
        if self.cache_dir:
            self._save_cache()
        
        self._initialized = True
        logger.info("SemanticRouter initialized")
    
    def _compute_embeddings(self) -> None:
        """Compute embeddings for all experts."""
        for expert_id, expert in self._experts.items():
            # Embed description
            if expert.description:
                expert.description_embedding = self.model.encode(
                    expert.description,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            
            # Embed examples
            if expert.examples:
                expert.example_embeddings = self.model.encode(
                    expert.examples,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            
            # Compute combined embedding (max pooling)
            expert.combined_embedding = self._max_pool_embeddings(expert)
            
            logger.debug(f"Computed embeddings for {expert_id}")
    
    def _max_pool_embeddings(self, expert: ExpertEmbedding) -> np.ndarray:
        """
        Max pool description and example embeddings.
        
        Args:
            expert: Expert with embeddings
            
        Returns:
            Max-pooled embedding vector
        """
        embeddings = []
        
        if expert.description_embedding is not None:
            embeddings.append(expert.description_embedding)
        
        if expert.example_embeddings is not None:
            embeddings.append(expert.example_embeddings)
        
        if not embeddings:
            return np.zeros(self.EMBEDDING_DIM)
        
        # Stack all embeddings
        if len(embeddings) == 1:
            if embeddings[0].ndim == 1:
                return embeddings[0]
            else:
                # Max pool over examples
                return np.max(embeddings[0], axis=0)
        
        # Combine description and examples
        all_embeds = []
        for emb in embeddings:
            if emb.ndim == 1:
                all_embeds.append(emb.reshape(1, -1))
            else:
                all_embeds.append(emb)
        
        stacked = np.vstack(all_embeds)
        pooled = np.max(stacked, axis=0)
        
        # Normalize
        norm = np.linalg.norm(pooled)
        if norm > 0:
            pooled = pooled / norm
        
        return pooled
    
    def route(self, query: str, top_k: int = 5) -> List[SemanticResult]:
        """
        Route query to appropriate experts using semantic similarity.
        
        Args:
            query: User query text
            top_k: Number of top results to return
            
        Returns:
            List of SemanticResult sorted by similarity (descending)
        """
        if not self._initialized:
            self.initialize()
        
        # Encode query
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        
        # Compute similarities
        results = []
        for expert_id, expert in self._experts.items():
            if expert.combined_embedding is None:
                continue
            
            # Cosine similarity (embeddings are normalized)
            similarity = float(np.dot(query_embedding, expert.combined_embedding))
            
            results.append(SemanticResult(
                expert_id=expert_id,
                similarity=similarity,
                embedding_source="combined",
            ))
        
        # Sort by similarity
        results.sort(key=lambda x: x.similarity, reverse=True)
        
        return results[:top_k]
    
    def route_batch(
        self, queries: List[str], top_k: int = 5
    ) -> List[List[SemanticResult]]:
        """
        Route multiple queries in batch.
        
        Args:
            queries: List of query texts
            top_k: Number of top results per query
            
        Returns:
            List of result lists
        """
        if not self._initialized:
            self.initialize()
        
        # Batch encode queries
        query_embeddings = self.model.encode(
            queries,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
        )
        
        # Build expert embedding matrix
        expert_ids = list(self._experts.keys())
        expert_matrix = np.vstack([
            self._experts[eid].combined_embedding
            for eid in expert_ids
            if self._experts[eid].combined_embedding is not None
        ])
        
        # Compute all similarities at once
        similarities = query_embeddings @ expert_matrix.T
        
        # Build results
        all_results = []
        for i, query in enumerate(queries):
            results = []
            for j, expert_id in enumerate(expert_ids):
                if self._experts[expert_id].combined_embedding is None:
                    continue
                results.append(SemanticResult(
                    expert_id=expert_id,
                    similarity=float(similarities[i, j]),
                    embedding_source="combined",
                ))
            results.sort(key=lambda x: x.similarity, reverse=True)
            all_results.append(results[:top_k])
        
        return all_results
    
    def get_similarity(self, query: str, expert_id: str) -> float:
        """
        Get similarity between query and specific expert.
        
        Args:
            query: Query text
            expert_id: Expert ID
            
        Returns:
            Cosine similarity score
        """
        if not self._initialized:
            self.initialize()
        
        if expert_id not in self._experts:
            return 0.0
        
        expert = self._experts[expert_id]
        if expert.combined_embedding is None:
            return 0.0
        
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        
        return float(np.dot(query_embedding, expert.combined_embedding))
    
    def add_expert(
        self,
        expert_id: str,
        description: str,
        examples: Optional[List[str]] = None,
    ) -> None:
        """
        Add a new expert dynamically.
        
        Args:
            expert_id: Unique expert ID
            description: Expert description
            examples: Example queries for this expert
        """
        expert = ExpertEmbedding(
            expert_id=expert_id,
            description=description,
            examples=examples or [],
        )
        
        # Compute embeddings if initialized
        if self._initialized:
            if description:
                expert.description_embedding = self.model.encode(
                    description,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            
            if examples:
                expert.example_embeddings = self.model.encode(
                    examples,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            
            expert.combined_embedding = self._max_pool_embeddings(expert)
        
        self._experts[expert_id] = expert
        logger.info(f"Added expert: {expert_id}")
    
    def remove_expert(self, expert_id: str) -> bool:
        """
        Remove an expert.
        
        Args:
            expert_id: Expert ID to remove
            
        Returns:
            True if removed, False if not found
        """
        if expert_id in self._experts:
            del self._experts[expert_id]
            logger.info(f"Removed expert: {expert_id}")
            return True
        return False
    
    def get_expert_ids(self) -> List[str]:
        """Get all expert IDs."""
        return list(self._experts.keys())
    
    def _load_cache(self) -> bool:
        """Load embeddings from cache."""
        if not self.cache_dir:
            return False
        
        cache_file = self.cache_dir / "semantic_router_cache.npz"
        meta_file = self.cache_dir / "semantic_router_meta.json"
        
        if not cache_file.exists() or not meta_file.exists():
            return False
        
        try:
            # Load metadata
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            
            # Check model compatibility
            if meta.get('model_name') != self.model_name:
                logger.info("Cache model mismatch, recomputing")
                return False
            
            # Load embeddings
            data = np.load(cache_file)
            
            for expert_id in meta.get('expert_ids', []):
                if expert_id not in self._experts:
                    continue
                
                expert = self._experts[expert_id]
                
                key_desc = f"{expert_id}_desc"
                key_examples = f"{expert_id}_examples"
                key_combined = f"{expert_id}_combined"
                
                if key_desc in data:
                    expert.description_embedding = data[key_desc]
                if key_examples in data:
                    expert.example_embeddings = data[key_examples]
                if key_combined in data:
                    expert.combined_embedding = data[key_combined]
            
            logger.info(f"Loaded embeddings from cache: {cache_file}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return False
    
    def _save_cache(self) -> None:
        """Save embeddings to cache."""
        if not self.cache_dir:
            return
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_file = self.cache_dir / "semantic_router_cache.npz"
        meta_file = self.cache_dir / "semantic_router_meta.json"
        
        try:
            # Prepare data
            arrays = {}
            expert_ids = []
            
            for expert_id, expert in self._experts.items():
                expert_ids.append(expert_id)
                
                if expert.description_embedding is not None:
                    arrays[f"{expert_id}_desc"] = expert.description_embedding
                if expert.example_embeddings is not None:
                    arrays[f"{expert_id}_examples"] = expert.example_embeddings
                if expert.combined_embedding is not None:
                    arrays[f"{expert_id}_combined"] = expert.combined_embedding
            
            # Save embeddings
            np.savez(cache_file, **arrays)
            
            # Save metadata
            meta = {
                'model_name': self.model_name,
                'expert_ids': expert_ids,
                'embedding_dim': self.EMBEDDING_DIM,
            }
            with open(meta_file, 'w') as f:
                json.dump(meta, f)
            
            logger.info(f"Saved embeddings to cache: {cache_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def get_status(self) -> Dict:
        """Get router status."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'initialized': self._initialized,
            'num_experts': len(self._experts),
            'expert_ids': list(self._experts.keys()),
            'embedding_dim': self.EMBEDDING_DIM,
            'cache_dir': str(self.cache_dir) if self.cache_dir else None,
        }

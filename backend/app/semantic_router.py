"""
Semantic Router - Embedding-based intelligent routing

Kullanıcı mesajını semantic similarity ile analiz edip uygun LoRA adaptörünü seçer.
Keyword matching yerine embedding-based similarity kullanır.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import logging

logger = logging.getLogger(__name__)


class SemanticRouter:
    """
    Semantic Router - Embedding-based intelligent routing
    
    Her adaptör için örnek prompt'ları embedding'e çevirir ve
    kullanıcı mesajının semantic similarity'sine göre en uygun adaptörü seçer.
    """
    
    def __init__(self, similarity_threshold: float = 0.7, use_semantic: bool = True):
        """
        Semantic Router'ı başlat.
        
        Args:
            similarity_threshold: Minimum similarity skoru (0.0-1.0)
            use_semantic: Semantic router kullanılsın mı? (False ise sadece keyword router)
        """
        self.use_semantic = use_semantic
        self.similarity_threshold = similarity_threshold
        
        # NOTE: Router fallback removed - SemanticRouter is now mandatory
        # If semantic router fails, it will raise RuntimeError instead of falling back
        
        # Embedding model (lazy loading)
        self.embedding_model: Optional[SentenceTransformer] = None
        
        # Route definitions (her adaptör için örnek prompt'lar)
        self.route_definitions: Dict[str, List[str]] = {
            'coder_adapter': [
                "How do I write a Python function?",
                "What's the syntax for JavaScript classes?",
                "How to debug this code error?",
                "Explain this algorithm step by step",
                "Fix this SQL query bug",
                "What's the best way to structure this API?",
                "How to optimize this code performance?",
                "Explain this programming concept",
                "How to implement a sorting algorithm?",
                "What's wrong with my code?",
                "How to use this library?",
                "Explain this data structure",
            ],
            'law_adapter': [
                "What are my legal rights in this situation?",
                "Explain this contract clause",
                "What does this law mean?",
                "How to file a lawsuit?",
                "What are the legal implications?",
                "Explain this legal term",
                "What's the jurisdiction for this case?",
                "How to draft a legal agreement?",
                "What are my legal obligations?",
                "Explain this legal document",
                "What does this statute mean?",
                "How to resolve this legal dispute?",
            ],
            'default_adapter': [
                "General question",
                "Help me understand",
                "What is",
                "Tell me about",
            ]
        }
        
        # Pre-computed route embeddings
        self.route_embeddings: Dict[str, np.ndarray] = {}
        
        # Initialize semantic router if enabled
        if self.use_semantic:
            try:
                self._initialize_semantic_routes()
                logger.info("Semantic router initialized successfully")
            except ImportError as e:
                logger.error(f"Failed to import required dependencies for semantic router: {e}")
                logger.error("Please install sentence-transformers: pip install sentence-transformers")
                raise RuntimeError(
                    "SemanticRouter initialization failed: missing dependencies. "
                    "Router fallback has been removed. Please install: pip install sentence-transformers"
                ) from e
            except Exception as e:
                logger.error(f"Failed to initialize semantic router: {e}", exc_info=True)
                raise RuntimeError(
                    "SemanticRouter initialization failed. Router fallback has been removed. "
                    "Please ensure all dependencies are installed and configured correctly."
                ) from e
    
    def _initialize_semantic_routes(self):
        """Her route için embedding'leri önceden hesapla."""
        # Load embedding model (lazy loading)
        if self.embedding_model is None:
            try:
                logger.info("Loading embedding model: all-MiniLM-L6-v2")
                # Use a lightweight model that doesn't require CUDA
                # all-MiniLM-L6-v2 is CPU-friendly and fast
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}", exc_info=True)
                raise
        
        # Pre-compute embeddings for each route
        for adapter_name, example_prompts in self.route_definitions.items():
            # Tüm örnek prompt'ları birleştir ve embedding'e çevir
            combined_text = " ".join(example_prompts)
            embedding = self.embedding_model.encode(
                combined_text, 
                normalize_embeddings=True,
                show_progress_bar=False
            )
            self.route_embeddings[adapter_name] = embedding
            logger.info(f"Initialized route embeddings for: {adapter_name}")
    
    def decide_adapter(self, prompt: str) -> Tuple[str, float]:
        """
        Semantic similarity ile en uygun adaptörü seç.
        
        Args:
            prompt: Kullanıcı mesajı
            
        Returns:
            (adapter_name, similarity_score) tuple
        """
        # If semantic router is disabled, raise error (fallback removed)
        if not self.use_semantic:
            raise RuntimeError(
                "SemanticRouter is disabled but Router fallback has been removed. "
                "Please enable semantic routing or use SemanticRouter with use_semantic=True."
            )
        
        try:
            # Ensure embedding model is loaded
            if self.embedding_model is None:
                self._initialize_semantic_routes()
            
            # Kullanıcı mesajını embedding'e çevir
            prompt_embedding = self.embedding_model.encode(
                prompt, 
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            # Her route ile cosine similarity hesapla
            similarities = {}
            for adapter_name, route_embedding in self.route_embeddings.items():
                if adapter_name == 'default_adapter':
                    continue
                
                # Cosine similarity (normalized embeddings için dot product yeterli)
                similarity = float(np.dot(prompt_embedding, route_embedding))
                similarities[adapter_name] = similarity
            
            # En yüksek similarity'ye sahip adaptörü bul
            if similarities:
                best_adapter = max(similarities.items(), key=lambda x: x[1])
                adapter_name, similarity_score = best_adapter
                
                # Threshold kontrolü
                if similarity_score >= self.similarity_threshold:
                    return (adapter_name, similarity_score)
            
            # Threshold altındaysa veya hiç eşleşme yoksa default
            return ('default_adapter', 0.0)
            
        except Exception as e:
            logger.error(f"Semantic router error: {e}", exc_info=True)
            # Router fallback removed - raise error instead
            raise RuntimeError(
                f"SemanticRouter failed to decide adapter: {e}. "
                "Router fallback has been removed. Please ensure semantic router is properly configured."
            ) from e
    
    def add_route_example(self, adapter_name: str, example_prompt: str):
        """
        Yeni bir route örneği ekle (runtime'da dinamik olarak).
        
        Args:
            adapter_name: Adaptör adı
            example_prompt: Örnek prompt
        """
        if adapter_name not in self.route_definitions:
            self.route_definitions[adapter_name] = []
        
        self.route_definitions[adapter_name].append(example_prompt)
        
        # Route embedding'ini yeniden hesapla (eğer semantic router aktifse)
        if self.use_semantic and self.embedding_model is not None:
            combined_text = " ".join(self.route_definitions[adapter_name])
            embedding = self.embedding_model.encode(
                combined_text, 
                normalize_embeddings=True,
                show_progress_bar=False
            )
            self.route_embeddings[adapter_name] = embedding
            logger.info(f"Added example to route: {adapter_name}")
    
    def get_available_adapters(self) -> List[str]:
        """
        Mevcut adaptörleri döndürür.
        
        Returns:
            Adaptör adları listesi
        """
        return list(self.route_definitions.keys())


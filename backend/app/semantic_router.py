"""
Semantic Router - Embedding-based intelligent routing

Kullanıcı mesajını semantic similarity ile analiz edip uygun LoRA adaptörünü seçer.
Keyword matching yerine embedding-based similarity kullanır.

NOTE: This module supports GPU-less deployment by providing a KeywordRouter fallback
when sentence-transformers is not available or when R3MES_INFERENCE_MODE != local.
"""

from typing import Dict, List, Tuple, Optional, Protocol
import os
import logging
import re

logger = logging.getLogger(__name__)

# Lazy import for sentence_transformers and numpy
_sentence_transformers_available: Optional[bool] = None
_SentenceTransformer = None
_np = None


def _ensure_embedding_libraries():
    """Lazy load embedding libraries when needed."""
    global _sentence_transformers_available, _SentenceTransformer, _np
    
    if _sentence_transformers_available is not None:
        return _sentence_transformers_available
    
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        _SentenceTransformer = SentenceTransformer
        _np = np
        _sentence_transformers_available = True
        logger.info("sentence-transformers loaded successfully")
        return True
    except ImportError as e:
        logger.warning(f"sentence-transformers not available: {e}")
        logger.warning("Will use KeywordRouter fallback")
        _sentence_transformers_available = False
        return False


class RouterProtocol(Protocol):
    """Protocol for router implementations."""
    
    def decide_adapter(self, prompt: str) -> Tuple[str, float]:
        """Decide which adapter to use for the given prompt."""
        ...
    
    def get_available_adapters(self) -> List[str]:
        """Get list of available adapters."""
        ...


class KeywordRouter:
    """
    Simple keyword-based router for GPU-less deployment.
    
    Uses regex patterns to match prompts to adapters.
    This is a fallback when sentence-transformers is not available.
    """
    
    def __init__(self, similarity_threshold: float = 0.7):
        """
        Initialize KeywordRouter.
        
        Args:
            similarity_threshold: Not used, kept for API compatibility
        """
        self.similarity_threshold = similarity_threshold
        
        # Keyword patterns for each adapter
        self.route_patterns: Dict[str, List[re.Pattern]] = {
            'coder_adapter': [
                re.compile(r'\b(code|coding|program|programming|function|class|method|api|debug|bug|error|syntax|algorithm|sql|database|library|framework|python|javascript|java|typescript|rust|go|c\+\+|html|css|react|vue|angular|node|django|flask|fastapi)\b', re.IGNORECASE),
                re.compile(r'\b(implement|optimize|refactor|compile|runtime|exception|stack\s*trace|variable|loop|array|list|dict|object|string|integer|float|boolean)\b', re.IGNORECASE),
                re.compile(r'\b(git|github|docker|kubernetes|aws|azure|gcp|ci/cd|devops|deploy|server|backend|frontend|fullstack)\b', re.IGNORECASE),
            ],
            'law_adapter': [
                re.compile(r'\b(legal|law|lawyer|attorney|court|judge|lawsuit|contract|agreement|clause|statute|regulation|jurisdiction|liability|rights|obligations|plaintiff|defendant)\b', re.IGNORECASE),
                re.compile(r'\b(sue|litigation|settlement|damages|negligence|breach|tort|criminal|civil|appeal|verdict|testimony|evidence|witness)\b', re.IGNORECASE),
                re.compile(r'\b(copyright|trademark|patent|intellectual\s*property|privacy|gdpr|compliance|terms\s*of\s*service|license)\b', re.IGNORECASE),
            ],
        }
        
        logger.info("KeywordRouter initialized (fallback mode)")
    
    def decide_adapter(self, prompt: str) -> Tuple[str, float]:
        """
        Decide adapter using keyword matching.
        
        Args:
            prompt: User message
            
        Returns:
            (adapter_name, confidence_score) tuple
        """
        scores: Dict[str, int] = {}
        
        for adapter_name, patterns in self.route_patterns.items():
            score = 0
            for pattern in patterns:
                matches = pattern.findall(prompt)
                score += len(matches)
            scores[adapter_name] = score
        
        if scores:
            best_adapter = max(scores.items(), key=lambda x: x[1])
            adapter_name, match_count = best_adapter
            
            if match_count > 0:
                # Normalize score to 0-1 range (rough approximation)
                confidence = min(1.0, match_count * 0.2)
                return (adapter_name, confidence)
        
        return ('default_adapter', 0.0)
    
    def get_available_adapters(self) -> List[str]:
        """Get list of available adapters."""
        return list(self.route_patterns.keys()) + ['default_adapter']


class SemanticRouter:
    """
    Semantic Router - Embedding-based intelligent routing
    
    Her adaptör için örnek prompt'ları embedding'e çevirir ve
    kullanıcı mesajının semantic similarity'sine göre en uygun adaptörü seçer.
    
    Falls back to KeywordRouter when sentence-transformers is not available.
    """
    
    def __init__(self, similarity_threshold: float = 0.7, use_semantic: bool = True):
        """
        Semantic Router'ı başlat.
        
        Args:
            similarity_threshold: Minimum similarity skoru (0.0-1.0)
            use_semantic: Semantic router kullanılsın mı? (False ise sadece keyword router)
        """
        self.similarity_threshold = similarity_threshold
        self.use_semantic = use_semantic
        
        # Check if we should use semantic routing
        self._fallback_router: Optional[KeywordRouter] = None
        self._semantic_available = False
        
        if self.use_semantic:
            # Try to load sentence-transformers
            if _ensure_embedding_libraries():
                self._semantic_available = True
            else:
                logger.warning("Falling back to KeywordRouter")
                self._fallback_router = KeywordRouter(similarity_threshold)
        else:
            # Explicitly disabled, use keyword router
            self._fallback_router = KeywordRouter(similarity_threshold)
        
        # Embedding model (lazy loading)
        self.embedding_model = None
        
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
        self.route_embeddings: Dict[str, any] = {}
        
        # Initialize semantic routes if available
        if self._semantic_available:
            try:
                self._initialize_semantic_routes()
                logger.info("Semantic router initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize semantic routes: {e}")
                logger.warning("Falling back to KeywordRouter")
                self._semantic_available = False
                self._fallback_router = KeywordRouter(similarity_threshold)
    
    def _initialize_semantic_routes(self):
        """Her route için embedding'leri önceden hesapla."""
        if not _sentence_transformers_available:
            raise RuntimeError("sentence-transformers not available")
        
        # Load embedding model (lazy loading)
        if self.embedding_model is None:
            try:
                logger.info("Loading embedding model: all-MiniLM-L6-v2")
                # Use a lightweight model that doesn't require CUDA
                # all-MiniLM-L6-v2 is CPU-friendly and fast
                self.embedding_model = _SentenceTransformer('all-MiniLM-L6-v2')
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
        # Use fallback router if semantic is not available
        if self._fallback_router is not None:
            return self._fallback_router.decide_adapter(prompt)
        
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
                similarity = float(_np.dot(prompt_embedding, route_embedding))
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
            # Fall back to keyword router on error
            if self._fallback_router is None:
                self._fallback_router = KeywordRouter(self.similarity_threshold)
            return self._fallback_router.decide_adapter(prompt)
    
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
        if self._semantic_available and self.embedding_model is not None:
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
    
    def is_using_semantic(self) -> bool:
        """
        Check if semantic routing is active.
        
        Returns:
            True if using semantic embeddings, False if using keyword fallback
        """
        return self._semantic_available and self._fallback_router is None


def get_router(similarity_threshold: float = 0.7, use_semantic: bool = True) -> RouterProtocol:
    """
    Factory function to get the appropriate router.
    
    Checks R3MES_INFERENCE_MODE and returns:
    - SemanticRouter if mode is LOCAL and sentence-transformers is available
    - KeywordRouter otherwise
    
    Args:
        similarity_threshold: Minimum similarity score
        use_semantic: Whether to try semantic routing
        
    Returns:
        Router instance (SemanticRouter or KeywordRouter)
    """
    # Check inference mode
    try:
        from .inference_mode import should_load_ai_libraries
        if not should_load_ai_libraries():
            logger.info("Inference mode is not LOCAL, using KeywordRouter")
            return KeywordRouter(similarity_threshold)
    except ImportError:
        pass
    
    # Try semantic router
    if use_semantic:
        return SemanticRouter(similarity_threshold, use_semantic=True)
    
    return KeywordRouter(similarity_threshold)

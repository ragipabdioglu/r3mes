"""
Hybrid Router for DoRA Expert Selection

Combines Keyword Router (fast) + Semantic Router (accurate) with score fusion.
Implements the 4-stage routing pipeline with VRAM-adaptive gating.

Pipeline:
1. Keyword Router (<1ms) - Fast pre-filter
2. Semantic Router (~10ms) - Deep understanding (skipped if keyword conf >= 0.85)
3. Score Fusion - Weighted combination
4. VRAM-Adaptive Gating - Resource-aware selection
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import time

from .keyword_router import KeywordRouter, RouterResult
from .semantic_router import SemanticRouter, SemanticResult
from .vram_adaptive_gating import VRAMAdaptiveGating, ExpertScore

logger = logging.getLogger(__name__)


@dataclass
class HybridRouterConfig:
    """Configuration for HybridRouter."""
    keyword_weight: float = 0.3
    semantic_weight: float = 0.7
    fast_path_threshold: float = 0.85
    fallback_threshold: float = 0.5
    semantic_model: str = "all-MiniLM-L6-v2"
    enable_semantic: bool = True
    cache_dir: Optional[str] = None


@dataclass
class RoutingMetrics:
    """Metrics from routing operation."""
    total_time_ms: float
    keyword_time_ms: float
    semantic_time_ms: float
    used_fast_path: bool
    keyword_confidence: float
    top_expert: str
    num_experts_selected: int


class HybridRouter:
    """
    Hybrid router combining keyword and semantic routing.
    
    Implements the design decision for multi-stage routing:
    - Stage 1: Keyword Router (fast, rule-based)
    - Stage 2: Semantic Router (embedding-based, skipped on fast path)
    - Stage 3: Score Fusion (weighted combination)
    - Stage 4: VRAM-Adaptive Gating (resource-aware selection)
    """
    
    def __init__(
        self,
        config: Optional[HybridRouterConfig] = None,
        keyword_router: Optional[KeywordRouter] = None,
        semantic_router: Optional[SemanticRouter] = None,
        gating: Optional[VRAMAdaptiveGating] = None,
    ):
        """
        Initialize hybrid router.
        
        Args:
            config: Router configuration
            keyword_router: Pre-configured keyword router (optional)
            semantic_router: Pre-configured semantic router (optional)
            gating: Pre-configured VRAM gating (optional)
        """
        self.config = config or HybridRouterConfig()
        
        # Initialize components
        self.keyword_router = keyword_router or KeywordRouter()
        
        if semantic_router:
            self.semantic_router = semantic_router
        elif self.config.enable_semantic:
            self.semantic_router = SemanticRouter(
                model_name=self.config.semantic_model,
                cache_dir=self.config.cache_dir,
            )
        else:
            self.semantic_router = None
        
        self.gating = gating or VRAMAdaptiveGating(
            confidence_threshold=self.config.fallback_threshold
        )
        
        # Metrics
        self._total_routes = 0
        self._fast_path_count = 0
        self._semantic_path_count = 0
        
        logger.info(
            f"HybridRouter initialized: "
            f"keyword_weight={self.config.keyword_weight}, "
            f"semantic_weight={self.config.semantic_weight}, "
            f"fast_path_threshold={self.config.fast_path_threshold}"
        )
    
    def initialize(self) -> None:
        """Initialize semantic router embeddings (lazy loading)."""
        if self.semantic_router:
            self.semantic_router.initialize()
            logger.info("Semantic router initialized")
    
    def route(
        self,
        query: str,
        force_semantic: bool = False,
    ) -> Tuple[List[Tuple[str, float]], RoutingMetrics]:
        """
        Route query to appropriate experts.
        
        Args:
            query: User query text
            force_semantic: Force semantic routing even on fast path
            
        Returns:
            Tuple of (selected_experts, metrics)
            - selected_experts: List of (expert_id, weight) tuples
            - metrics: Routing metrics
        """
        start_time = time.perf_counter()
        
        # Stage 1: Keyword Router
        keyword_start = time.perf_counter()
        keyword_results = self.keyword_router.route(query)
        keyword_time = (time.perf_counter() - keyword_start) * 1000
        
        max_keyword_conf = max(
            (r.confidence for r in keyword_results),
            default=0.0
        )
        
        # Check fast path
        use_fast_path = (
            max_keyword_conf >= self.config.fast_path_threshold
            and not force_semantic
            and self.config.enable_semantic
        )
        
        semantic_time = 0.0
        
        if use_fast_path:
            # Fast path: Skip semantic router
            self._fast_path_count += 1
            expert_scores = self._keyword_to_scores(keyword_results)
            logger.debug(f"Fast path: keyword_conf={max_keyword_conf:.2f}")
        else:
            # Stage 2: Semantic Router
            if self.semantic_router and self.config.enable_semantic:
                semantic_start = time.perf_counter()
                semantic_results = self.semantic_router.route(query)
                semantic_time = (time.perf_counter() - semantic_start) * 1000
                
                # Stage 3: Score Fusion
                expert_scores = self._fuse_scores(keyword_results, semantic_results)
                self._semantic_path_count += 1
            else:
                # Semantic disabled, use keyword only
                expert_scores = self._keyword_to_scores(keyword_results)
        
        # Stage 4: VRAM-Adaptive Gating
        selected = self.gating.select(expert_scores)
        
        total_time = (time.perf_counter() - start_time) * 1000
        self._total_routes += 1
        
        # Build metrics
        metrics = RoutingMetrics(
            total_time_ms=total_time,
            keyword_time_ms=keyword_time,
            semantic_time_ms=semantic_time,
            used_fast_path=use_fast_path,
            keyword_confidence=max_keyword_conf,
            top_expert=selected[0][0] if selected else "none",
            num_experts_selected=len(selected),
        )
        
        logger.debug(
            f"Route completed: {metrics.top_expert} "
            f"({metrics.total_time_ms:.1f}ms, fast_path={use_fast_path})"
        )
        
        return selected, metrics
    
    def route_simple(self, query: str) -> List[Tuple[str, float]]:
        """
        Simple routing without metrics.
        
        Args:
            query: User query text
            
        Returns:
            List of (expert_id, weight) tuples
        """
        selected, _ = self.route(query)
        return selected
    
    def _keyword_to_scores(
        self, keyword_results: List[RouterResult]
    ) -> List[ExpertScore]:
        """Convert keyword results to ExpertScore list."""
        return [
            ExpertScore(
                expert_id=r.expert_id,
                score=r.confidence,
                source="keyword",
            )
            for r in keyword_results
        ]
    
    def _fuse_scores(
        self,
        keyword_results: List[RouterResult],
        semantic_results: List[SemanticResult],
    ) -> List[ExpertScore]:
        """
        Fuse keyword and semantic scores.
        
        Uses weighted combination:
        final_score = keyword_weight * keyword_score + semantic_weight * semantic_score
        """
        scores: Dict[str, float] = {}
        sources: Dict[str, str] = {}
        
        # Add keyword scores
        for r in keyword_results:
            scores[r.expert_id] = self.config.keyword_weight * r.confidence
            sources[r.expert_id] = "keyword"
        
        # Add semantic scores
        for r in semantic_results:
            if r.expert_id in scores:
                scores[r.expert_id] += self.config.semantic_weight * r.similarity
                sources[r.expert_id] = "combined"
            else:
                scores[r.expert_id] = self.config.semantic_weight * r.similarity
                sources[r.expert_id] = "semantic"
        
        # Convert to ExpertScore list
        result = [
            ExpertScore(
                expert_id=expert_id,
                score=score,
                source=sources[expert_id],
            )
            for expert_id, score in scores.items()
        ]
        
        return sorted(result, reverse=True)
    
    def get_stats(self) -> Dict:
        """Get routing statistics."""
        fast_path_rate = (
            self._fast_path_count / self._total_routes
            if self._total_routes > 0 else 0.0
        )
        
        return {
            'total_routes': self._total_routes,
            'fast_path_count': self._fast_path_count,
            'semantic_path_count': self._semantic_path_count,
            'fast_path_rate': fast_path_rate,
            'config': {
                'keyword_weight': self.config.keyword_weight,
                'semantic_weight': self.config.semantic_weight,
                'fast_path_threshold': self.config.fast_path_threshold,
                'enable_semantic': self.config.enable_semantic,
            },
            'gating': self.gating.get_status(),
        }
    
    def reset_stats(self) -> None:
        """Reset routing statistics."""
        self._total_routes = 0
        self._fast_path_count = 0
        self._semantic_path_count = 0

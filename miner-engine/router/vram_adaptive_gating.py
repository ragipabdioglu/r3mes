"""
VRAM-Adaptive Gating for DoRA Expert Selection

Implements dynamic Top-K expert selection based on available VRAM:
- VRAM < 8GB: Top-1 (single expert)
- VRAM 8-16GB: Top-2 (two experts)
- VRAM > 16GB: Top-3 (three experts)

Features:
- Dynamic expert count based on hardware
- Weighted expert combination
- Fallback to general expert when confidence is low
"""

import torch
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExpertScore:
    """Score for a DoRA expert."""
    expert_id: str
    score: float
    source: str = "unknown"  # keyword, semantic, learned
    
    def __lt__(self, other):
        return self.score < other.score


class VRAMAdaptiveGating:
    """
    VRAM-adaptive gating for expert selection.
    
    Selects Top-K experts based on available VRAM capacity.
    Implements the design decision for resource-aware inference.
    """
    
    # VRAM thresholds (GB)
    LOW_VRAM_GB = 8
    HIGH_VRAM_GB = 16
    
    # Default expert sizes (MB) - can be overridden
    DEFAULT_EXPERT_SIZE_MB = 50
    
    def __init__(
        self,
        fallback_expert: str = "general_dora",
        confidence_threshold: float = 0.5,
        device: str = "cuda",
    ):
        """
        Initialize VRAM-adaptive gating.
        
        Args:
            fallback_expert: Expert to use when confidence is low
            confidence_threshold: Minimum confidence to use expert
            device: CUDA device
        """
        self.fallback_expert = fallback_expert
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # Detect VRAM
        self._vram_gb = self._get_vram_gb()
        self._max_experts = self._compute_max_experts()
        
        logger.info(
            f"VRAMAdaptiveGating: {self._vram_gb:.1f}GB VRAM, "
            f"max_experts={self._max_experts}"
        )
    
    def _get_vram_gb(self) -> float:
        """Get total VRAM in GB."""
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / (1024 ** 3)
        except Exception:
            return 0.0
    
    def _compute_max_experts(self) -> int:
        """Compute max experts based on VRAM."""
        if self._vram_gb < self.LOW_VRAM_GB:
            return 1
        elif self._vram_gb < self.HIGH_VRAM_GB:
            return 2
        else:
            return 3
    
    @property
    def max_experts(self) -> int:
        """Get maximum number of experts."""
        return self._max_experts
    
    @property
    def vram_gb(self) -> float:
        """Get VRAM in GB."""
        return self._vram_gb
    
    def select(
        self,
        expert_scores: List[ExpertScore],
        expert_sizes: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Select experts based on scores and VRAM constraints.
        
        Args:
            expert_scores: List of ExpertScore (should be sorted by score desc)
            expert_sizes: Optional dict of expert_id -> size_mb
            
        Returns:
            List of (expert_id, weight) tuples
        """
        if not expert_scores:
            return [(self.fallback_expert, 1.0)]
        
        # Sort by score descending
        sorted_scores = sorted(expert_scores, reverse=True)
        
        # Filter out fallback from main selection
        candidates = [
            es for es in sorted_scores
            if es.expert_id != self.fallback_expert
        ]
        
        # Select top-K
        selected = candidates[:self._max_experts]
        
        # Check if we need fallback
        needs_fallback = (
            not selected or
            max(es.score for es in selected) < self.confidence_threshold
        )
        
        if needs_fallback:
            # Add fallback expert
            fallback_score = ExpertScore(
                expert_id=self.fallback_expert,
                score=self.confidence_threshold,
                source="fallback",
            )
            selected.append(fallback_score)
        
        # Compute weights (softmax-like normalization)
        weights = self._compute_weights([es.score for es in selected])
        
        result = [
            (es.expert_id, w)
            for es, w in zip(selected, weights)
        ]
        
        logger.debug(f"Selected experts: {result}")
        return result
    
    def _compute_weights(self, scores: List[float]) -> List[float]:
        """
        Compute normalized weights from scores.
        
        Uses softmax-like normalization with temperature.
        """
        if not scores:
            return []
        
        if len(scores) == 1:
            return [1.0]
        
        # Simple normalization (sum to 1)
        total = sum(scores)
        if total > 0:
            return [s / total for s in scores]
        else:
            # Equal weights if all scores are 0
            return [1.0 / len(scores)] * len(scores)
    
    def select_with_budget(
        self,
        expert_scores: List[ExpertScore],
        expert_sizes: Dict[str, float],
        vram_budget_mb: float,
    ) -> List[Tuple[str, float]]:
        """
        Select experts within a VRAM budget.
        
        Args:
            expert_scores: List of ExpertScore
            expert_sizes: Dict of expert_id -> size_mb
            vram_budget_mb: Maximum VRAM to use
            
        Returns:
            List of (expert_id, weight) tuples
        """
        sorted_scores = sorted(expert_scores, reverse=True)
        
        selected = []
        total_size = 0.0
        
        for es in sorted_scores:
            if es.expert_id == self.fallback_expert:
                continue
            
            size = expert_sizes.get(es.expert_id, self.DEFAULT_EXPERT_SIZE_MB)
            
            if total_size + size <= vram_budget_mb:
                selected.append(es)
                total_size += size
            
            if len(selected) >= self._max_experts:
                break
        
        # Check fallback
        if not selected or max(es.score for es in selected) < self.confidence_threshold:
            fallback_size = expert_sizes.get(
                self.fallback_expert, self.DEFAULT_EXPERT_SIZE_MB
            )
            if total_size + fallback_size <= vram_budget_mb:
                selected.append(ExpertScore(
                    expert_id=self.fallback_expert,
                    score=self.confidence_threshold,
                    source="fallback",
                ))
        
        # Compute weights
        weights = self._compute_weights([es.score for es in selected])
        
        return [(es.expert_id, w) for es, w in zip(selected, weights)]
    
    def merge_router_results(
        self,
        keyword_results: List[Tuple[str, float]],
        semantic_results: Optional[List[Tuple[str, float]]] = None,
        keyword_weight: float = 0.6,
        semantic_weight: float = 0.4,
    ) -> List[ExpertScore]:
        """
        Merge results from multiple routers.
        
        Args:
            keyword_results: Results from keyword router
            semantic_results: Results from semantic router (optional)
            keyword_weight: Weight for keyword results
            semantic_weight: Weight for semantic results
            
        Returns:
            Merged ExpertScore list
        """
        scores: Dict[str, float] = {}
        sources: Dict[str, str] = {}
        
        # Add keyword results
        for expert_id, score in keyword_results:
            scores[expert_id] = score * keyword_weight
            sources[expert_id] = "keyword"
        
        # Add semantic results
        if semantic_results:
            for expert_id, score in semantic_results:
                if expert_id in scores:
                    scores[expert_id] += score * semantic_weight
                    sources[expert_id] = "combined"
                else:
                    scores[expert_id] = score * semantic_weight
                    sources[expert_id] = "semantic"
        
        # Convert to ExpertScore list
        result = [
            ExpertScore(expert_id=eid, score=score, source=sources[eid])
            for eid, score in scores.items()
        ]
        
        return sorted(result, reverse=True)
    
    def get_status(self) -> Dict[str, any]:
        """Get gating status."""
        return {
            'vram_gb': self._vram_gb,
            'max_experts': self._max_experts,
            'fallback_expert': self.fallback_expert,
            'confidence_threshold': self.confidence_threshold,
            'tier': (
                'low' if self._vram_gb < self.LOW_VRAM_GB
                else 'medium' if self._vram_gb < self.HIGH_VRAM_GB
                else 'high'
            ),
        }

"""
Router Module for DoRA Expert Selection

Provides multi-stage routing for selecting appropriate DoRA experts:
- Stage 1: Keyword Router (fast, rule-based)
- Stage 2: Semantic Router (embedding-based)
- Stage 3: Score Fusion (weighted combination)
- Stage 4: VRAM-Adaptive Gating (resource-aware selection)

Main entry point: HybridRouter
"""

from .keyword_router import KeywordRouter, RouterResult
from .semantic_router import SemanticRouter, SemanticResult, ExpertEmbedding
from .vram_adaptive_gating import VRAMAdaptiveGating, ExpertScore
from .hybrid_router import HybridRouter, HybridRouterConfig, RoutingMetrics

__all__ = [
    # Main router
    'HybridRouter',
    'HybridRouterConfig',
    'RoutingMetrics',
    # Sub-routers
    'KeywordRouter',
    'RouterResult',
    'SemanticRouter',
    'SemanticResult',
    'ExpertEmbedding',
    'VRAMAdaptiveGating',
    'ExpertScore',
]

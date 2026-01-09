"""
Inference Pipeline - BitNet + DoRA + RAG Integration

Main pipeline that orchestrates all components:
1. RAG Retriever - Context retrieval for query augmentation
2. Hybrid Router - Expert selection (Keyword + Semantic + VRAM Gating)
3. Tiered Cache - DoRA adapter caching (VRAM → RAM → Disk)
4. Inference Backend - Model execution with selected adapters

Pipeline Flow:
    User Query
        │
        ▼
    ┌─────────────────────────────────────┐
    │ STAGE 1: RAG Context Retrieval      │
    │ - Embed query                       │
    │ - Search FAISS index                │
    │ - Augment prompt with context       │
    └─────────────────────────────────────┘
        │
        ▼
    ┌─────────────────────────────────────┐
    │ STAGE 2: Expert Routing             │
    │ - Keyword Router (fast path)        │
    │ - Semantic Router (if needed)       │
    │ - Score Fusion                      │
    │ - VRAM-Adaptive Gating              │
    └─────────────────────────────────────┘
        │
        ▼
    ┌─────────────────────────────────────┐
    │ STAGE 3: Adapter Loading            │
    │ - Check tiered cache                │
    │ - Load from disk/IPFS if needed     │
    │ - Promote to VRAM                   │
    └─────────────────────────────────────┘
        │
        ▼
    ┌─────────────────────────────────────┐
    │ STAGE 4: Inference Execution        │
    │ - Apply DoRA adapters               │
    │ - Run BitNet forward pass           │
    │ - Return result                     │
    └─────────────────────────────────────┘
        │
        ▼
    Output + Metrics
"""

import asyncio
import time
import logging
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline execution stages."""
    RAG_RETRIEVAL = "rag_retrieval"
    EXPERT_ROUTING = "expert_routing"
    ADAPTER_LOADING = "adapter_loading"
    INFERENCE = "inference"


@dataclass
class PipelineConfig:
    """Configuration for inference pipeline."""
    # RAG settings
    enable_rag: bool = True
    rag_top_k: int = 3
    rag_threshold: float = 0.5
    rag_context_template: str = "Context:\n{context}\n\nQuery: {query}"
    
    # Router settings
    router_strategy: str = "hybrid"  # keyword, semantic, hybrid
    keyword_weight: float = 0.3
    semantic_weight: float = 0.7
    fast_path_threshold: float = 0.85
    
    # Cache settings
    vram_capacity_mb: int = 2048
    ram_capacity_mb: int = 8192
    disk_cache_dir: str = ".r3mes/dora_cache"
    enable_predictive_loading: bool = False
    
    # Inference settings
    max_batch_size: int = 1
    max_seq_length: int = 2048
    default_temperature: float = 0.7
    
    # Adapter settings
    adapter_dir: str = ".r3mes/adapters"
    fallback_expert: str = "general_dora"
    
    # Timeouts (ms)
    rag_timeout_ms: float = 100.0
    routing_timeout_ms: float = 50.0
    loading_timeout_ms: float = 500.0
    inference_timeout_ms: float = 30000.0


@dataclass
class PipelineMetrics:
    """Metrics from pipeline execution."""
    total_time_ms: float = 0.0
    rag_time_ms: float = 0.0
    routing_time_ms: float = 0.0
    loading_time_ms: float = 0.0
    inference_time_ms: float = 0.0
    
    # RAG metrics
    rag_docs_retrieved: int = 0
    rag_context_length: int = 0
    
    # Routing metrics
    used_fast_path: bool = False
    keyword_confidence: float = 0.0
    
    # Adapter metrics
    adapters_loaded: List[str] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Inference metrics
    tokens_generated: int = 0
    backend_used: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_time_ms': self.total_time_ms,
            'rag_time_ms': self.rag_time_ms,
            'routing_time_ms': self.routing_time_ms,
            'loading_time_ms': self.loading_time_ms,
            'inference_time_ms': self.inference_time_ms,
            'rag_docs_retrieved': self.rag_docs_retrieved,
            'rag_context_length': self.rag_context_length,
            'used_fast_path': self.used_fast_path,
            'keyword_confidence': self.keyword_confidence,
            'adapters_loaded': self.adapters_loaded,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'tokens_generated': self.tokens_generated,
            'backend_used': self.backend_used,
        }


@dataclass
class PipelineResult:
    """Result from pipeline execution."""
    output: torch.Tensor
    text: Optional[str] = None
    metrics: PipelineMetrics = field(default_factory=PipelineMetrics)
    experts_used: List[Tuple[str, float]] = field(default_factory=list)
    rag_context: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


class InferencePipeline:
    """
    Main inference pipeline integrating all components.
    
    Orchestrates:
    - RAG retrieval for context augmentation
    - Hybrid routing for expert selection
    - Tiered caching for adapter management
    - Backend inference execution
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        router=None,
        cache=None,
        backend=None,
        retriever=None,
    ):
        """
        Initialize inference pipeline.
        
        Args:
            config: Pipeline configuration
            router: Pre-configured HybridRouter (optional)
            cache: Pre-configured TieredDoRACache (optional)
            backend: Pre-configured InferenceBackend (optional)
            retriever: Pre-configured RAGRetriever (optional)
        """
        self.config = config or PipelineConfig()
        
        # Components (lazy initialization)
        self._router = router
        self._cache = cache
        self._backend = backend
        self._retriever = retriever
        
        # State
        self._initialized = False
        self._model_loaded = False
        
        # Statistics
        self._total_requests = 0
        self._total_errors = 0
        
        logger.info("InferencePipeline created")
    
    async def initialize(self) -> bool:
        """
        Initialize all pipeline components.
        
        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True
        
        try:
            logger.info("Initializing inference pipeline...")
            
            # Initialize router
            if self._router is None:
                self._router = self._create_router()
            
            # Initialize semantic router embeddings
            if hasattr(self._router, 'initialize'):
                self._router.initialize()
            
            # Initialize cache
            if self._cache is None:
                self._cache = self._create_cache()
            
            # Initialize backend
            if self._backend is None:
                self._backend = self._create_backend()
            
            # Initialize RAG retriever (optional)
            if self._retriever is None and self.config.enable_rag:
                self._retriever = self._create_retriever()
            
            self._initialized = True
            logger.info("Pipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            return False
    
    def _create_router(self):
        """Create hybrid router."""
        # Import here to avoid circular imports
        from router.hybrid_router import HybridRouter, HybridRouterConfig
        
        config = HybridRouterConfig(
            keyword_weight=self.config.keyword_weight,
            semantic_weight=self.config.semantic_weight,
            fast_path_threshold=self.config.fast_path_threshold,
            fallback_threshold=0.5,
            enable_semantic=(self.config.router_strategy in ["semantic", "hybrid"]),
        )
        
        return HybridRouter(config=config)
    
    def _create_cache(self):
        """Create tiered cache."""
        from cache.tiered_cache import TieredDoRACache
        
        return TieredDoRACache(
            vram_capacity_mb=self.config.vram_capacity_mb,
            ram_capacity_mb=self.config.ram_capacity_mb,
            disk_cache_dir=self.config.disk_cache_dir,
            enable_predictive=self.config.enable_predictive_loading,
        )
    
    def _create_backend(self):
        """Create inference backend."""
        from core.inference_backend import get_best_backend
        
        backend = get_best_backend()
        if backend is None:
            raise RuntimeError("No inference backend available")
        
        return backend
    
    def _create_retriever(self):
        """Create RAG retriever."""
        from rag.retriever import RAGRetriever, RetrieverConfig, IndexType
        
        config = RetrieverConfig(
            default_top_k=self.config.rag_top_k,
            similarity_threshold=self.config.rag_threshold,
        )
        
        return RAGRetriever(config=config)

    async def load_model(self, model_path: str) -> bool:
        """
        Load base model.
        
        Args:
            model_path: Path to model or IPFS hash
            
        Returns:
            True if successful
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            success = self._backend.load_model(model_path)
            if success:
                self._model_loaded = True
                logger.info(f"Model loaded: {model_path}")
            return success
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    async def run(
        self,
        query: str,
        input_ids: Optional[torch.Tensor] = None,
        skip_rag: bool = False,
        skip_routing: bool = False,
        force_experts: Optional[List[str]] = None,
        **kwargs
    ) -> PipelineResult:
        """
        Run full inference pipeline.
        
        Args:
            query: User query text
            input_ids: Pre-tokenized input (optional)
            skip_rag: Skip RAG retrieval
            skip_routing: Skip expert routing
            force_experts: Force specific experts (bypasses routing)
            **kwargs: Additional inference options
            
        Returns:
            PipelineResult with output and metrics
        """
        start_time = time.perf_counter()
        metrics = PipelineMetrics()
        
        self._total_requests += 1
        
        try:
            # Ensure initialized
            if not self._initialized:
                await self.initialize()
            
            # Stage 1: RAG Context Retrieval
            augmented_query = query
            rag_context = None
            
            if self.config.enable_rag and not skip_rag and self._retriever:
                rag_start = time.perf_counter()
                rag_context, augmented_query = await self._retrieve_context(query)
                metrics.rag_time_ms = (time.perf_counter() - rag_start) * 1000
                metrics.rag_context_length = len(rag_context) if rag_context else 0
            
            # Stage 2: Expert Routing
            if force_experts:
                # Use forced experts
                experts = [(e, 1.0 / len(force_experts)) for e in force_experts]
                metrics.used_fast_path = False
            elif skip_routing:
                # Use fallback expert
                experts = [(self.config.fallback_expert, 1.0)]
                metrics.used_fast_path = True
            else:
                routing_start = time.perf_counter()
                experts, routing_metrics = self._router.route(augmented_query)
                metrics.routing_time_ms = (time.perf_counter() - routing_start) * 1000
                metrics.used_fast_path = routing_metrics.used_fast_path
                metrics.keyword_confidence = routing_metrics.keyword_confidence
            
            # Stage 3: Adapter Loading
            loading_start = time.perf_counter()
            await self._ensure_adapters_loaded(experts, metrics)
            metrics.loading_time_ms = (time.perf_counter() - loading_start) * 1000
            metrics.adapters_loaded = [e[0] for e in experts]
            
            # Stage 4: Inference Execution
            inference_start = time.perf_counter()
            
            # Prepare input
            if input_ids is None:
                # Simple tokenization placeholder
                # In production, use proper tokenizer
                input_ids = self._simple_tokenize(augmented_query)
            
            # Run inference
            adapter_ids = [e[0] for e in experts]
            adapter_weights = [e[1] for e in experts]
            
            result = self._backend.inference(
                input_ids=input_ids,
                adapter_ids=adapter_ids,
                adapter_weights=adapter_weights,
                **kwargs
            )
            
            metrics.inference_time_ms = (time.perf_counter() - inference_start) * 1000
            metrics.backend_used = result.backend_used
            metrics.total_time_ms = (time.perf_counter() - start_time) * 1000
            
            return PipelineResult(
                output=result.output,
                metrics=metrics,
                experts_used=experts,
                rag_context=rag_context,
                success=True,
            )
            
        except Exception as e:
            self._total_errors += 1
            logger.error(f"Pipeline error: {e}")
            
            metrics.total_time_ms = (time.perf_counter() - start_time) * 1000
            
            return PipelineResult(
                output=torch.tensor([]),
                metrics=metrics,
                success=False,
                error=str(e),
            )
    
    async def _retrieve_context(
        self, query: str
    ) -> Tuple[Optional[str], str]:
        """
        Retrieve context from RAG and augment query.
        
        Args:
            query: Original query
            
        Returns:
            Tuple of (context, augmented_query)
        """
        if not self._retriever:
            return None, query
        
        try:
            results = self._retriever.search(
                query=query,
                top_k=self.config.rag_top_k,
                threshold=self.config.rag_threshold,
            )
            
            if not results:
                return None, query
            
            # Build context from retrieved documents
            context_parts = []
            for i, r in enumerate(results, 1):
                context_parts.append(f"[{i}] {r.content}")
            
            context = "\n".join(context_parts)
            
            # Augment query with context
            augmented = self.config.rag_context_template.format(
                context=context,
                query=query,
            )
            
            logger.debug(f"RAG retrieved {len(results)} documents")
            return context, augmented
            
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
            return None, query
    
    async def _ensure_adapters_loaded(
        self,
        experts: List[Tuple[str, float]],
        metrics: PipelineMetrics,
    ):
        """
        Ensure required adapters are loaded in cache.
        
        Args:
            experts: List of (expert_id, weight) tuples
            metrics: Metrics to update
        """
        for expert_id, _ in experts:
            # Check cache
            adapter = await self._cache.get(expert_id)
            
            if adapter is not None:
                metrics.cache_hits += 1
                continue
            
            metrics.cache_misses += 1
            
            # Load from disk
            adapter_path = Path(self.config.adapter_dir) / f"{expert_id}.pt"
            
            if adapter_path.exists():
                try:
                    # Load adapter
                    success = self._backend.load_adapter(expert_id, str(adapter_path))
                    
                    if success:
                        # Get adapter data and cache it
                        adapter_data = self._backend.adapters.get(expert_id)
                        if adapter_data:
                            size_mb = adapter_data.estimate_size_mb() if hasattr(adapter_data, 'estimate_size_mb') else 50.0
                            await self._cache.put(expert_id, adapter_data, size_mb)
                            logger.debug(f"Loaded and cached adapter: {expert_id}")
                except Exception as e:
                    logger.warning(f"Failed to load adapter {expert_id}: {e}")
            else:
                logger.warning(f"Adapter not found: {adapter_path}")
    
    def _simple_tokenize(self, text: str) -> torch.Tensor:
        """
        Simple tokenization placeholder.
        
        In production, use proper tokenizer from model.
        """
        # Simple character-level tokenization for testing
        tokens = [ord(c) % 1000 for c in text[:self.config.max_seq_length]]
        return torch.tensor([tokens], dtype=torch.long)

    async def run_batch(
        self,
        queries: List[str],
        **kwargs
    ) -> List[PipelineResult]:
        """
        Run pipeline for multiple queries.
        
        Args:
            queries: List of query strings
            **kwargs: Additional inference options
            
        Returns:
            List of PipelineResult
        """
        results = []
        for query in queries:
            result = await self.run(query, **kwargs)
            results.append(result)
        return results
    
    async def warmup(self, sample_query: str = "Hello, how are you?"):
        """
        Warmup pipeline with sample inference.
        
        Args:
            sample_query: Sample query for warmup
        """
        logger.info("Warming up pipeline...")
        
        # Initialize if needed
        if not self._initialized:
            await self.initialize()
        
        # Run sample inference
        await self.run(sample_query, skip_rag=True)
        
        # Warmup backend
        if hasattr(self._backend, 'warmup'):
            self._backend.warmup()
        
        logger.info("Pipeline warmup complete")
    
    def add_rag_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Add document to RAG index.
        
        Args:
            doc_id: Document ID
            content: Document content
            metadata: Optional metadata
        """
        if self._retriever:
            self._retriever.add_document(doc_id, content, metadata)
    
    def add_rag_documents(
        self,
        documents: List[Tuple[str, str, Optional[Dict[str, Any]]]],
    ) -> int:
        """
        Add multiple documents to RAG index.
        
        Args:
            documents: List of (doc_id, content, metadata) tuples
            
        Returns:
            Number of documents added
        """
        if self._retriever:
            return self._retriever.add_documents(documents)
        return 0
    
    async def preload_adapters(self, adapter_ids: List[str]):
        """
        Preload adapters into cache.
        
        Args:
            adapter_ids: List of adapter IDs to preload
        """
        logger.info(f"Preloading {len(adapter_ids)} adapters...")
        
        for adapter_id in adapter_ids:
            adapter_path = Path(self.config.adapter_dir) / f"{adapter_id}.pt"
            
            if adapter_path.exists():
                try:
                    success = self._backend.load_adapter(adapter_id, str(adapter_path))
                    if success:
                        adapter_data = self._backend.adapters.get(adapter_id)
                        if adapter_data:
                            size_mb = adapter_data.estimate_size_mb() if hasattr(adapter_data, 'estimate_size_mb') else 50.0
                            await self._cache.put(adapter_id, adapter_data, size_mb)
                            logger.info(f"Preloaded adapter: {adapter_id}")
                except Exception as e:
                    logger.warning(f"Failed to preload adapter {adapter_id}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = {
            'initialized': self._initialized,
            'model_loaded': self._model_loaded,
            'total_requests': self._total_requests,
            'total_errors': self._total_errors,
            'error_rate': self._total_errors / max(1, self._total_requests),
            'config': {
                'enable_rag': self.config.enable_rag,
                'router_strategy': self.config.router_strategy,
                'vram_capacity_mb': self.config.vram_capacity_mb,
            },
        }
        
        # Add component stats
        if self._router:
            stats['router'] = self._router.get_stats()
        
        if self._cache:
            stats['cache'] = self._cache.get_usage()
        
        if self._backend:
            stats['backend'] = {
                'capabilities': self._backend.get_capabilities().to_dict(),
                'vram_usage': self._backend.get_vram_usage(),
            }
        
        if self._retriever:
            stats['retriever'] = self._retriever.get_stats()
        
        return stats
    
    def reset_stats(self):
        """Reset pipeline statistics."""
        self._total_requests = 0
        self._total_errors = 0
        
        if self._router:
            self._router.reset_stats()
    
    async def shutdown(self):
        """Shutdown pipeline and cleanup resources."""
        logger.info("Shutting down pipeline...")
        
        # Clear cache
        if self._cache:
            self._cache.clear()
        
        # Unload adapters
        if self._backend:
            for adapter_id in list(self._backend.adapters.keys()):
                self._backend.unload_adapter(adapter_id)
        
        self._initialized = False
        self._model_loaded = False
        
        logger.info("Pipeline shutdown complete")
    
    # Properties for component access
    @property
    def router(self):
        """Get router component."""
        return self._router
    
    @property
    def cache(self):
        """Get cache component."""
        return self._cache
    
    @property
    def backend(self):
        """Get backend component."""
        return self._backend
    
    @property
    def retriever(self):
        """Get retriever component."""
        return self._retriever


class StreamingInferencePipeline(InferencePipeline):
    """
    Streaming variant of inference pipeline.
    
    Yields tokens as they are generated for real-time output.
    """
    
    async def run_streaming(
        self,
        query: str,
        **kwargs
    ):
        """
        Run inference with streaming output.
        
        Args:
            query: User query
            **kwargs: Additional options
            
        Yields:
            Generated tokens/text chunks
        """
        # Run pipeline stages up to inference
        if not self._initialized:
            await self.initialize()
        
        # RAG retrieval
        augmented_query = query
        if self.config.enable_rag and self._retriever:
            _, augmented_query = await self._retrieve_context(query)
        
        # Expert routing
        experts, _ = self._router.route(augmented_query)
        
        # Load adapters
        metrics = PipelineMetrics()
        await self._ensure_adapters_loaded(experts, metrics)
        
        # Prepare input
        input_ids = self._simple_tokenize(augmented_query)
        
        # Streaming inference (if backend supports it)
        # Check for actual callable method, not just attribute existence
        streaming_method = getattr(self._backend, 'inference_streaming', None)
        if streaming_method is not None and callable(streaming_method):
            try:
                adapter_ids = [e[0] for e in experts]
                adapter_weights = [e[1] for e in experts]
                
                async for token in streaming_method(
                    input_ids=input_ids,
                    adapter_ids=adapter_ids,
                    adapter_weights=adapter_weights,
                    **kwargs
                ):
                    yield token
                return
            except (TypeError, AttributeError):
                # Fallback if streaming fails
                pass
        
        # Fallback to non-streaming
        result = await self.run(query, **kwargs)
        yield result.output


# Factory functions
def create_pipeline(
    config: Optional[PipelineConfig] = None,
    **kwargs
) -> InferencePipeline:
    """
    Create inference pipeline with default configuration.
    
    Args:
        config: Pipeline configuration
        **kwargs: Override config values
        
    Returns:
        Configured InferencePipeline
    """
    if config is None:
        config = PipelineConfig(**kwargs)
    
    return InferencePipeline(config=config)


def create_streaming_pipeline(
    config: Optional[PipelineConfig] = None,
    **kwargs
) -> StreamingInferencePipeline:
    """
    Create streaming inference pipeline.
    
    Args:
        config: Pipeline configuration
        **kwargs: Override config values
        
    Returns:
        Configured StreamingInferencePipeline
    """
    if config is None:
        config = PipelineConfig(**kwargs)
    
    return StreamingInferencePipeline(config=config)


async def quick_inference(
    query: str,
    model_path: Optional[str] = None,
    enable_rag: bool = False,
) -> PipelineResult:
    """
    Quick inference helper for simple use cases.
    
    Args:
        query: User query
        model_path: Path to model (optional)
        enable_rag: Enable RAG retrieval
        
    Returns:
        PipelineResult
    """
    config = PipelineConfig(enable_rag=enable_rag)
    pipeline = InferencePipeline(config=config)
    
    await pipeline.initialize()
    
    if model_path:
        await pipeline.load_model(model_path)
    
    return await pipeline.run(query)

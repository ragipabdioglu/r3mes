"""
R3MES Inference API Endpoints

FAZ 3: Backend API ↔ ServingEngine/InferencePipeline Entegrasyonu

Provides REST API endpoints for AI inference using the BitNet + DoRA + RAG pipeline.
Supports both local inference and remote serving node proxying.
"""

import asyncio
import logging
import time
import os
from typing import Optional, Dict, Any, List, AsyncGenerator
from fastapi import APIRouter, HTTPException, Request, Depends, Header, Query, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter
from slowapi.util import get_remote_address

from .config_manager import get_config_manager
from .database_async import AsyncDatabase
from .inference_mode import get_inference_mode, InferenceMode, is_inference_available
from .exceptions import (
    InferenceError,
    InvalidInputError,
    InvalidWalletAddressError,
    InsufficientCreditsError,
)
from .input_validation import validate_wallet_address

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/inference", tags=["inference"])

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

config_manager = get_config_manager()
config = config_manager.load()
database = AsyncDatabase(db_path=config.database_path, chain_json_path=config.chain_json_path)

# Lazy-loaded ServingEngine instance (singleton)
_serving_engine = None
_engine_lock = asyncio.Lock()


async def get_serving_engine():
    """
    Lazy-load ServingEngine singleton.
    
    Only initializes when inference mode is LOCAL.
    """
    global _serving_engine
    
    inference_mode = get_inference_mode()
    if inference_mode != InferenceMode.LOCAL:
        return None
    
    async with _engine_lock:
        if _serving_engine is None:
            try:
                # Import ServingEngine from miner-engine
                import sys
                miner_engine_path = os.path.join(os.path.dirname(__file__), "..", "..", "miner-engine")
                if miner_engine_path not in sys.path:
                    sys.path.insert(0, miner_engine_path)
                
                from r3mes.serving.engine import ServingEngine
                
                # Get configuration from environment
                private_key = os.getenv("R3MES_SERVING_PRIVATE_KEY", "")
                blockchain_url = os.getenv("R3MES_BLOCKCHAIN_URL", "localhost:9090")
                chain_id = os.getenv("R3MES_CHAIN_ID", "remes-test")
                model_ipfs_hash = os.getenv("R3MES_MODEL_IPFS_HASH")
                
                # Pipeline configuration
                enable_rag = os.getenv("R3MES_ENABLE_RAG", "true").lower() == "true"
                enable_caching = os.getenv("R3MES_ENABLE_CACHING", "true").lower() == "true"
                vram_capacity = int(os.getenv("R3MES_VRAM_CAPACITY_MB", "2048"))
                ram_capacity = int(os.getenv("R3MES_RAM_CAPACITY_MB", "8192"))
                adapter_dir = os.getenv("R3MES_ADAPTER_DIR")
                
                _serving_engine = ServingEngine(
                    private_key=private_key,
                    blockchain_url=blockchain_url,
                    chain_id=chain_id,
                    model_ipfs_hash=model_ipfs_hash,
                    enable_rag=enable_rag,
                    enable_caching=enable_caching,
                    vram_capacity_mb=vram_capacity,
                    ram_capacity_mb=ram_capacity,
                    adapter_dir=adapter_dir,
                )
                
                # Initialize pipeline
                await _serving_engine.initialize_pipeline()
                
                # Load model if hash provided
                if model_ipfs_hash:
                    await _serving_engine.load_model()
                
                logger.info("ServingEngine initialized successfully")
                
            except ImportError as e:
                logger.error(f"Failed to import ServingEngine: {e}")
                raise HTTPException(
                    status_code=503,
                    detail="Inference engine not available. Check miner-engine installation."
                )
            except Exception as e:
                logger.error(f"Failed to initialize ServingEngine: {e}", exc_info=True)
                raise HTTPException(
                    status_code=503,
                    detail=f"Failed to initialize inference engine: {str(e)}"
                )
    
    return _serving_engine


# ============================================================================
# Request/Response Models
# ============================================================================

class InferenceRequest(BaseModel):
    """Inference request model."""
    prompt: str = Field(..., min_length=1, max_length=32000, description="Input prompt for inference")
    wallet_address: Optional[str] = Field(None, description="Wallet address for credit deduction")
    
    # Inference options
    max_tokens: int = Field(default=512, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: int = Field(default=50, ge=0, le=100, description="Top-k sampling")
    
    # Pipeline options
    skip_rag: bool = Field(default=False, description="Skip RAG context retrieval")
    force_experts: Optional[List[str]] = Field(default=None, description="Force specific DoRA experts")
    stream: bool = Field(default=False, description="Enable streaming response")
    
    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        if not v or not v.strip():
            raise InvalidInputError("Prompt cannot be empty")
        v = v.replace('\x00', '')
        return v.strip()
    
    @field_validator("wallet_address")
    @classmethod
    def validate_wallet(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        if not v.startswith("remes"):
            raise InvalidWalletAddressError("Invalid address format: must start with 'remes'")
        return v


class InferenceResponse(BaseModel):
    """Inference response model."""
    request_id: str
    text: str
    tokens_generated: int
    latency_ms: float
    experts_used: List[Dict[str, Any]]
    rag_context_used: bool
    model_version: str
    credits_used: float = 0.0


class InferenceHealthResponse(BaseModel):
    """Inference health response model."""
    status: str
    inference_mode: str
    is_ready: bool
    is_healthy: bool
    pipeline_initialized: bool
    model_loaded: bool
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    error_message: Optional[str] = None


class InferenceMetricsResponse(BaseModel):
    """Inference metrics response model."""
    serving_engine_requests_total: int
    serving_engine_requests_success: int
    serving_engine_requests_failed: int
    serving_engine_latency_avg_ms: float
    serving_engine_ready: int
    serving_engine_healthy: int
    pipeline_total_requests: int = 0
    pipeline_error_rate: float = 0.0
    cache_vram_used_mb: float = 0.0
    cache_ram_used_mb: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0


# ============================================================================
# Inference Endpoints
# ============================================================================

@router.post("/generate", response_model=InferenceResponse)
@limiter.limit(config.rate_limit_chat)
async def generate_inference(
    request: Request,
    inference_request: InferenceRequest,
    background_tasks: BackgroundTasks,
):
    """
    Generate AI inference using BitNet + DoRA + RAG pipeline.
    
    Supports:
    - Local inference (R3MES_INFERENCE_MODE=local)
    - Remote serving node proxy (R3MES_INFERENCE_MODE=remote)
    - Mock responses for testing (R3MES_INFERENCE_MODE=mock)
    
    Rate Limit: Configurable via RATE_LIMIT_CHAT
    """
    inference_mode = get_inference_mode()
    start_time = time.perf_counter()
    request_id = f"inf_{int(time.time() * 1000)}_{id(request)}"
    
    # Handle disabled mode
    if inference_mode == InferenceMode.DISABLED:
        raise HTTPException(
            status_code=503,
            detail="Inference is disabled on this server"
        )
    
    # Handle mock mode
    if inference_mode == InferenceMode.MOCK:
        return _generate_mock_response(request_id, inference_request, start_time)
    
    # Handle remote mode (proxy to serving nodes)
    if inference_mode == InferenceMode.REMOTE:
        return await _proxy_to_serving_node(request_id, inference_request, start_time)
    
    # Local inference mode
    try:
        engine = await get_serving_engine()
        if not engine:
            raise HTTPException(
                status_code=503,
                detail="Inference engine not available"
            )
        
        if not engine.is_ready():
            raise HTTPException(
                status_code=503,
                detail="Inference engine not ready. Pipeline may still be initializing."
            )
        
        # Run inference through pipeline
        result = await engine.infer(
            query=inference_request.prompt,
            skip_rag=inference_request.skip_rag,
            force_experts=inference_request.force_experts,
            temperature=inference_request.temperature,
            max_tokens=inference_request.max_tokens,
            top_p=inference_request.top_p,
            top_k=inference_request.top_k,
        )
        
        if not result.success:
            raise InferenceError(f"Inference failed: {result.error}")
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Calculate credits (background task)
        credits_used = 0.0
        if inference_request.wallet_address:
            credits_used = _calculate_credits(result.metrics.tokens_generated)
            background_tasks.add_task(
                _deduct_credits,
                inference_request.wallet_address,
                credits_used
            )
        
        return InferenceResponse(
            request_id=request_id,
            text=result.text or "",
            tokens_generated=result.metrics.tokens_generated,
            latency_ms=latency_ms,
            experts_used=[{"id": e[0], "weight": e[1]} for e in result.experts_used],
            rag_context_used=result.rag_context is not None,
            model_version=engine.model_version,
            credits_used=credits_used,
        )
        
    except HTTPException:
        raise
    except InferenceError as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal inference error")


@router.post("/generate/stream")
@limiter.limit(config.rate_limit_chat)
async def generate_inference_stream(
    request: Request,
    inference_request: InferenceRequest,
):
    """
    Generate streaming AI inference.
    
    Returns Server-Sent Events (SSE) stream of generated tokens.
    """
    inference_mode = get_inference_mode()
    
    if inference_mode == InferenceMode.DISABLED:
        raise HTTPException(status_code=503, detail="Inference is disabled")
    
    if inference_mode == InferenceMode.MOCK:
        return StreamingResponse(
            _mock_stream_generator(inference_request.prompt),
            media_type="text/event-stream"
        )
    
    if inference_mode == InferenceMode.REMOTE:
        # For remote mode, proxy streaming to serving node
        return StreamingResponse(
            _proxy_stream_to_serving_node(inference_request),
            media_type="text/event-stream"
        )
    
    # Local streaming inference
    try:
        engine = await get_serving_engine()
        if not engine:
            raise HTTPException(status_code=503, detail="Inference engine not available")
        
        async def stream_generator():
            try:
                async for token in engine.infer_streaming(
                    query=inference_request.prompt,
                    skip_rag=inference_request.skip_rag,
                    force_experts=inference_request.force_experts,
                    temperature=inference_request.temperature,
                    max_tokens=inference_request.max_tokens,
                ):
                    yield f"data: {token}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: [ERROR] {str(e)}\n\n"
        
        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Streaming inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Streaming inference failed")


# ============================================================================
# Health & Metrics Endpoints
# ============================================================================

@router.get("/health", response_model=InferenceHealthResponse)
async def get_inference_health():
    """
    Get inference engine health status.
    
    Used for:
    - Kubernetes liveness/readiness probes
    - Load balancer health checks
    - Monitoring dashboards
    """
    inference_mode = get_inference_mode()
    
    if inference_mode == InferenceMode.DISABLED:
        return InferenceHealthResponse(
            status="disabled",
            inference_mode=inference_mode.value,
            is_ready=False,
            is_healthy=True,
            pipeline_initialized=False,
            model_loaded=False,
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            avg_latency_ms=0.0,
        )
    
    if inference_mode == InferenceMode.MOCK:
        return InferenceHealthResponse(
            status="mock",
            inference_mode=inference_mode.value,
            is_ready=True,
            is_healthy=True,
            pipeline_initialized=True,
            model_loaded=True,
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            avg_latency_ms=0.0,
        )
    
    if inference_mode == InferenceMode.REMOTE:
        return InferenceHealthResponse(
            status="remote",
            inference_mode=inference_mode.value,
            is_ready=True,
            is_healthy=True,
            pipeline_initialized=False,
            model_loaded=False,
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            avg_latency_ms=0.0,
        )
    
    # Local mode - get actual engine health
    try:
        engine = await get_serving_engine()
        if not engine:
            return InferenceHealthResponse(
                status="unavailable",
                inference_mode=inference_mode.value,
                is_ready=False,
                is_healthy=False,
                pipeline_initialized=False,
                model_loaded=False,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                avg_latency_ms=0.0,
                error_message="Engine not initialized",
            )
        
        health = engine.get_health()
        return InferenceHealthResponse(
            status=health.get("state", "unknown"),
            inference_mode=inference_mode.value,
            is_ready=health.get("is_ready", False),
            is_healthy=health.get("is_healthy", False),
            pipeline_initialized=health.get("pipeline_initialized", False),
            model_loaded=health.get("model_loaded", False),
            total_requests=health.get("total_requests", 0),
            successful_requests=health.get("successful_requests", 0),
            failed_requests=health.get("failed_requests", 0),
            avg_latency_ms=health.get("avg_latency_ms", 0.0),
            error_message=health.get("error_message"),
        )
        
    except Exception as e:
        logger.error(f"Error getting inference health: {e}")
        return InferenceHealthResponse(
            status="error",
            inference_mode=inference_mode.value,
            is_ready=False,
            is_healthy=False,
            pipeline_initialized=False,
            model_loaded=False,
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            avg_latency_ms=0.0,
            error_message=str(e),
        )


@router.get("/health/ready")
async def get_inference_readiness():
    """
    Kubernetes readiness probe endpoint.
    
    Returns 200 if ready to serve requests, 503 otherwise.
    """
    inference_mode = get_inference_mode()
    
    if inference_mode == InferenceMode.DISABLED:
        raise HTTPException(status_code=503, detail="Inference disabled")
    
    if inference_mode in (InferenceMode.MOCK, InferenceMode.REMOTE):
        return {"ready": True}
    
    try:
        engine = await get_serving_engine()
        if engine and engine.is_ready():
            return {"ready": True}
        raise HTTPException(status_code=503, detail="Engine not ready")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/health/live")
async def get_inference_liveness():
    """
    Kubernetes liveness probe endpoint.
    
    Returns 200 if engine is alive, 503 otherwise.
    """
    inference_mode = get_inference_mode()
    
    if inference_mode == InferenceMode.DISABLED:
        return {"alive": True, "mode": "disabled"}
    
    if inference_mode in (InferenceMode.MOCK, InferenceMode.REMOTE):
        return {"alive": True, "mode": inference_mode.value}
    
    try:
        engine = await get_serving_engine()
        if engine and engine.is_healthy():
            return {"alive": True, "mode": "local"}
        raise HTTPException(status_code=503, detail="Engine unhealthy")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/metrics", response_model=InferenceMetricsResponse)
async def get_inference_metrics():
    """
    Get Prometheus-compatible inference metrics.
    """
    inference_mode = get_inference_mode()
    
    if inference_mode != InferenceMode.LOCAL:
        return InferenceMetricsResponse(
            serving_engine_requests_total=0,
            serving_engine_requests_success=0,
            serving_engine_requests_failed=0,
            serving_engine_latency_avg_ms=0.0,
            serving_engine_ready=1 if inference_mode != InferenceMode.DISABLED else 0,
            serving_engine_healthy=1,
        )
    
    try:
        engine = await get_serving_engine()
        if not engine:
            return InferenceMetricsResponse(
                serving_engine_requests_total=0,
                serving_engine_requests_success=0,
                serving_engine_requests_failed=0,
                serving_engine_latency_avg_ms=0.0,
                serving_engine_ready=0,
                serving_engine_healthy=0,
            )
        
        metrics = engine.get_metrics()
        return InferenceMetricsResponse(**metrics)
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return InferenceMetricsResponse(
            serving_engine_requests_total=0,
            serving_engine_requests_success=0,
            serving_engine_requests_failed=0,
            serving_engine_latency_avg_ms=0.0,
            serving_engine_ready=0,
            serving_engine_healthy=0,
        )


# ============================================================================
# Pipeline Management Endpoints
# ============================================================================

@router.post("/pipeline/warmup")
async def warmup_pipeline():
    """
    Warmup inference pipeline.
    
    Pre-loads models and caches for faster first inference.
    """
    inference_mode = get_inference_mode()
    
    if inference_mode != InferenceMode.LOCAL:
        return {"status": "skipped", "reason": f"Mode is {inference_mode.value}"}
    
    try:
        engine = await get_serving_engine()
        if not engine:
            raise HTTPException(status_code=503, detail="Engine not available")
        
        if engine._pipeline:
            await engine._pipeline.warmup()
            return {"status": "success", "message": "Pipeline warmed up"}
        
        return {"status": "skipped", "reason": "Pipeline not initialized"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Warmup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/adapters/preload")
async def preload_adapters(adapter_ids: List[str]):
    """
    Preload DoRA adapters into cache.
    
    Args:
        adapter_ids: List of adapter IDs to preload
    """
    inference_mode = get_inference_mode()
    
    if inference_mode != InferenceMode.LOCAL:
        return {"status": "skipped", "reason": f"Mode is {inference_mode.value}"}
    
    try:
        engine = await get_serving_engine()
        if not engine:
            raise HTTPException(status_code=503, detail="Engine not available")
        
        await engine.preload_adapters(adapter_ids)
        return {"status": "success", "preloaded": adapter_ids}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Preload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/document")
async def add_rag_document(
    doc_id: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Add document to RAG index.
    
    Args:
        doc_id: Unique document ID
        content: Document content
        metadata: Optional metadata
    """
    inference_mode = get_inference_mode()
    
    if inference_mode != InferenceMode.LOCAL:
        return {"status": "skipped", "reason": f"Mode is {inference_mode.value}"}
    
    try:
        engine = await get_serving_engine()
        if not engine:
            raise HTTPException(status_code=503, detail="Engine not available")
        
        engine.add_rag_document(doc_id, content, metadata)
        return {"status": "success", "doc_id": doc_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Add document error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Helper Functions
# ============================================================================

def _generate_mock_response(
    request_id: str,
    inference_request: InferenceRequest,
    start_time: float
) -> InferenceResponse:
    """Generate mock response for testing."""
    mock_text = f"[MOCK] Response to: {inference_request.prompt[:100]}..."
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    return InferenceResponse(
        request_id=request_id,
        text=mock_text,
        tokens_generated=len(mock_text.split()),
        latency_ms=latency_ms,
        experts_used=[{"id": "mock_expert", "weight": 1.0}],
        rag_context_used=False,
        model_version="mock-v1.0.0",
        credits_used=0.0,
    )


async def _mock_stream_generator(prompt: str) -> AsyncGenerator[str, None]:
    """Generate mock streaming response."""
    mock_response = f"[MOCK] Streaming response to: {prompt[:50]}..."
    for word in mock_response.split():
        yield f"data: {word} \n\n"
        await asyncio.sleep(0.05)
    yield "data: [DONE]\n\n"


async def _proxy_to_serving_node(
    request_id: str,
    inference_request: InferenceRequest,
    start_time: float
) -> InferenceResponse:
    """Proxy inference request to remote serving node."""
    from .serving_node_registry import ServingNodeRegistry
    
    try:
        # Get available serving node
        registry = ServingNodeRegistry(database)
        node = await registry.get_available_node()
        
        if not node:
            raise HTTPException(
                status_code=503,
                detail="No serving nodes available"
            )
        
        # Proxy request to serving node
        import httpx
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{node['endpoint']}/inference",
                json={
                    "prompt": inference_request.prompt,
                    "max_tokens": inference_request.max_tokens,
                    "temperature": inference_request.temperature,
                    "skip_rag": inference_request.skip_rag,
                    "force_experts": inference_request.force_experts,
                }
            )
            response.raise_for_status()
            result = response.json()
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return InferenceResponse(
            request_id=request_id,
            text=result.get("text", ""),
            tokens_generated=result.get("tokens_generated", 0),
            latency_ms=latency_ms,
            experts_used=result.get("experts_used", []),
            rag_context_used=result.get("rag_context_used", False),
            model_version=result.get("model_version", "remote"),
            credits_used=0.0,
        )
        
    except httpx.HTTPError as e:
        logger.error(f"Proxy error: {e}")
        raise HTTPException(status_code=502, detail="Serving node error")
    except Exception as e:
        logger.error(f"Proxy error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Proxy failed")


async def _proxy_stream_to_serving_node(
    inference_request: InferenceRequest
) -> AsyncGenerator[str, None]:
    """Proxy streaming request to serving node."""
    from .serving_node_registry import ServingNodeRegistry
    
    try:
        registry = ServingNodeRegistry(database)
        node = await registry.get_available_node()
        
        if not node:
            yield "data: [ERROR] No serving nodes available\n\n"
            return
        
        import httpx
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{node['endpoint']}/inference/stream",
                json={
                    "prompt": inference_request.prompt,
                    "max_tokens": inference_request.max_tokens,
                    "temperature": inference_request.temperature,
                }
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        yield f"{line}\n"
                        
    except Exception as e:
        logger.error(f"Stream proxy error: {e}")
        yield f"data: [ERROR] {str(e)}\n\n"


def _calculate_credits(tokens_generated: int) -> float:
    """Calculate credits based on tokens generated."""
    # 1 credit per 1000 tokens
    credit_rate = float(os.getenv("R3MES_CREDIT_RATE", "0.001"))
    return tokens_generated * credit_rate


async def _deduct_credits(wallet_address: str, credits: float):
    """Deduct credits from wallet (background task)."""
    try:
        await database.deduct_credits(wallet_address, credits)
        logger.debug(f"Deducted {credits} credits from {wallet_address}")
    except Exception as e:
        logger.error(f"Failed to deduct credits: {e}")

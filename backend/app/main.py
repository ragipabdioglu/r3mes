"""
R3MES Backend Inference Service - FastAPI Application

Web sitesinin (Frontend) bağlanacağı kapıları açar.
"""

from fastapi import FastAPI, HTTPException, Request, Depends, Header, Path as FastAPIPath
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
from typing import Optional, AsyncGenerator, List
import httpx
import random
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
import os
import logging

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Import inference mode configuration (GPU-less deployment support)
from .inference_mode import (
    get_inference_mode, 
    should_load_ai_libraries, 
    is_inference_available,
    InferenceMode,
    validate_inference_mode_for_startup,
)

# Conditional imports based on inference mode
# These modules require GPU libraries (torch, transformers, etc.)
# Only import them when R3MES_INFERENCE_MODE=local
_model_manager = None
_semantic_router = None
_task_queue = None

def get_model_manager():
    """Lazy load model manager only when needed."""
    global _model_manager
    if _model_manager is None:
        if not should_load_ai_libraries():
            return None
        from .model_manager import AIModelManager
        from .config_manager import get_config_manager
        config = get_config_manager().load()
        _model_manager = AIModelManager(base_model_path=config.base_model_path)
    return _model_manager

def get_semantic_router():
    """Lazy load semantic router only when needed."""
    global _semantic_router
    if _semantic_router is None:
        if not should_load_ai_libraries():
            # Return a simple keyword-based router for non-local modes
            return None
        from .semantic_router import SemanticRouter
        similarity_threshold = float(os.getenv("SEMANTIC_ROUTER_THRESHOLD", "0.7"))
        _semantic_router = SemanticRouter(
            similarity_threshold=similarity_threshold,
            use_semantic=True
        )
    return _semantic_router

def get_task_queue():
    """Lazy load task queue only when needed."""
    global _task_queue
    if _task_queue is None:
        if not should_load_ai_libraries():
            return None
        from .task_queue import TaskQueue
        max_workers = int(os.getenv("MAX_WORKERS", "1"))
        _task_queue = TaskQueue(max_workers=max_workers)
    return _task_queue

from .database_async import AsyncDatabase
from .setup_logging import setup_logging
from .config_manager import get_config_manager
from .config_endpoints import router as config_router
from .leaderboard_endpoints import router as leaderboard_router
from .advanced_analytics import router as analytics_router
from .debug_endpoints import router as debug_router
from .faucet import router as faucet_router
from .websocket_endpoints import router as websocket_router
from .system_endpoints import router as system_router
from .health_endpoints import router as health_router
from .serving_endpoints import router as serving_router
from .proposer_endpoints import router as proposer_router
from .role_endpoints import router as role_router
from .serving_node_registry import ServingNodeRegistry
from .notifications import get_notification_service, NotificationPriority
from .cache import get_cache_manager
from .cache_middleware import cache_middleware
from .metrics import get_metrics_response
from .sentry import init_sentry
from .debug_middleware import DebugMiddleware
from .panic_recovery import PanicRecoveryMiddleware
from .graceful_shutdown import get_graceful_shutdown
from .opentelemetry_setup import (
    setup_opentelemetry,
    instrument_fastapi,
    instrument_http_clients,
    instrument_databases,
)
from .trace_middleware import TraceMiddleware
from .url_validator import get_ssrf_protector, validate_serving_endpoint
from .exceptions import (
    ProductionConfigurationError,
    InvalidConfigurationError,
    MissingEnvironmentVariableError,
    InvalidAPIKeyError,
    MissingCredentialsError,
    InsufficientCreditsError,
    CreditDeductionError,
    ModelLoadError,
    AdapterSelectionError,
    InferenceError,
    InvalidInputError,
    InvalidWalletAddressError,
)

# Initialize Sentry first (before logging)
init_sentry()

# Setup logging first
setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    enable_file_logging=os.getenv("ENABLE_FILE_LOGGING", "true").lower() == "true"
)

logger = logging.getLogger(__name__)

# Load configuration
config_manager = get_config_manager()
config = config_manager.load()

# Initialize components using config (needed for lifespan function)
# Model manager is now lazy-loaded based on inference mode
database = AsyncDatabase(db_path=config.database_path, chain_json_path=config.chain_json_path)
cache_manager = get_cache_manager()
serving_node_registry = ServingNodeRegistry(database)

# Initialize authentication with database
from .auth import init_auth
init_auth(database)

# Initialize services
from .services import UserService, APIKeyService, ChatService
from .cache_invalidation_strategy import get_cache_invalidation_manager

user_service = UserService(database, cache_manager)
api_key_service = APIKeyService(database, cache_manager)
chat_service = ChatService(database, cache_manager)
cache_invalidation_manager = get_cache_invalidation_manager(cache_manager)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager for startup and shutdown operations.
    Replaces the deprecated @app.on_event("startup") and @app.on_event("shutdown") decorators.
    Ensures sequential execution of startup tasks.
    """
    # Setup graceful shutdown
    graceful_shutdown = get_graceful_shutdown()
    graceful_shutdown.setup_signal_handlers()
    
    # Register shutdown handlers
    async def shutdown_database():
        await database.close()
    
    async def shutdown_cache():
        await cache_manager.close()
    
    async def shutdown_inference():
        # Only shutdown inference executor if it was initialized
        if should_load_ai_libraries():
            try:
                from .inference_executor import shutdown_inference_executor
                await shutdown_inference_executor()
            except ImportError:
                pass  # Inference executor not available
    
    async def shutdown_websockets():
        from .websocket_manager import connection_manager
        # Close all WebSocket connections gracefully
        for channel, connections in list(connection_manager.active_connections.items()):
            for websocket in list(connections):
                try:
                    await websocket.close(code=1001, reason="Server shutting down")
                except Exception as e:
                    logger.warning(f"Error closing WebSocket: {e}")
                connection_manager.disconnect(websocket, channel)
    
    async def shutdown_error_rate_monitor():
        try:
            from .error_rate_monitor import get_error_rate_monitor
            error_rate_monitor = get_error_rate_monitor()
            await error_rate_monitor.stop()
        except Exception as e:
            logger.warning(f"Error stopping error rate monitor: {e}")
    
    graceful_shutdown.register_shutdown_handler(shutdown_database)
    graceful_shutdown.register_shutdown_handler(shutdown_cache)
    graceful_shutdown.register_shutdown_handler(shutdown_inference)
    graceful_shutdown.register_shutdown_handler(shutdown_websockets)
    graceful_shutdown.register_shutdown_handler(shutdown_error_rate_monitor)
    
    # Store graceful shutdown in app state
    app.state.graceful_shutdown = graceful_shutdown
    app.state.is_production = os.getenv("R3MES_ENV", "development").lower() == "production"
    
    # ========== STARTUP ==========
    logger.info("Starting R3MES Backend Service...")
    
    # 0. Validate inference mode configuration
    inference_mode = get_inference_mode()
    logger.info(f"Inference mode: {inference_mode.value}")
    
    is_valid, mode_message = validate_inference_mode_for_startup()
    if not is_valid:
        logger.error(f"❌ Inference mode validation failed: {mode_message}")
        raise RuntimeError(mode_message)
    logger.info(f"✅ {mode_message}")
    
    # 1. Validate environment variables
    try:
        from .env_validator import validate_environment
        validate_environment()
        logger.info("✅ Environment variables validated successfully")
    except Exception as e:
        logger.error(f"❌ Environment validation failed: {e}")
        raise
    
    # 2. Validate production environment configuration
    is_production = os.getenv("R3MES_ENV", "development").lower() == "production"
    if is_production:
        # Check that mock model is not enabled in production
        use_mock_model = os.getenv("R3MES_USE_MOCK_MODEL", "false").lower() == "true"
        if use_mock_model:
            logger.error("FATAL: R3MES_USE_MOCK_MODEL=true detected in production environment")
            raise ProductionConfigurationError(
                "FATAL: R3MES_USE_MOCK_MODEL=true is not allowed in production. "
                "Mock models are for development/testing only. "
                "Set R3MES_USE_MOCK_MODEL=false and ensure a real model is configured."
            )
        # Only check model manager if inference mode is LOCAL
        if inference_mode == InferenceMode.LOCAL:
            model_manager = get_model_manager()
            if model_manager is None or model_manager.base_model is None:
                logger.error("FATAL: Model manager initialized in mock mode in production")
                raise ModelLoadError(
                    "FATAL: Model could not be loaded in production environment. "
                    "Ensure R3MES_MODEL_IPFS_HASH, R3MES_MODEL_NAME, or BASE_MODEL_PATH is set correctly. "
                    "Model loading failed during initialization."
                )
            logger.info("✅ Production environment validated: Real model loaded successfully")
        else:
            logger.info(f"✅ Production environment validated: Inference mode is {inference_mode.value} (no local model required)")
    
    # 2. Connect to database
    logger.info("Connecting to database...")
    await database.connect()
    
    # 3. Connect to cache (Redis)
    logger.info("Connecting to Redis cache...")
    await cache_manager.connect()
    
    # 4. Initialize inference executor (only if local inference mode)
    if inference_mode == InferenceMode.LOCAL:
        logger.info("Initializing inference executor...")
        max_workers = int(os.getenv("MAX_WORKERS", "1"))
        from .inference_executor import get_inference_executor
        await get_inference_executor(max_workers=max_workers)
        logger.info("✅ Inference executor initialized")
    else:
        logger.info(f"Skipping inference executor initialization (mode: {inference_mode.value})")
    
    # 5. Load adapters on startup (only if local inference mode)
    if inference_mode == InferenceMode.LOCAL:
        logger.info("Loading adapters from checkpoints...")
        notification_service = get_notification_service()
        model_manager = get_model_manager()
        try:
            # Resolve relative path from current working directory to avoid hardcoded path issues
            checkpoints_dir = (Path.cwd() / "backend" / "checkpoints").resolve()
            
            if checkpoints_dir.exists() and model_manager is not None:
                loaded_count = 0
                for adapter_dir in checkpoints_dir.iterdir():
                    if adapter_dir.is_dir():
                        adapter_name = adapter_dir.name
                        adapter_path = str(adapter_dir)
                        
                        if model_manager.load_adapter(adapter_name, adapter_path):
                            logger.info(f"Auto-loaded adapter: {adapter_name}")
                            loaded_count += 1
                
                if loaded_count > 0:
                    await notification_service.send_system_alert(
                        component="Backend",
                        alert_type="startup",
                        message=f"Backend started successfully. Loaded {loaded_count} adapters.",
                        priority=NotificationPriority.LOW
                    )
                    logger.info(f"✅ Loaded {loaded_count} adapters on startup")
        except Exception as e:
            logger.error(f"Error loading adapters during startup: {e}")
            await notification_service.send_system_alert(
                component="Backend",
                alert_type="startup_error",
                message=f"Error loading adapters during startup: {str(e)}",
                priority=NotificationPriority.CRITICAL
            )
    else:
        logger.info(f"Skipping adapter loading (mode: {inference_mode.value})")
    
    # 6. Start blockchain indexer (if using PostgreSQL)
    if database.config.is_postgresql():
        logger.info("Starting blockchain indexer...")
        try:
            from .indexer import get_indexer
            indexer = get_indexer(database)
            await indexer.start()
            logger.info("✅ Blockchain indexer started")
        except Exception as e:
            logger.warning(f"Failed to start blockchain indexer: {e} (continuing without indexer)")
    
    # 6.5. Start error rate monitor
    logger.info("Starting error rate monitor...")
    try:
        from .error_rate_monitor import get_error_rate_monitor
        error_rate_monitor = get_error_rate_monitor()
        await error_rate_monitor.start()
        logger.info("✅ Error rate monitor started")
    except Exception as e:
        logger.warning(f"Failed to start error rate monitor: {e} (continuing without error rate monitoring)")
    
    # 7. Warm cache on startup
    try:
        from .cache_warming import get_cache_warmer
        from .blockchain_rpc_client import get_blockchain_rpc_client
        from .blockchain_query_client import BlockchainQueryClient
        
        rpc_client = get_blockchain_rpc_client()
        query_client = BlockchainQueryClient()
        cache_warmer = get_cache_warmer(database, rpc_client, query_client)
        
        # Warm cache asynchronously (don't block startup)
        asyncio.create_task(cache_warmer.warm_on_startup())
        logger.info("✅ Cache warming initiated")
    except Exception as e:
        logger.warning(f"Failed to warm cache on startup: {e} (continuing without cache warming)")
    
    # 8. Start system metrics collector
    try:
        from .system_metrics_collector import get_system_metrics_collector
        metrics_interval = float(os.getenv("BACKEND_METRICS_INTERVAL", "10.0"))
        metrics_collector = get_system_metrics_collector(interval=metrics_interval)
        metrics_collector.start()
        logger.info("✅ System metrics collector started")
    except Exception as e:
        logger.warning(f"Failed to start system metrics collector: {e} (continuing without metrics)")
    
    # 9. Start serving node cleanup (event-driven via WebSocket disconnect)
    logger.info("Setting up serving node cleanup (event-driven)...")
    try:
        from .websocket_manager import connection_manager
        
        async def on_websocket_disconnect(channel: str):
            """Event-driven cleanup: triggered when WebSocket disconnects."""
            try:
                # Cleanup stale nodes when serving node WebSocket disconnects
                if channel == "serving" or channel.startswith("serving_"):
                    count = await serving_node_registry.cleanup_stale_nodes(max_age_seconds=120)
                    if count > 0:
                        logger.info(f"Cleaned up {count} stale serving nodes after WebSocket disconnect on {channel}")
            except Exception as e:
                logger.warning(f"Error in serving node cleanup on disconnect: {e}")
        
        # Register disconnect callback for event-driven cleanup
        connection_manager.register_disconnect_callback(on_websocket_disconnect)
        logger.info("✅ Serving node cleanup registered (event-driven via WebSocket disconnect)")
        
        # Fallback: Periodic cleanup as backup (less frequent, only if no events)
        async def periodic_cleanup_fallback():
            """Periodic cleanup as fallback (runs every 5 minutes instead of 1 minute)."""
            import asyncio
            while True:
                try:
                    await asyncio.sleep(300)  # Run every 5 minutes (reduced from 1 minute)
                    count = await serving_node_registry.cleanup_stale_nodes(max_age_seconds=120)
                    if count > 0:
                        logger.info(f"Periodic cleanup: removed {count} stale serving nodes")
                except asyncio.CancelledError:
                    logger.info("Periodic cleanup task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in periodic cleanup task: {e}", exc_info=True)
                    await asyncio.sleep(300)  # Wait before retrying
        
        cleanup_task = asyncio.create_task(periodic_cleanup_fallback())
        logger.info("✅ Periodic cleanup fallback started (5 minute interval)")
    except Exception as e:
        logger.warning(f"Failed to start serving node cleanup task: {e} (continuing without cleanup)")
    
    # 10. Setup OpenTelemetry for distributed tracing
    try:
        jaeger_endpoint = os.getenv("JAEGER_ENDPOINT")
        otlp_endpoint = os.getenv("OTLP_ENDPOINT")
        enable_console_tracing = os.getenv("ENABLE_CONSOLE_TRACING", "false").lower() == "true"
        
        if jaeger_endpoint or otlp_endpoint or enable_console_tracing:
            setup_opentelemetry(
                service_name="r3mes-backend",
                jaeger_endpoint=jaeger_endpoint,
                otlp_endpoint=otlp_endpoint,
                enable_console_exporter=enable_console_tracing
            )
            instrument_http_clients()
            instrument_databases()
            logger.info("✅ OpenTelemetry tracing initialized")
        else:
            logger.info("OpenTelemetry tracing disabled (no endpoint configured)")
    except Exception as e:
        logger.warning(f"Failed to setup OpenTelemetry: {e} (continuing without tracing)")
    
    logger.info("✅ R3MES Backend Service started successfully")
    
    # Yield control to the application
    yield
    
    # ========== SHUTDOWN ==========
    logger.info("Shutting down R3MES Backend Service...")
    
    # Stop system metrics collector
    try:
        from .system_metrics_collector import get_system_metrics_collector
        metrics_collector = get_system_metrics_collector()
        metrics_collector.stop()
        logger.info("✅ System metrics collector stopped")
    except Exception as e:
        logger.warning(f"Error stopping metrics collector: {e}")
    
    # Stop blockchain indexer first
    if database.config.is_postgresql():
        try:
            from .indexer import get_indexer
            indexer = get_indexer(database)
            await indexer.stop()
            logger.info("✅ Blockchain indexer stopped")
        except Exception as e:
            logger.warning(f"Error stopping indexer: {e}")
    
    # Execute all shutdown handlers with timeout
    shutdown_timeout = float(os.getenv("BACKEND_SHUTDOWN_TIMEOUT", "30.0"))
    await graceful_shutdown.execute_shutdown_handlers(timeout=shutdown_timeout)
    
    logger.info("✅ R3MES Backend Service shut down successfully")

app = FastAPI(
    title="R3MES Inference Service",
    version="1.0.0",
    description="AI inference service with Multi-LoRA support and credit system",
    lifespan=lifespan
)

# Rate Limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration - Production ready
# In production, NEVER allow wildcard origins
is_production = os.getenv("R3MES_ENV", "development").lower() == "production"
is_test_mode = os.getenv("R3MES_TEST_MODE", "false").lower() == "true"

if is_production and not is_test_mode:
    # Production: Strict CORS - only allow specific origins
    # In production, CORS_ALLOWED_ORIGINS must be explicitly set (no localhost fallback)
    cors_origins = os.getenv("CORS_ALLOWED_ORIGINS")
    if not cors_origins:
        raise MissingEnvironmentVariableError(
            "CORS_ALLOWED_ORIGINS must be set in production mode. "
            "Do not use localhost in production."
        )
    allowed_origins = cors_origins.split(",")
    # Remove any wildcard entries in production
    allowed_origins = [origin for origin in allowed_origins if origin != "*"]
    # Validate that no localhost origins are in production
    for origin in allowed_origins:
        if "localhost" in origin or "127.0.0.1" in origin:
            raise ProductionConfigurationError(
                f"Localhost origin '{origin}' is not allowed in production. "
                "Set CORS_ALLOWED_ORIGINS to production URLs only."
            )
    if not allowed_origins:
        raise InvalidConfigurationError("CORS_ALLOWED_ORIGINS must contain at least one valid origin in production mode")
else:
    # Development/Test mode
    cors_allowed = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000")
    allowed_origins = cors_allowed.split(",")
    
    # Only allow wildcard in explicit test mode
    if os.getenv("CORS_ALLOW_ALL", "false").lower() == "true" and is_test_mode:
        allowed_origins = ["*"]
        logger.warning("CORS: Allowing all origins (TEST MODE ONLY)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Requested-With", "Accept"],
    expose_headers=["X-Request-ID", "X-Trace-ID", "X-Span-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

# OpenTelemetry instrumentation (must be before other middleware)
try:
    instrument_fastapi(app)
    logger.info("✅ FastAPI instrumented with OpenTelemetry")
except Exception as e:
    logger.warning(f"Failed to instrument FastAPI with OpenTelemetry: {e}")

# Trace middleware (for log correlation - must be early in chain)
app.add_middleware(TraceMiddleware)

# Panic recovery middleware (must be first to catch all exceptions)
app.add_middleware(PanicRecoveryMiddleware)

# Cache middleware (only for GET requests)
app.middleware("http")(cache_middleware)

# Debug middleware (only active when debug mode is enabled)
app.add_middleware(DebugMiddleware)

# Include configuration router
app.include_router(config_router)

# Include leaderboard router
app.include_router(leaderboard_router)

# Include advanced analytics router
app.include_router(analytics_router)

# Include WebSocket router
app.include_router(websocket_router)

# Include debug router (only works when debug mode is enabled)
app.include_router(debug_router)

# Include faucet router
app.include_router(faucet_router)

# Include system router (version, time)
app.include_router(system_router)

# Include health check router (detailed health endpoints)
app.include_router(health_router)

# Include serving router
app.include_router(serving_router)

# Include proposer router
app.include_router(proposer_router)

# Include role management router
app.include_router(role_router)

# Include validator endpoints (trust scores)
from .validator_endpoints import router as validator_router
app.include_router(validator_router)

# Include miner endpoints (locations, tiers, leaderboard)
from .miner_endpoints import router as miner_router
app.include_router(miner_router)

# Note: Components (model_manager, database, cache_manager) are initialized above for lifespan function
# Model manager, task queue, and semantic router are now lazy-loaded based on inference mode

# Task Queue and Semantic Router are lazy-loaded via get_task_queue() and get_semantic_router()
# They are only initialized when R3MES_INFERENCE_MODE=local

# Simple keyword-based adapter routing for non-local modes
def simple_keyword_router(message: str) -> str:
    """Simple keyword-based adapter routing when SemanticRouter is not available."""
    message_lower = message.lower()
    
    # Simple keyword matching
    if any(word in message_lower for word in ["code", "program", "function", "debug", "error"]):
        return "coding"
    elif any(word in message_lower for word in ["legal", "law", "contract", "court"]):
        return "legal"
    elif any(word in message_lower for word in ["medical", "health", "doctor", "symptom"]):
        return "medical"
    elif any(word in message_lower for word in ["finance", "money", "invest", "stock"]):
        return "finance"
    else:
        return "general"

# API Key Authentication
security = HTTPBearer(auto_error=False)

async def get_wallet_from_auth(
    request: Request,
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> Optional[str]:
    """
    API key veya wallet_address'ten wallet adresini alır.
    
    Öncelik sırası:
    1. X-API-Key header
    2. Authorization Bearer token
    3. Request body'deki wallet_address
    """
    # Try API key from header
    api_key = x_api_key or (authorization.credentials if authorization else None)
    
    if api_key:
        api_key_info = await database.validate_api_key(api_key)
        if api_key_info and api_key_info["is_active"]:
            return api_key_info["wallet_address"]
        else:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired API key"
            ) from InvalidAPIKeyError("Invalid or expired API key")
    
    return None

# Request models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000, description="User message for AI inference")
    wallet_address: Optional[str] = Field(None, description="Wallet address (optional if API key is provided)")
    
    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Validate and sanitize message input."""
        if not v or not v.strip():
            raise InvalidInputError("Message cannot be empty")
        # Remove null bytes and control characters (except newlines and tabs)
        v = v.replace('\x00', '')
        # Limit message length to prevent DoS
        if len(v) > 10000:
            raise InvalidInputError("Message too long (max 10000 characters)")
        return v.strip()
    
    @field_validator("wallet_address")
    @classmethod
    def validate_wallet_address(cls, v: Optional[str]) -> Optional[str]:
        """Validate Cosmos wallet address format."""
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        # Basic validation: Cosmos addresses start with specific prefixes
        if not v.startswith("remes"):
            raise InvalidWalletAddressError("Invalid address format: must start with 'remes'")
        if len(v) < 20 or len(v) > 60:
            raise InvalidWalletAddressError("Invalid address length (must be 20-60 characters)")
        # Check for invalid characters
        if not all(c.isalnum() or c in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'] for c in v):
            raise InvalidWalletAddressError("Invalid address format: contains invalid characters")
        return v

class UserInfoResponse(BaseModel):
    wallet_address: str
    credits: float
    is_miner: bool

class NetworkStatsResponse(BaseModel):
    active_miners: int
    total_users: int
    total_credits: float
    block_height: Optional[int] = None

class BlockResponse(BaseModel):
    height: int
    miner: Optional[str] = None
    timestamp: Optional[str] = None
    hash: Optional[str] = None

class BlocksResponse(BaseModel):
    blocks: list[BlockResponse]
    limit: int
    total: int

class MinerStatsResponse(BaseModel):
    wallet_address: str
    total_earnings: float
    hashrate: float
    gpu_temperature: float
    blocks_found: int
    uptime_percentage: float
    network_difficulty: float

class EarningsHistoryResponse(BaseModel):
    earnings: list[dict]

class HashrateHistoryResponse(BaseModel):
    hashrate: list[dict]

@app.post("/chat")
@limiter.limit(config.rate_limit_chat)
async def chat(request: Request, chat_request: ChatRequest):
    """
    Chat endpoint - AI inference with credit system.
    
    Rate Limit: 10 requests per minute per IP (configurable via RATE_LIMIT_CHAT)
    
    Behavior depends on R3MES_INFERENCE_MODE:
    - disabled: Returns 503 error
    - mock: Returns mock responses
    - remote: Proxies to Serving Nodes
    - local: Runs inference locally (requires GPU)
    """
    # Check inference mode first
    inference_mode = get_inference_mode()
    
    # Handle disabled mode
    if inference_mode == InferenceMode.DISABLED:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Inference service disabled",
                "code": "INFERENCE_DISABLED",
                "message": "AI inference is not available on this server. Please try again later or contact support."
            }
        )
    
    # Handle mock mode
    if inference_mode == InferenceMode.MOCK:
        async def generate_mock_response():
            mock_response = f"[MOCK MODE] This is a simulated response to: {chat_request.message[:50]}..."
            for word in mock_response.split():
                yield word + " "
        return StreamingResponse(generate_mock_response(), media_type="text/plain")
    
    # Get wallet address from API key or request body
    wallet_from_auth = await get_wallet_from_auth(request)
    wallet_address = wallet_from_auth or chat_request.wallet_address
    
    if not wallet_address:
        raise HTTPException(
            status_code=401,
            detail="Either provide wallet_address in request body or valid API key in X-API-Key header"
        ) from MissingCredentialsError("Either provide wallet_address in request body or valid API key in X-API-Key header")
    
    # ATOMIC CREDIT RESERVATION: Reserve credit before starting inference
    # This prevents race conditions where concurrent requests could overdraw credits
    reservation = await database.reserve_credit_atomic(wallet_address, 1.0)
    
    if not reservation["success"]:
        notification_service = get_notification_service()
        await notification_service.send_notification(
            title="Low Credits Alert",
            message=f"Wallet {wallet_address} attempted to use chat with insufficient credits.",
            priority=NotificationPriority.LOW,
            metadata={"wallet_address": wallet_address, "error": reservation.get("error")}
        )
        raise HTTPException(
            status_code=402,
            detail="Insufficient credits. Please mine blocks to earn credits."
        ) from InsufficientCreditsError(f"Wallet {wallet_address}: {reservation.get('error')}")
    
    reservation_id = reservation["reservation_id"]
    logger.debug(f"Credit reserved for {wallet_address}, reservation_id: {reservation_id}")
    
    # Decide adapter using semantic router (if available) or simple keyword router
    semantic_router = get_semantic_router()
    if semantic_router is not None:
        adapter_result = semantic_router.decide_adapter(chat_request.message)
        if isinstance(adapter_result, tuple):
            adapter_name, similarity_score = adapter_result
            if similarity_score > 0:
                logger.debug(f"Semantic router: {adapter_name} (similarity: {similarity_score:.3f})")
        else:
            adapter_name = adapter_result
    else:
        # Use simple keyword router for remote mode
        adapter_name = simple_keyword_router(chat_request.message)
        logger.debug(f"Keyword router: {adapter_name}")
    
    # Try to route to serving node first (for both remote and local modes)
    serving_nodes = await serving_node_registry.get_serving_nodes_for_lora(
        lora_name=adapter_name,
        max_age_seconds=60
    )
    
    # Local inference function with atomic credit handling (only for local mode)
    async def _generate_local_inference_atomic():
        stream_started = False
        reservation_confirmed = False
        try:
            # Get inference executor
            from .inference_executor import get_inference_executor
            inference_executor = await get_inference_executor(max_workers=int(os.getenv("MAX_WORKERS", "1")))
            
            # Get model manager
            model_manager = get_model_manager()
            if model_manager is None:
                raise RuntimeError("Model manager not available. Set R3MES_INFERENCE_MODE=local and ensure GPU is available.")
            
            # Use inference executor to run in separate thread/process
            async for token in inference_executor.run_inference_streaming(
                chat_request.message,
                adapter_name,
                model_manager
            ):
                # First token received - stream has successfully started
                if not stream_started:
                    stream_started = True
                    # Confirm credit reservation after stream successfully starts
                    try:
                        confirmed = await database.confirm_credit_reservation(reservation_id)
                        if confirmed:
                            reservation_confirmed = True
                            logger.info(f"Credit reservation confirmed for {wallet_address} (reservation: {reservation_id})")
                        else:
                            logger.error(f"Failed to confirm credit reservation {reservation_id}")
                    except Exception as e:
                        logger.error(f"Error confirming credit reservation for {wallet_address}: {e}")
                
                yield token
                
        except Exception as e:
            # If stream fails before starting, rollback reservation
            if not stream_started and not reservation_confirmed:
                logger.warning(f"Stream failed before starting for {wallet_address}: {e}. Rolling back reservation.")
                await database.rollback_credit_reservation(reservation_id)
            # If stream fails after starting but reservation was confirmed, log it
            elif reservation_confirmed:
                logger.warning(f"Stream failed after starting for {wallet_address}: {e}. Credit already deducted.")
            raise
        finally:
            # If stream started but reservation wasn't confirmed, try to confirm now
            if stream_started and not reservation_confirmed:
                try:
                    await database.confirm_credit_reservation(reservation_id)
                    logger.info(f"Credit reservation confirmed in finally block for {wallet_address}")
                except Exception as e:
                    logger.error(f"Failed to confirm reservation in finally block: {e}")
    
    # If serving nodes available, route to one of them
    if serving_nodes:
        # Load balancing: round-robin (simple random selection for now)
        selected_node = random.choice(serving_nodes)
        endpoint_url = selected_node["endpoint_url"]
        
        # SSRF Protection: Validate serving node endpoint before proxying
        is_valid_url, validation_error = validate_serving_endpoint(endpoint_url)
        if not is_valid_url:
            logger.warning(
                f"SSRF Protection: Blocked request to serving node {selected_node['wallet_address']} "
                f"with invalid endpoint: {endpoint_url}. Reason: {validation_error}"
            )
            # Fall back to local inference instead of proxying to potentially malicious URL
            logger.info(f"Falling back to local inference due to invalid serving node endpoint")
            return StreamingResponse(_generate_local_inference_atomic(), media_type="text/plain")
        
        logger.info(f"Routing chat request to serving node: {selected_node['wallet_address']} at {endpoint_url}")
        
        # Proxy request to serving node with atomic credit handling
        async def generate_from_node_atomic():
            stream_started = False
            reservation_confirmed = False
            try:
                http_client_timeout = float(os.getenv("BACKEND_HTTP_CLIENT_TIMEOUT", "30.0"))
                async with httpx.AsyncClient(timeout=http_client_timeout) as client:
                    # Make streaming request to serving node
                    async with client.stream(
                        "POST",
                        f"{endpoint_url}/chat",
                        json={
                            "message": chat_request.message,
                            "wallet_address": wallet_address
                        },
                        headers={"Content-Type": "application/json"}
                    ) as response:
                        response.raise_for_status()
                        
                        # Stream response from serving node
                        async for chunk in response.aiter_text():
                            if chunk:
                                if not stream_started:
                                    stream_started = True
                                    # Confirm credit reservation after stream successfully starts
                                    try:
                                        confirmed = await database.confirm_credit_reservation(reservation_id)
                                        if confirmed:
                                            reservation_confirmed = True
                                            logger.info(f"Credit reservation confirmed for {wallet_address} (reservation: {reservation_id})")
                                        else:
                                            logger.error(f"Failed to confirm credit reservation {reservation_id}")
                                    except Exception as e:
                                        logger.error(f"Error confirming credit reservation for {wallet_address}: {e}")
                                
                                yield chunk
            except Exception as e:
                # If serving node fails, fall back to local inference
                logger.warning(f"Serving node request failed: {e}. Falling back to local inference.")
                if not stream_started:
                    # Retry with local inference (reservation still valid)
                    async for token in _generate_local_inference_atomic():
                        yield token
                else:
                    raise
            finally:
                # If stream started but reservation wasn't confirmed, try to confirm now
                if stream_started and not reservation_confirmed:
                    try:
                        await database.confirm_credit_reservation(reservation_id)
                        logger.info(f"Credit reservation confirmed in finally block for {wallet_address}")
                    except Exception as e:
                        logger.error(f"Failed to confirm reservation in finally block: {e}")
        
        return StreamingResponse(generate_from_node_atomic(), media_type="text/plain")
    
    # Handle case when no serving nodes available
    if inference_mode == InferenceMode.REMOTE:
        # In remote mode, we MUST have serving nodes
        logger.warning(f"No serving nodes available for {adapter_name} in remote mode")
        # Rollback the credit reservation since we can't process the request
        await database.rollback_credit_reservation(reservation_id)
        raise HTTPException(
            status_code=503,
            detail={
                "error": "No serving nodes available",
                "code": "NO_SERVING_NODES",
                "message": f"No serving nodes are currently available for adapter '{adapter_name}'. Please try again later."
            }
        )
    
    # Fallback to local inference (only for local mode)
    logger.debug(f"No serving nodes available for {adapter_name}, using local inference")
    return StreamingResponse(_generate_local_inference_atomic(), media_type="text/plain")

@app.get("/user/info/{wallet_address}")
@limiter.limit(config.rate_limit_get)
async def get_user_info(
    request: Request,
    wallet_address: str = FastAPIPath(..., description="Wallet address", min_length=20, max_length=60, pattern="^remes[a-z0-9]+$")
) -> UserInfoResponse:
    """
    Kullanıcı bilgilerini döndürür.
    
    Cüzdanın kalan kredisini ve madenci olup olmadığını (VIP durumu) JSON döndür.
    """
    user_info = await database.get_user_info(wallet_address)
    
    if not user_info:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserInfoResponse(
        wallet_address=user_info['wallet_address'],
        credits=user_info['credits'],
        is_miner=user_info['is_miner']
    )

@app.get("/network/stats")
@limiter.limit(config.rate_limit_get)
async def get_network_stats(request: Request) -> NetworkStatsResponse:
    """
    Ağ istatistiklerini döndürür.
    
    Aktif madenci sayısı, toplam blok sayısı gibi genel istatistikleri döndür.
    """
    stats = await database.get_network_stats()
    
    return NetworkStatsResponse(
        active_miners=stats['active_miners'],
        total_users=stats['total_users'],
        total_credits=stats['total_credits'],
        block_height=stats.get('block_height')
    )

@app.get("/blocks")
@limiter.limit(config.rate_limit_get)
async def get_blocks(
    request: Request,
    limit: int = Query(default=10, ge=1, le=100, description="Number of blocks to return (1-100)")
) -> BlocksResponse:
    """
    Son blokları döndürür.
    
    Args:
        limit: Döndürülecek blok sayısı (default: 10, max: 100)
    
    Returns:
        Blok listesi
    """
    # Field validation already ensures 1 <= limit <= 100, no manual validation needed
    
    blocks_data = await database.get_recent_blocks(limit=limit)
    
    blocks = [
        BlockResponse(
            height=block['height'],
            miner=block.get('miner'),
            timestamp=block.get('timestamp'),
            hash=block.get('hash')
        )
        for block in blocks_data
    ]
    
    return BlocksResponse(
        blocks=blocks,
        limit=limit,
        total=len(blocks)
    )

@app.get("/miner/stats/{wallet_address}")
@limiter.limit(config.rate_limit_get)
async def get_miner_stats(
    request: Request,
    wallet_address: str = PathParam(..., description="Miner wallet address", min_length=20, max_length=60, pattern="^remes[a-z0-9]+$")
) -> MinerStatsResponse:
    """
    Miner istatistiklerini döndürür.
    
    Args:
        wallet_address: Miner cüzdan adresi
    
    Returns:
        Miner istatistikleri
    """
    stats = await database.get_miner_stats(wallet_address)
    
    return MinerStatsResponse(
        wallet_address=stats['wallet_address'],
        total_earnings=stats['total_earnings'],
        hashrate=stats['hashrate'],
        gpu_temperature=stats['gpu_temperature'],
        blocks_found=stats['blocks_found'],
        uptime_percentage=stats['uptime_percentage'],
        network_difficulty=stats['network_difficulty']
    )

@app.get("/miner/earnings/{wallet_address}")
@limiter.limit(config.rate_limit_get)
async def get_miner_earnings(
    request: Request,
    wallet_address: str = PathParam(..., description="Miner wallet address", min_length=20, max_length=60, pattern="^remes[a-z0-9]+$"),
    days: int = Query(default=7, ge=1, le=365, description="Number of days to retrieve (1-365)")
) -> EarningsHistoryResponse:
    """
    Miner earnings geçmişini döndürür.
    
    Args:
        wallet_address: Miner cüzdan adresi
        days: Kaç günlük geçmiş (default: 7)
    
    Returns:
        Earnings geçmişi
    """
    earnings = await database.get_earnings_history(wallet_address, days=days)
    
    return EarningsHistoryResponse(earnings=earnings)

@app.get("/miner/hashrate/{wallet_address}")
@limiter.limit(config.rate_limit_get)
async def get_miner_hashrate(
    request: Request,
    wallet_address: str = PathParam(..., description="Miner wallet address", min_length=20, max_length=60, pattern="^remes[a-z0-9]+$"),
    days: int = Query(default=7, ge=1, le=365, description="Number of days to retrieve (1-365)")
) -> HashrateHistoryResponse:
    """
    Miner hashrate geçmişini döndürür.
    
    Args:
        wallet_address: Miner cüzdan adresi
        days: Kaç günlük geçmiş (default: 7)
    
    Returns:
        Hashrate geçmişi
    """
    hashrate = await database.get_hashrate_history(wallet_address, days=days)
    
    return HashrateHistoryResponse(hashrate=hashrate)

# ========== API Key Management Endpoints ==========

class CreateAPIKeyRequest(BaseModel):
    wallet_address: str = Field(..., description="Wallet address for API key")
    name: Optional[str] = Field(None, max_length=100, description="API key name (optional)")
    expires_days: Optional[int] = Field(None, ge=1, le=365, description="Expiration in days (1-365, optional)")
    
    @field_validator("wallet_address")
    @classmethod
    def validate_wallet_address(cls, v: str) -> str:
        """Validate Cosmos wallet address format."""
        v = v.strip()
        if not v:
            raise ValueError("Wallet address cannot be empty")
        if not v.startswith("remes"):
            raise ValueError("Invalid address format: must start with 'remes'")
        if len(v) < 20 or len(v) > 60:
            raise ValueError("Invalid address length (must be 20-60 characters)")
        return v
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: Optional[str]) -> Optional[str]:
        """Validate API key name."""
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        if len(v) > 100:
            raise ValueError("API key name too long (max 100 characters)")
        # Prevent XSS in name
        if any(char in v for char in ['<', '>', '"', "'", '&']):
            raise ValueError("API key name contains invalid characters")
        return v

class RevokeAPIKeyRequest(BaseModel):
    api_key_id: int = Field(..., ge=1, description="API key ID to revoke")
    wallet_address: str = Field(..., description="Wallet address that owns the API key")
    
    @field_validator("wallet_address")
    @classmethod
    def validate_wallet_address(cls, v: str) -> str:
        """Validate Cosmos wallet address format."""
        v = v.strip()
        if not v:
            raise ValueError("Wallet address cannot be empty")
        if not v.startswith("remes"):
            raise ValueError("Invalid address format: must start with 'remes'")
        if len(v) < 20 or len(v) > 60:
            raise ValueError("Invalid address length (must be 20-60 characters)")
        return v

# ========== Authentication Endpoints ==========

class LoginRequest(BaseModel):
    wallet_address: str = Field(..., description="Wallet address for login")
    
    @field_validator("wallet_address")
    @classmethod
    def validate_wallet_address(cls, v: str) -> str:
        """Validate Cosmos wallet address format."""
        v = v.strip()
        if not v:
            raise ValueError("Wallet address cannot be empty")
        if not v.startswith("remes"):
            raise ValueError("Invalid address format: must start with 'remes'")
        if len(v) < 20 or len(v) > 60:
            raise ValueError("Invalid address length (must be 20-60 characters)")
        return v

@app.post("/auth/login")
@limiter.limit("10/minute")  # Rate limit for login attempts
async def login(request: Request, login_request: LoginRequest):
    """
    Login endpoint - creates JWT token for wallet address.
    
    Args:
        login_request: Login request with wallet address
        
    Returns:
        JWT token and user info
    """
    from .auth import create_jwt_token
    
    wallet_address = login_request.wallet_address
    
    # Check if user exists in database
    user_info = await database.get_user_info(wallet_address)
    if not user_info:
        # Create user if doesn't exist
        await database.create_user(wallet_address)
        user_info = await database.get_user_info(wallet_address)
    
    # Create JWT token
    token = create_jwt_token(wallet_address)
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": 86400,  # 24 hours in seconds
        "user": {
            "wallet_address": user_info['wallet_address'],
            "credits": user_info['credits'],
            "is_miner": user_info['is_miner']
        }
    }

@app.get("/auth/me")
@limiter.limit("30/minute")
async def get_current_user(request: Request):
    """
    Get current user info from JWT token.
    
    Returns:
        Current user information
    """
    from .auth import require_auth
    
    wallet_address = await require_auth(request)
    user_info = await database.get_user_info(wallet_address)
    
    if not user_info:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "wallet_address": user_info['wallet_address'],
        "credits": user_info['credits'],
        "is_miner": user_info['is_miner']
    }

# ========== API Key Management Endpoints ==========

@app.post("/api-keys/create")
@limiter.limit("5/minute")  # Specific rate limit for API key creation
async def create_api_key(request: Request, key_request: CreateAPIKeyRequest):
    """
    Yeni bir API key oluşturur.
    
    Args:
        key_request: API key oluşturma isteği
        
    Returns:
        API key bilgileri
    """
    api_key = await database.create_api_key(
        wallet_address=key_request.wallet_address,
        name=key_request.name or "Default",
        expires_in_days=key_request.expires_days
    )
    
    # Format response (api_key is plaintext, but we don't store it)
    api_key_data = {
        "api_key": api_key,
        "name": key_request.name or "Default",
        "created_at": datetime.now().isoformat(),
        "expires_at": None
    }
    
    return {
        "api_key": api_key_data["api_key"],
        "name": api_key_data["name"],
        "created_at": api_key_data["created_at"],
        "expires_at": api_key_data["expires_at"],
        "message": "⚠️  Save this API key securely. It will not be shown again."
    }

@app.get("/api-keys/list/{wallet_address}")
@limiter.limit(config.rate_limit_get)
async def list_api_keys(request: Request, wallet_address: str):
    """
    Bir cüzdan için tüm API key'leri listeler.
    
    Args:
        wallet_address: Cüzdan adresi
        
    Returns:
        API key listesi
    """
    keys = await database.list_api_keys(wallet_address)
    return {"wallet_address": wallet_address, "api_keys": keys}

@app.post("/api-keys/revoke")
@limiter.limit("10/minute")  # Specific rate limit for API key revocation
async def revoke_api_key(request: Request, revoke_request: RevokeAPIKeyRequest):
    """
    Bir API key'i iptal eder.
    
    Args:
        revoke_request: API key iptal isteği
        
    Returns:
        İşlem sonucu
    """
    success = await database.revoke_api_key(
        api_key_id=revoke_request.api_key_id,
        wallet_address=revoke_request.wallet_address
    )
    
    if success:
        return {"message": "API key revoked successfully"}
    else:
        raise HTTPException(
            status_code=404,
            detail="API key not found or you don't have permission to revoke it"
        )

@app.delete("/api-keys/delete")
@limiter.limit(config.rate_limit_get)
async def delete_api_key(request: Request, revoke_request: RevokeAPIKeyRequest):
    """
    Bir API key'i tamamen siler.
    
    Args:
        revoke_request: API key silme isteği
        
    Returns:
        İşlem sonucu
    """
    success = await database.delete_api_key(
        api_key_id=revoke_request.api_key_id,
        wallet_address=revoke_request.wallet_address
    )
    
    if success:
        return {"message": "API key deleted successfully"}
    else:
        raise HTTPException(
            status_code=404,
            detail="API key not found or you don't have permission to delete it"
        )

@app.get("/health")
@limiter.limit("100/minute")  # Health check için daha yüksek limit
async def health_check(request: Request):
    """Health check endpoint."""
    inference_mode = get_inference_mode()
    model_manager = get_model_manager()
    
    return {
        "status": "healthy",
        "inference_mode": inference_mode.value,
        "model_loaded": model_manager.base_model is not None if model_manager else False,
        "adapters_count": len(model_manager.adapters) if model_manager else 0
    }

@app.get("/queue/stats")
@limiter.limit(config.rate_limit_get)
async def get_queue_stats(request: Request):
    """Get task queue statistics"""
    task_queue = get_task_queue()
    if task_queue is None:
        return {"error": "Task queue not available", "inference_mode": get_inference_mode().value}
    return task_queue.get_queue_stats()

@app.get("/metrics")
async def metrics_endpoint():
    """
    Prometheus metrics endpoint.
    
    Returns Prometheus-formatted metrics for monitoring.
    """
    return get_metrics_response()

# LoRA Registry and Serving Node Endpoints

@app.get("/api/lora/list", response_model=List[LoRARegistryResponse])
@limiter.limit(config.rate_limit_get)
async def get_lora_list(request: Request, active_only: bool = True):
    """
    Get list of available LoRA adapters.
    
    Query Parameters:
        active_only: Only return active LoRAs (default: True)
    """
    loras = await serving_node_registry.get_lora_list(active_only=active_only)
    # Return list directly (FastAPI will serialize it)
    return loras

class LoRARegisterRequest(BaseModel):
    name: str
    ipfs_hash: str
    description: Optional[str] = None
    category: Optional[str] = None
    version: Optional[str] = None

@app.post("/api/lora/register")
@limiter.limit(config.rate_limit_post)
async def register_lora(
    request: Request,
    lora_request: LoRARegisterRequest
):
    """
    Register a new LoRA adapter (admin endpoint).
    
    Body Parameters:
        name: LoRA adapter name (unique)
        ipfs_hash: IPFS hash of the LoRA file
        description: Optional description
        category: Optional category (e.g., "coding", "legal")
        version: Optional version string
    """
    result = await serving_node_registry.register_lora(
        name=lora_request.name,
        ipfs_hash=lora_request.ipfs_hash,
        description=lora_request.description,
        category=lora_request.category,
        version=lora_request.version
    )
    if result.get("success"):
        return result
    else:
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to register LoRA"))

class ServingNodeRegisterRequest(BaseModel):
    wallet_address: str
    endpoint_url: str
    available_lora_list: List[str]

@app.post("/api/serving-node/register")
@limiter.limit(config.rate_limit_post)
async def register_serving_node(
    request: Request,
    register_request: ServingNodeRegisterRequest
):
    """
    Register or update a serving node.
    
    Body Parameters:
        wallet_address: Miner wallet address
        endpoint_url: Serving node HTTP endpoint URL
        available_lora_list: List of LoRA adapter names this node can serve
    """
    result = await serving_node_registry.register_serving_node(
        wallet_address=register_request.wallet_address,
        endpoint_url=register_request.endpoint_url,
        available_lora_list=register_request.available_lora_list
    )
    if result.get("success"):
        return result
    else:
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to register serving node"))

class ServingNodeHeartbeatRequest(BaseModel):
    wallet_address: str
    status: Optional[str] = None
    current_load: Optional[int] = None

@app.post("/api/serving-node/heartbeat")
@limiter.limit(config.rate_limit_post)
async def update_serving_node_heartbeat(
    request: Request,
    heartbeat_request: ServingNodeHeartbeatRequest
):
    """
    Update serving node heartbeat.
    
    Body Parameters:
        wallet_address: Miner wallet address
        status: Optional status update (active, idle, busy)
        current_load: Optional current load (number of active requests)
    """
    result = await serving_node_registry.update_serving_node_heartbeat(
        wallet_address=heartbeat_request.wallet_address,
        status=heartbeat_request.status,
        current_load=heartbeat_request.current_load
    )
    if result.get("success"):
        return result
    else:
        raise HTTPException(status_code=404, detail=result.get("error", "Serving node not found"))

@app.get("/api/serving-node/list")
@limiter.limit(config.rate_limit_get)
async def get_serving_nodes_for_lora(request: Request, lora: Optional[str] = None, max_age_seconds: int = 60):
    """
    Get list of serving nodes.
    
    Query Parameters:
        lora: Filter by LoRA adapter name (optional)
        max_age_seconds: Maximum age of last heartbeat (default: 60s)
    """
    if lora:
        nodes = await serving_node_registry.get_serving_nodes_for_lora(
            lora_name=lora,
            max_age_seconds=max_age_seconds
        )
    else:
        # Return all active nodes (if no LoRA specified)
        # For now, return empty list if no LoRA specified
        # Could add a method to get all nodes if needed
        nodes = []
    
    return {"nodes": nodes, "count": len(nodes)}

# Note: Startup logic (including adapter loading) is now handled by the lifespan context manager

if __name__ == "__main__":
    import uvicorn
    # Get port from environment variable, default to 8000 for development
    port = int(os.getenv("BACKEND_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


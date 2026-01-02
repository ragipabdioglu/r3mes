"""
R3MES Backend Inference Service - Refactored Main Application

Clean, modular FastAPI application with proper separation of concerns.
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Import lifespan management
from .lifespan import create_lifespan

# Import middleware
from .middleware.error_handler import create_error_handler_middleware
from .cache_middleware import cache_middleware
from .debug_middleware import DebugMiddleware
from .panic_recovery import PanicRecoveryMiddleware

# Import routers
from .api.chat import router as chat_router
from .api.users import router as users_router
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

# Import utilities
from .setup_logging import setup_logging
from .sentry import init_sentry
from .metrics import get_metrics_response

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize Sentry for error tracking
init_sentry()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create FastAPI application with lifespan
app = FastAPI(
    title="R3MES Inference Service",
    description="Decentralized AI inference service with blockchain integration",
    version="1.0.0",
    lifespan=create_lifespan,
    docs_url="/docs" if os.getenv("R3MES_ENV", "development") != "production" else None,
    redoc_url="/redoc" if os.getenv("R3MES_ENV", "development") != "production" else None
)

# Add rate limiting error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure middleware stack (order matters!)
def setup_middleware():
    """Setup middleware stack in correct order."""
    
    # 1. Panic recovery (outermost)
    app.add_middleware(PanicRecoveryMiddleware)
    
    # 2. Error handling
    debug_mode = os.getenv("R3MES_ENV", "development") == "development"
    error_handler = create_error_handler_middleware(debug=debug_mode)
    app.add_middleware(error_handler)
    
    # 3. Debug middleware (development only)
    if debug_mode:
        app.add_middleware(DebugMiddleware)
    
    # 4. Cache middleware
    app.add_middleware(cache_middleware)
    
    # 5. CORS (innermost - closest to routes)
    setup_cors()

def setup_cors():
    """Setup CORS middleware with environment-specific configuration."""
    
    env = os.getenv("R3MES_ENV", "development")
    
    if env == "production":
        # Production CORS - restrictive
        allowed_origins = os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
        allowed_origins = [origin.strip() for origin in allowed_origins if origin.strip()]
        
        if not allowed_origins:
            logger.warning("No CORS origins configured for production")
            allowed_origins = []
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID"]
        )
        
        logger.info(f"Production CORS configured with origins: {allowed_origins}")
        
    else:
        # Development CORS - permissive
        development_origins = [
            "http://localhost:3000",
            "http://localhost:5173", 
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173"
        ]
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=development_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID"]
        )
        
        logger.info(f"Development CORS configured with origins: {development_origins}")

def include_routers():
    """Include all API routers."""
    
    # Core API routes
    app.include_router(chat_router, prefix="/api", tags=["chat"])
    app.include_router(users_router, prefix="/api", tags=["users"])
    
    # Feature routes
    app.include_router(config_router, prefix="/api", tags=["config"])
    app.include_router(leaderboard_router, prefix="/api", tags=["leaderboard"])
    app.include_router(analytics_router, prefix="/api", tags=["analytics"])
    app.include_router(faucet_router, prefix="/api", tags=["faucet"])
    app.include_router(serving_router, prefix="/api", tags=["serving"])
    app.include_router(proposer_router, prefix="/api", tags=["proposer"])
    app.include_router(role_router, prefix="/api", tags=["roles"])
    
    # System routes
    app.include_router(health_router, prefix="/api", tags=["health"])
    app.include_router(system_router, prefix="/api", tags=["system"])
    app.include_router(websocket_router, prefix="/ws", tags=["websocket"])
    
    # Debug routes (development only)
    if os.getenv("R3MES_ENV", "development") == "development":
        app.include_router(debug_router, prefix="/debug", tags=["debug"])

# Setup application
setup_middleware()
include_routers()

# Add metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return get_metrics_response()

# Add root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "service": "R3MES Inference Service",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs" if os.getenv("R3MES_ENV", "development") != "production" else None
    }

# Health check endpoint (simple)
@app.get("/ping")
async def ping():
    """Simple health check endpoint."""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))
    
    # Run server
    uvicorn.run(
        "main_refactored:app",
        host=host,
        port=port,
        workers=workers,
        reload=os.getenv("R3MES_ENV", "development") == "development",
        log_level="info"
    )
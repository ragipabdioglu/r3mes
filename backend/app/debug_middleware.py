"""
Debug Middleware for R3MES Backend

Provides request/response logging, database query logging, and cache hit/miss logging for debug mode.
"""

import time
import logging
from typing import Callable, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .debug_config import get_debug_config

logger = logging.getLogger(__name__)


class DebugMiddleware(BaseHTTPMiddleware):
    """Middleware for debug mode request/response logging"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.debug_config = get_debug_config()
        self.enabled = self.debug_config.enabled and self.debug_config.is_backend_enabled() and self.debug_config.logging
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log debug information"""
        if not self.enabled:
            return await call_next(request)
        
        # Generate request ID
        request_id = f"{int(time.time() * 1000000)}"  # Microsecond timestamp
        
        # Log request
        start_time = time.perf_counter()
        logger.trace(
            "Request received",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "client": request.client.host if request.client else None,
            }
        )
        
        # Add request ID to request state
        request.state.request_id = request_id
        request.state.start_time = start_time
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Log response
            logger.trace(
                "Response sent",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                }
            )
            
            # Add debug headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time-MS"] = f"{duration_ms:.2f}"
            
            return response
            
        except Exception as e:
            # Log error
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": duration_ms,
                    "error": str(e),
                },
                exc_info=True
            )
            raise


# Database query logging decorator
def log_database_query(query_name: str):
    """
    Decorator for logging database queries in debug mode.
    
    Usage:
        @log_database_query("get_user")
        async def get_user(user_id: int):
            ...
    """
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            debug_config = get_debug_config()
            if debug_config.enabled and debug_config.is_backend_enabled() and debug_config.logging:
                start = time.perf_counter()
                logger.trace(f"Database query started: {query_name}", extra={"query": query_name})
                try:
                    result = await func(*args, **kwargs)
                    duration_ms = (time.perf_counter() - start) * 1000
                    logger.trace(
                        f"Database query completed: {query_name}",
                        extra={
                            "query": query_name,
                            "duration_ms": duration_ms,
                            "success": True,
                        }
                    )
                    return result
                except Exception as e:
                    duration_ms = (time.perf_counter() - start) * 1000
                    logger.error(
                        f"Database query failed: {query_name}",
                        extra={
                            "query": query_name,
                            "duration_ms": duration_ms,
                            "error": str(e),
                        },
                        exc_info=True
                    )
                    raise
            else:
                return await func(*args, **kwargs)
        return wrapper
    return decorator


# Cache logging helpers
def log_cache_hit(key: str):
    """Log a cache hit"""
    debug_config = get_debug_config()
    if debug_config.enabled and debug_config.is_backend_enabled() and debug_config.logging:
        logger.trace("Cache hit", extra={"cache_key": key, "cache_result": "hit"})


def log_cache_miss(key: str):
    """Log a cache miss"""
    debug_config = get_debug_config()
    if debug_config.enabled and debug_config.is_backend_enabled() and debug_config.logging:
        logger.trace("Cache miss", extra={"cache_key": key, "cache_result": "miss"})


def log_cache_set(key: str, ttl: Optional[int] = None):
    """Log a cache set operation"""
    debug_config = get_debug_config()
    if debug_config.enabled and debug_config.is_backend_enabled() and debug_config.logging:
        logger.trace("Cache set", extra={"cache_key": key, "ttl": ttl})

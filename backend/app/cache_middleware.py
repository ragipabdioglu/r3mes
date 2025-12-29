"""
Cache Middleware for FastAPI

Provides caching decorators and middleware for API endpoints.
"""

from fastapi import Request, Response
from typing import Callable, Optional
import hashlib
import json
import logging
from functools import wraps

from .cache import get_cache_manager
from .cache_keys import CacheTTL

logger = logging.getLogger(__name__)


def cache_response(ttl: int = 3600, key_prefix: str = ""):
    """
    Decorator to cache API endpoint responses.
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache = get_cache_manager()
            
            # Generate cache key from function name and arguments
            cache_key = _generate_cache_key(func.__name__, key_prefix, args, kwargs)
            
            # Try to get from cache
            cached = await cache.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for {cache_key}")
                return cached
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            if isinstance(result, dict):
                await cache.set(cache_key, result, ttl)
                logger.debug(f"Cached result for {cache_key}")
            
            return result
        
        return wrapper
    return decorator


def _generate_cache_key(func_name: str, prefix: str, args: tuple, kwargs: dict) -> str:
    """Generate cache key from function name and arguments."""
    # Create a hash of the arguments
    key_data = {
        "func": func_name,
        "args": str(args),
        "kwargs": str(sorted(kwargs.items()))
    }
    key_string = json.dumps(key_data, sort_keys=True)
    key_hash = hashlib.md5(key_string.encode()).hexdigest()
    
    if prefix:
        return f"{prefix}:{func_name}:{key_hash}"
    return f"cache:{func_name}:{key_hash}"


async def cache_middleware(request: Request, call_next):
    """
    FastAPI middleware for response caching.
    
    Caches GET requests based on URL and query parameters.
    """
    # Only cache GET requests
    if request.method != "GET":
        response = await call_next(request)
        return response
    
    cache = get_cache_manager()
    
    # Generate cache key from request
    cache_key = _generate_request_cache_key(request)
    
    # Try to get from cache
    cached = await cache.get(cache_key)
    if cached:
        logger.debug(f"Cache hit for {cache_key}")
        return Response(
            content=json.dumps(cached),
            media_type="application/json",
            headers={"X-Cache": "HIT"}
        )
    
    # Process request
    response = await call_next(request)
    
    # Cache successful responses
    if response.status_code == 200:
        try:
            # Parse response body
            body = await response.body()
            if body:
                data = json.loads(body)
                # Determine TTL based on endpoint
                ttl = _get_ttl_for_endpoint(request.url.path)
                await cache.set(cache_key, data, ttl)
                logger.debug(f"Cached response for {cache_key}")
        except Exception as e:
            logger.error(f"Error caching response: {e}")
    
    # Add cache header
    response.headers["X-Cache"] = "MISS"
    return response


def _generate_request_cache_key(request: Request) -> str:
    """Generate cache key from request."""
    # Include path and query parameters
    key_data = {
        "path": request.url.path,
        "query": str(sorted(request.query_params.items()))
    }
    key_string = json.dumps(key_data, sort_keys=True)
    key_hash = hashlib.md5(key_string.encode()).hexdigest()
    return f"http:{request.url.path}:{key_hash}"


def _get_ttl_for_endpoint(path: str) -> int:
    """Get appropriate TTL for endpoint."""
    if "/network/stats" in path:
        return CacheTTL.NETWORK_STATS
    elif "/blocks" in path:
        return CacheTTL.BLOCKS
    elif "/miner/stats" in path:
        return CacheTTL.MINER_STATS
    elif "/user/info" in path:
        return CacheTTL.USER_INFO
    elif "/miner/earnings" in path or "/miner/hashrate" in path:
        return CacheTTL.MINER_HISTORY
    else:
        return 60  # Default 1 minute


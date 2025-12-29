"""
Centralized Error Handlers for R3MES Backend

Provides consistent error handling and logging across all modules.
"""

import logging
import traceback
from typing import Optional, Dict, Any, Type, Callable, TypeVar
from functools import wraps

from fastapi import HTTPException, status
from fastapi.responses import JSONResponse

from .exceptions import (
    R3MESException,
    AuthenticationError,
    AuthorizationError,
    InvalidAPIKeyError,
    DatabaseError,
    DatabaseConnectionError,
    BlockchainError,
    BlockchainConnectionError,
    ModelError,
    ModelNotFoundError,
    InferenceError,
    CreditError,
    InsufficientCreditsError,
    ValidationError,
    InvalidInputError,
    RateLimitError,
    CacheError,
    NetworkError,
    TimeoutError as R3MESTimeoutError,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


# Exception to HTTP status code mapping
EXCEPTION_STATUS_MAP: Dict[Type[R3MESException], int] = {
    # Authentication/Authorization - 401/403
    AuthenticationError: status.HTTP_401_UNAUTHORIZED,
    InvalidAPIKeyError: status.HTTP_401_UNAUTHORIZED,
    AuthorizationError: status.HTTP_403_FORBIDDEN,
    
    # Not Found - 404
    ModelNotFoundError: status.HTTP_404_NOT_FOUND,
    
    # Validation - 400
    ValidationError: status.HTTP_400_BAD_REQUEST,
    InvalidInputError: status.HTTP_400_BAD_REQUEST,
    
    # Payment Required - 402
    InsufficientCreditsError: status.HTTP_402_PAYMENT_REQUIRED,
    
    # Rate Limit - 429
    RateLimitError: status.HTTP_429_TOO_MANY_REQUESTS,
    
    # Service Unavailable - 503
    DatabaseConnectionError: status.HTTP_503_SERVICE_UNAVAILABLE,
    BlockchainConnectionError: status.HTTP_503_SERVICE_UNAVAILABLE,
    CacheError: status.HTTP_503_SERVICE_UNAVAILABLE,
    
    # Gateway Timeout - 504
    R3MESTimeoutError: status.HTTP_504_GATEWAY_TIMEOUT,
    
    # Internal Server Error - 500 (default for other R3MES exceptions)
    DatabaseError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    BlockchainError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    ModelError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    InferenceError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    CreditError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    NetworkError: status.HTTP_500_INTERNAL_SERVER_ERROR,
}


def get_status_code_for_exception(exc: Exception) -> int:
    """
    Get appropriate HTTP status code for an exception.
    
    Args:
        exc: The exception to map
        
    Returns:
        HTTP status code
    """
    # Check exact type first, then base classes
    for exc_type, status_code in EXCEPTION_STATUS_MAP.items():
        if isinstance(exc, exc_type):
            return status_code
    
    # Default to 500 for unknown R3MES exceptions
    if isinstance(exc, R3MESException):
        return status.HTTP_500_INTERNAL_SERVER_ERROR
    
    # Default to 500 for all other exceptions
    return status.HTTP_500_INTERNAL_SERVER_ERROR


def format_error_response(
    exc: Exception,
    include_traceback: bool = False,
) -> Dict[str, Any]:
    """
    Format exception into a consistent error response.
    
    Args:
        exc: The exception to format
        include_traceback: Whether to include traceback (dev only)
        
    Returns:
        Error response dictionary
    """
    response = {
        "error": True,
        "error_type": type(exc).__name__,
        "message": str(exc),
    }
    
    # Add error code if available
    if hasattr(exc, 'code'):
        response["code"] = exc.code
    
    # Add details if available
    if hasattr(exc, 'details'):
        response["details"] = exc.details
    
    # Add traceback in development
    if include_traceback:
        response["traceback"] = traceback.format_exc()
    
    return response


def log_exception(
    exc: Exception,
    context: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log an exception with consistent formatting.
    
    Args:
        exc: The exception to log
        context: Optional context string
        extra: Optional extra data to include
    """
    exc_type = type(exc).__name__
    exc_message = str(exc)
    
    # Build log message
    log_parts = [f"{exc_type}: {exc_message}"]
    if context:
        log_parts.insert(0, f"[{context}]")
    
    log_message = " ".join(log_parts)
    
    # Determine log level based on exception type
    if isinstance(exc, (ValidationError, InvalidInputError, RateLimitError)):
        # Client errors - warning level
        logger.warning(log_message, extra=extra)
    elif isinstance(exc, (AuthenticationError, AuthorizationError)):
        # Auth errors - warning level (could be attack attempts)
        logger.warning(log_message, extra=extra)
    elif isinstance(exc, (DatabaseConnectionError, BlockchainConnectionError, CacheError)):
        # Infrastructure errors - critical level
        logger.critical(log_message, exc_info=True, extra=extra)
    elif isinstance(exc, R3MESException):
        # Other R3MES errors - error level
        logger.error(log_message, exc_info=True, extra=extra)
    else:
        # Unknown errors - error level with full traceback
        logger.error(log_message, exc_info=True, extra=extra)


def handle_exception(
    exc: Exception,
    context: Optional[str] = None,
    reraise: bool = True,
    include_traceback: bool = False,
) -> Optional[JSONResponse]:
    """
    Handle an exception with logging and optional HTTP response.
    
    Args:
        exc: The exception to handle
        context: Optional context string for logging
        reraise: Whether to reraise the exception
        include_traceback: Whether to include traceback in response
        
    Returns:
        JSONResponse if not reraising, None otherwise
        
    Raises:
        HTTPException: If reraise is True
    """
    # Log the exception
    log_exception(exc, context)
    
    # Get status code
    status_code = get_status_code_for_exception(exc)
    
    # Format response
    error_response = format_error_response(exc, include_traceback)
    
    if reraise:
        raise HTTPException(
            status_code=status_code,
            detail=error_response,
        )
    
    return JSONResponse(
        status_code=status_code,
        content=error_response,
    )


def with_error_handling(
    context: Optional[str] = None,
    reraise: bool = True,
    default_return: Any = None,
):
    """
    Decorator for consistent error handling.
    
    Args:
        context: Context string for logging
        reraise: Whether to reraise exceptions
        default_return: Default return value if not reraising
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                # Don't wrap HTTPExceptions
                raise
            except Exception as e:
                handle_exception(e, context or func.__name__, reraise)
                return default_return
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except HTTPException:
                # Don't wrap HTTPExceptions
                raise
            except Exception as e:
                handle_exception(e, context or func.__name__, reraise)
                return default_return
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


class ErrorContext:
    """
    Context manager for error handling with automatic logging.
    
    Example:
        with ErrorContext("processing user request", user_id=123):
            # code that might raise exceptions
            pass
    """
    
    def __init__(
        self,
        context: str,
        reraise: bool = True,
        **extra,
    ):
        self.context = context
        self.reraise = reraise
        self.extra = extra
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            log_exception(exc_val, self.context, self.extra)
            if not self.reraise:
                return True  # Suppress exception
        return False

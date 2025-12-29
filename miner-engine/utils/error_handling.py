"""
Error handling utilities with retry mechanisms and specific exception types.
"""

import time
import logging
from typing import Callable, TypeVar, Optional, List, Type
from functools import wraps

T = TypeVar("T")
logger = logging.getLogger(__name__)


class RetryableError(Exception):
    """Base class for errors that can be retried."""
    pass


class NetworkError(RetryableError):
    """Network-related errors (connection, timeout, etc.)."""
    pass


class AuthenticationError(Exception):
    """Authentication/authorization errors (non-retryable)."""
    pass


class ResourceError(Exception):
    """Resource errors (memory, disk, etc.)."""
    pass


def exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: Optional[List[Type[Exception]]] = None,
):
    """
    Decorator for exponential backoff retry mechanism.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Multiplier for delay after each retry
        retryable_exceptions: List of exception types that should trigger retry
    
    Returns:
        Decorated function with retry logic
    """
    if retryable_exceptions is None:
        retryable_exceptions = [NetworkError, ConnectionError, TimeoutError]
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Check if exception is retryable
                    is_retryable = any(
                        isinstance(e, exc_type) for exc_type in retryable_exceptions
                    )
                    
                    if not is_retryable or attempt >= max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {attempt + 1} attempts: {e}",
                            exc_info=True
                        )
                        raise
                    
                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    
                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected error in retry mechanism")
        
        return wrapper
    return decorator


def handle_specific_errors(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to handle specific error types and convert them to appropriate exceptions.
    
    Args:
        func: Function to wrap
    
    Returns:
        Wrapped function with error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        try:
            return func(*args, **kwargs)
        except ConnectionError as e:
            raise NetworkError(f"Network connection failed: {e}") from e
        except TimeoutError as e:
            raise NetworkError(f"Operation timed out: {e}") from e
        except PermissionError as e:
            raise AuthenticationError(f"Permission denied: {e}") from e
        except MemoryError as e:
            raise ResourceError(f"Insufficient memory: {e}") from e
        except OSError as e:
            if e.errno == 28:  # No space left on device
                raise ResourceError(f"Disk space exhausted: {e}") from e
            raise
        except Exception as e:
            # Re-raise unknown exceptions as-is
            raise
    
    return wrapper


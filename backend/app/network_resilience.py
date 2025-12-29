"""
Network Resilience Utilities

Provides retry mechanisms, circuit breakers, and timeout configurations
for robust network communication.
"""

import asyncio
import logging
import os
import time
from enum import Enum
from typing import Callable, Optional, TypeVar, Any, Dict
from functools import wraps
from datetime import datetime, timedelta
import requests
from requests.exceptions import RequestException, ConnectionError, Timeout

from .exceptions import (
    NetworkError,
    ConnectionError as R3MESConnectionError,
    TimeoutError as R3MESTimeoutError,
    BlockchainConnectionError,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests immediately
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    
    Prevents cascading failures by stopping requests to a failing service
    and periodically testing if it has recovered.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
        name: str = "circuit_breaker"
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type that counts as failure
            name: Name for logging
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED
        self._lock = asyncio.Lock()
        import threading
        self._sync_lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute sync function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: If function raises an exception
        """
        with self._sync_lock:
            # Check if circuit should transition to half-open
            if self.state == CircuitState.OPEN:
                if self.last_failure_time:
                    time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
                    if time_since_failure >= self.recovery_timeout:
                        logger.info(f"Circuit breaker {self.name}: Transitioning to HALF_OPEN")
                        self.state = CircuitState.HALF_OPEN
                        self.failure_count = 0
                    else:
                        raise CircuitBreakerOpenError(
                            f"Circuit breaker {self.name} is OPEN. "
                            f"Retry after {self.recovery_timeout - time_since_failure:.1f} seconds"
                        )
                else:
                    raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
        
        # Try to execute function
        try:
            result = func(*args, **kwargs)
            
            # Success - reset failure count
            with self._sync_lock:
                if self.state == CircuitState.HALF_OPEN:
                    logger.info(f"Circuit breaker {self.name}: Recovered, transitioning to CLOSED")
                    self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.last_failure_time = None
            
            return result
            
        except self.expected_exception as e:
            # Failure - increment count
            with self._sync_lock:
                self.failure_count += 1
                self.last_failure_time = datetime.now()
                
                if self.failure_count >= self.failure_threshold:
                    if self.state != CircuitState.OPEN:
                        logger.warning(
                            f"Circuit breaker {self.name}: Opening circuit "
                            f"after {self.failure_count} failures"
                        )
                        self.state = CircuitState.OPEN
                elif self.state == CircuitState.HALF_OPEN:
                    logger.warning(
                        f"Circuit breaker {self.name}: Still failing, reopening circuit"
                    )
                    self.state = CircuitState.OPEN
            
            raise
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: If function raises an exception
        """
        async with self._lock:
            # Check if circuit should transition to half-open
            if self.state == CircuitState.OPEN:
                if self.last_failure_time:
                    time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
                    if time_since_failure >= self.recovery_timeout:
                        logger.info(f"Circuit breaker {self.name}: Transitioning to HALF_OPEN")
                        self.state = CircuitState.HALF_OPEN
                        self.failure_count = 0
                    else:
                        raise CircuitBreakerOpenError(
                            f"Circuit breaker {self.name} is OPEN. "
                            f"Retry after {self.recovery_timeout - time_since_failure:.1f} seconds"
                        )
                else:
                    raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
        
        # Try to execute function
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Success - reset failure count
            async with self._lock:
                if self.state == CircuitState.HALF_OPEN:
                    logger.info(f"Circuit breaker {self.name}: Recovered, transitioning to CLOSED")
                    self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.last_failure_time = None
            
            return result
            
        except self.expected_exception as e:
            # Failure - increment count
            async with self._lock:
                self.failure_count += 1
                self.last_failure_time = datetime.now()
                
                if self.failure_count >= self.failure_threshold:
                    if self.state != CircuitState.OPEN:
                        logger.warning(
                            f"Circuit breaker {self.name}: Opening circuit "
                            f"after {self.failure_count} failures"
                        )
                        self.state = CircuitState.OPEN
                elif self.state == CircuitState.HALF_OPEN:
                    logger.warning(
                        f"Circuit breaker {self.name}: Still failing, reopening circuit"
                    )
                    self.state = CircuitState.OPEN
            
            raise


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


class RetryConfig:
    """Configuration for retry mechanism"""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        retryable_exceptions: tuple = (RequestException,),
        retryable_status_codes: tuple = (500, 502, 503, 504),
    ):
        """
        Initialize retry configuration.
        
        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay before first retry (seconds)
            max_delay: Maximum delay between retries (seconds)
            backoff_factor: Multiplier for exponential backoff
            retryable_exceptions: Exception types that should trigger retry
            retryable_status_codes: HTTP status codes that should trigger retry
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.retryable_exceptions = retryable_exceptions
        self.retryable_status_codes = retryable_status_codes


def retry_with_backoff(
    func: Callable,
    config: Optional[RetryConfig] = None,
    operation_name: str = "operation"
) -> Callable:
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        func: Function to retry
        config: Retry configuration (uses default if None)
        operation_name: Name for logging
        
    Returns:
        Wrapped function with retry logic
    """
    if config is None:
        config = RetryConfig()
    
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        last_exception = None
        
        for attempt in range(config.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except config.retryable_exceptions as e:
                last_exception = e
                
                # Check if it's a retryable HTTP error
                if hasattr(e, 'response') and e.response is not None:
                    status_code = e.response.status_code
                    if status_code not in config.retryable_status_codes:
                        # Not retryable, raise immediately
                        logger.error(f"{operation_name} failed (non-retryable): {e}")
                        raise
                
                # Check if we should retry
                if attempt < config.max_retries:
                    delay = min(
                        config.initial_delay * (config.backoff_factor ** attempt),
                        config.max_delay
                    )
                    logger.warning(
                        f"{operation_name} failed (attempt {attempt + 1}/{config.max_retries + 1}), "
                        f"retrying in {delay:.2f}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"{operation_name} failed after {config.max_retries + 1} attempts: {e}")
                    raise
                    
            except Exception as e:
                # Non-retryable exception, raise immediately
                logger.error(f"{operation_name} failed (non-retryable): {e}")
                raise
        
        # Should never reach here, but just in case
        if last_exception:
            raise last_exception
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        last_exception = None
        
        for attempt in range(config.max_retries + 1):
            try:
                return func(*args, **kwargs)
                    
            except config.retryable_exceptions as e:
                last_exception = e
                
                # Check if it's a retryable HTTP error
                if hasattr(e, 'response') and e.response is not None:
                    status_code = e.response.status_code
                    if status_code not in config.retryable_status_codes:
                        # Not retryable, raise immediately
                        logger.error(f"{operation_name} failed (non-retryable): {e}")
                        raise
                
                # Check if we should retry
                if attempt < config.max_retries:
                    delay = min(
                        config.initial_delay * (config.backoff_factor ** attempt),
                        config.max_delay
                    )
                    logger.warning(
                        f"{operation_name} failed (attempt {attempt + 1}/{config.max_retries + 1}), "
                        f"retrying in {delay:.2f}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"{operation_name} failed after {config.max_retries + 1} attempts: {e}")
                    raise
                    
            except Exception as e:
                # Non-retryable exception, raise immediately
                logger.error(f"{operation_name} failed (non-retryable): {e}")
                raise
        
        # Should never reach here, but just in case
        if last_exception:
            raise last_exception
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def with_timeout(timeout: float, operation_name: str = "operation"):
    """
    Decorator for adding timeout to functions.
    
    Args:
        timeout: Timeout in seconds
        operation_name: Name for logging
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.error(f"{operation_name} timed out after {timeout}s")
                raise R3MESTimeoutError(f"{operation_name} timed out after {timeout}s")
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we can't easily add timeout without threading
            # Just execute normally and log a warning
            logger.warning(f"Timeout not supported for sync function {operation_name}")
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Global circuit breakers for different services
_blockchain_circuit_breaker: Optional[CircuitBreaker] = None
_ipfs_circuit_breaker: Optional[CircuitBreaker] = None


def get_blockchain_circuit_breaker() -> CircuitBreaker:
    """Get or create blockchain circuit breaker."""
    global _blockchain_circuit_breaker
    if _blockchain_circuit_breaker is None:
        _blockchain_circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
            expected_exception=(RequestException, BlockchainConnectionError),
            name="blockchain"
        )
    return _blockchain_circuit_breaker


def get_ipfs_circuit_breaker() -> CircuitBreaker:
    """Get or create IPFS circuit breaker."""
    global _ipfs_circuit_breaker
    if _ipfs_circuit_breaker is None:
        _ipfs_circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
            expected_exception=(RequestException, ConnectionError),
            name="ipfs"
        )
    return _ipfs_circuit_breaker


# Default retry configurations (can be overridden via environment variables)
DEFAULT_RETRY_CONFIG = RetryConfig(
    max_retries=int(os.getenv("BACKEND_RETRY_MAX_RETRIES", "3")),
    initial_delay=float(os.getenv("BACKEND_RETRY_INITIAL_DELAY", "1.0")),
    max_delay=float(os.getenv("BACKEND_RETRY_MAX_DELAY", "60.0")),
    backoff_factor=float(os.getenv("BACKEND_RETRY_BACKOFF_FACTOR", "2.0")),
)

BLOCKCHAIN_RETRY_CONFIG = RetryConfig(
    max_retries=int(os.getenv("BACKEND_BLOCKCHAIN_RETRY_MAX_RETRIES", "3")),
    initial_delay=float(os.getenv("BACKEND_RETRY_INITIAL_DELAY", "1.0")),
    max_delay=float(os.getenv("BACKEND_BLOCKCHAIN_RETRY_MAX_DELAY", "30.0")),
    backoff_factor=float(os.getenv("BACKEND_RETRY_BACKOFF_FACTOR", "2.0")),
)

IPFS_RETRY_CONFIG = RetryConfig(
    max_retries=int(os.getenv("BACKEND_IPFS_RETRY_MAX_RETRIES", "5")),
    initial_delay=float(os.getenv("BACKEND_IPFS_RETRY_INITIAL_DELAY", "0.5")),
    max_delay=float(os.getenv("BACKEND_IPFS_RETRY_MAX_DELAY", "10.0")),
    backoff_factor=float(os.getenv("BACKEND_IPFS_RETRY_BACKOFF_FACTOR", "1.5")),
)


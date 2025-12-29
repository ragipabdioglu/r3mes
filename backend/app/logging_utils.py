"""
Logging Utilities for R3MES Backend

Provides consistent logging helpers and structured logging support.
"""

import logging
import time
import functools
from typing import Optional, Dict, Any, Callable, TypeVar
from contextlib import contextmanager

T = TypeVar('T')


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class StructuredLogger:
    """
    Wrapper for structured logging with consistent formatting.
    
    Ensures all log messages use the same format and include
    relevant context information.
    """
    
    def __init__(self, name: str):
        self._logger = logging.getLogger(name)
        self._context: Dict[str, Any] = {}
    
    def with_context(self, **kwargs) -> "StructuredLogger":
        """
        Create a new logger with additional context.
        
        Args:
            **kwargs: Context key-value pairs
            
        Returns:
            New StructuredLogger with merged context
        """
        new_logger = StructuredLogger(self._logger.name)
        new_logger._context = {**self._context, **kwargs}
        return new_logger
    
    def _format_message(self, message: str, **kwargs) -> str:
        """Format message with context."""
        context = {**self._context, **kwargs}
        if context:
            context_str = " ".join(f"{k}={v}" for k, v in context.items())
            return f"{message} [{context_str}]"
        return message
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._logger.debug(self._format_message(message, **kwargs))
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._logger.info(self._format_message(message, **kwargs))
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._logger.warning(self._format_message(message, **kwargs))
    
    def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log error message."""
        self._logger.error(self._format_message(message, **kwargs), exc_info=exc_info)
    
    def critical(self, message: str, exc_info: bool = True, **kwargs) -> None:
        """Log critical message."""
        self._logger.critical(self._format_message(message, **kwargs), exc_info=exc_info)
    
    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback."""
        self._logger.exception(self._format_message(message, **kwargs))


def get_structured_logger(name: str) -> StructuredLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name)


@contextmanager
def log_duration(
    logger: logging.Logger,
    operation: str,
    level: int = logging.DEBUG,
    threshold_ms: Optional[float] = None,
):
    """
    Context manager to log operation duration.
    
    Args:
        logger: Logger instance
        operation: Operation name
        level: Log level (default: DEBUG)
        threshold_ms: Only log if duration exceeds this threshold
        
    Example:
        with log_duration(logger, "database query"):
            result = await db.query(...)
    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        if threshold_ms is None or duration_ms >= threshold_ms:
            logger.log(level, f"{operation} completed in {duration_ms:.2f}ms")


def log_function_call(
    logger: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
    log_args: bool = False,
    log_result: bool = False,
):
    """
    Decorator to log function calls.
    
    Args:
        logger: Logger instance (uses function's module logger if None)
        level: Log level
        log_args: Whether to log function arguments
        log_result: Whether to log function result
        
    Example:
        @log_function_call(log_args=True)
        async def process_request(user_id: int):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        nonlocal logger
        if logger is None:
            logger = logging.getLogger(func.__module__)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            func_name = func.__name__
            
            # Log entry
            if log_args:
                logger.log(level, f"Calling {func_name} with args={args}, kwargs={kwargs}")
            else:
                logger.log(level, f"Calling {func_name}")
            
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                # Log exit
                if log_result:
                    logger.log(level, f"{func_name} returned {result} in {duration_ms:.2f}ms")
                else:
                    logger.log(level, f"{func_name} completed in {duration_ms:.2f}ms")
                
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.error(f"{func_name} failed after {duration_ms:.2f}ms: {e}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            func_name = func.__name__
            
            # Log entry
            if log_args:
                logger.log(level, f"Calling {func_name} with args={args}, kwargs={kwargs}")
            else:
                logger.log(level, f"Calling {func_name}")
            
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                # Log exit
                if log_result:
                    logger.log(level, f"{func_name} returned {result} in {duration_ms:.2f}ms")
                else:
                    logger.log(level, f"{func_name} completed in {duration_ms:.2f}ms")
                
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.error(f"{func_name} failed after {duration_ms:.2f}ms: {e}")
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


class RequestLogger:
    """
    Logger for HTTP request/response logging.
    
    Provides consistent logging for API requests with timing
    and status information.
    """
    
    def __init__(self, logger: logging.Logger):
        self._logger = logger
    
    def log_request(
        self,
        method: str,
        path: str,
        client_ip: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> float:
        """
        Log incoming request.
        
        Args:
            method: HTTP method
            path: Request path
            client_ip: Client IP address
            user_id: User ID if authenticated
            
        Returns:
            Start timestamp for duration calculation
        """
        parts = [f"{method} {path}"]
        if client_ip:
            parts.append(f"client={client_ip}")
        if user_id:
            parts.append(f"user={user_id}")
        
        self._logger.info(" ".join(parts))
        return time.perf_counter()
    
    def log_response(
        self,
        method: str,
        path: str,
        status_code: int,
        start_time: float,
    ) -> None:
        """
        Log outgoing response.
        
        Args:
            method: HTTP method
            path: Request path
            status_code: Response status code
            start_time: Request start timestamp
        """
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Determine log level based on status code
        if status_code >= 500:
            level = logging.ERROR
        elif status_code >= 400:
            level = logging.WARNING
        else:
            level = logging.INFO
        
        self._logger.log(
            level,
            f"{method} {path} -> {status_code} ({duration_ms:.2f}ms)"
        )

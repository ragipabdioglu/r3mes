"""
Base Service Class

Provides common functionality for all services including error handling,
logging, and validation patterns.
"""

import logging
from abc import ABC
from typing import Optional, Dict, Any

from ..exceptions import R3MESException, InvalidInputError

logger = logging.getLogger(__name__)


class BaseService(ABC):
    """
    Base service with common business logic patterns.
    
    Provides standardized error handling, logging, and validation patterns
    for all service implementations.
    """
    
    def __init__(self, database, cache_manager=None):
        """
        Initialize base service.
        
        Args:
            database: Database instance
            cache_manager: Cache manager instance (optional)
        """
        self.db = database
        self.cache = cache_manager
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def _log_operation(self, operation: str, details: Dict[str, Any] = None):
        """
        Log service operation with structured logging.
        
        Args:
            operation: Operation name
            details: Additional details to log
        """
        log_data = {
            "service": self.__class__.__name__,
            "operation": operation,
            **(details or {})
        }
        self.logger.info(f"Service operation: {operation}", extra=log_data)
    
    async def _handle_error(self, error: Exception, operation: str, context: Dict[str, Any] = None):
        """
        Handle service errors with proper logging and re-raising.
        
        Args:
            error: Original exception
            operation: Operation that failed
            context: Additional context for debugging
        """
        error_data = {
            "service": self.__class__.__name__,
            "operation": operation,
            "error_type": type(error).__name__,
            "error_message": str(error),
            **(context or {})
        }
        
        self.logger.error(f"Service error in {operation}: {error}", extra=error_data, exc_info=True)
        
        # Re-raise R3MES exceptions as-is
        if isinstance(error, R3MESException):
            raise
        
        # Wrap other exceptions
        raise R3MESException(
            message=f"Service operation failed: {operation}",
            details={"original_error": str(error), "context": context}
        ) from error
    
    async def _validate_required_params(self, params: Dict[str, Any], required_fields: list):
        """
        Validate that required parameters are present.
        
        Args:
            params: Parameters to validate
            required_fields: List of required field names
            
        Raises:
            InvalidInputError: If required fields are missing
        """
        missing_fields = []
        for field in required_fields:
            if field not in params or params[field] is None:
                missing_fields.append(field)
        
        if missing_fields:
            raise InvalidInputError(
                message=f"Missing required fields: {', '.join(missing_fields)}",
                field="required_fields",
                value=missing_fields
            )
    
    async def _cache_get(self, key: str) -> Optional[Any]:
        """
        Get value from cache if cache manager is available.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if self.cache is None:
            return None
        
        try:
            return await self.cache.get(key)
        except Exception as e:
            self.logger.warning(f"Cache get failed for key {key}: {e}")
            return None
    
    async def _cache_set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """
        Set value in cache if cache manager is available.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            True if cached successfully, False otherwise
        """
        if self.cache is None:
            return False
        
        try:
            await self.cache.set(key, value, ttl=ttl)
            return True
        except Exception as e:
            self.logger.warning(f"Cache set failed for key {key}: {e}")
            return False
    
    async def _cache_delete(self, key: str) -> bool:
        """
        Delete value from cache if cache manager is available.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted successfully, False otherwise
        """
        if self.cache is None:
            return False
        
        try:
            await self.cache.delete(key)
            return True
        except Exception as e:
            self.logger.warning(f"Cache delete failed for key {key}: {e}")
            return False
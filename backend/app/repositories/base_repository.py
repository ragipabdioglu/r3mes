"""
Base Repository Class

Provides common functionality for all repositories including error handling,
logging, and validation patterns.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Callable
import logging
from datetime import datetime

from ..exceptions import DatabaseError, InvalidInputError, R3MESException

logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_PAGE_SIZE = int(os.getenv("DEFAULT_PAGE_SIZE", "50"))
MAX_PAGE_SIZE = int(os.getenv("MAX_PAGE_SIZE", "1000"))


class BaseRepository(ABC):
    """
    Base repository with common CRUD operations and error handling.
    
    Provides standardized error handling, logging, and validation patterns
    for all repository implementations.
    """
    
    def __init__(self, database):
        """
        Initialize base repository.
        
        Args:
            database: Database connection instance
        """
        self.database = database
        self.entity_name = self.__class__.__name__.replace('Repository', '')
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def _execute_query(
        self, 
        operation_name: str,
        query_func: Callable,
        *args, 
        **kwargs
    ) -> Any:
        """
        Execute database query with standardized error handling.
        
        Args:
            operation_name: Name of the operation for logging
            query_func: Database function to execute
            *args: Arguments for the query function
            **kwargs: Keyword arguments for the query function
            
        Returns:
            Query result
            
        Raises:
            DatabaseError: If database operation fails
        """
        start_time = datetime.now()
        
        try:
            self.logger.debug(f"Executing {operation_name} for {self.entity_name}")
            result = await query_func(*args, **kwargs)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.debug(
                f"Completed {operation_name} for {self.entity_name} "
                f"in {execution_time:.3f}s"
            )
            
            return result
            
        except R3MESException:
            # Re-raise R3MES exceptions without wrapping
            raise
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(
                f"Failed {operation_name} for {self.entity_name} "
                f"after {execution_time:.3f}s: {e}",
                exc_info=True
            )
            raise DatabaseError(
                message=f"Failed to execute {operation_name} for {self.entity_name}",
                details={
                    "operation": operation_name,
                    "entity": self.entity_name,
                    "execution_time": execution_time,
                    "error": str(e)
                },
                cause=e
            ) from e
    
    async def _validate_input(
        self, 
        value: Any, 
        field_name: str, 
        validator_func: Callable[[Any], Any],
        required: bool = True
    ) -> Any:
        """
        Validate input with standardized error handling.
        
        Args:
            value: Value to validate
            field_name: Name of the field being validated
            validator_func: Validation function
            required: Whether the field is required
            
        Returns:
            Validated value
            
        Raises:
            InvalidInputError: If validation fails
        """
        if value is None:
            if required:
                raise InvalidInputError(
                    message=f"{field_name} is required",
                    field=field_name,
                    value=value
                )
            return None
        
        try:
            validated_value = validator_func(value)
            self.logger.debug(f"Validated {field_name} for {self.entity_name}")
            return validated_value
            
        except R3MESException:
            # Re-raise R3MES exceptions without wrapping
            raise
        except Exception as e:
            self.logger.warning(
                f"Validation failed for {field_name} in {self.entity_name}: {e}"
            )
            raise InvalidInputError(
                message=f"Invalid {field_name}: {str(e)}",
                field=field_name,
                value=value,
                details={"validation_error": str(e)},
                cause=e
            ) from e
    
    async def _validate_pagination(
        self, 
        limit: Optional[int] = None, 
        offset: Optional[int] = None,
        max_limit: int = MAX_PAGE_SIZE
    ) -> tuple[int, int]:
        """
        Validate pagination parameters.
        
        Args:
            limit: Number of items to return
            offset: Number of items to skip
            max_limit: Maximum allowed limit
            
        Returns:
            Tuple of (validated_limit, validated_offset)
            
        Raises:
            InvalidInputError: If pagination parameters are invalid
        """
        # Validate limit
        if limit is None:
            limit = DEFAULT_PAGE_SIZE  # Default limit
        elif not isinstance(limit, int) or limit < 1:
            raise InvalidInputError(
                message="Limit must be a positive integer",
                field="limit",
                value=limit
            )
        elif limit > max_limit:
            raise InvalidInputError(
                message=f"Limit cannot exceed {max_limit}",
                field="limit",
                value=limit
            )
        
        # Validate offset
        if offset is None:
            offset = 0  # Default offset
        elif not isinstance(offset, int) or offset < 0:
            raise InvalidInputError(
                message="Offset must be a non-negative integer",
                field="offset",
                value=offset
            )
        
        return limit, offset
    
    async def _check_exists(
        self, 
        entity_id: Any, 
        entity_name: Optional[str] = None
    ) -> bool:
        """
        Check if entity exists (to be implemented by subclasses).
        
        Args:
            entity_id: ID of the entity to check
            entity_name: Name of the entity (defaults to self.entity_name)
            
        Returns:
            True if entity exists, False otherwise
        """
        # This is a placeholder - subclasses should implement this
        return True
    
    def _log_operation(
        self, 
        operation: str, 
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Log repository operation.
        
        Args:
            operation: Operation name
            details: Additional details to log
        """
        log_data = {
            "entity": self.entity_name,
            "operation": operation
        }
        if details:
            log_data.update(details)
        
        self.logger.info(f"{self.entity_name} {operation}", extra=log_data)
    
    def _log_error(
        self, 
        operation: str, 
        error: Exception, 
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Log repository error.
        
        Args:
            operation: Operation name
            error: Exception that occurred
            details: Additional details to log
        """
        log_data = {
            "entity": self.entity_name,
            "operation": operation,
            "error": str(error),
            "error_type": type(error).__name__
        }
        if details:
            log_data.update(details)
        
        self.logger.error(
            f"{self.entity_name} {operation} failed: {error}",
            extra=log_data,
            exc_info=True
        )
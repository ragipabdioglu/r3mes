"""
Error Handler Middleware

Provides centralized error handling for all API endpoints with consistent
error responses and proper logging.
"""

import logging
import traceback
import uuid
from datetime import datetime
from typing import Dict, Any

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..exceptions import (
    R3MESException,
    InvalidInputError,
    ValidationError,
    DatabaseError,
    AuthenticationError,
    ResourceNotFoundError,
    InsufficientCreditsError,
    NetworkError,
    TimeoutError,
    ErrorCode
)

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle all exceptions consistently across the API.
    
    Converts exceptions to appropriate HTTP responses with standardized format.
    """
    
    def __init__(self, app, debug: bool = False):
        """
        Initialize error handler middleware.
        
        Args:
            app: FastAPI application
            debug: Whether to include debug information in responses
        """
        super().__init__(app)
        self.debug = debug
    
    async def dispatch(self, request: Request, call_next):
        """
        Process request and handle any exceptions.
        
        Args:
            request: HTTP request
            call_next: Next middleware/endpoint in chain
            
        Returns:
            HTTP response
        """
        # Generate request ID for tracking
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        try:
            response = await call_next(request)
            return response
            
        except HTTPException:
            # Let FastAPI handle HTTPExceptions normally
            raise
            
        except R3MESException as e:
            return await self._handle_r3mes_exception(request, e, request_id)
            
        except Exception as e:
            return await self._handle_unexpected_exception(request, e, request_id)
    
    async def _handle_r3mes_exception(
        self, 
        request: Request, 
        exception: R3MESException, 
        request_id: str
    ) -> JSONResponse:
        """
        Handle R3MES-specific exceptions.
        
        Args:
            request: HTTP request
            exception: R3MES exception
            request_id: Request ID for tracking
            
        Returns:
            JSON error response
        """
        # Determine HTTP status code based on exception type
        status_code = self._get_status_code_for_exception(exception)
        
        # Create error response
        error_response = {
            "error": True,
            "error_code": exception.error_code.value,
            "message": exception.user_message,
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "path": str(request.url.path)
        }
        
        # Add details if available and not sensitive
        if exception.details and not self._contains_sensitive_data(exception.details):
            error_response["details"] = exception.details
        
        # Add debug information if enabled
        if self.debug:
            error_response["debug"] = {
                "exception_type": type(exception).__name__,
                "internal_message": exception.message,
                "cause": str(exception.cause) if exception.cause else None
            }
        
        # Log the error
        self._log_exception(request, exception, request_id, status_code)
        
        return JSONResponse(
            status_code=status_code,
            content=error_response
        )
    
    async def _handle_unexpected_exception(
        self, 
        request: Request, 
        exception: Exception, 
        request_id: str
    ) -> JSONResponse:
        """
        Handle unexpected exceptions.
        
        Args:
            request: HTTP request
            exception: Unexpected exception
            request_id: Request ID for tracking
            
        Returns:
            JSON error response
        """
        # Create generic error response
        error_response = {
            "error": True,
            "error_code": ErrorCode.UNKNOWN_ERROR.value,
            "message": "Internal server error",
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "path": str(request.url.path)
        }
        
        # Add debug information if enabled
        if self.debug:
            error_response["debug"] = {
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
                "traceback": traceback.format_exc()
            }
        
        # Log the error
        logger.critical(
            f"Unexpected exception in {request.method} {request.url.path}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
                "user_agent": request.headers.get("user-agent"),
                "client_ip": self._get_client_ip(request)
            },
            exc_info=True
        )
        
        return JSONResponse(
            status_code=500,
            content=error_response
        )
    
    def _get_status_code_for_exception(self, exception: R3MESException) -> int:
        """
        Get appropriate HTTP status code for R3MES exception.
        
        Args:
            exception: R3MES exception
            
        Returns:
            HTTP status code
        """
        status_code_map = {
            InvalidInputError: 400,
            ValidationError: 400,
            AuthenticationError: 401,
            InsufficientCreditsError: 402,  # Payment Required
            ResourceNotFoundError: 404,
            TimeoutError: 408,
            DatabaseError: 500,
            NetworkError: 502,
        }
        
        return status_code_map.get(type(exception), 500)
    
    def _contains_sensitive_data(self, details: Dict[str, Any]) -> bool:
        """
        Check if details contain sensitive data that shouldn't be exposed.
        
        Args:
            details: Exception details dictionary
            
        Returns:
            True if contains sensitive data
        """
        sensitive_keys = {
            'password', 'api_key', 'token', 'secret', 'private_key',
            'wallet_private_key', 'mnemonic', 'seed', 'hash'
        }
        
        def check_dict(d: Dict[str, Any]) -> bool:
            for key, value in d.items():
                if isinstance(key, str) and any(sensitive in key.lower() for sensitive in sensitive_keys):
                    return True
                if isinstance(value, dict) and check_dict(value):
                    return True
                if isinstance(value, str) and len(value) > 50 and any(char in value for char in ['=', '+', '/']):
                    # Might be base64 encoded sensitive data
                    return True
            return False
        
        return check_dict(details)
    
    def _log_exception(
        self, 
        request: Request, 
        exception: R3MESException, 
        request_id: str,
        status_code: int
    ):
        """
        Log exception with appropriate level and context.
        
        Args:
            request: HTTP request
            exception: R3MES exception
            request_id: Request ID for tracking
            status_code: HTTP status code
        """
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": status_code,
            "error_code": exception.error_code.value,
            "exception_type": type(exception).__name__,
            "user_agent": request.headers.get("user-agent"),
            "client_ip": self._get_client_ip(request)
        }
        
        # Add wallet address if available
        if hasattr(request.state, 'wallet_address'):
            log_data["wallet_address"] = request.state.wallet_address
        
        # Determine log level based on exception type and status code
        if status_code >= 500:
            log_level = logging.ERROR
        elif status_code >= 400:
            log_level = logging.WARNING
        else:
            log_level = logging.INFO
        
        logger.log(
            log_level,
            f"{exception.error_code.value}: {exception.user_message}",
            extra=log_data,
            exc_info=status_code >= 500
        )
    
    def _get_client_ip(self, request: Request) -> str:
        """
        Get client IP address from request.
        
        Args:
            request: HTTP request
            
        Returns:
            Client IP address
        """
        # Check for forwarded headers (behind proxy)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        if request.client:
            return request.client.host
        
        return "unknown"


def create_error_handler_middleware(debug: bool = False):
    """
    Create error handler middleware instance.
    
    Args:
        debug: Whether to include debug information in responses
        
    Returns:
        Error handler middleware class
    """
    class ConfiguredErrorHandlerMiddleware(ErrorHandlerMiddleware):
        def __init__(self, app):
            super().__init__(app, debug=debug)
    
    return ConfiguredErrorHandlerMiddleware
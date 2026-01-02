#!/usr/bin/env python3
"""
Unit tests for middleware components.

Tests error handling middleware, authentication, and request processing.
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.middleware.error_handler import (
    ErrorHandlerMiddleware,
    generate_request_id,
    filter_sensitive_data,
)
from app.exceptions import (
    R3MESException,
    InvalidInputError,
    AuthenticationError,
    DatabaseError,
    ErrorCode,
)


class TestErrorHandlerMiddleware:
    """Test cases for ErrorHandlerMiddleware."""
    
    @pytest.fixture
    def mock_app(self):
        """Mock FastAPI application."""
        app = Mock()
        return app
    
    @pytest.fixture
    def middleware(self, mock_app):
        """Create ErrorHandlerMiddleware instance."""
        return ErrorHandlerMiddleware(mock_app)
    
    @pytest.fixture
    def mock_request(self):
        """Mock HTTP request."""
        request = Mock(spec=Request)
        request.method = "GET"
        request.url = Mock()
        request.url.path = "/api/test"
        request.headers = {"user-agent": "test-client"}
        request.client = Mock()
        request.client.host = "127.0.0.1"
        return request
    
    def test_generate_request_id(self):
        """Test request ID generation."""
        request_id = generate_request_id()
        
        assert isinstance(request_id, str)
        assert len(request_id) == 8
        assert request_id.isalnum()
    
    def test_filter_sensitive_data_basic(self):
        """Test basic sensitive data filtering."""
        data = {
            "username": "test_user",
            "password": "secret123",
            "api_key": "r3mes_secret_key",
            "token": "jwt_token_here"
        }
        
        filtered = filter_sensitive_data(data)
        
        assert filtered["username"] == "test_user"
        assert filtered["password"] == "***FILTERED***"
        assert filtered["api_key"] == "***FILTERED***"
        assert filtered["token"] == "***FILTERED***"
    
    def test_filter_sensitive_data_nested(self):
        """Test sensitive data filtering in nested structures."""
        data = {
            "user": {
                "name": "test_user",
                "credentials": {
                    "password": "secret123",
                    "api_key": "r3mes_secret_key"
                }
            },
            "config": {
                "database_url": "postgresql://user:pass@localhost/db"
            }
        }
        
        filtered = filter_sensitive_data(data)
        
        assert filtered["user"]["name"] == "test_user"
        assert filtered["user"]["credentials"]["password"] == "***FILTERED***"
        assert filtered["user"]["credentials"]["api_key"] == "***FILTERED***"
        assert filtered["config"]["database_url"] == "***FILTERED***"
    
    def test_filter_sensitive_data_list(self):
        """Test sensitive data filtering in lists."""
        data = {
            "users": [
                {"name": "user1", "password": "pass1"},
                {"name": "user2", "api_key": "key2"}
            ]
        }
        
        filtered = filter_sensitive_data(data)
        
        assert filtered["users"][0]["name"] == "user1"
        assert filtered["users"][0]["password"] == "***FILTERED***"
        assert filtered["users"][1]["name"] == "user2"
        assert filtered["users"][1]["api_key"] == "***FILTERED***"
    
    @pytest.mark.asyncio
    async def test_dispatch_success(self, middleware, mock_request):
        """Test successful request processing."""
        async def mock_call_next(request):
            return Response("Success", status_code=200)
        
        response = await middleware.dispatch(mock_request, mock_call_next)
        
        assert response.status_code == 200
        assert hasattr(mock_request.state, 'request_id')
    
    @pytest.mark.asyncio
    async def test_dispatch_r3mes_exception(self, middleware, mock_request):
        """Test handling of R3MES exceptions."""
        async def mock_call_next(request):
            raise InvalidInputError(
                message="Invalid input provided",
                field="username",
                value="invalid_user"
            )
        
        response = await middleware.dispatch(mock_request, mock_call_next)
        
        assert response.status_code == 400
        
        # Parse response body
        body = json.loads(response.body.decode())
        assert body["error"] is True
        assert body["error_code"] == "R3MES_1001"
        assert "Invalid input provided" in body["message"]
        assert "request_id" in body
    
    @pytest.mark.asyncio
    async def test_dispatch_http_exception(self, middleware, mock_request):
        """Test handling of HTTP exceptions."""
        async def mock_call_next(request):
            raise HTTPException(status_code=404, detail="Not found")
        
        response = await middleware.dispatch(mock_request, mock_call_next)
        
        assert response.status_code == 404
        
        body = json.loads(response.body.decode())
        assert body["error"] is True
        assert body["message"] == "Not found"
        assert "request_id" in body
    
    @pytest.mark.asyncio
    async def test_dispatch_validation_error(self, middleware, mock_request):
        """Test handling of Pydantic validation errors."""
        from pydantic import ValidationError
        
        async def mock_call_next(request):
            # Create a mock validation error
            error = ValidationError.from_exception_data(
                "TestModel",
                [
                    {
                        "type": "missing",
                        "loc": ("field1",),
                        "msg": "Field required",
                        "input": {},
                    }
                ]
            )
            raise error
        
        response = await middleware.dispatch(mock_request, mock_call_next)
        
        assert response.status_code == 422
        
        body = json.loads(response.body.decode())
        assert body["error"] is True
        assert "Validation failed" in body["message"]
        assert "request_id" in body
    
    @pytest.mark.asyncio
    async def test_dispatch_generic_exception(self, middleware, mock_request):
        """Test handling of generic exceptions."""
        async def mock_call_next(request):
            raise Exception("Unexpected error")
        
        response = await middleware.dispatch(mock_request, mock_call_next)
        
        assert response.status_code == 500
        
        body = json.loads(response.body.decode())
        assert body["error"] is True
        assert body["message"] == "Internal server error"
        assert "request_id" in body
    
    @pytest.mark.asyncio
    async def test_dispatch_database_error(self, middleware, mock_request):
        """Test handling of database errors."""
        async def mock_call_next(request):
            raise DatabaseError(
                message="Database connection failed",
                operation="user_lookup"
            )
        
        response = await middleware.dispatch(mock_request, mock_call_next)
        
        assert response.status_code == 500
        
        body = json.loads(response.body.decode())
        assert body["error"] is True
        assert body["error_code"] == "R3MES_1201"
        assert "Database operation failed" in body["message"]
    
    @pytest.mark.asyncio
    async def test_dispatch_authentication_error(self, middleware, mock_request):
        """Test handling of authentication errors."""
        async def mock_call_next(request):
            raise AuthenticationError(
                message="Invalid credentials",
                auth_method="api_key"
            )
        
        response = await middleware.dispatch(mock_request, mock_call_next)
        
        assert response.status_code == 401
        
        body = json.loads(response.body.decode())
        assert body["error"] is True
        assert body["error_code"] == "R3MES_1100"
        assert "Authentication failed" in body["message"]
    
    @pytest.mark.asyncio
    async def test_error_context_logging(self, middleware, mock_request):
        """Test that error context is properly logged."""
        async def mock_call_next(request):
            raise InvalidInputError(
                message="Test error for logging",
                field="test_field"
            )
        
        with patch('app.middleware.error_handler.logger') as mock_logger:
            response = await middleware.dispatch(mock_request, mock_call_next)
            
            # Verify error was logged
            mock_logger.error.assert_called_once()
            log_call = mock_logger.error.call_args
            
            # Check log message contains error details
            log_message = log_call[0][0]
            assert "Request failed" in log_message
            
            # Check log context
            log_context = log_call[1]
            assert "request_id" in log_context
            assert log_context["method"] == "GET"
            assert log_context["path"] == "/api/test"
            assert log_context["status_code"] == 400
    
    @pytest.mark.asyncio
    async def test_request_id_propagation(self, middleware, mock_request):
        """Test that request ID is properly propagated."""
        async def mock_call_next(request):
            # Verify request has request_id in state
            assert hasattr(request.state, 'request_id')
            assert isinstance(request.state.request_id, str)
            return Response("Success", status_code=200)
        
        response = await middleware.dispatch(mock_request, mock_call_next)
        
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_sensitive_data_not_logged(self, middleware, mock_request):
        """Test that sensitive data is not logged in errors."""
        # Add sensitive data to request
        mock_request.json = AsyncMock(return_value={
            "username": "test_user",
            "password": "secret123",
            "api_key": "r3mes_secret_key"
        })
        
        async def mock_call_next(request):
            raise InvalidInputError("Test error with sensitive data")
        
        with patch('app.middleware.error_handler.logger') as mock_logger:
            response = await middleware.dispatch(mock_request, mock_call_next)
            
            # Check that sensitive data was filtered in logs
            log_call = mock_logger.error.call_args
            log_str = str(log_call)
            
            assert "secret123" not in log_str
            assert "r3mes_secret_key" not in log_str
            assert "***FILTERED***" in log_str or "test_user" in log_str
    
    def test_error_response_format(self):
        """Test error response format consistency."""
        # Test with R3MES exception
        exception = InvalidInputError(
            message="Test error",
            field="test_field",
            value="test_value"
        )
        
        # The middleware should format this consistently
        expected_format = {
            "error": True,
            "error_code": "R3MES_1001",
            "message": "Invalid input provided",
            "request_id": "test_id",
            "details": {
                "field": "test_field",
                "value": "test_value"
            }
        }
        
        # Verify the exception has the expected structure
        exception_dict = exception.to_dict()
        assert exception_dict["error"] is True
        assert exception_dict["error_code"] == "R3MES_1001"
        assert exception_dict["details"]["field"] == "test_field"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
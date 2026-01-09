"""
Custom exceptions for R3MES Backend
"""

from fastapi import HTTPException, status


class R3MESException(Exception):
    """Base exception for R3MES backend."""
    pass


class InvalidAPIKeyError(R3MESException):
    """Raised when API key is invalid."""
    pass


class MissingCredentialsError(R3MESException):
    """Raised when required credentials are missing."""
    pass


class ProductionConfigurationError(R3MESException):
    """Raised when production configuration is invalid."""
    pass


class InvalidInputError(R3MESException):
    """Raised when input validation fails."""
    pass


class ValidationError(R3MESException):
    """Raised when data validation fails."""
    pass


class AuthenticationError(HTTPException):
    """Authentication error."""
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"}
        )


class AuthorizationError(HTTPException):
    """Authorization error."""
    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail
        )

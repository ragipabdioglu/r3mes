"""
R3MES SDK Error Classes

Custom exception hierarchy for the R3MES SDK.
"""

from typing import Optional, Dict, Any


class R3MESError(Exception):
    """Base exception for all R3MES SDK errors."""

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}

    def __str__(self) -> str:
        if self.code:
            return f"[{self.code}] {self.message}"
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, code={self.code!r})"


class ConnectionError(R3MESError):
    """Raised when connection to R3MES services fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="CONNECTION_ERROR", details=details)


class AuthenticationError(R3MESError):
    """Raised when authentication fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="AUTHENTICATION_ERROR", details=details)


class InsufficientCreditsError(R3MESError):
    """Raised when user has insufficient credits for an operation."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="INSUFFICIENT_CREDITS", details=details)


class NotFoundError(R3MESError):
    """Raised when a requested resource is not found."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="NOT_FOUND", details=details)


class RateLimitError(R3MESError):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="RATE_LIMIT", details=details)


class ValidationError(R3MESError):
    """Raised when input validation fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="VALIDATION_ERROR", details=details)


class TimeoutError(R3MESError):
    """Raised when a request times out."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="TIMEOUT_ERROR", details=details)

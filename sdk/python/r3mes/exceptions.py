"""
R3MES SDK Exceptions

Custom exceptions for the R3MES Python SDK.
"""


class R3MESError(Exception):
    """Base exception for all R3MES SDK errors."""
    
    def __init__(self, message: str, code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
    
    def __str__(self):
        if self.code:
            return f"[{self.code}] {self.message}"
        return self.message


class ConnectionError(R3MESError):
    """Raised when connection to R3MES network fails."""
    pass


class AuthenticationError(R3MESError):
    """Raised when authentication fails."""
    pass


class InsufficientCreditsError(R3MESError):
    """Raised when wallet has insufficient credits for operation."""
    pass


class InvalidWalletError(R3MESError):
    """Raised when wallet address or key is invalid."""
    pass


class TransactionError(R3MESError):
    """Raised when a blockchain transaction fails."""
    pass


class InferenceError(R3MESError):
    """Raised when inference request fails."""
    pass


class TimeoutError(R3MESError):
    """Raised when an operation times out."""
    pass


class RateLimitError(R3MESError):
    """Raised when rate limit is exceeded."""
    pass


class ValidationError(R3MESError):
    """Raised when input validation fails."""
    pass


class BlockchainError(R3MESError):
    """Raised when blockchain query or transaction fails."""
    pass


class NodeNotFoundError(R3MESError):
    """Raised when a node is not found."""
    pass


class ConfigurationError(R3MESError):
    """Raised when SDK configuration is invalid."""
    pass

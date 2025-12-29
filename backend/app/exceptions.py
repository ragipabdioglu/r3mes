"""
Custom Exception Classes for R3MES Backend

Specific exception types for better error handling and debugging.
"""


class R3MESException(Exception):
    """Base exception for all R3MES-specific errors"""
    pass


# Authentication & Authorization Exceptions
class AuthenticationError(R3MESException):
    """Raised when authentication fails"""
    pass


class AuthorizationError(R3MESException):
    """Raised when user lacks required permissions"""
    pass


class InvalidAPIKeyError(AuthenticationError):
    """Raised when API key is invalid or expired"""
    pass


class MissingCredentialsError(AuthenticationError):
    """Raised when required credentials are missing"""
    pass


# Database Exceptions
class DatabaseError(R3MESException):
    """Base exception for database-related errors"""
    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails"""
    pass


class DatabaseQueryError(DatabaseError):
    """Raised when database query fails"""
    pass


class UserNotFoundError(DatabaseError):
    """Raised when user is not found in database"""
    pass


# Blockchain Exceptions
class BlockchainError(R3MESException):
    """Base exception for blockchain-related errors"""
    pass


class BlockchainConnectionError(BlockchainError):
    """Raised when blockchain node connection fails"""
    pass


class BlockchainQueryError(BlockchainError):
    """Raised when blockchain query fails"""
    pass


class BlockchainTransactionError(BlockchainError):
    """Raised when blockchain transaction fails"""
    pass


class InvalidBlockchainResponseError(BlockchainError):
    """Raised when blockchain returns invalid response"""
    pass


# Model & Inference Exceptions
class ModelError(R3MESException):
    """Base exception for model-related errors"""
    pass


class ModelLoadError(ModelError):
    """Raised when model fails to load"""
    pass


class ModelNotFoundError(ModelError):
    """Raised when model is not found"""
    pass


class AdapterNotFoundError(ModelError):
    """Raised when adapter is not found"""
    pass


class InferenceError(ModelError):
    """Raised when inference fails"""
    pass


class InsufficientVRAMError(ModelError):
    """Raised when insufficient VRAM for model operation"""
    pass


# Credit System Exceptions
class CreditError(R3MESException):
    """Base exception for credit system errors"""
    pass


class InsufficientCreditsError(CreditError):
    """Raised when user has insufficient credits"""
    pass


class CreditDeductionError(CreditError):
    """Raised when credit deduction fails"""
    pass


# Configuration Exceptions
class ConfigurationError(R3MESException):
    """Base exception for configuration errors"""
    pass


class MissingEnvironmentVariableError(ConfigurationError):
    """Raised when required environment variable is missing"""
    pass


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration is invalid"""
    pass


class InvalidEnvironmentVariableError(ConfigurationError):
    """Raised when environment variable has invalid value"""
    pass


class ProductionConfigurationError(ConfigurationError):
    """Raised when production configuration is invalid or insecure"""
    pass


# IPFS Exceptions
class IPFSError(R3MESException):
    """Base exception for IPFS-related errors"""
    pass


class IPFSConnectionError(IPFSError):
    """Raised when IPFS connection fails"""
    pass


class IPFSUploadError(IPFSError):
    """Raised when IPFS upload fails"""
    pass


class IPFSDownloadError(IPFSError):
    """Raised when IPFS download fails"""
    pass


# Validation Exceptions
class ValidationError(R3MESException):
    """Base exception for validation errors"""
    pass


class InvalidInputError(ValidationError):
    """Raised when input validation fails"""
    pass


class InvalidWalletAddressError(ValidationError):
    """Raised when wallet address format is invalid"""
    pass


class InvalidAmountError(ValidationError):
    """Raised when amount format is invalid"""
    pass


# Rate Limiting Exceptions
class RateLimitError(R3MESException):
    """Raised when rate limit is exceeded"""
    pass


# Cache Exceptions
class CacheError(R3MESException):
    """Base exception for cache-related errors"""
    pass


class CacheConnectionError(CacheError):
    """Raised when cache connection fails"""
    pass


# Network Exceptions
class NetworkError(R3MESException):
    """Base exception for network-related errors"""
    pass


class TimeoutError(NetworkError):
    """Raised when network request times out"""
    pass


class ConnectionError(NetworkError):
    """Raised when network connection fails"""
    pass


# Router Exceptions
class RouterError(R3MESException):
    """Base exception for router-related errors"""
    pass


class AdapterSelectionError(RouterError):
    """Raised when adapter selection fails"""
    pass


# Faucet Exceptions
class FaucetError(R3MESException):
    """Base exception for faucet-related errors"""
    pass


class FaucetDisabledError(FaucetError):
    """Raised when faucet is disabled"""
    pass


class FaucetRateLimitError(FaucetError):
    """Raised when faucet rate limit is exceeded"""
    pass


class FaucetTransactionError(FaucetError):
    """Raised when faucet transaction fails"""
    pass


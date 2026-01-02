"""
R3MES Exception Classes - Standardized Error Handling

Production-ready exception hierarchy with proper error codes and context.
"""

import logging
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorCode(Enum):
    """Standardized error codes for R3MES system."""
    
    # General errors (1000-1099)
    UNKNOWN_ERROR = "R3MES_1000"
    INVALID_INPUT = "R3MES_1001"
    VALIDATION_ERROR = "R3MES_1002"
    CONFIGURATION_ERROR = "R3MES_1003"
    
    # Authentication & Authorization (1100-1199)
    AUTHENTICATION_FAILED = "R3MES_1100"
    INVALID_API_KEY = "R3MES_1101"
    INSUFFICIENT_PERMISSIONS = "R3MES_1102"
    TOKEN_EXPIRED = "R3MES_1103"
    INVALID_SIGNATURE = "R3MES_1104"
    
    # Database errors (1200-1299)
    DATABASE_CONNECTION_ERROR = "R3MES_1200"
    DATABASE_QUERY_ERROR = "R3MES_1201"
    DATABASE_CONSTRAINT_ERROR = "R3MES_1202"
    DATABASE_TIMEOUT = "R3MES_1203"
    
    # Blockchain errors (1300-1399)
    BLOCKCHAIN_CONNECTION_ERROR = "R3MES_1300"
    INVALID_WALLET_ADDRESS = "R3MES_1301"
    INSUFFICIENT_BALANCE = "R3MES_1302"
    TRANSACTION_FAILED = "R3MES_1303"
    INVALID_NONCE = "R3MES_1304"
    INSUFFICIENT_STAKE = "R3MES_1305"
    
    # Mining errors (1400-1499)
    MINING_ERROR = "R3MES_1400"
    INVALID_GRADIENT = "R3MES_1401"
    GPU_ERROR = "R3MES_1402"
    MODEL_LOADING_ERROR = "R3MES_1403"
    TRAINING_ERROR = "R3MES_1404"
    
    # Network & Communication (1500-1599)
    NETWORK_ERROR = "R3MES_1500"
    TIMEOUT_ERROR = "R3MES_1501"
    IPFS_ERROR = "R3MES_1502"
    RPC_ERROR = "R3MES_1503"
    
    # Resource errors (1600-1699)
    INSUFFICIENT_CREDITS = "R3MES_1600"
    RESOURCE_NOT_FOUND = "R3MES_1601"
    RESOURCE_LOCKED = "R3MES_1602"
    QUOTA_EXCEEDED = "R3MES_1603"
    
    # Production configuration (1700-1799)
    PRODUCTION_CONFIG_ERROR = "R3MES_1700"
    MISSING_ENVIRONMENT_VARIABLE = "R3MES_1701"
    INVALID_PRODUCTION_SETTING = "R3MES_1702"
    SECRETS_MANAGEMENT_ERROR = "R3MES_1703"


class R3MESException(Exception):
    """
    Base exception class for all R3MES errors.
    
    Provides structured error handling with error codes, context, and logging.
    """
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        user_message: Optional[str] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.cause = cause
        self.user_message = user_message or message
        
        # Log the error
        self._log_error()
        
        super().__init__(self.message)
    
    def _log_error(self):
        """Log the error with appropriate level and context."""
        log_data = {
            "error_code": self.error_code.value,
            "message": self.message,
            "details": self.details,
        }
        
        if self.cause:
            log_data["cause"] = str(self.cause)
        
        # Log level based on error type
        if self.error_code.value.startswith("R3MES_10"):  # General errors
            logger.error("R3MES Error: %s", log_data)
        elif self.error_code.value.startswith("R3MES_11"):  # Auth errors
            logger.warning("Authentication Error: %s", log_data)
        elif self.error_code.value.startswith("R3MES_12"):  # Database errors
            logger.error("Database Error: %s", log_data, exc_info=self.cause)
        elif self.error_code.value.startswith("R3MES_17"):  # Production config
            logger.critical("Production Configuration Error: %s", log_data)
        else:
            logger.error("R3MES Error: %s", log_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": True,
            "error_code": self.error_code.value,
            "message": self.user_message,
            "details": self.details,
        }
    
    def __str__(self) -> str:
        return f"[{self.error_code.value}] {self.message}"


# Specific exception classes

class InvalidInputError(R3MESException):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        
        super().__init__(
            message=message,
            error_code=ErrorCode.INVALID_INPUT,
            details=details,
            user_message="Invalid input provided"
        )


class ValidationError(R3MESException):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, validation_errors: Optional[Dict[str, str]] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            details={"validation_errors": validation_errors or {}},
            user_message="Validation failed"
        )


class DatabaseError(R3MESException):
    """Raised when database operations fail."""
    
    def __init__(self, message: str, operation: Optional[str] = None, cause: Optional[Exception] = None):
        details = {}
        if operation:
            details["operation"] = operation
        
        super().__init__(
            message=message,
            error_code=ErrorCode.DATABASE_QUERY_ERROR,
            details=details,
            cause=cause,
            user_message="Database operation failed"
        )


class DatabaseConnectionError(R3MESException):
    """Raised when database connection fails."""
    
    def __init__(self, message: str, database_url: Optional[str] = None, cause: Optional[Exception] = None):
        details = {}
        if database_url:
            # Don't log full URL for security
            details["database_type"] = "postgresql" if "postgresql" in database_url else "sqlite"
        
        super().__init__(
            message=message,
            error_code=ErrorCode.DATABASE_CONNECTION_ERROR,
            details=details,
            cause=cause,
            user_message="Unable to connect to database"
        )


class AuthenticationError(R3MESException):
    """Raised when authentication fails."""
    
    def __init__(self, message: str, auth_method: Optional[str] = None):
        details = {}
        if auth_method:
            details["auth_method"] = auth_method
        
        super().__init__(
            message=message,
            error_code=ErrorCode.AUTHENTICATION_FAILED,
            details=details,
            user_message="Authentication failed"
        )


class InvalidAPIKeyError(R3MESException):
    """Raised when API key is invalid or expired."""
    
    def __init__(self, message: str = "Invalid or expired API key"):
        super().__init__(
            message=message,
            error_code=ErrorCode.INVALID_API_KEY,
            user_message="Invalid API key"
        )


class BlockchainQueryError(R3MESException):
    """Raised when blockchain query operations fail."""
    
    def __init__(self, message: str, query_type: Optional[str] = None, cause: Optional[Exception] = None):
        details = {}
        if query_type:
            details["query_type"] = query_type
        
        super().__init__(
            message=message,
            error_code=ErrorCode.BLOCKCHAIN_CONNECTION_ERROR,
            details=details,
            cause=cause,
            user_message="Blockchain query failed"
        )


class InsufficientVRAMError(R3MESException):
    """Raised when insufficient VRAM for model operations."""
    
    def __init__(self, message: str, required_mb: Optional[int] = None, available_mb: Optional[int] = None):
        details = {}
        if required_mb:
            details["required_mb"] = required_mb
        if available_mb:
            details["available_mb"] = available_mb
        
        super().__init__(
            message=message,
            error_code=ErrorCode.GPU_ERROR,
            details=details,
            user_message="Insufficient GPU memory"
        )


class InvalidEnvironmentVariableError(R3MESException):
    """Raised when environment variable value is invalid."""
    
    def __init__(self, message: str, variable_name: Optional[str] = None, value: Any = None):
        details = {}
        if variable_name:
            details["variable_name"] = variable_name
        if value is not None:
            details["value"] = str(value)
        
        super().__init__(
            message=message,
            error_code=ErrorCode.CONFIGURATION_ERROR,
            details=details,
            user_message="Invalid environment variable value"
        )


class BlockchainConnectionError(R3MESException):
    """Raised when blockchain connection fails."""
    
    def __init__(self, message: str, endpoint: Optional[str] = None, cause: Optional[Exception] = None):
        details = {}
        if endpoint:
            details["endpoint"] = endpoint
        
        super().__init__(
            message=message,
            error_code=ErrorCode.BLOCKCHAIN_CONNECTION_ERROR,
            details=details,
            cause=cause,
            user_message="Unable to connect to blockchain"
        )


class AdapterNotFoundError(R3MESException):
    """Raised when AI adapter is not found."""
    
    def __init__(self, message: str, adapter_name: Optional[str] = None):
        details = {}
        if adapter_name:
            details["adapter_name"] = adapter_name
        
        super().__init__(
            message=message,
            error_code=ErrorCode.MODEL_LOADING_ERROR,
            details=details,
            user_message="Adapter not found"
        )


class BlockchainError(R3MESException):
    """Raised when blockchain operations fail."""
    
    def __init__(self, message: str, operation: Optional[str] = None, cause: Optional[Exception] = None):
        details = {}
        if operation:
            details["operation"] = operation
        
        super().__init__(
            message=message,
            error_code=ErrorCode.BLOCKCHAIN_CONNECTION_ERROR,
            details=details,
            cause=cause,
            user_message="Blockchain operation failed"
        )


class ResourceNotFoundError(R3MESException):
    """Raised when a requested resource is not found."""
    
    def __init__(self, resource_type: str, resource_id: str, message: Optional[str] = None):
        if not message:
            message = f"{resource_type.title()} not found: {resource_id}"
        
        super().__init__(
            message=message,
            error_code=ErrorCode.RESOURCE_NOT_FOUND,
            details={
                "resource_type": resource_type,
                "resource_id": resource_id
            },
            user_message=f"{resource_type.title()} not found"
        )


class InvalidWalletAddressError(R3MESException):
    """Raised when wallet address format is invalid."""
    
    def __init__(self, address: str):
        super().__init__(
            message=f"Invalid wallet address format: {address}",
            error_code=ErrorCode.INVALID_WALLET_ADDRESS,
            details={"address": address},
            user_message="Invalid wallet address format"
        )


class InsufficientCreditsError(R3MESException):
    """Raised when user has insufficient credits."""
    
    def __init__(self, required: float, available: float, wallet: str):
        super().__init__(
            message=f"Insufficient credits: required {required}, available {available}",
            error_code=ErrorCode.INSUFFICIENT_CREDITS,
            details={
                "required": required,
                "available": available,
                "wallet": wallet
            },
            user_message=f"Insufficient credits. Required: {required}, Available: {available}"
        )


class MiningError(R3MESException):
    """Raised when mining operations fail."""
    
    def __init__(self, message: str, miner: Optional[str] = None, cause: Optional[Exception] = None):
        details = {}
        if miner:
            details["miner"] = miner
        
        super().__init__(
            message=message,
            error_code=ErrorCode.MINING_ERROR,
            details=details,
            cause=cause,
            user_message="Mining operation failed"
        )


class ProductionConfigurationError(R3MESException):
    """Raised when production configuration is invalid."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        details = {}
        if config_key:
            details["config_key"] = config_key
        
        super().__init__(
            message=message,
            error_code=ErrorCode.PRODUCTION_CONFIG_ERROR,
            details=details,
            user_message="Configuration error"
        )


class MissingEnvironmentVariableError(R3MESException):
    """Raised when required environment variable is missing."""
    
    def __init__(self, variable_name: str, context: Optional[str] = None):
        message = f"Required environment variable not set: {variable_name}"
        if context:
            message += f" (context: {context})"
        
        super().__init__(
            message=message,
            error_code=ErrorCode.MISSING_ENVIRONMENT_VARIABLE,
            details={
                "variable_name": variable_name,
                "context": context
            },
            user_message="Configuration error: missing required setting"
        )


class NetworkError(R3MESException):
    """Raised when network operations fail."""
    
    def __init__(self, message: str, endpoint: Optional[str] = None, cause: Optional[Exception] = None):
        details = {}
        if endpoint:
            details["endpoint"] = endpoint
        
        super().__init__(
            message=message,
            error_code=ErrorCode.NETWORK_ERROR,
            details=details,
            cause=cause,
            user_message="Network operation failed"
        )


class IPFSError(R3MESException):
    """Raised when IPFS operations fail."""
    
    def __init__(self, message: str, ipfs_hash: Optional[str] = None, cause: Optional[Exception] = None):
        details = {}
        if ipfs_hash:
            details["ipfs_hash"] = ipfs_hash
        
        super().__init__(
            message=message,
            error_code=ErrorCode.IPFS_ERROR,
            details=details,
            cause=cause,
            user_message="IPFS operation failed"
        )


class ConnectionError(R3MESException):
    """Raised when connection operations fail."""
    
    def __init__(self, message: str, endpoint: Optional[str] = None, cause: Optional[Exception] = None):
        details = {}
        if endpoint:
            details["endpoint"] = endpoint
        
        super().__init__(
            message=message,
            error_code=ErrorCode.NETWORK_ERROR,
            details=details,
            cause=cause,
            user_message="Connection failed"
        )


class ModelLoadError(R3MESException):
    """Raised when AI model loading fails."""
    
    def __init__(self, message: str, model_path: Optional[str] = None, cause: Optional[Exception] = None):
        details = {}
        if model_path:
            details["model_path"] = model_path
        
        super().__init__(
            message=message,
            error_code=ErrorCode.MODEL_LOADING_ERROR,
            details=details,
            cause=cause,
            user_message="Model loading failed"
        )


class AdapterSelectionError(R3MESException):
    """Raised when adapter selection fails."""
    
    def __init__(self, message: str, adapter_name: Optional[str] = None):
        details = {}
        if adapter_name:
            details["adapter_name"] = adapter_name
        
        super().__init__(
            message=message,
            error_code=ErrorCode.MODEL_LOADING_ERROR,
            details=details,
            user_message="Adapter selection failed"
        )


class InferenceError(R3MESException):
    """Raised when AI inference fails."""
    
    def __init__(self, message: str, model_name: Optional[str] = None, cause: Optional[Exception] = None):
        details = {}
        if model_name:
            details["model_name"] = model_name
        
        super().__init__(
            message=message,
            error_code=ErrorCode.MODEL_LOADING_ERROR,
            details=details,
            cause=cause,
            user_message="AI inference failed"
        )


class InvalidConfigurationError(R3MESException):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        details = {}
        if config_key:
            details["config_key"] = config_key
        
        super().__init__(
            message=message,
            error_code=ErrorCode.CONFIGURATION_ERROR,
            details=details,
            user_message="Configuration error"
        )


class MissingCredentialsError(R3MESException):
    """Raised when required credentials are missing."""
    
    def __init__(self, message: str, credential_type: Optional[str] = None):
        details = {}
        if credential_type:
            details["credential_type"] = credential_type
        
        super().__init__(
            message=message,
            error_code=ErrorCode.AUTHENTICATION_FAILED,
            details=details,
            user_message="Missing credentials"
        )


class CreditDeductionError(R3MESException):
    """Raised when credit deduction fails."""
    
    def __init__(self, message: str, wallet: Optional[str] = None, amount: Optional[float] = None):
        details = {}
        if wallet:
            details["wallet"] = wallet
        if amount:
            details["amount"] = amount
        
        super().__init__(
            message=message,
            error_code=ErrorCode.INSUFFICIENT_CREDITS,
            details=details,
            user_message="Credit deduction failed"
        )


class TimeoutError(R3MESException):
    """Raised when operations timeout."""
    
    def __init__(self, message: str, timeout_seconds: Optional[float] = None, operation: Optional[str] = None):
        details = {}
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds
        if operation:
            details["operation"] = operation
        
        super().__init__(
            message=message,
            error_code=ErrorCode.TIMEOUT_ERROR,
            details=details,
            user_message="Operation timed out"
        )


class ModelNotFoundError(R3MESException):
    """Raised when AI model is not found."""
    
    def __init__(self, message: str, model_name: Optional[str] = None):
        details = {}
        if model_name:
            details["model_name"] = model_name
        
        super().__init__(
            message=message,
            error_code=ErrorCode.MODEL_LOADING_ERROR,
            details=details,
            user_message="Model not found"
        )


# Exception handling utilities

def handle_database_exception(func):
    """Decorator to handle database exceptions consistently."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "connection" in str(e).lower():
                raise DatabaseConnectionError(
                    message=f"Database connection failed in {func.__name__}",
                    cause=e
                )
            else:
                raise DatabaseError(
                    message=f"Database operation failed in {func.__name__}",
                    operation=func.__name__,
                    cause=e
                )
    return wrapper


def handle_blockchain_exception(func):
    """Decorator to handle blockchain exceptions consistently."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise BlockchainError(
                message=f"Blockchain operation failed in {func.__name__}",
                operation=func.__name__,
                cause=e
            )
    return wrapper


def validate_wallet_address(address: str) -> str:
    """
    Validate wallet address format.
    
    Args:
        address: Wallet address to validate
        
    Returns:
        Validated address
        
    Raises:
        InvalidWalletAddressError: If address format is invalid
    """
    if not address:
        raise InvalidWalletAddressError("Empty wallet address")
    
    if not address.startswith("remes1"):
        raise InvalidWalletAddressError(address)
    
    if len(address) != 44:  # remes1 + 38 characters
        raise InvalidWalletAddressError(address)
    
    # Basic character validation (bech32)
    allowed_chars = set("023456789acdefghjklmnpqrstuvwxyz")
    address_chars = set(address[6:].lower())  # Skip "remes1" prefix
    
    if not address_chars.issubset(allowed_chars):
        raise InvalidWalletAddressError(address)
    
    return address


def validate_positive_number(value: Any, field_name: str) -> float:
    """
    Validate that a value is a positive number.
    
    Args:
        value: Value to validate
        field_name: Name of the field for error messages
        
    Returns:
        Validated number as float
        
    Raises:
        InvalidInputError: If value is not a positive number
    """
    try:
        num_value = float(value)
    except (ValueError, TypeError):
        raise InvalidInputError(
            message=f"{field_name} must be a number",
            field=field_name,
            value=value
        )
    
    if num_value <= 0:
        raise InvalidInputError(
            message=f"{field_name} must be positive",
            field=field_name,
            value=value
        )
    
    return num_value
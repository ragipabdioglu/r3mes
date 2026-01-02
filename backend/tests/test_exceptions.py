#!/usr/bin/env python3
"""
Unit tests for exception handling system.

Tests the R3MES exception hierarchy, error codes, and validation utilities.
"""

import pytest
from unittest.mock import patch, Mock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.exceptions import (
    R3MESException,
    ErrorCode,
    InvalidInputError,
    ValidationError,
    DatabaseError,
    DatabaseConnectionError,
    AuthenticationError,
    InvalidAPIKeyError,
    BlockchainError,
    InvalidWalletAddressError,
    InsufficientCreditsError,
    MiningError,
    ProductionConfigurationError,
    MissingEnvironmentVariableError,
    NetworkError,
    IPFSError,
    ConnectionError,
    ModelLoadError,
    validate_wallet_address,
    validate_positive_number,
)


class TestErrorCode:
    """Test cases for ErrorCode enum."""
    
    def test_error_code_values(self):
        """Test that error codes have correct values."""
        assert ErrorCode.UNKNOWN_ERROR.value == "R3MES_1000"
        assert ErrorCode.INVALID_INPUT.value == "R3MES_1001"
        assert ErrorCode.AUTHENTICATION_FAILED.value == "R3MES_1100"
        assert ErrorCode.DATABASE_CONNECTION_ERROR.value == "R3MES_1200"
        assert ErrorCode.BLOCKCHAIN_CONNECTION_ERROR.value == "R3MES_1300"
        assert ErrorCode.MINING_ERROR.value == "R3MES_1400"
        assert ErrorCode.NETWORK_ERROR.value == "R3MES_1500"
        assert ErrorCode.INSUFFICIENT_CREDITS.value == "R3MES_1600"
        assert ErrorCode.PRODUCTION_CONFIG_ERROR.value == "R3MES_1700"
    
    def test_error_code_categories(self):
        """Test error code categorization."""
        # General errors (1000-1099)
        assert ErrorCode.UNKNOWN_ERROR.value.startswith("R3MES_10")
        assert ErrorCode.INVALID_INPUT.value.startswith("R3MES_10")
        
        # Auth errors (1100-1199)
        assert ErrorCode.AUTHENTICATION_FAILED.value.startswith("R3MES_11")
        assert ErrorCode.INVALID_API_KEY.value.startswith("R3MES_11")
        
        # Database errors (1200-1299)
        assert ErrorCode.DATABASE_CONNECTION_ERROR.value.startswith("R3MES_12")
        assert ErrorCode.DATABASE_QUERY_ERROR.value.startswith("R3MES_12")


class TestR3MESException:
    """Test cases for base R3MESException class."""
    
    def test_basic_exception_creation(self):
        """Test basic exception creation."""
        exc = R3MESException(
            message="Test error",
            error_code=ErrorCode.UNKNOWN_ERROR
        )
        
        assert exc.message == "Test error"
        assert exc.error_code == ErrorCode.UNKNOWN_ERROR
        assert exc.details == {}
        assert exc.cause is None
        assert exc.user_message == "Test error"
    
    def test_exception_with_details(self):
        """Test exception creation with details."""
        details = {"field": "username", "value": "invalid"}
        exc = R3MESException(
            message="Test error with details",
            error_code=ErrorCode.INVALID_INPUT,
            details=details,
            user_message="User-friendly message"
        )
        
        assert exc.details == details
        assert exc.user_message == "User-friendly message"
    
    def test_exception_with_cause(self):
        """Test exception creation with cause."""
        cause = ValueError("Original error")
        exc = R3MESException(
            message="Wrapped error",
            error_code=ErrorCode.UNKNOWN_ERROR,
            cause=cause
        )
        
        assert exc.cause == cause
    
    def test_exception_to_dict(self):
        """Test exception serialization to dictionary."""
        details = {"field": "test", "value": "invalid"}
        exc = R3MESException(
            message="Test error",
            error_code=ErrorCode.INVALID_INPUT,
            details=details,
            user_message="User message"
        )
        
        result = exc.to_dict()
        
        assert result["error"] is True
        assert result["error_code"] == "R3MES_1001"
        assert result["message"] == "User message"
        assert result["details"] == details
    
    def test_exception_str_representation(self):
        """Test string representation of exception."""
        exc = R3MESException(
            message="Test error",
            error_code=ErrorCode.INVALID_INPUT
        )
        
        str_repr = str(exc)
        assert "[R3MES_1001]" in str_repr
        assert "Test error" in str_repr
    
    @patch('app.exceptions.logger')
    def test_exception_logging(self, mock_logger):
        """Test that exceptions are properly logged."""
        exc = R3MESException(
            message="Test error for logging",
            error_code=ErrorCode.DATABASE_QUERY_ERROR
        )
        
        # Verify logging was called
        mock_logger.error.assert_called_once()
        log_call = mock_logger.error.call_args[0][1]
        
        assert log_call["error_code"] == "R3MES_1201"
        assert log_call["message"] == "Test error for logging"


class TestSpecificExceptions:
    """Test cases for specific exception classes."""
    
    def test_invalid_input_error(self):
        """Test InvalidInputError creation and properties."""
        exc = InvalidInputError(
            message="Invalid username",
            field="username",
            value="invalid_user"
        )
        
        assert exc.error_code == ErrorCode.INVALID_INPUT
        assert exc.details["field"] == "username"
        assert exc.details["value"] == "invalid_user"
        assert exc.user_message == "Invalid input provided"
    
    def test_validation_error(self):
        """Test ValidationError creation and properties."""
        validation_errors = {"username": "Required field", "email": "Invalid format"}
        exc = ValidationError(
            message="Validation failed",
            validation_errors=validation_errors
        )
        
        assert exc.error_code == ErrorCode.VALIDATION_ERROR
        assert exc.details["validation_errors"] == validation_errors
        assert exc.user_message == "Validation failed"
    
    def test_database_error(self):
        """Test DatabaseError creation and properties."""
        cause = Exception("Connection timeout")
        exc = DatabaseError(
            message="Database operation failed",
            operation="user_lookup",
            cause=cause
        )
        
        assert exc.error_code == ErrorCode.DATABASE_QUERY_ERROR
        assert exc.details["operation"] == "user_lookup"
        assert exc.cause == cause
        assert exc.user_message == "Database operation failed"
    
    def test_database_connection_error(self):
        """Test DatabaseConnectionError creation and properties."""
        exc = DatabaseConnectionError(
            message="Cannot connect to database",
            database_url="postgresql://user:pass@localhost/db"
        )
        
        assert exc.error_code == ErrorCode.DATABASE_CONNECTION_ERROR
        assert exc.details["database_type"] == "postgresql"
        assert exc.user_message == "Unable to connect to database"
        # Ensure full URL is not logged for security
        assert "pass" not in str(exc.details)
    
    def test_authentication_error(self):
        """Test AuthenticationError creation and properties."""
        exc = AuthenticationError(
            message="Invalid credentials",
            auth_method="api_key"
        )
        
        assert exc.error_code == ErrorCode.AUTHENTICATION_FAILED
        assert exc.details["auth_method"] == "api_key"
        assert exc.user_message == "Authentication failed"
    
    def test_invalid_api_key_error(self):
        """Test InvalidAPIKeyError creation and properties."""
        exc = InvalidAPIKeyError()
        
        assert exc.error_code == ErrorCode.INVALID_API_KEY
        assert exc.user_message == "Invalid API key"
        
        # Test with custom message
        exc_custom = InvalidAPIKeyError("Custom API key error")
        assert exc_custom.message == "Custom API key error"
    
    def test_blockchain_error(self):
        """Test BlockchainError creation and properties."""
        cause = Exception("Network timeout")
        exc = BlockchainError(
            message="Blockchain operation failed",
            operation="submit_transaction",
            cause=cause
        )
        
        assert exc.error_code == ErrorCode.BLOCKCHAIN_CONNECTION_ERROR
        assert exc.details["operation"] == "submit_transaction"
        assert exc.cause == cause
    
    def test_invalid_wallet_address_error(self):
        """Test InvalidWalletAddressError creation and properties."""
        invalid_address = "invalid_address_format"
        exc = InvalidWalletAddressError(invalid_address)
        
        assert exc.error_code == ErrorCode.INVALID_WALLET_ADDRESS
        assert exc.details["address"] == invalid_address
        assert exc.user_message == "Invalid wallet address format"
    
    def test_insufficient_credits_error(self):
        """Test InsufficientCreditsError creation and properties."""
        exc = InsufficientCreditsError(
            required=100.0,
            available=50.0,
            wallet="remes1testaddress"
        )
        
        assert exc.error_code == ErrorCode.INSUFFICIENT_CREDITS
        assert exc.details["required"] == 100.0
        assert exc.details["available"] == 50.0
        assert exc.details["wallet"] == "remes1testaddress"
        assert "Required: 100.0, Available: 50.0" in exc.user_message
    
    def test_mining_error(self):
        """Test MiningError creation and properties."""
        cause = Exception("GPU memory error")
        exc = MiningError(
            message="Mining operation failed",
            miner="remes1miner",
            cause=cause
        )
        
        assert exc.error_code == ErrorCode.MINING_ERROR
        assert exc.details["miner"] == "remes1miner"
        assert exc.cause == cause
    
    def test_production_configuration_error(self):
        """Test ProductionConfigurationError creation and properties."""
        exc = ProductionConfigurationError(
            message="Invalid production config",
            config_key="JWT_SECRET"
        )
        
        assert exc.error_code == ErrorCode.PRODUCTION_CONFIG_ERROR
        assert exc.details["config_key"] == "JWT_SECRET"
        assert exc.user_message == "Configuration error"
    
    def test_missing_environment_variable_error(self):
        """Test MissingEnvironmentVariableError creation and properties."""
        exc = MissingEnvironmentVariableError(
            variable_name="DATABASE_URL",
            context="database initialization"
        )
        
        assert exc.error_code == ErrorCode.MISSING_ENVIRONMENT_VARIABLE
        assert exc.details["variable_name"] == "DATABASE_URL"
        assert exc.details["context"] == "database initialization"
        assert "DATABASE_URL" in exc.message
        assert "database initialization" in exc.message
    
    def test_network_error(self):
        """Test NetworkError creation and properties."""
        cause = Exception("Connection refused")
        exc = NetworkError(
            message="Network request failed",
            endpoint="https://api.example.com",
            cause=cause
        )
        
        assert exc.error_code == ErrorCode.NETWORK_ERROR
        assert exc.details["endpoint"] == "https://api.example.com"
        assert exc.cause == cause
    
    def test_ipfs_error(self):
        """Test IPFSError creation and properties."""
        exc = IPFSError(
            message="IPFS upload failed",
            ipfs_hash="QmTestHash123"
        )
        
        assert exc.error_code == ErrorCode.IPFS_ERROR
        assert exc.details["ipfs_hash"] == "QmTestHash123"
    
    def test_model_load_error(self):
        """Test ModelLoadError creation and properties."""
        exc = ModelLoadError(
            message="Model loading failed",
            model_path="/path/to/model"
        )
        
        assert exc.error_code == ErrorCode.MODEL_LOADING_ERROR
        assert exc.details["model_path"] == "/path/to/model"


class TestValidationUtilities:
    """Test cases for validation utility functions."""
    
    def test_validate_wallet_address_valid(self):
        """Test wallet address validation with valid addresses."""
        valid_addresses = [
            "remes1testaddress234567890234567890234567890",  # 44 chars total
            "remes1acdefghjklmnpqrstuvwxyz023456789023456",  # 44 chars total
            "remes1234567890acdefghjklmnpqrstuvwxyz023456",  # 44 chars total
        ]
        
        for address in valid_addresses:
            # Should not raise exception
            result = validate_wallet_address(address)
            assert result == address
    
    def test_validate_wallet_address_invalid_empty(self):
        """Test wallet address validation with empty address."""
        with pytest.raises(InvalidWalletAddressError) as exc_info:
            validate_wallet_address("")
        
        assert "Empty wallet address" in str(exc_info.value)
    
    def test_validate_wallet_address_invalid_prefix(self):
        """Test wallet address validation with invalid prefix."""
        with pytest.raises(InvalidWalletAddressError):
            validate_wallet_address("cosmos1testaddress123456789012345678901234")
    
    def test_validate_wallet_address_invalid_length(self):
        """Test wallet address validation with invalid length."""
        # Too short
        with pytest.raises(InvalidWalletAddressError):
            validate_wallet_address("remes1short")
        
        # Too long
        with pytest.raises(InvalidWalletAddressError):
            validate_wallet_address("remes1" + "a" * 50)
    
    def test_validate_wallet_address_invalid_characters(self):
        """Test wallet address validation with invalid characters."""
        # Contains uppercase (not allowed in bech32)
        with pytest.raises(InvalidWalletAddressError):
            validate_wallet_address("remes1TESTADDRESS123456789012345678901234")
        
        # Contains invalid characters
        with pytest.raises(InvalidWalletAddressError):
            validate_wallet_address("remes1testaddress!@#$%^&*()1234567890123")
    
    def test_validate_positive_number_valid(self):
        """Test positive number validation with valid values."""
        valid_values = [1, 1.0, "1", "1.5", 100, 0.1]
        
        for value in valid_values:
            result = validate_positive_number(value, "test_field")
            assert result > 0
            assert isinstance(result, float)
    
    def test_validate_positive_number_invalid_type(self):
        """Test positive number validation with invalid types."""
        invalid_values = ["not_a_number", None, [], {}]
        
        for value in invalid_values:
            with pytest.raises(InvalidInputError) as exc_info:
                validate_positive_number(value, "test_field")
            
            assert "must be a number" in str(exc_info.value)
            assert exc_info.value.details["field"] == "test_field"
    
    def test_validate_positive_number_non_positive(self):
        """Test positive number validation with non-positive values."""
        non_positive_values = [0, -1, -0.5, "-1"]
        
        for value in non_positive_values:
            with pytest.raises(InvalidInputError) as exc_info:
                validate_positive_number(value, "test_field")
            
            assert "must be positive" in str(exc_info.value)
            assert exc_info.value.details["field"] == "test_field"


class TestExceptionInheritance:
    """Test exception inheritance and polymorphism."""
    
    def test_all_exceptions_inherit_from_r3mes_exception(self):
        """Test that all custom exceptions inherit from R3MESException."""
        exception_classes = [
            InvalidInputError,
            ValidationError,
            DatabaseError,
            DatabaseConnectionError,
            AuthenticationError,
            InvalidAPIKeyError,
            BlockchainError,
            InvalidWalletAddressError,
            InsufficientCreditsError,
            MiningError,
            ProductionConfigurationError,
            MissingEnvironmentVariableError,
            NetworkError,
            IPFSError,
            ConnectionError,
            ModelLoadError,
        ]
        
        for exc_class in exception_classes:
            assert issubclass(exc_class, R3MESException)
            assert issubclass(exc_class, Exception)
    
    def test_exception_polymorphism(self):
        """Test exception polymorphism."""
        exceptions = [
            InvalidInputError("Test input error"),
            DatabaseError("Test database error"),
            AuthenticationError("Test auth error"),
        ]
        
        for exc in exceptions:
            # All should be instances of R3MESException
            assert isinstance(exc, R3MESException)
            
            # All should have to_dict method
            result = exc.to_dict()
            assert isinstance(result, dict)
            assert "error" in result
            assert "error_code" in result
            assert "message" in result
            
            # All should have proper string representation
            str_repr = str(exc)
            assert "[R3MES_" in str_repr


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
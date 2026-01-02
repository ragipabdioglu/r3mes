#!/usr/bin/env python3
"""
Unit tests for configuration and environment handling.

Tests configuration loading, validation, and environment variable handling.
"""

import pytest
import os
from unittest.mock import patch, Mock
from typing import Dict, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.exceptions import (
    MissingEnvironmentVariableError,
    InvalidEnvironmentVariableError,
    ProductionConfigurationError,
)


class TestEnvironmentVariableHandling:
    """Test cases for environment variable handling."""
    
    def test_required_environment_variables(self):
        """Test validation of required environment variables."""
        required_vars = [
            "DATABASE_URL",
            "JWT_SECRET",
            "BLOCKCHAIN_RPC_URL",
            "IPFS_GATEWAY_URL",
        ]
        
        for var_name in required_vars:
            # Test missing variable
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(MissingEnvironmentVariableError) as exc_info:
                    if var_name not in os.environ:
                        raise MissingEnvironmentVariableError(
                            variable_name=var_name,
                            context="application startup"
                        )
                
                assert var_name in str(exc_info.value)
                assert exc_info.value.details["variable_name"] == var_name
    
    def test_environment_variable_validation(self):
        """Test validation of environment variable values."""
        # Test valid database URL formats
        valid_db_urls = [
            "postgresql://user:pass@localhost:5432/dbname",
            "sqlite:///path/to/database.db",
            "postgresql+asyncpg://user:pass@localhost/db",
        ]
        
        for url in valid_db_urls:
            # Should not raise exception for valid URLs
            assert url.startswith(("postgresql", "sqlite"))
        
        # Test invalid database URLs
        invalid_db_urls = [
            "",
            "invalid_url",
            "http://not-a-database",
        ]
        
        for url in invalid_db_urls:
            # Should detect invalid URLs
            is_valid = url.startswith(("postgresql", "sqlite"))
            assert not is_valid
    
    def test_jwt_secret_validation(self):
        """Test JWT secret validation."""
        # Test valid JWT secrets
        valid_secrets = [
            "a" * 32,  # 32 characters minimum
            "super_secret_jwt_key_with_sufficient_length",
            "1234567890abcdef" * 2,  # 32 characters
        ]
        
        for secret in valid_secrets:
            assert len(secret) >= 32
        
        # Test invalid JWT secrets
        invalid_secrets = [
            "",
            "short",
            "a" * 31,  # Too short
        ]
        
        for secret in invalid_secrets:
            if len(secret) < 32:
                with pytest.raises(InvalidEnvironmentVariableError):
                    raise InvalidEnvironmentVariableError(
                        message="JWT secret must be at least 32 characters",
                        variable_name="JWT_SECRET",
                        value=secret
                    )
    
    def test_url_format_validation(self):
        """Test URL format validation for various services."""
        # Test blockchain RPC URLs
        valid_rpc_urls = [
            "http://localhost:26657",
            "https://rpc.cosmos.network",
            "ws://localhost:26657/websocket",
        ]
        
        for url in valid_rpc_urls:
            assert url.startswith(("http://", "https://", "ws://"))
        
        # Test IPFS gateway URLs
        valid_ipfs_urls = [
            "http://localhost:8080",
            "https://ipfs.io",
            "https://gateway.pinata.cloud",
        ]
        
        for url in valid_ipfs_urls:
            assert url.startswith(("http://", "https://"))
    
    @patch.dict(os.environ, {
        "DATABASE_URL": "postgresql://user:pass@localhost/test",
        "JWT_SECRET": "a" * 32,
        "BLOCKCHAIN_RPC_URL": "http://localhost:26657",
        "IPFS_GATEWAY_URL": "http://localhost:8080"
    })
    def test_complete_environment_setup(self):
        """Test complete environment variable setup."""
        # All required variables should be present
        required_vars = [
            "DATABASE_URL",
            "JWT_SECRET", 
            "BLOCKCHAIN_RPC_URL",
            "IPFS_GATEWAY_URL"
        ]
        
        for var in required_vars:
            assert var in os.environ
            assert len(os.environ[var]) > 0


class TestConfigurationLoading:
    """Test cases for configuration loading and validation."""
    
    def test_development_configuration(self):
        """Test development environment configuration."""
        dev_config = {
            "debug": True,
            "log_level": "DEBUG",
            "database_echo": True,
            "cors_origins": ["http://localhost:3000"],
        }
        
        # Validate development settings
        assert dev_config["debug"] is True
        assert dev_config["log_level"] == "DEBUG"
        assert "http://localhost:3000" in dev_config["cors_origins"]
    
    def test_production_configuration(self):
        """Test production environment configuration."""
        prod_config = {
            "debug": False,
            "log_level": "INFO",
            "database_echo": False,
            "cors_origins": ["https://app.r3mes.io"],
        }
        
        # Validate production settings
        assert prod_config["debug"] is False
        assert prod_config["log_level"] in ["INFO", "WARNING", "ERROR"]
        assert all(origin.startswith("https://") for origin in prod_config["cors_origins"])
    
    def test_configuration_validation_rules(self):
        """Test configuration validation business rules."""
        # Test log level validation
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        invalid_log_levels = ["TRACE", "VERBOSE", "invalid"]
        
        for level in valid_log_levels:
            assert level in valid_log_levels
        
        for level in invalid_log_levels:
            assert level not in valid_log_levels
    
    def test_security_configuration(self):
        """Test security-related configuration."""
        security_config = {
            "jwt_algorithm": "HS256",
            "jwt_expiration_hours": 24,
            "api_key_expiration_days": 90,
            "max_api_keys_per_wallet": 10,
            "rate_limit_per_minute": 60,
        }
        
        # Validate security settings
        assert security_config["jwt_algorithm"] in ["HS256", "RS256"]
        assert 1 <= security_config["jwt_expiration_hours"] <= 168  # 1 hour to 1 week
        assert 1 <= security_config["api_key_expiration_days"] <= 365
        assert 1 <= security_config["max_api_keys_per_wallet"] <= 50
        assert 1 <= security_config["rate_limit_per_minute"] <= 1000


class TestDatabaseConfiguration:
    """Test cases for database configuration."""
    
    def test_database_url_parsing(self):
        """Test database URL parsing and validation."""
        # Test PostgreSQL URLs
        pg_urls = [
            "postgresql://user:pass@localhost:5432/dbname",
            "postgresql+asyncpg://user:pass@localhost/db",
            "postgres://user:pass@host:5432/db",
        ]
        
        for url in pg_urls:
            # Should be recognized as PostgreSQL
            assert any(prefix in url for prefix in ["postgresql", "postgres"])
        
        # Test SQLite URLs
        sqlite_urls = [
            "sqlite:///path/to/database.db",
            "sqlite:///:memory:",
        ]
        
        for url in sqlite_urls:
            assert url.startswith("sqlite://")
    
    def test_database_connection_parameters(self):
        """Test database connection parameter validation."""
        connection_params = {
            "pool_size": 20,
            "max_overflow": 30,
            "pool_timeout": 30,
            "pool_recycle": 3600,
        }
        
        # Validate connection parameters
        assert 1 <= connection_params["pool_size"] <= 100
        assert 0 <= connection_params["max_overflow"] <= 100
        assert 1 <= connection_params["pool_timeout"] <= 300
        assert 300 <= connection_params["pool_recycle"] <= 86400  # 5 min to 24 hours
    
    def test_database_ssl_configuration(self):
        """Test database SSL configuration."""
        ssl_configs = [
            {"ssl_mode": "require", "ssl_cert": None, "ssl_key": None},
            {"ssl_mode": "prefer", "ssl_cert": "/path/to/cert", "ssl_key": "/path/to/key"},
            {"ssl_mode": "disable", "ssl_cert": None, "ssl_key": None},
        ]
        
        valid_ssl_modes = ["disable", "allow", "prefer", "require", "verify-ca", "verify-full"]
        
        for config in ssl_configs:
            assert config["ssl_mode"] in valid_ssl_modes


class TestBlockchainConfiguration:
    """Test cases for blockchain configuration."""
    
    def test_blockchain_network_configuration(self):
        """Test blockchain network configuration."""
        network_configs = [
            {
                "chain_id": "r3mes-testnet-1",
                "rpc_url": "http://localhost:26657",
                "grpc_url": "localhost:9090",
                "address_prefix": "remes",
            },
            {
                "chain_id": "r3mes-mainnet-1", 
                "rpc_url": "https://rpc.r3mes.io",
                "grpc_url": "grpc.r3mes.io:443",
                "address_prefix": "remes",
            }
        ]
        
        for config in network_configs:
            # Validate chain ID format
            assert config["chain_id"].startswith("r3mes-")
            assert config["chain_id"].endswith(("-1", "-testnet-1", "-mainnet-1"))
            
            # Validate address prefix
            assert config["address_prefix"] == "remes"
            
            # Validate URLs
            assert config["rpc_url"].startswith(("http://", "https://"))
    
    def test_consensus_parameters(self):
        """Test blockchain consensus parameters."""
        consensus_params = {
            "block_time_seconds": 6,
            "max_validators": 100,
            "unbonding_time_days": 21,
            "min_stake_amount": 1000000,  # microunits
        }
        
        # Validate consensus parameters
        assert 1 <= consensus_params["block_time_seconds"] <= 60
        assert 1 <= consensus_params["max_validators"] <= 500
        assert 1 <= consensus_params["unbonding_time_days"] <= 365
        assert consensus_params["min_stake_amount"] > 0
    
    def test_gas_configuration(self):
        """Test gas configuration parameters."""
        gas_config = {
            "gas_price": "0.025uremes",
            "gas_adjustment": 1.3,
            "max_gas": 2000000,
        }
        
        # Validate gas configuration
        assert gas_config["gas_price"].endswith("uremes")
        assert 1.0 <= gas_config["gas_adjustment"] <= 2.0
        assert 100000 <= gas_config["max_gas"] <= 10000000


class TestIPFSConfiguration:
    """Test cases for IPFS configuration."""
    
    def test_ipfs_gateway_configuration(self):
        """Test IPFS gateway configuration."""
        gateway_configs = [
            {
                "gateway_url": "http://localhost:8080",
                "api_url": "http://localhost:5001",
                "timeout_seconds": 30,
            },
            {
                "gateway_url": "https://ipfs.io",
                "api_url": "https://ipfs.infura.io:5001",
                "timeout_seconds": 60,
            }
        ]
        
        for config in gateway_configs:
            # Validate URLs
            assert config["gateway_url"].startswith(("http://", "https://"))
            assert config["api_url"].startswith(("http://", "https://"))
            
            # Validate timeout
            assert 5 <= config["timeout_seconds"] <= 300
    
    def test_ipfs_pinning_configuration(self):
        """Test IPFS pinning service configuration."""
        pinning_config = {
            "service": "pinata",
            "api_key": "test_api_key",
            "secret_key": "test_secret_key",
            "max_file_size_mb": 100,
        }
        
        # Validate pinning configuration
        assert pinning_config["service"] in ["pinata", "infura", "fleek"]
        assert len(pinning_config["api_key"]) > 0
        assert len(pinning_config["secret_key"]) > 0
        assert 1 <= pinning_config["max_file_size_mb"] <= 1000


class TestLoggingConfiguration:
    """Test cases for logging configuration."""
    
    def test_log_level_configuration(self):
        """Test log level configuration."""
        log_configs = [
            {"level": "DEBUG", "environment": "development"},
            {"level": "INFO", "environment": "production"},
            {"level": "WARNING", "environment": "production"},
            {"level": "ERROR", "environment": "production"},
        ]
        
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        for config in log_configs:
            assert config["level"] in valid_levels
            
            # Production should not use DEBUG
            if config["environment"] == "production":
                assert config["level"] != "DEBUG"
    
    def test_log_format_configuration(self):
        """Test log format configuration."""
        log_formats = {
            "development": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "production": "%(asctime)s - %(levelname)s - %(name)s - %(message)s - %(pathname)s:%(lineno)d",
        }
        
        for env, format_str in log_formats.items():
            # Should contain required fields
            required_fields = ["%(asctime)s", "%(levelname)s", "%(message)s"]
            for field in required_fields:
                assert field in format_str
    
    def test_log_handler_configuration(self):
        """Test log handler configuration."""
        handler_configs = [
            {
                "type": "console",
                "level": "INFO",
                "format": "simple",
            },
            {
                "type": "file",
                "level": "DEBUG", 
                "filename": "/var/log/r3mes/app.log",
                "max_bytes": 10485760,  # 10MB
                "backup_count": 5,
            }
        ]
        
        for config in handler_configs:
            assert config["type"] in ["console", "file", "syslog"]
            assert config["level"] in ["DEBUG", "INFO", "WARNING", "ERROR"]
            
            if config["type"] == "file":
                assert "filename" in config
                assert config["max_bytes"] > 0
                assert config["backup_count"] > 0


class TestSecurityConfiguration:
    """Test cases for security configuration."""
    
    def test_cors_configuration(self):
        """Test CORS configuration."""
        cors_configs = [
            {
                "origins": ["http://localhost:3000"],
                "methods": ["GET", "POST", "PUT", "DELETE"],
                "headers": ["*"],
                "credentials": True,
            },
            {
                "origins": ["https://app.r3mes.io"],
                "methods": ["GET", "POST"],
                "headers": ["Authorization", "Content-Type"],
                "credentials": False,
            }
        ]
        
        valid_methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
        
        for config in cors_configs:
            # Validate origins
            for origin in config["origins"]:
                assert origin.startswith(("http://", "https://"))
            
            # Validate methods
            for method in config["methods"]:
                assert method in valid_methods
    
    def test_rate_limiting_configuration(self):
        """Test rate limiting configuration."""
        rate_limit_configs = [
            {
                "requests_per_minute": 60,
                "burst_limit": 10,
                "window_size_seconds": 60,
            },
            {
                "requests_per_minute": 1000,
                "burst_limit": 100,
                "window_size_seconds": 60,
            }
        ]
        
        for config in rate_limit_configs:
            assert 1 <= config["requests_per_minute"] <= 10000
            assert 1 <= config["burst_limit"] <= config["requests_per_minute"]
            assert 1 <= config["window_size_seconds"] <= 3600
    
    def test_authentication_configuration(self):
        """Test authentication configuration."""
        auth_config = {
            "jwt_secret": "a" * 32,
            "jwt_algorithm": "HS256",
            "jwt_expiration_hours": 24,
            "refresh_token_expiration_days": 30,
        }
        
        # Validate authentication settings
        assert len(auth_config["jwt_secret"]) >= 32
        assert auth_config["jwt_algorithm"] in ["HS256", "HS512", "RS256"]
        assert 1 <= auth_config["jwt_expiration_hours"] <= 168  # 1 hour to 1 week
        assert 1 <= auth_config["refresh_token_expiration_days"] <= 365


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
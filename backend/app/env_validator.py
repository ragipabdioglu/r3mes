"""
Environment Variable Validator

Centralized validation for all environment variables across the R3MES backend.
"""

import os
import re
import logging
from typing import Dict, List, Optional, Tuple, Callable, Any
from urllib.parse import urlparse
from dataclasses import dataclass

from .exceptions import (
    MissingEnvironmentVariableError,
    InvalidEnvironmentVariableError,
    ProductionConfigurationError,
)

logger = logging.getLogger(__name__)


@dataclass
class EnvVarRule:
    """Rule for validating an environment variable."""
    name: str
    required: bool = False
    required_in_production: bool = False
    validator: Optional[Callable[[str], Tuple[bool, Optional[str]]]] = None
    default: Optional[str] = None
    description: str = ""
    example: str = ""
    no_localhost_in_production: bool = False


class EnvironmentValidator:
    """
    Centralized environment variable validator.
    
    Validates all environment variables according to defined rules,
    with different requirements for development, staging, and production.
    """
    
    def __init__(self):
        self.is_production = os.getenv("R3MES_ENV", "development").lower() == "production"
        self.is_staging = os.getenv("R3MES_ENV", "development").lower() == "staging"
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_url(self, value: str, allow_localhost: bool = False) -> Tuple[bool, Optional[str]]:
        """
        Validate URL format.
        
        Args:
            value: URL to validate
            allow_localhost: Whether to allow localhost URLs
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            parsed = urlparse(value)
            if not parsed.scheme:
                return False, "URL must include scheme (http:// or https://)"
            if not parsed.netloc:
                return False, "URL must include hostname"
            if not allow_localhost:
                # Use exact hostname matching (not substring)
                hostname = parsed.hostname or ""
                hostname_lower = hostname.lower()
                # Check for exact localhost matches (case-insensitive)
                if hostname_lower in ("localhost", "127.0.0.1", "::1"):
                    return False, "URL cannot use localhost or 127.0.0.1"
                # Check for 127.x.x.x IP addresses
                if hostname.startswith("127."):
                    return False, "URL cannot use 127.x.x.x IP addresses"
            return True, None
        except Exception as e:
            return False, f"Invalid URL format: {e}"
    
    def validate_port(self, value: str) -> Tuple[bool, Optional[str]]:
        """Validate port number."""
        try:
            port = int(value)
            if port < 1 or port > 65535:
                return False, "Port must be between 1 and 65535"
            return True, None
        except ValueError:
            return False, "Port must be a valid integer"
    
    def validate_positive_int(self, value: str) -> Tuple[bool, Optional[str]]:
        """Validate positive integer."""
        try:
            num = int(value)
            if num <= 0:
                return False, "Value must be a positive integer"
            return True, None
        except ValueError:
            return False, "Value must be a valid integer"
    
    def validate_positive_float(self, value: str) -> Tuple[bool, Optional[str]]:
        """Validate positive float."""
        try:
            num = float(value)
            if num <= 0:
                return False, "Value must be a positive number"
            return True, None
        except ValueError:
            return False, "Value must be a valid number"
    
    def validate_boolean(self, value: str) -> Tuple[bool, Optional[str]]:
        """Validate boolean value."""
        if value.lower() in ("true", "false", "1", "0", "yes", "no"):
            return True, None
        return False, "Value must be a boolean (true/false, 1/0, yes/no)"
    
    def validate_path(self, value: str, must_exist: bool = False) -> Tuple[bool, Optional[str]]:
        """Validate file/directory path."""
        from pathlib import Path
        path = Path(value)
        if must_exist and not path.exists():
            return False, f"Path does not exist: {value}"
        return True, None
    
    def validate_wallet_address(self, value: str) -> Tuple[bool, Optional[str]]:
        """Validate Cosmos wallet address format."""
        if not value.startswith("remes"):
            return False, "Wallet address must start with 'remes'"
        if len(value) < 20 or len(value) > 60:
            return False, "Wallet address must be between 20 and 60 characters"
        if not re.match(r"^remes[a-zA-Z0-9]{19,59}$", value):
            return False, "Wallet address contains invalid characters"
        return True, None
    
    def validate_hex_string(self, value: str, length: Optional[int] = None) -> Tuple[bool, Optional[str]]:
        """Validate hexadecimal string."""
        if not re.match(r"^[0-9a-fA-F]+$", value):
            return False, "Value must be a valid hexadecimal string"
        if length and len(value) != length:
            return False, f"Value must be exactly {length} characters"
        return True, None
    
    def get_rules(self) -> List[EnvVarRule]:
        """Get all environment variable validation rules."""
        return [
            # Core Environment
            EnvVarRule(
                name="R3MES_ENV",
                required=False,
                default="development",
                validator=self.validate_environment_mode,
                description="Environment mode: development, staging, or production",
                example="production"
            ),
            
            # Database Configuration
            EnvVarRule(
                name="DATABASE_TYPE",
                required=False,
                default="sqlite",
                validator=lambda v: (True, None) if v in ("sqlite", "postgresql") else (False, "Must be 'sqlite' or 'postgresql'"),
                description="Database type: sqlite or postgresql",
                example="postgresql"
            ),
            EnvVarRule(
                name="DATABASE_URL",
                required_in_production=True,
                validator=lambda v: self.validate_url(v, allow_localhost=not self.is_production),
                description="Database connection URL",
                example="postgresql://user:password@host:5432/dbname",
                no_localhost_in_production=True
            ),
            EnvVarRule(
                name="DATABASE_PATH",
                required=False,
                default="backend/database.db",
                description="SQLite database file path (for DATABASE_TYPE=sqlite)",
                example="backend/database.db"
            ),
            
            # Redis Configuration
            EnvVarRule(
                name="REDIS_URL",
                required_in_production=True,
                validator=lambda v: self.validate_url(v, allow_localhost=True),  # Redis can be localhost in containers
                description="Redis connection URL",
                example="redis://localhost:6379/0"
            ),
            
            # Blockchain Configuration
            EnvVarRule(
                name="BLOCKCHAIN_RPC_URL",
                required_in_production=True,
                validator=lambda v: self.validate_url(v, allow_localhost=not self.is_production),
                description="Blockchain RPC URL (Tendermint)",
                example="http://blockchain.example.com:26657",
                no_localhost_in_production=True
            ),
            EnvVarRule(
                name="BLOCKCHAIN_GRPC_URL",
                required_in_production=True,
                validator=lambda v: (True, None) if ":" in v else (False, "Must include hostname:port"),
                description="Blockchain gRPC URL",
                example="blockchain.example.com:9090",
                no_localhost_in_production=True
            ),
            EnvVarRule(
                name="BLOCKCHAIN_REST_URL",
                required_in_production=True,
                validator=lambda v: self.validate_url(v, allow_localhost=not self.is_production),
                description="Blockchain REST API URL",
                example="http://blockchain.example.com:1317",
                no_localhost_in_production=True
            ),
            
            # Backend API Configuration
            EnvVarRule(
                name="BACKEND_PORT",
                required=False,
                default="8000",
                validator=self.validate_port,
                description="Backend API server port",
                example="8000"
            ),
            EnvVarRule(
                name="CORS_ALLOWED_ORIGINS",
                required_in_production=True,
                validator=lambda v: (True, None) if v and "*" not in v else (False, "Cannot use wildcard '*' in production"),
                description="Comma-separated list of allowed CORS origins",
                example="https://app.r3mes.network,https://www.r3mes.network",
                no_localhost_in_production=True
            ),
            
            # Faucet Configuration
            EnvVarRule(
                name="FAUCET_ENABLED",
                required=False,
                default="false",
                validator=self.validate_boolean,
                description="Enable faucet for test token distribution",
                example="false"
            ),
            EnvVarRule(
                name="FAUCET_KEY_NAME",
                required=False,
                description="Name of the faucet key in remesd keyring",
                example="faucet_key"
            ),
            EnvVarRule(
                name="REMESD_PATH",
                required=False,
                description="Path to remesd binary",
                example="/usr/local/bin/remesd"
            ),
            EnvVarRule(
                name="REMESD_HOME",
                required=False,
                description="Path to remesd home directory",
                example="/home/user/.remes"
            ),
            
            # Model Configuration
            EnvVarRule(
                name="R3MES_USE_MOCK_MODEL",
                required=False,
                default="false",
                validator=self.validate_boolean,
                description="Use mock model instead of real AI model (for testing)",
                example="false"
            ),
            EnvVarRule(
                name="BASE_MODEL_PATH",
                required=False,
                description="Path to base AI model",
                example="/models/bitnet_b1_58"
            ),
            EnvVarRule(
                name="IPFS_GATEWAY_URL",
                required_in_production=True,
                validator=lambda v: self.validate_url(v, allow_localhost=not self.is_production),
                description="IPFS gateway URL for model downloads",
                example="https://ipfs.io/ipfs/",
                no_localhost_in_production=True
            ),
            
            # Logging Configuration
            EnvVarRule(
                name="LOG_LEVEL",
                required=False,
                default="INFO",
                validator=lambda v: (True, None) if v in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL") else (False, "Must be DEBUG, INFO, WARNING, ERROR, or CRITICAL"),
                description="Logging level",
                example="INFO"
            ),
            EnvVarRule(
                name="LOG_FILE",
                required=False,
                description="Path to log file (optional, enables file logging)",
                example="/var/log/r3mes/backend.log"
            ),
            
            # Security Configuration
            EnvVarRule(
                name="API_KEY_SECRET",
                required_in_production=True,
                validator=lambda v: (True, None) if len(v) >= 32 else (False, "Must be at least 32 characters"),
                description="Secret key for API key hashing (must be at least 32 characters)",
                example="your-secret-key-here-min-32-chars"
            ),
            EnvVarRule(
                name="JWT_SECRET",
                required=False,
                validator=lambda v: (True, None) if len(v) >= 32 else (False, "Must be at least 32 characters"),
                description="Secret key for JWT signing (must be at least 32 characters)",
                example="your-jwt-secret-key-min-32-chars"
            ),
        ]
    
    def validate_environment_mode(self, value: str) -> Tuple[bool, Optional[str]]:
        """Validate R3MES_ENV value."""
        valid_modes = ("development", "staging", "production")
        if value.lower() not in valid_modes:
            return False, f"Must be one of: {', '.join(valid_modes)}"
        return True, None
    
    def validate_secret_management(self) -> Tuple[bool, List[str]]:
        """
        Validate secret management service configuration.
        
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        if not self.is_production:
            # Development mode doesn't require secret management
            return True, errors
        
        # In production, check if secret management service is configured
        has_gcp = bool(os.getenv("GOOGLE_CLOUD_PROJECT"))
        has_aws = bool(os.getenv("AWS_SECRETS_MANAGER_REGION") or os.getenv("AWS_DEFAULT_REGION"))
        has_vault = bool(os.getenv("VAULT_ADDR"))
        
        if not has_gcp and not has_aws and not has_vault:
            errors.append(
                "Secret management service required in production. "
                "Set GOOGLE_CLOUD_PROJECT, AWS_SECRETS_MANAGER_REGION, or VAULT_ADDR environment variable."
            )
            return False, errors
        
        # Test connection to secret management service
        try:
            from .secrets import get_secret_manager
            secret_manager = get_secret_manager()
            if not secret_manager.test_connection():
                errors.append(
                    f"Failed to connect to secret management service. "
                    f"Please verify configuration."
                )
                return False, errors
        except Exception as e:
            errors.append(
                f"Secret management service initialization failed: {e}"
            )
            return False, errors
        
        return True, errors
    
    def validate_all(self) -> Tuple[bool, List[str], List[str]]:
        """
        Validate all environment variables.
        
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        # Validate secret management first (in production)
        if self.is_production:
            secret_mgmt_valid, secret_mgmt_errors = self.validate_secret_management()
            if not secret_mgmt_valid:
                self.errors.extend(secret_mgmt_errors)
        
        rules = self.get_rules()
        
        for rule in rules:
            value = os.getenv(rule.name)
            
            # Check if required
            if rule.required or (rule.required_in_production and self.is_production):
                if not value:
                    if rule.default:
                        value = rule.default
                        self.warnings.append(f"{rule.name}: Using default value '{rule.default}'")
                    else:
                        self.errors.append(
                            f"{rule.name}: Required but not set. {rule.description}"
                        )
                        continue
            
            # Skip validation if not set and not required
            if not value:
                continue
            
            # Check localhost restriction in production
            if rule.no_localhost_in_production and self.is_production:
                if "localhost" in value or "127.0.0.1" in value:
                    self.errors.append(
                        f"{rule.name}: Cannot use localhost in production. {rule.description}"
                    )
                    continue
            
            # Run custom validator if provided
            if rule.validator:
                is_valid, error_msg = rule.validator(value)
                if not is_valid:
                    self.errors.append(
                        f"{rule.name}: {error_msg}. {rule.description}"
                    )
        
        return len(self.errors) == 0, self.errors, self.warnings
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get a detailed validation report."""
        is_valid, errors, warnings = self.validate_all()
        
        return {
            "valid": is_valid,
            "environment": os.getenv("R3MES_ENV", "development"),
            "is_production": self.is_production,
            "is_staging": self.is_staging,
            "errors": errors,
            "warnings": warnings,
            "error_count": len(errors),
            "warning_count": len(warnings),
        }
    
    def validate_and_raise(self):
        """
        Validate all environment variables and raise exceptions on errors.
        
        Raises:
            MissingEnvironmentVariableError: If required variables are missing
            InvalidEnvironmentVariableError: If variables have invalid values
            ProductionConfigurationError: If production configuration is invalid
        """
        is_valid, errors, warnings = self.validate_all()
        
        # Log warnings
        for warning in warnings:
            logger.warning(warning)
        
        # Raise exceptions for errors
        if not is_valid:
            missing_vars = [e for e in errors if "Required but not set" in e]
            invalid_vars = [e for e in errors if "Required but not set" not in e]
            
            if missing_vars:
                var_names = [e.split(":")[0] for e in missing_vars]
                raise MissingEnvironmentVariableError(
                    f"Missing required environment variables: {', '.join(var_names)}"
                )
            
            if invalid_vars:
                if self.is_production:
                    raise ProductionConfigurationError(
                        f"Invalid production configuration:\n" + "\n".join(invalid_vars)
                    )
                else:
                    raise InvalidEnvironmentVariableError(
                        f"Invalid environment variables:\n" + "\n".join(invalid_vars)
                    )


# Global validator instance
_validator: Optional[EnvironmentValidator] = None


def get_env_validator() -> EnvironmentValidator:
    """Get or create global environment validator."""
    global _validator
    if _validator is None:
        _validator = EnvironmentValidator()
    return _validator


def validate_environment() -> None:
    """
    Validate all environment variables on startup.
    
    Raises exceptions if validation fails.
    """
    validator = get_env_validator()
    validator.validate_and_raise()


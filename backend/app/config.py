"""
Centralized Configuration Management for R3MES Backend

Production-ready configuration with Vault integration and environment validation.
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class ProductionConfigurationError(Exception):
    """Raised when production configuration is invalid."""
    pass


class MissingEnvironmentVariableError(Exception):
    """Raised when required environment variable is missing."""
    pass


class VaultSecretManager:
    """HashiCorp Vault integration for secret management."""
    
    def __init__(self, vault_addr: Optional[str] = None, vault_token: Optional[str] = None):
        self.vault_addr = vault_addr or os.getenv("VAULT_ADDR")
        self.vault_token = vault_token or os.getenv("VAULT_TOKEN")
        self._client = None
        
    def get_secret(self, path: str) -> Optional[str]:
        """Get secret from Vault."""
        if not self.vault_addr or not self.vault_token:
            logger.warning("Vault not configured, skipping secret retrieval")
            return None
            
        try:
            import hvac
            if not self._client:
                self._client = hvac.Client(url=self.vault_addr, token=self.vault_token)
                
            response = self._client.secrets.kv.v2.read_secret_version(path=path)
            return response['data']['data'].get('value')
        except ImportError:
            logger.warning("hvac library not installed, cannot use Vault")
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve secret from Vault: {e}")
            return None


class R3MESConfig(BaseSettings):
    """
    R3MES Configuration with production-ready validation.
    
    Supports multiple configuration sources:
    1. Environment variables
    2. HashiCorp Vault (production)
    3. .env files (development)
    """
    
    # Environment Configuration
    ENV: str = Field(default="development", env="R3MES_ENV")
    DEBUG: bool = Field(default=False)
    
    # Network Configuration
    HOST: str = Field(default="0.0.0.0", env="R3MES_HOST")
    PORT: int = Field(default=8000, env="R3MES_PORT")
    
    # Blockchain Configuration
    CHAIN_ID: str = Field(..., env="CHAIN_ID")
    RPC_URL: str = Field(..., env="RPC_URL")
    REST_URL: str = Field(..., env="REST_URL")
    BLOCKCHAIN_RPC_URL: Optional[str] = Field(None, env="BLOCKCHAIN_RPC_URL")
    
    # Database Configuration
    DATABASE_URL: str = Field(..., env="DATABASE_URL")
    DATABASE_TIMEOUT: int = Field(default=30, env="BACKEND_DATABASE_TIMEOUT")
    
    # Security Configuration
    JWT_SECRET: str = Field(..., env="JWT_SECRET")
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_EXPIRATION_HOURS: int = Field(default=24)
    
    # Redis Configuration
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    REDIS_PASSWORD: Optional[str] = Field(None, env="REDIS_PASSWORD")
    
    # Model Configuration
    MODEL_PATH: str = Field(..., env="MODEL_PATH")
    IPFS_NODE: str = Field(..., env="IPFS_NODE")
    
    # Secrets Management
    MNEMONIC: Optional[str] = Field(None, env="MNEMONIC")
    
    # Vault Configuration
    VAULT_ADDR: Optional[str] = Field(None, env="VAULT_ADDR")
    VAULT_TOKEN: Optional[str] = Field(None, env="VAULT_TOKEN")
    
    # Monitoring Configuration
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    METRICS_ENABLED: bool = Field(default=True, env="METRICS_ENABLED")
    PROMETHEUS_PORT: int = Field(default=9090, env="PROMETHEUS_PORT")
    
    # Performance Configuration
    WORKER_PROCESSES: int = Field(default=1, env="WORKER_PROCESSES")
    MAX_CONNECTIONS: int = Field(default=100, env="MAX_CONNECTIONS")
    CACHE_TTL: int = Field(default=300, env="CACHE_TTL")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="allow"  # Allow extra fields for test environment
    )
        
    @field_validator("ENV")
    @classmethod
    def validate_environment(cls, v):
        """Validate environment setting."""
        valid_envs = ["development", "testing", "staging", "production"]
        if v not in valid_envs:
            raise ValueError(f"ENV must be one of {valid_envs}")
        return v
    
    @field_validator("DEBUG", mode="before")
    @classmethod
    def set_debug_mode(cls, v, info):
        """Set debug mode based on environment."""
        env = info.data.get("ENV", "development") if info.data else "development"
        if env == "production":
            return False
        return v if v is not None else (env == "development")
    
    @field_validator("RPC_URL", "REST_URL", "BLOCKCHAIN_RPC_URL")
    @classmethod
    def validate_no_localhost_in_production(cls, v, info):
        """Ensure no localhost URLs in production."""
        if not v:
            return v
            
        env = info.data.get("ENV", "development") if info.data else "development"
        if env == "production":
            localhost_indicators = ["localhost", "127.0.0.1", "::1", "0.0.0.0"]
            if any(indicator in v.lower() for indicator in localhost_indicators):
                from .exceptions import ProductionConfigurationError
                raise ProductionConfigurationError(
                    f"URL cannot use localhost in production: {v}"
                )
        return v
    
    @field_validator("JWT_SECRET")
    @classmethod
    def validate_jwt_secret(cls, v, info):
        """Validate JWT secret strength."""
        env = info.data.get("ENV", "development") if info.data else "development"
        if env == "production":
            if len(v) < 32:
                from .exceptions import ProductionConfigurationError
                raise ProductionConfigurationError(
                    "JWT_SECRET must be at least 32 characters in production"
                )
            if v in ["your-secret-key", "change-me", "jwt-secret"]:
                from .exceptions import ProductionConfigurationError
                raise ProductionConfigurationError(
                    "JWT_SECRET cannot use default/example values in production"
                )
        return v
    
    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, v, info):
        """Validate database URL."""
        env = info.data.get("ENV", "development") if info.data else "development"
        if env == "production":
            if "localhost" in v or "127.0.0.1" in v:
                from .exceptions import ProductionConfigurationError
                raise ProductionConfigurationError(
                    f"DATABASE_URL cannot use localhost in production: {v}"
                )
            if "password123" in v or "admin" in v:
                from .exceptions import ProductionConfigurationError
                raise ProductionConfigurationError(
                    "DATABASE_URL contains weak credentials"
                )
        return v
    
    @field_validator("MNEMONIC")
    @classmethod
    def validate_mnemonic(cls, v, info):
        """Validate mnemonic phrase."""
        if not v:
            return v
            
        env = info.data.get("ENV", "development") if info.data else "development"
        if env == "production":
            # Check for example/test mnemonics
            test_words = ["test", "example", "sample", "demo"]
            if any(word in v.lower() for word in test_words):
                from .exceptions import ProductionConfigurationError
                raise ProductionConfigurationError(
                    "MNEMONIC cannot contain test/example words in production"
                )
        return v
    
    def load_secrets_from_vault(self) -> None:
        """Load secrets from HashiCorp Vault in production."""
        if self.ENV != "production" or not self.VAULT_ADDR:
            return
            
        vault = VaultSecretManager(self.VAULT_ADDR, self.VAULT_TOKEN)
        
        # Load secrets from Vault
        secret_mappings = {
            "JWT_SECRET": "secret/r3mes/production/jwt_secret",
            "DATABASE_URL": "secret/r3mes/production/database_url",
            "REDIS_PASSWORD": "secret/r3mes/production/redis_password",
            "MNEMONIC": "secret/r3mes/production/mnemonic",
        }
        
        for attr_name, vault_path in secret_mappings.items():
            secret_value = vault.get_secret(vault_path)
            if secret_value:
                setattr(self, attr_name, secret_value)
                logger.info(f"Loaded {attr_name} from Vault")
            else:
                logger.warning(f"Failed to load {attr_name} from Vault: {vault_path}")
    
    def validate_production_requirements(self) -> None:
        """Validate all production requirements are met."""
        if self.ENV != "production":
            return
            
        required_secrets = ["JWT_SECRET", "DATABASE_URL"]
        missing_secrets = []
        
        for secret in required_secrets:
            value = getattr(self, secret, None)
            if not value:
                missing_secrets.append(secret)
        
        if missing_secrets:
            raise MissingEnvironmentVariableError(
                f"Missing required production secrets: {missing_secrets}"
            )
        
        # Validate network configuration
        if not self.BLOCKCHAIN_RPC_URL:
            raise MissingEnvironmentVariableError(
                "BLOCKCHAIN_RPC_URL is required in production"
            )
        
        logger.info("✅ Production configuration validation passed")


# Global configuration instance
_config: Optional[R3MESConfig] = None


def get_config() -> R3MESConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = R3MESConfig()
        
        # Load secrets from Vault in production
        _config.load_secrets_from_vault()
        
        # Validate production requirements
        _config.validate_production_requirements()
        
        logger.info(f"Configuration loaded for environment: {_config.ENV}")
    
    return _config


def reload_config() -> R3MESConfig:
    """Reload configuration (useful for testing)."""
    global _config
    _config = None
    return get_config()


# Configuration validation utilities
def is_production() -> bool:
    """Check if running in production environment."""
    return get_config().ENV == "production"


def is_development() -> bool:
    """Check if running in development environment."""
    return get_config().ENV == "development"


def validate_localhost_usage(url: str, context: str = "") -> None:
    """Validate that localhost is not used in production."""
    if is_production():
        localhost_indicators = ["localhost", "127.0.0.1", "::1", "0.0.0.0"]
        if any(indicator in url.lower() for indicator in localhost_indicators):
            raise ProductionConfigurationError(
                f"Cannot use localhost in production {context}: {url}"
            )
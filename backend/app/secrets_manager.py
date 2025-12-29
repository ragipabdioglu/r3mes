"""
R3MES Secrets Manager

Unified secrets management supporting multiple backends:
- Docker Secrets (file-based)
- Docker Swarm Secrets
- Environment Variables
- HashiCorp Vault (optional)
- AWS Secrets Manager (optional)
- Kubernetes Secrets (optional)

Usage:
    from app.secrets_manager import get_secrets_manager, SecretKey
    
    secrets = get_secrets_manager()
    db_password = secrets.get(SecretKey.POSTGRES_PASSWORD)
"""

import os
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Dict, Any
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


class SecretKey(Enum):
    """Enumeration of all secret keys used in R3MES."""
    
    # Database
    POSTGRES_PASSWORD = "postgres_password"
    POSTGRES_USER = "postgres_user"
    
    # Cache
    REDIS_PASSWORD = "redis_password"
    
    # Authentication
    JWT_SECRET = "jwt_secret"
    API_SECRET_KEY = "api_secret_key"
    
    # Monitoring
    GRAFANA_ADMIN_PASSWORD = "grafana_admin_password"
    
    # Blockchain
    VALIDATOR_KEY = "validator_key"
    NODE_KEY = "node_key"
    
    # External Services
    SENTRY_DSN = "sentry_dsn"
    SLACK_WEBHOOK_URL = "slack_webhook_url"
    
    # SSL
    SSL_CERTIFICATE = "ssl_certificate"
    SSL_PRIVATE_KEY = "ssl_private_key"


class SecretsBackend(ABC):
    """Abstract base class for secrets backends."""
    
    @abstractmethod
    def get_secret(self, key: str) -> Optional[str]:
        """Retrieve a secret by key."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name for logging."""
        pass


class DockerSecretsBackend(SecretsBackend):
    """
    Docker Secrets backend.
    Reads secrets from /run/secrets/ directory (Docker Compose/Swarm).
    """
    
    SECRETS_PATH = Path("/run/secrets")
    
    @property
    def name(self) -> str:
        return "Docker Secrets"
    
    def is_available(self) -> bool:
        return self.SECRETS_PATH.exists() and self.SECRETS_PATH.is_dir()
    
    def get_secret(self, key: str) -> Optional[str]:
        secret_file = self.SECRETS_PATH / key
        if secret_file.exists():
            try:
                return secret_file.read_text().strip()
            except Exception as e:
                logger.warning(f"Failed to read Docker secret {key}: {e}")
        return None


class FileSecretsBackend(SecretsBackend):
    """
    File-based secrets backend.
    Reads secrets from _FILE environment variables pointing to files.
    """
    
    @property
    def name(self) -> str:
        return "File Secrets"
    
    def is_available(self) -> bool:
        # Always available as fallback
        return True
    
    def get_secret(self, key: str) -> Optional[str]:
        # Check for _FILE environment variable
        file_env_key = f"{key.upper()}_FILE"
        secret_file_path = os.environ.get(file_env_key)
        
        if secret_file_path:
            try:
                return Path(secret_file_path).read_text().strip()
            except Exception as e:
                logger.warning(f"Failed to read secret file {secret_file_path}: {e}")
        
        return None


class EnvironmentSecretsBackend(SecretsBackend):
    """
    Environment variables backend.
    Reads secrets directly from environment variables.
    """
    
    @property
    def name(self) -> str:
        return "Environment Variables"
    
    def is_available(self) -> bool:
        return True
    
    def get_secret(self, key: str) -> Optional[str]:
        return os.environ.get(key.upper())


class VaultSecretsBackend(SecretsBackend):
    """
    HashiCorp Vault backend.
    Requires hvac library and VAULT_ADDR, VAULT_TOKEN environment variables.
    """
    
    def __init__(self):
        self._client = None
        self._mount_point = os.environ.get("VAULT_MOUNT_POINT", "secret")
        self._path_prefix = os.environ.get("VAULT_PATH_PREFIX", "r3mes")
    
    @property
    def name(self) -> str:
        return "HashiCorp Vault"
    
    def is_available(self) -> bool:
        if not os.environ.get("VAULT_ADDR"):
            return False
        
        try:
            import hvac
            self._client = hvac.Client(
                url=os.environ.get("VAULT_ADDR"),
                token=os.environ.get("VAULT_TOKEN"),
            )
            return self._client.is_authenticated()
        except ImportError:
            logger.debug("hvac library not installed, Vault backend unavailable")
            return False
        except Exception as e:
            logger.warning(f"Vault connection failed: {e}")
            return False
    
    def get_secret(self, key: str) -> Optional[str]:
        if not self._client:
            return None
        
        try:
            path = f"{self._path_prefix}/{key}"
            response = self._client.secrets.kv.v2.read_secret_version(
                path=path,
                mount_point=self._mount_point,
            )
            return response["data"]["data"].get("value")
        except Exception as e:
            logger.warning(f"Failed to read Vault secret {key}: {e}")
            return None


class AWSSecretsBackend(SecretsBackend):
    """
    AWS Secrets Manager backend.
    Requires boto3 library and AWS credentials.
    """
    
    def __init__(self):
        self._client = None
        self._prefix = os.environ.get("AWS_SECRETS_PREFIX", "r3mes")
        self._region = os.environ.get("AWS_REGION", "us-east-1")
    
    @property
    def name(self) -> str:
        return "AWS Secrets Manager"
    
    def is_available(self) -> bool:
        if not os.environ.get("AWS_SECRETS_MANAGER_ENABLED", "").lower() == "true":
            return False
        
        try:
            import boto3
            self._client = boto3.client("secretsmanager", region_name=self._region)
            return True
        except ImportError:
            logger.debug("boto3 library not installed, AWS Secrets Manager unavailable")
            return False
        except Exception as e:
            logger.warning(f"AWS Secrets Manager connection failed: {e}")
            return False
    
    def get_secret(self, key: str) -> Optional[str]:
        if not self._client:
            return None
        
        try:
            secret_name = f"{self._prefix}/{key}"
            response = self._client.get_secret_value(SecretId=secret_name)
            return response.get("SecretString")
        except Exception as e:
            logger.warning(f"Failed to read AWS secret {key}: {e}")
            return None


class KubernetesSecretsBackend(SecretsBackend):
    """
    Kubernetes Secrets backend.
    Reads secrets mounted as files in /var/run/secrets/kubernetes.io/serviceaccount/
    or custom mount paths.
    """
    
    def __init__(self):
        self._secrets_path = Path(
            os.environ.get("K8S_SECRETS_PATH", "/var/run/secrets/r3mes")
        )
    
    @property
    def name(self) -> str:
        return "Kubernetes Secrets"
    
    def is_available(self) -> bool:
        return self._secrets_path.exists() and self._secrets_path.is_dir()
    
    def get_secret(self, key: str) -> Optional[str]:
        secret_file = self._secrets_path / key
        if secret_file.exists():
            try:
                return secret_file.read_text().strip()
            except Exception as e:
                logger.warning(f"Failed to read K8s secret {key}: {e}")
        return None


class SecretsManager:
    """
    Unified secrets manager that tries multiple backends in order.
    
    Priority order:
    1. Docker Secrets (/run/secrets/)
    2. Kubernetes Secrets
    3. HashiCorp Vault
    4. AWS Secrets Manager
    5. File-based secrets (_FILE env vars)
    6. Environment variables (fallback)
    """
    
    def __init__(self):
        self._backends: list[SecretsBackend] = []
        self._cache: Dict[str, str] = {}
        self._cache_enabled = os.environ.get("SECRETS_CACHE_ENABLED", "true").lower() == "true"
        
        # Initialize backends in priority order
        self._init_backends()
    
    def _init_backends(self):
        """Initialize available backends."""
        backends_to_try = [
            DockerSecretsBackend(),
            KubernetesSecretsBackend(),
            VaultSecretsBackend(),
            AWSSecretsBackend(),
            FileSecretsBackend(),
            EnvironmentSecretsBackend(),
        ]
        
        for backend in backends_to_try:
            if backend.is_available():
                self._backends.append(backend)
                logger.info(f"Secrets backend available: {backend.name}")
        
        if not self._backends:
            logger.warning("No secrets backends available!")
    
    def get(self, key: SecretKey, default: Optional[str] = None) -> Optional[str]:
        """
        Get a secret value.
        
        Args:
            key: SecretKey enum value
            default: Default value if secret not found
            
        Returns:
            Secret value or default
        """
        key_str = key.value
        
        # Check cache first
        if self._cache_enabled and key_str in self._cache:
            return self._cache[key_str]
        
        # Try each backend
        for backend in self._backends:
            value = backend.get_secret(key_str)
            if value is not None:
                logger.debug(f"Secret {key_str} found in {backend.name}")
                if self._cache_enabled:
                    self._cache[key_str] = value
                return value
        
        logger.debug(f"Secret {key_str} not found, using default")
        return default
    
    def get_raw(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a secret by raw string key (not SecretKey enum).
        
        Args:
            key: String key name
            default: Default value if secret not found
            
        Returns:
            Secret value or default
        """
        # Check cache first
        if self._cache_enabled and key in self._cache:
            return self._cache[key]
        
        # Try each backend
        for backend in self._backends:
            value = backend.get_secret(key)
            if value is not None:
                if self._cache_enabled:
                    self._cache[key] = value
                return value
        
        return default
    
    def clear_cache(self):
        """Clear the secrets cache."""
        self._cache.clear()
        logger.info("Secrets cache cleared")
    
    def get_database_url(self) -> str:
        """
        Build PostgreSQL database URL from secrets.
        
        Returns:
            PostgreSQL connection URL
        """
        user = self.get(SecretKey.POSTGRES_USER) or os.environ.get("POSTGRES_USER", "r3mes")
        password = self.get(SecretKey.POSTGRES_PASSWORD) or os.environ.get("POSTGRES_PASSWORD", "")
        host = os.environ.get("POSTGRES_HOST", "localhost")
        port = os.environ.get("POSTGRES_PORT", "5432")
        db = os.environ.get("POSTGRES_DB", "r3mes")
        
        if not password:
            logger.warning("PostgreSQL password not found in secrets!")
        
        return f"postgresql://{user}:{password}@{host}:{port}/{db}"
    
    def get_redis_url(self) -> str:
        """
        Build Redis URL from secrets.
        
        Returns:
            Redis connection URL
        """
        password = self.get(SecretKey.REDIS_PASSWORD) or os.environ.get("REDIS_PASSWORD", "")
        host = os.environ.get("REDIS_HOST", "localhost")
        port = os.environ.get("REDIS_PORT", "6379")
        db = os.environ.get("REDIS_DB", "0")
        
        if password:
            return f"redis://:{password}@{host}:{port}/{db}"
        return f"redis://{host}:{port}/{db}"
    
    @property
    def available_backends(self) -> list[str]:
        """Get list of available backend names."""
        return [b.name for b in self._backends]


# Singleton instance
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager() -> SecretsManager:
    """
    Get the singleton SecretsManager instance.
    
    Returns:
        SecretsManager instance
    """
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager


# Convenience functions
def get_secret(key: SecretKey, default: Optional[str] = None) -> Optional[str]:
    """Convenience function to get a secret."""
    return get_secrets_manager().get(key, default)


def get_database_url() -> str:
    """Convenience function to get database URL."""
    return get_secrets_manager().get_database_url()


def get_redis_url() -> str:
    """Convenience function to get Redis URL."""
    return get_secrets_manager().get_redis_url()

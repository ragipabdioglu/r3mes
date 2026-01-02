"""
HashiCorp Vault Client - Production-ready Vault integration

Provides secure secrets management using HashiCorp Vault with:
- Automatic authentication
- Connection pooling
- Error handling and retries
- Secret caching with TTL
- Health checks
"""

import os
import time
import logging
import asyncio
from typing import Dict, Any, Optional, Union
from functools import lru_cache
from dataclasses import dataclass
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


@dataclass
class VaultConfig:
    """Vault configuration."""
    url: str
    token: Optional[str] = None
    mount_point: str = "secret"
    path_prefix: str = "r3mes"
    timeout: int = 30
    max_retries: int = 3
    cache_ttl: int = 300  # 5 minutes
    verify_ssl: bool = True


class VaultError(Exception):
    """Base Vault error."""
    pass


class VaultConnectionError(VaultError):
    """Vault connection error."""
    pass


class VaultSecretNotFound(VaultError):
    """Secret not found in Vault."""
    pass


class VaultAuthenticationError(VaultError):
    """Vault authentication error."""
    pass


class VaultClient:
    """
    Production-ready HashiCorp Vault client.
    
    Features:
    - Automatic token authentication
    - Connection health checks
    - Secret caching with TTL
    - Retry logic with exponential backoff
    - Secure error handling
    """
    
    def __init__(self, config: Optional[VaultConfig] = None):
        """
        Initialize Vault client.
        
        Args:
            config: Vault configuration (defaults to environment variables)
        """
        self.config = config or self._load_config_from_env()
        self._client = None
        self._cache: Dict[str, tuple] = {}  # {path: (value, timestamp)}
        self._initialized = False
        
        # Initialize hvac client
        self._init_client()
    
    def _load_config_from_env(self) -> VaultConfig:
        """Load Vault configuration from environment variables."""
        vault_url = os.getenv("VAULT_ADDR")
        if not vault_url:
            raise VaultConnectionError("VAULT_ADDR environment variable must be set")
        
        vault_token = os.getenv("VAULT_TOKEN")
        if not vault_token:
            raise VaultAuthenticationError("VAULT_TOKEN environment variable must be set")
        
        return VaultConfig(
            url=vault_url,
            token=vault_token,
            mount_point=os.getenv("VAULT_MOUNT_POINT", "secret"),
            path_prefix=os.getenv("VAULT_PATH_PREFIX", "r3mes"),
            timeout=int(os.getenv("VAULT_TIMEOUT", "30")),
            max_retries=int(os.getenv("VAULT_MAX_RETRIES", "3")),
            cache_ttl=int(os.getenv("VAULT_CACHE_TTL", "300")),
            verify_ssl=os.getenv("VAULT_VERIFY_SSL", "true").lower() == "true"
        )
    
    def _init_client(self):
        """Initialize hvac client."""
        try:
            import hvac
            
            self._client = hvac.Client(
                url=self.config.url,
                token=self.config.token,
                timeout=self.config.timeout,
                verify=self.config.verify_ssl
            )
            
            logger.info(f"Vault client initialized (url: {self.config.url})")
            
        except ImportError:
            raise ImportError("hvac library is required. Install with: pip install hvac")
        except Exception as e:
            raise VaultConnectionError(f"Failed to initialize Vault client: {e}")
    
    async def initialize(self) -> None:
        """Initialize and authenticate with Vault."""
        if self._initialized:
            return
        
        try:
            # Test authentication
            if not self._client.is_authenticated():
                raise VaultAuthenticationError("Vault authentication failed")
            
            # Test basic connectivity
            await self._test_connectivity()
            
            self._initialized = True
            logger.info("✅ Vault client initialized and authenticated")
            
        except Exception as e:
            logger.error(f"❌ Vault initialization failed: {e}")
            raise VaultConnectionError(f"Vault initialization failed: {e}")
    
    async def _test_connectivity(self) -> None:
        """Test Vault connectivity."""
        try:
            # Try to read sys/health endpoint
            health = self._client.sys.read_health_status()
            if not health.get('initialized', False):
                raise VaultConnectionError("Vault is not initialized")
            
            if health.get('sealed', True):
                raise VaultConnectionError("Vault is sealed")
                
        except Exception as e:
            raise VaultConnectionError(f"Vault connectivity test failed: {e}")
    
    def _get_cache_key(self, path: str) -> str:
        """Generate cache key for secret path."""
        return f"{self.config.mount_point}/{self.config.path_prefix}/{path}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached value is still valid."""
        if cache_key not in self._cache:
            return False
        
        _, timestamp = self._cache[cache_key]
        return time.time() - timestamp < self.config.cache_ttl
    
    async def get_secret(self, path: str, key: Optional[str] = None) -> Union[str, Dict[str, Any]]:
        """
        Get secret from Vault.
        
        Args:
            path: Secret path (e.g., 'database/credentials')
            key: Specific key within secret (optional)
            
        Returns:
            Secret value or dictionary of all secret data
            
        Raises:
            VaultSecretNotFound: If secret doesn't exist
            VaultConnectionError: If connection fails
        """
        if not self._initialized:
            await self.initialize()
        
        cache_key = self._get_cache_key(path)
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            cached_data, _ = self._cache[cache_key]
            if key:
                if key not in cached_data:
                    raise VaultSecretNotFound(f"Key '{key}' not found in secret '{path}'")
                return cached_data[key]
            return cached_data
        
        # Fetch from Vault with retry logic
        secret_data = await self._fetch_secret_with_retry(path)
        
        # Cache the result
        self._cache[cache_key] = (secret_data, time.time())
        
        if key:
            if key not in secret_data:
                raise VaultSecretNotFound(f"Key '{key}' not found in secret '{path}'")
            return secret_data[key]
        
        return secret_data
    
    async def _fetch_secret_with_retry(self, path: str) -> Dict[str, Any]:
        """Fetch secret from Vault with retry logic."""
        full_path = f"{self.config.path_prefix}/{path}"
        
        for attempt in range(self.config.max_retries):
            try:
                # Use KV v2 engine
                response = self._client.secrets.kv.v2.read_secret_version(
                    path=full_path,
                    mount_point=self.config.mount_point
                )
                
                if not response or 'data' not in response:
                    raise VaultSecretNotFound(f"Secret not found: {path}")
                
                secret_data = response['data'].get('data', {})
                if not secret_data:
                    raise VaultSecretNotFound(f"Secret data is empty: {path}")
                
                return secret_data
                
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    if "not found" in str(e).lower():
                        raise VaultSecretNotFound(f"Secret not found: {path}")
                    raise VaultConnectionError(f"Failed to fetch secret '{path}': {e}")
                
                # Exponential backoff
                wait_time = 2 ** attempt
                logger.warning(f"Vault request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
        
        raise VaultConnectionError(f"Failed to fetch secret after {self.config.max_retries} attempts")
    
    async def put_secret(self, path: str, secret_data: Dict[str, Any]) -> None:
        """
        Store secret in Vault.
        
        Args:
            path: Secret path
            secret_data: Dictionary of secret key-value pairs
            
        Raises:
            VaultConnectionError: If operation fails
        """
        if not self._initialized:
            await self.initialize()
        
        full_path = f"{self.config.path_prefix}/{path}"
        
        try:
            self._client.secrets.kv.v2.create_or_update_secret(
                path=full_path,
                secret=secret_data,
                mount_point=self.config.mount_point
            )
            
            # Invalidate cache
            cache_key = self._get_cache_key(path)
            if cache_key in self._cache:
                del self._cache[cache_key]
            
            logger.info(f"Secret stored successfully: {path}")
            
        except Exception as e:
            raise VaultConnectionError(f"Failed to store secret '{path}': {e}")
    
    async def delete_secret(self, path: str) -> None:
        """
        Delete secret from Vault.
        
        Args:
            path: Secret path
            
        Raises:
            VaultConnectionError: If operation fails
        """
        if not self._initialized:
            await self.initialize()
        
        full_path = f"{self.config.path_prefix}/{path}"
        
        try:
            self._client.secrets.kv.v2.delete_metadata_and_all_versions(
                path=full_path,
                mount_point=self.config.mount_point
            )
            
            # Invalidate cache
            cache_key = self._get_cache_key(path)
            if cache_key in self._cache:
                del self._cache[cache_key]
            
            logger.info(f"Secret deleted successfully: {path}")
            
        except Exception as e:
            raise VaultConnectionError(f"Failed to delete secret '{path}': {e}")
    
    async def list_secrets(self, path: str = "") -> list[str]:
        """
        List secrets under a path.
        
        Args:
            path: Path prefix to list (optional)
            
        Returns:
            List of secret paths
            
        Raises:
            VaultConnectionError: If operation fails
        """
        if not self._initialized:
            await self.initialize()
        
        full_path = f"{self.config.path_prefix}/{path}" if path else self.config.path_prefix
        
        try:
            response = self._client.secrets.kv.v2.list_secrets(
                path=full_path,
                mount_point=self.config.mount_point
            )
            
            return response.get('data', {}).get('keys', [])
            
        except Exception as e:
            if "not found" in str(e).lower():
                return []
            raise VaultConnectionError(f"Failed to list secrets at '{path}': {e}")
    
    def clear_cache(self) -> None:
        """Clear the secret cache."""
        self._cache.clear()
        logger.info("Vault secret cache cleared")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform Vault health check.
        
        Returns:
            Health status information
        """
        try:
            if not self._initialized:
                return {"status": "not_initialized", "healthy": False}
            
            # Check authentication
            if not self._client.is_authenticated():
                return {"status": "authentication_failed", "healthy": False}
            
            # Check Vault health
            health = self._client.sys.read_health_status()
            
            return {
                "status": "healthy",
                "healthy": True,
                "initialized": health.get('initialized', False),
                "sealed": health.get('sealed', True),
                "standby": health.get('standby', False),
                "server_time_utc": health.get('server_time_utc'),
                "version": health.get('version'),
                "cache_size": len(self._cache)
            }
            
        except Exception as e:
            logger.error(f"Vault health check failed: {e}")
            return {
                "status": "unhealthy",
                "healthy": False,
                "error": str(e)
            }


# Global Vault client instance
_vault_client: Optional[VaultClient] = None


def get_vault_client() -> VaultClient:
    """
    Get or create global Vault client instance.
    
    Returns:
        VaultClient instance
    """
    global _vault_client
    
    if _vault_client is None:
        _vault_client = VaultClient()
    
    return _vault_client


async def initialize_vault() -> VaultClient:
    """
    Initialize Vault client and test connectivity.
    
    Returns:
        Initialized VaultClient instance
        
    Raises:
        VaultConnectionError: If initialization fails
    """
    vault_client = get_vault_client()
    await vault_client.initialize()
    return vault_client


# Convenience functions
async def get_secret(path: str, key: Optional[str] = None) -> Union[str, Dict[str, Any]]:
    """Convenience function to get a secret."""
    vault_client = get_vault_client()
    return await vault_client.get_secret(path, key)


async def put_secret(path: str, secret_data: Dict[str, Any]) -> None:
    """Convenience function to store a secret."""
    vault_client = get_vault_client()
    await vault_client.put_secret(path, secret_data)


async def vault_health_check() -> Dict[str, Any]:
    """Convenience function for health check."""
    vault_client = get_vault_client()
    return await vault_client.health_check()
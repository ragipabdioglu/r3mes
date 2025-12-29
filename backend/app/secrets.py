"""
Secret Management Service Integration

Provides unified interface for accessing secrets from various secret management services:
- Google Cloud Secret Manager
- AWS Secrets Manager
- HashiCorp Vault
- Environment Variables (fallback for development)

All secrets are cached with TTL to reduce API calls.
"""

import os
import json
import logging
import time
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from functools import lru_cache

logger = logging.getLogger(__name__)


class SecretManager(ABC):
    """Base class for secret management providers."""
    
    @abstractmethod
    def get_secret(self, secret_name: str) -> str:
        """Get a secret value by name."""
        pass
    
    @abstractmethod
    def get_secrets(self, secret_path: str) -> Dict[str, Any]:
        """Get multiple secrets from a path."""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test connection to secret management service."""
        pass


class AWSSecretsManager(SecretManager):
    """AWS Secrets Manager implementation."""
    
    def __init__(self, region: Optional[str] = None):
        """
        Initialize AWS Secrets Manager client.
        
        Args:
            region: AWS region (defaults to AWS_DEFAULT_REGION env var)
        """
        try:
            import boto3
            self.region = region or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
            self.client = boto3.client('secretsmanager', region_name=self.region)
            self._cache: Dict[str, tuple] = {}  # {secret_name: (value, timestamp)}
            self._cache_ttl = 300  # 5 minutes
            logger.info(f"AWS Secrets Manager initialized (region: {self.region})")
        except ImportError:
            raise ImportError("boto3 is required for AWS Secrets Manager. Install with: pip install boto3")
        except Exception as e:
            logger.error(f"Failed to initialize AWS Secrets Manager: {e}")
            raise
    
    def get_secret(self, secret_name: str) -> str:
        """
        Get a secret value from AWS Secrets Manager.
        
        Args:
            secret_name: Secret name or ARN
            
        Returns:
            Secret value as string
        """
        # Check cache first
        if secret_name in self._cache:
            value, timestamp = self._cache[secret_name]
            if time.time() - timestamp < self._cache_ttl:
                return value
        
        try:
            response = self.client.get_secret_value(SecretId=secret_name)
            secret_string = response.get('SecretString', '')
            
            # Try to parse as JSON (AWS Secrets Manager stores JSON)
            try:
                secret_dict = json.loads(secret_string)
                # If it's a dict, try to get the value by the secret name key
                if isinstance(secret_dict, dict):
                    # Common pattern: secret name is the key
                    if secret_name.split('/')[-1] in secret_dict:
                        value = secret_dict[secret_name.split('/')[-1]]
                    # Or use the first value if only one key
                    elif len(secret_dict) == 1:
                        value = list(secret_dict.values())[0]
                    else:
                        # Return the whole JSON string if multiple keys
                        value = secret_string
                else:
                    value = secret_string
            except json.JSONDecodeError:
                # Not JSON, return as-is
                value = secret_string
            
            # Cache the value
            self._cache[secret_name] = (value, time.time())
            return value
            
        except self.client.exceptions.ResourceNotFoundException:
            logger.error(f"Secret not found: {secret_name}")
            raise ValueError(f"Secret {secret_name} not found in AWS Secrets Manager")
        except Exception as e:
            logger.error(f"Failed to get secret {secret_name} from AWS Secrets Manager: {e}")
            raise
    
    def get_secrets(self, secret_path: str) -> Dict[str, Any]:
        """
        Get multiple secrets from a path (all secrets under a prefix).
        
        Args:
            secret_path: Secret path prefix (e.g., 'r3mes/production')
            
        Returns:
            Dictionary of secret names to values
        """
        try:
            # List secrets with the prefix
            paginator = self.client.get_paginator('list_secrets')
            secrets = {}
            
            for page in paginator.paginate(
                Filters=[{'Key': 'name', 'Values': [secret_path]}]
            ):
                for secret in page.get('SecretList', []):
                    secret_name = secret['Name']
                    try:
                        secrets[secret_name] = self.get_secret(secret_name)
                    except Exception as e:
                        logger.warning(f"Failed to get secret {secret_name}: {e}")
            
            return secrets
            
        except Exception as e:
            logger.error(f"Failed to list secrets from path {secret_path}: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test connection to AWS Secrets Manager."""
        try:
            # Try to list secrets (minimal operation)
            self.client.list_secrets(MaxResults=1)
            return True
        except Exception as e:
            logger.error(f"AWS Secrets Manager connection test failed: {e}")
            return False


class HashiCorpVault(SecretManager):
    """HashiCorp Vault implementation."""
    
    def __init__(self, vault_url: Optional[str] = None, vault_token: Optional[str] = None):
        """
        Initialize HashiCorp Vault client.
        
        Args:
            vault_url: Vault server URL (defaults to VAULT_ADDR env var)
            vault_token: Vault token (defaults to VAULT_TOKEN env var)
        """
        try:
            import hvac
            self.vault_url = vault_url or os.getenv("VAULT_ADDR", "http://localhost:8200")
            self.vault_token = vault_token or os.getenv("VAULT_TOKEN")
            
            if not self.vault_token:
                raise ValueError("VAULT_TOKEN environment variable must be set for HashiCorp Vault")
            
            self.client = hvac.Client(url=self.vault_url, token=self.vault_token)
            self._cache: Dict[str, tuple] = {}
            self._cache_ttl = 300  # 5 minutes
            logger.info(f"HashiCorp Vault initialized (url: {self.vault_url})")
        except ImportError:
            raise ImportError("hvac is required for HashiCorp Vault. Install with: pip install hvac")
        except Exception as e:
            logger.error(f"Failed to initialize HashiCorp Vault: {e}")
            raise
    
    def get_secret(self, secret_name: str) -> str:
        """
        Get a secret value from HashiCorp Vault.
        
        Args:
            secret_name: Secret path (e.g., 'secret/data/r3mes/production/database_url')
            
        Returns:
            Secret value as string
        """
        # Check cache first
        if secret_name in self._cache:
            value, timestamp = self._cache[secret_name]
            if time.time() - timestamp < self._cache_ttl:
                return value
        
        try:
            # Parse secret path (format: secret/data/path/to/secret)
            parts = secret_name.split('/')
            if len(parts) < 3:
                raise ValueError(f"Invalid Vault secret path format: {secret_name}")
            
            mount_point = parts[0]  # Usually 'secret'
            path = '/'.join(parts[2:])  # Path after 'data'
            
            # Read secret from KV v2 engine
            response = self.client.secrets.kv.v2.read_secret_version(path=path, mount_point=mount_point)
            secret_data = response.get('data', {}).get('data', {})
            
            # Get the value (if path ends with key name, use that key, otherwise use first value)
            if '/' in path:
                key_name = path.split('/')[-1]
                value = secret_data.get(key_name, list(secret_data.values())[0] if secret_data else '')
            else:
                value = list(secret_data.values())[0] if secret_data else ''
            
            # Cache the value
            self._cache[secret_name] = (str(value), time.time())
            return str(value)
            
        except Exception as e:
            logger.error(f"Failed to get secret {secret_name} from HashiCorp Vault: {e}")
            raise
    
    def get_secrets(self, secret_path: str) -> Dict[str, Any]:
        """
        Get multiple secrets from a path.
        
        Args:
            secret_path: Secret path (e.g., 'secret/data/r3mes/production')
            
        Returns:
            Dictionary of secret names to values
        """
        try:
            parts = secret_path.split('/')
            if len(parts) < 3:
                raise ValueError(f"Invalid Vault secret path format: {secret_path}")
            
            mount_point = parts[0]
            path = '/'.join(parts[2:])
            
            # List secrets under the path
            response = self.client.secrets.kv.v2.list_secrets(path=path, mount_point=mount_point)
            secrets = {}
            
            for key in response.get('data', {}).get('keys', []):
                full_path = f"{secret_path}/{key}"
                try:
                    secrets[full_path] = self.get_secret(full_path)
                except Exception as e:
                    logger.warning(f"Failed to get secret {full_path}: {e}")
            
            return secrets
            
        except Exception as e:
            logger.error(f"Failed to list secrets from path {secret_path}: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test connection to HashiCorp Vault."""
        try:
            # Check if client is authenticated
            return self.client.is_authenticated()
        except Exception as e:
            logger.error(f"HashiCorp Vault connection test failed: {e}")
            return False


class GoogleCloudSecretManager(SecretManager):
    """Google Cloud Secret Manager implementation."""
    
    def __init__(self, project_id: Optional[str] = None):
        """
        Initialize Google Cloud Secret Manager client.
        
        Args:
            project_id: GCP project ID (defaults to GOOGLE_CLOUD_PROJECT env var)
        """
        try:
            from google.cloud import secretmanager
            self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
            
            if not self.project_id:
                raise ValueError("GOOGLE_CLOUD_PROJECT environment variable must be set for Google Cloud Secret Manager")
            
            self.client = secretmanager.SecretManagerServiceClient()
            self._cache: Dict[str, tuple] = {}  # {secret_name: (value, timestamp)}
            self._cache_ttl = 300  # 5 minutes
            logger.info(f"Google Cloud Secret Manager initialized (project: {self.project_id})")
        except ImportError:
            raise ImportError("google-cloud-secret-manager is required. Install with: pip install google-cloud-secret-manager")
        except Exception as e:
            logger.error(f"Failed to initialize Google Cloud Secret Manager: {e}")
            raise
    
    def get_secret(self, secret_name: str, version: str = "latest") -> str:
        """
        Get a secret value from Google Cloud Secret Manager.
        
        Args:
            secret_name: Secret name (e.g., 'r3mes-production-database-url')
            version: Secret version (default: 'latest')
            
        Returns:
            Secret value as string
        """
        # Check cache first
        cache_key = f"{secret_name}:{version}"
        if cache_key in self._cache:
            value, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return value
        
        try:
            # Build the resource name of the secret version
            name = f"projects/{self.project_id}/secrets/{secret_name}/versions/{version}"
            
            # Access the secret version
            response = self.client.access_secret_version(request={"name": name})
            
            # Decode the secret value
            secret_value = response.payload.data.decode("UTF-8")
            
            # Cache the value
            self._cache[cache_key] = (secret_value, time.time())
            return secret_value
            
        except Exception as e:
            logger.error(f"Failed to get secret {secret_name} from Google Cloud Secret Manager: {e}")
            raise ValueError(f"Secret {secret_name} not found in Google Cloud Secret Manager: {e}")
    
    def get_secrets(self, secret_path: str) -> Dict[str, Any]:
        """
        Get multiple secrets with a prefix.
        
        Args:
            secret_path: Secret name prefix (e.g., 'r3mes-production')
            
        Returns:
            Dictionary of secret names to values
        """
        try:
            # List secrets with the prefix
            parent = f"projects/{self.project_id}"
            secrets = {}
            
            # List all secrets
            for secret in self.client.list_secrets(request={"parent": parent}):
                secret_name = secret.name.split("/")[-1]  # Extract just the name
                
                # Filter by prefix
                if secret_name.startswith(secret_path):
                    try:
                        secrets[secret_name] = self.get_secret(secret_name)
                    except Exception as e:
                        logger.warning(f"Failed to get secret {secret_name}: {e}")
            
            return secrets
            
        except Exception as e:
            logger.error(f"Failed to list secrets from path {secret_path}: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test connection to Google Cloud Secret Manager."""
        try:
            # Try to list secrets (minimal operation)
            parent = f"projects/{self.project_id}"
            # Get first page of secrets to test connection
            list(self.client.list_secrets(request={"parent": parent}).pages)[:1]
            return True
        except Exception as e:
            logger.error(f"Google Cloud Secret Manager connection test failed: {e}")
            return False


class EnvironmentVariableSecretManager(SecretManager):
    """Environment variable fallback (for development)."""
    
    def __init__(self):
        """Initialize environment variable secret manager."""
        logger.info("Using environment variables for secret management (development mode)")
    
    def get_secret(self, secret_name: str) -> str:
        """
        Get a secret value from environment variable.
        
        Args:
            secret_name: Environment variable name
            
        Returns:
            Secret value as string
        """
        value = os.getenv(secret_name)
        if value is None:
            raise ValueError(f"Environment variable {secret_name} not found")
        return value
    
    def get_secrets(self, secret_path: str) -> Dict[str, Any]:
        """
        Get multiple secrets with a prefix.
        
        Args:
            secret_path: Prefix for environment variable names
            
        Returns:
            Dictionary of secret names to values
        """
        secrets = {}
        prefix = f"{secret_path}_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                secret_name = key[len(prefix):]
                secrets[secret_name] = value
        
        return secrets
    
    def test_connection(self) -> bool:
        """Environment variables are always available."""
        return True


# Global secret manager instance
_secret_manager: Optional[SecretManager] = None


def get_secret_manager() -> SecretManager:
    """
    Get or create global secret manager instance.
    
    Priority:
    1. Google Cloud Secret Manager (if GOOGLE_CLOUD_PROJECT is set)
    2. AWS Secrets Manager (if AWS_SECRETS_MANAGER_REGION is set)
    3. HashiCorp Vault (if VAULT_ADDR is set)
    4. Environment Variables (fallback)
    """
    global _secret_manager
    
    if _secret_manager is not None:
        return _secret_manager
    
    env_mode = os.getenv("R3MES_ENV", "development").lower()
    is_production = env_mode == "production"
    
    # Try Google Cloud Secret Manager first
    if os.getenv("GOOGLE_CLOUD_PROJECT"):
        try:
            _secret_manager = GoogleCloudSecretManager()
            logger.info("Using Google Cloud Secret Manager for secret management")
            return _secret_manager
        except Exception as e:
            if is_production:
                raise ValueError(f"Failed to initialize Google Cloud Secret Manager in production: {e}")
            logger.warning(f"Failed to initialize Google Cloud Secret Manager: {e}, falling back to other providers")
    
    # Try AWS Secrets Manager
    if os.getenv("AWS_SECRETS_MANAGER_REGION") or os.getenv("AWS_DEFAULT_REGION"):
        try:
            _secret_manager = AWSSecretsManager()
            logger.info("Using AWS Secrets Manager for secret management")
            return _secret_manager
        except Exception as e:
            if is_production:
                raise ValueError(f"Failed to initialize AWS Secrets Manager in production: {e}")
            logger.warning(f"Failed to initialize AWS Secrets Manager: {e}, falling back to environment variables")
    
    # Try HashiCorp Vault
    if os.getenv("VAULT_ADDR"):
        try:
            _secret_manager = HashiCorpVault()
            logger.info("Using HashiCorp Vault for secret management")
            return _secret_manager
        except Exception as e:
            if is_production:
                raise ValueError(f"Failed to initialize HashiCorp Vault in production: {e}")
            logger.warning(f"Failed to initialize HashiCorp Vault: {e}, falling back to environment variables")
    
    # Fallback to environment variables
    if is_production:
        logger.warning("Production mode detected but no secret management service configured. Using environment variables (not recommended).")
    
    _secret_manager = EnvironmentVariableSecretManager()
    return _secret_manager


def get_secret(secret_name: str, default: Optional[str] = None) -> str:
    """
    Get a secret value.
    
    Priority:
    1. Secret management service (production)
    2. Environment variable
    3. Default value (development only)
    
    Args:
        secret_name: Secret name or environment variable name
        default: Default value (only used in development)
        
    Returns:
        Secret value as string
    """
    env_mode = os.getenv("R3MES_ENV", "development").lower()
    is_production = env_mode == "production"
    
    # In production, use secret management service
    if is_production:
        try:
            secret_manager = get_secret_manager()
            return secret_manager.get_secret(secret_name)
        except Exception as e:
            logger.error(f"Failed to get secret {secret_name} from secret manager: {e}")
            raise ValueError(f"Secret {secret_name} must be available in production")
    
    # In development, try secret manager first, then environment variable, then default
    try:
        secret_manager = get_secret_manager()
        if not isinstance(secret_manager, EnvironmentVariableSecretManager):
            return secret_manager.get_secret(secret_name)
    except Exception:
        pass
    
    # Fallback to environment variable
    value = os.getenv(secret_name, default)
    if value is None:
        raise ValueError(f"Secret {secret_name} not found in environment and no default provided")
    return value


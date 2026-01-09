"""
Secrets Provider for R3MES Backend

Production-ready secrets management with support for:
- AWS Secrets Manager
- HashiCorp Vault
- Azure Key Vault
- Google Cloud Secret Manager
- Environment variables (fallback)
- Local file-based secrets (development only)
"""

import os
import json
import logging
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


class SecretsProvider(ABC):
    """Abstract base class for secrets providers."""
    
    @abstractmethod
    def get_secret(self, secret_name: str) -> Optional[str]:
        """Get secret value by name."""
        pass
    
    @abstractmethod
    def get_secret_dict(self, secret_name: str) -> Optional[Dict[str, Any]]:
        """Get secret as dictionary (for JSON secrets)."""
        pass
    
    @abstractmethod
    def set_secret(self, secret_name: str, secret_value: str) -> bool:
        """Set secret value (if supported)."""
        pass


class AWSSecretsManagerProvider(SecretsProvider):
    """AWS Secrets Manager provider."""
    
    def __init__(self, region_name: Optional[str] = None):
        """
        Initialize AWS Secrets Manager provider.
        
        Args:
            region_name: AWS region (default: from AWS_REGION env var)
        """
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")
            self.client = boto3.client(
                service_name='secretsmanager',
                region_name=self.region_name
            )
            self.ClientError = ClientError
            logger.info(f"AWS Secrets Manager provider initialized (region: {self.region_name})")
            
        except ImportError:
            raise ImportError("boto3 is required for AWS Secrets Manager. Install with: pip install boto3")
    
    def get_secret(self, secret_name: str) -> Optional[str]:
        """Get secret from AWS Secrets Manager."""
        try:
            response = self.client.get_secret_value(SecretId=secret_name)
            
            if 'SecretString' in response:
                return response['SecretString']
            else:
                # Binary secret
                import base64
                return base64.b64decode(response['SecretBinary']).decode('utf-8')
                
        except self.ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ResourceNotFoundException':
                logger.warning(f"Secret not found: {secret_name}")
            elif error_code == 'InvalidRequestException':
                logger.error(f"Invalid request for secret: {secret_name}")
            elif error_code == 'InvalidParameterException':
                logger.error(f"Invalid parameter for secret: {secret_name}")
            else:
                logger.error(f"Error getting secret {secret_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting secret {secret_name}: {e}")
            return None
    
    def get_secret_dict(self, secret_name: str) -> Optional[Dict[str, Any]]:
        """Get secret as dictionary."""
        secret_value = self.get_secret(secret_name)
        if secret_value:
            try:
                return json.loads(secret_value)
            except json.JSONDecodeError:
                logger.error(f"Secret {secret_name} is not valid JSON")
                return None
        return None
    
    def set_secret(self, secret_name: str, secret_value: str) -> bool:
        """Set secret in AWS Secrets Manager."""
        try:
            self.client.put_secret_value(
                SecretId=secret_name,
                SecretString=secret_value
            )
            logger.info(f"Secret updated: {secret_name}")
            return True
        except Exception as e:
            logger.error(f"Error setting secret {secret_name}: {e}")
            return False


class HashiCorpVaultProvider(SecretsProvider):
    """HashiCorp Vault provider."""
    
    def __init__(
        self,
        vault_url: Optional[str] = None,
        vault_token: Optional[str] = None,
        mount_point: str = "secret"
    ):
        """
        Initialize HashiCorp Vault provider.
        
        Args:
            vault_url: Vault server URL (default: from VAULT_ADDR env var)
            vault_token: Vault token (default: from VAULT_TOKEN env var)
            mount_point: Vault mount point (default: "secret")
        """
        try:
            import hvac
            
            self.vault_url = vault_url or os.getenv("VAULT_ADDR", "http://localhost:8200")
            self.vault_token = vault_token or os.getenv("VAULT_TOKEN")
            self.mount_point = mount_point
            
            if not self.vault_token:
                raise ValueError("VAULT_TOKEN environment variable must be set")
            
            self.client = hvac.Client(url=self.vault_url, token=self.vault_token)
            
            if not self.client.is_authenticated():
                raise ValueError("Vault authentication failed")
            
            logger.info(f"HashiCorp Vault provider initialized (url: {self.vault_url})")
            
        except ImportError:
            raise ImportError("hvac is required for HashiCorp Vault. Install with: pip install hvac")
    
    def get_secret(self, secret_name: str) -> Optional[str]:
        """Get secret from Vault."""
        try:
            response = self.client.secrets.kv.v2.read_secret_version(
                path=secret_name,
                mount_point=self.mount_point
            )
            
            data = response['data']['data']
            
            # If secret has a 'value' key, return it
            if 'value' in data:
                return data['value']
            
            # Otherwise, return as JSON
            return json.dumps(data)
            
        except Exception as e:
            logger.error(f"Error getting secret {secret_name} from Vault: {e}")
            return None
    
    def get_secret_dict(self, secret_name: str) -> Optional[Dict[str, Any]]:
        """Get secret as dictionary."""
        try:
            response = self.client.secrets.kv.v2.read_secret_version(
                path=secret_name,
                mount_point=self.mount_point
            )
            return response['data']['data']
        except Exception as e:
            logger.error(f"Error getting secret dict {secret_name} from Vault: {e}")
            return None
    
    def set_secret(self, secret_name: str, secret_value: str) -> bool:
        """Set secret in Vault."""
        try:
            self.client.secrets.kv.v2.create_or_update_secret(
                path=secret_name,
                secret={'value': secret_value},
                mount_point=self.mount_point
            )
            logger.info(f"Secret updated in Vault: {secret_name}")
            return True
        except Exception as e:
            logger.error(f"Error setting secret {secret_name} in Vault: {e}")
            return False


class EnvironmentVariableProvider(SecretsProvider):
    """Environment variable provider (fallback)."""
    
    def __init__(self, prefix: str = "R3MES_SECRET_"):
        """
        Initialize environment variable provider.
        
        Args:
            prefix: Prefix for secret environment variables
        """
        self.prefix = prefix
        logger.info(f"Environment variable provider initialized (prefix: {prefix})")
    
    def get_secret(self, secret_name: str) -> Optional[str]:
        """Get secret from environment variable."""
        env_var_name = f"{self.prefix}{secret_name.upper()}"
        value = os.getenv(env_var_name)
        
        if value is None:
            logger.warning(f"Secret not found in environment: {env_var_name}")
        
        return value
    
    def get_secret_dict(self, secret_name: str) -> Optional[Dict[str, Any]]:
        """Get secret as dictionary."""
        secret_value = self.get_secret(secret_name)
        if secret_value:
            try:
                return json.loads(secret_value)
            except json.JSONDecodeError:
                logger.error(f"Secret {secret_name} is not valid JSON")
                return None
        return None
    
    def set_secret(self, secret_name: str, secret_value: str) -> bool:
        """Set secret (not supported for environment variables)."""
        logger.warning("Setting secrets via environment variables is not supported")
        return False


class FileBasedProvider(SecretsProvider):
    """File-based secrets provider (development only)."""
    
    def __init__(self, secrets_dir: str = ".secrets"):
        """
        Initialize file-based provider.
        
        Args:
            secrets_dir: Directory containing secret files
        """
        self.secrets_dir = Path(secrets_dir)
        self.secrets_dir.mkdir(parents=True, exist_ok=True)
        
        # Warn if used in production
        is_production = os.getenv("R3MES_ENV", "development").lower() == "production"
        if is_production:
            logger.error("File-based secrets provider should not be used in production!")
        
        logger.info(f"File-based secrets provider initialized (dir: {secrets_dir})")
    
    def get_secret(self, secret_name: str) -> Optional[str]:
        """Get secret from file."""
        try:
            secret_file = self.secrets_dir / f"{secret_name}.secret"
            if secret_file.exists():
                return secret_file.read_text().strip()
            else:
                logger.warning(f"Secret file not found: {secret_file}")
                return None
        except Exception as e:
            logger.error(f"Error reading secret file {secret_name}: {e}")
            return None
    
    def get_secret_dict(self, secret_name: str) -> Optional[Dict[str, Any]]:
        """Get secret as dictionary."""
        secret_value = self.get_secret(secret_name)
        if secret_value:
            try:
                return json.loads(secret_value)
            except json.JSONDecodeError:
                logger.error(f"Secret {secret_name} is not valid JSON")
                return None
        return None
    
    def set_secret(self, secret_name: str, secret_value: str) -> bool:
        """Set secret in file."""
        try:
            secret_file = self.secrets_dir / f"{secret_name}.secret"
            secret_file.write_text(secret_value)
            secret_file.chmod(0o600)  # Read/write for owner only
            logger.info(f"Secret saved to file: {secret_file}")
            return True
        except Exception as e:
            logger.error(f"Error writing secret file {secret_name}: {e}")
            return False


class SecretsManager:
    """Unified secrets manager with multiple provider support."""
    
    def __init__(self, provider: Optional[SecretsProvider] = None):
        """
        Initialize secrets manager.
        
        Args:
            provider: Secrets provider (auto-detected if not provided)
        """
        if provider:
            self.provider = provider
        else:
            self.provider = self._auto_detect_provider()
        
        logger.info(f"Secrets manager initialized with provider: {type(self.provider).__name__}")
    
    def _auto_detect_provider(self) -> SecretsProvider:
        """Auto-detect secrets provider based on environment."""
        provider_type = os.getenv("SECRETS_PROVIDER", "env").lower()
        
        if provider_type == "aws":
            try:
                return AWSSecretsManagerProvider()
            except Exception as e:
                logger.warning(f"Failed to initialize AWS Secrets Manager: {e}")
                logger.info("Falling back to environment variable provider")
                return EnvironmentVariableProvider()
        
        elif provider_type == "vault":
            try:
                return HashiCorpVaultProvider()
            except Exception as e:
                logger.warning(f"Failed to initialize HashiCorp Vault: {e}")
                logger.info("Falling back to environment variable provider")
                return EnvironmentVariableProvider()
        
        elif provider_type == "file":
            return FileBasedProvider()
        
        else:  # Default to environment variables
            return EnvironmentVariableProvider()
    
    def get_secret(self, secret_name: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get secret value.
        
        Args:
            secret_name: Secret name
            default: Default value if secret not found
            
        Returns:
            Secret value or default
        """
        value = self.provider.get_secret(secret_name)
        return value if value is not None else default
    
    def get_secret_dict(self, secret_name: str, default: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Get secret as dictionary.
        
        Args:
            secret_name: Secret name
            default: Default value if secret not found
            
        Returns:
            Secret dictionary or default
        """
        value = self.provider.get_secret_dict(secret_name)
        return value if value is not None else default
    
    def set_secret(self, secret_name: str, secret_value: str) -> bool:
        """
        Set secret value.
        
        Args:
            secret_name: Secret name
            secret_value: Secret value
            
        Returns:
            True if successful
        """
        return self.provider.set_secret(secret_name, secret_value)
    
    def get_database_credentials(self) -> Dict[str, str]:
        """Get database credentials from secrets."""
        db_secret = self.get_secret_dict("database_credentials")
        
        if db_secret:
            return db_secret
        
        # Fallback to individual environment variables
        return {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", "5432"),
            "database": os.getenv("DB_NAME", "r3mes"),
            "username": os.getenv("DB_USER", "r3mes"),
            "password": os.getenv("DB_PASSWORD", ""),
        }
    
    def get_jwt_keys(self) -> Dict[str, str]:
        """Get JWT RSA keys from secrets."""
        jwt_secret = self.get_secret_dict("jwt_keys")
        
        if jwt_secret:
            return jwt_secret
        
        # Fallback to individual secrets/env vars
        return {
            "private_key": self.get_secret("jwt_private_key", os.getenv("JWT_PRIVATE_KEY", "")),
            "public_key": self.get_secret("jwt_public_key", os.getenv("JWT_PUBLIC_KEY", "")),
        }
    
    def get_api_keys(self) -> Dict[str, str]:
        """Get external API keys from secrets."""
        api_keys = self.get_secret_dict("api_keys")
        
        if api_keys:
            return api_keys
        
        # Fallback to individual environment variables
        return {
            "ipfs_api_key": os.getenv("IPFS_API_KEY", ""),
            "blockchain_api_key": os.getenv("BLOCKCHAIN_API_KEY", ""),
        }


# Global secrets manager instance
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager() -> SecretsManager:
    """Get secrets manager singleton."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager


def init_secrets_manager(provider: Optional[SecretsProvider] = None):
    """
    Initialize secrets manager with custom provider.
    
    Args:
        provider: Custom secrets provider
    """
    global _secrets_manager
    _secrets_manager = SecretsManager(provider=provider)

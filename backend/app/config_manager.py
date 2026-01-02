"""
Configuration Manager - Centralized configuration management

Manages application settings with environment variable support, UI-configurable options,
and HashiCorp Vault integration for secure secrets management.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


def normalize_path(path: str, base_dir: Optional[Path] = None) -> str:
    """
    Normalize a file path to be Docker volume mount compatible.
    
    Args:
        path: Path to normalize (can be absolute or relative)
        base_dir: Base directory for resolving relative paths (default: current working directory)
    
    Returns:
        Normalized absolute path as string
    """
    if not path:
        return path
    
    # If path starts with ~, expand user home directory
    if path.startswith("~/"):
        expanded = os.path.expanduser(path)
        return str(Path(expanded).resolve())
    
    # If path is already absolute, use it as-is
    path_obj = Path(path)
    if path_obj.is_absolute():
        return str(path_obj.resolve())
    
    # If relative, resolve against base_dir or current working directory
    if base_dir is None:
        base_dir = Path.cwd()
    
    resolved = (base_dir / path).resolve()
    return str(resolved)


@dataclass
class AppConfig:
    """Application configuration dataclass."""
    # Model settings
    base_model_path: str = "checkpoints/base_model"
    model_download_dir: str = "~/.r3mes/models"
    
    # Database settings
    database_path: str = "backend/database.db"
    chain_json_path: str = "chain.json"
    
    # Mining settings
    mining_difficulty: float = 1234.0  # Default fallback, will be fetched from blockchain if available
    gpu_memory_limit_mb: Optional[int] = None  # None = auto-detect
    p2p_port: int = 9090
    
    # API settings (can be overridden via environment variables in load() method)
    rate_limit_chat: str = "10/minute"
    rate_limit_get: str = "30/minute"
    rate_limit_post: str = "20/minute"
    
    # Network settings
    # Note: Defaults are None for production safety. In production, these must be set via environment variables.
    blockchain_rpc_url: Optional[str] = None
    blockchain_grpc_url: Optional[str] = None
    
    # Feature flags
    auto_start_mining: bool = False
    enable_notifications: bool = True
    
    # Multi-GPU settings
    use_multi_gpu: bool = False
    multi_gpu_strategy: str = "data_parallel"  # "data_parallel" or "model_parallel"
    multi_gpu_device_ids: Optional[str] = None  # Comma-separated device IDs (e.g., "0,1,2")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        """Create config from dictionary."""
        return cls(**data)


class ConfigManager:
    """
    Centralized configuration manager.
    
    Loads configuration from:
    1. Environment variables (highest priority)
    2. HashiCorp Vault (production secrets)
    3. Config file (~/.r3mes/config.json)
    4. Default values
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to config file (default: ~/.r3mes/config.json in dev, env var in prod)
        """
        # In production, disable user home config file unless explicitly provided
        is_production = os.getenv("R3MES_ENV", "development").lower() == "production"
        
        if config_file is None:
            # Allow override via environment variable
            config_file = os.getenv("R3MES_CONFIG_FILE")
            
            if config_file is None:
                if is_production:
                    # Production: use working directory or disable config file
                    # Don't use ~/.r3mes in production
                    logger.info("Production mode: config file disabled (use environment variables or R3MES_CONFIG_FILE)")
                    self.config_file = None
                else:
                    # Development: use ~/.r3mes/config.json
                    config_dir = Path.home() / ".r3mes"
                    config_dir.mkdir(parents=True, exist_ok=True)
                    config_file = str(config_dir / "config.json")
        
        if config_file:
            # Normalize config file path
            self.config_file = Path(normalize_path(config_file))
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.config_file = None
        self._config: Optional[AppConfig] = None
        self._vault_client = None
    
    async def _get_vault_client(self):
        """Get or create Vault client for secure configuration."""
        if self._vault_client is None:
            # Only initialize Vault in production or when explicitly configured
            vault_addr = os.getenv("VAULT_ADDR")
            if vault_addr:
                try:
                    from .vault_client import get_vault_client
                    self._vault_client = get_vault_client()
                    await self._vault_client.initialize()
                    logger.info("✅ Vault client initialized for configuration")
                except Exception as e:
                    logger.warning(f"Failed to initialize Vault client: {e}")
                    self._vault_client = None
        return self._vault_client
    
    async def _load_from_vault(self, config: AppConfig) -> AppConfig:
        """Load sensitive configuration from Vault."""
        vault_client = await self._get_vault_client()
        if not vault_client:
            return config
        
        try:
            # Load database configuration from Vault
            try:
                db_config = await vault_client.get_secret("database")
                if isinstance(db_config, dict):
                    # Build database URL from Vault secrets
                    db_user = db_config.get("user", "r3mes")
                    db_password = db_config.get("password")
                    db_host = db_config.get("host", "localhost")
                    db_port = db_config.get("port", "5432")
                    db_name = db_config.get("database", "r3mes")
                    
                    if db_password:
                        # Override database path with PostgreSQL URL from Vault
                        config.database_path = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
                        logger.info("✅ Database configuration loaded from Vault")
            except Exception as e:
                logger.debug(f"Database config not found in Vault: {e}")
            
            # Load blockchain configuration from Vault
            try:
                blockchain_config = await vault_client.get_secret("blockchain")
                if isinstance(blockchain_config, dict):
                    rpc_url = blockchain_config.get("rpc_url")
                    grpc_url = blockchain_config.get("grpc_url")
                    
                    if rpc_url:
                        config.blockchain_rpc_url = rpc_url
                        logger.info("✅ Blockchain RPC URL loaded from Vault")
                    
                    if grpc_url:
                        config.blockchain_grpc_url = grpc_url
                        logger.info("✅ Blockchain gRPC URL loaded from Vault")
            except Exception as e:
                logger.debug(f"Blockchain config not found in Vault: {e}")
            
        except Exception as e:
            logger.warning(f"Failed to load configuration from Vault: {e}")
        
        return config
    
    def load(self) -> AppConfig:
        """Load configuration from file and environment."""
        # Start with defaults
        config = AppConfig()
        
        # Load from file if exists
        if self.config_file and self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    file_data = json.load(f)
                    config = AppConfig.from_dict(file_data)
                    logger.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}, using defaults")
        
        # Override with environment variables and normalize paths
        base_dir = Path.cwd()
        
        # Normalize base_model_path
        base_model_path = os.getenv("BASE_MODEL_PATH", config.base_model_path)
        config.base_model_path = normalize_path(base_model_path, base_dir)
        
        # Normalize model_download_dir (expand ~ if present)
        model_download_dir = os.getenv("MODEL_DOWNLOAD_DIR", config.model_download_dir)
        config.model_download_dir = normalize_path(model_download_dir, base_dir)
        
        # Normalize database_path
        database_path = os.getenv("DATABASE_PATH", config.database_path)
        config.database_path = normalize_path(database_path, base_dir)
        
        # Normalize chain_json_path
        chain_json_path = os.getenv("CHAIN_JSON_PATH", config.chain_json_path)
        config.chain_json_path = normalize_path(chain_json_path, base_dir)
        
        # Environment-based URL configuration (production overrides)
        env_mode = os.getenv("R3MES_ENV", "development").lower()
        if env_mode == "production":
            # Production URLs (override defaults)
            # In production, BLOCKCHAIN_RPC_URL must be set (no localhost fallback)
            rpc_url = os.getenv("PRODUCTION_BLOCKCHAIN_RPC_URL") or os.getenv("BLOCKCHAIN_RPC_URL")
            if not rpc_url:
                raise ValueError(
                    "BLOCKCHAIN_RPC_URL or PRODUCTION_BLOCKCHAIN_RPC_URL must be set in production. "
                    "Do not use localhost in production."
                )
            config.blockchain_rpc_url = rpc_url
            # Validate that production doesn't use localhost (exact hostname match)
            from urllib.parse import urlparse
            parsed = urlparse(config.blockchain_rpc_url)
            hostname = parsed.hostname or ""
            if hostname.lower() in ("localhost", "127.0.0.1", "::1") or hostname.startswith("127."):
                raise ValueError(
                    f"BLOCKCHAIN_RPC_URL cannot use localhost in production: {config.blockchain_rpc_url}"
                )
            
            grpc_url = os.getenv("PRODUCTION_BLOCKCHAIN_GRPC_URL") or os.getenv("BLOCKCHAIN_GRPC_URL")
            if not grpc_url:
                raise ValueError(
                    "BLOCKCHAIN_GRPC_URL or PRODUCTION_BLOCKCHAIN_GRPC_URL must be set in production. "
                    "Do not use localhost in production."
                )
            config.blockchain_grpc_url = grpc_url
            # Validate that production doesn't use localhost (exact hostname match)
            # For gRPC URL, it's hostname:port format, so split on ':'
            grpc_hostname = grpc_url.split(":")[0] if ":" in grpc_url else grpc_url
            if grpc_hostname.lower() in ("localhost", "127.0.0.1", "::1") or grpc_hostname.startswith("127."):
                raise ValueError(
                    f"BLOCKCHAIN_GRPC_URL cannot use localhost in production: {config.blockchain_grpc_url}"
                )
        else:
            # Development/staging: use environment variables or fallback to localhost defaults
            config.blockchain_rpc_url = os.getenv("BLOCKCHAIN_RPC_URL", "http://localhost:26657")
            config.blockchain_grpc_url = os.getenv("BLOCKCHAIN_GRPC_URL", "localhost:9090")
        
        # Mining settings
        if os.getenv("MINING_DIFFICULTY"):
            try:
                config.mining_difficulty = float(os.getenv("MINING_DIFFICULTY"))
            except ValueError:
                logger.warning("Invalid MINING_DIFFICULTY, using default")
        else:
            # Try to fetch from blockchain if available
            try:
                from .blockchain_query_client import get_blockchain_client
                blockchain_client = get_blockchain_client()
                # Query params from blockchain
                params_data = blockchain_client._query_rest("/remes/remes/v1/params")
                if "params" in params_data:
                    params = params_data["params"]
                    # Extract mining difficulty from params (if available)
                    difficulty_str = params.get("mining_difficulty", params.get("network_difficulty"))
                    if difficulty_str:
                        config.mining_difficulty = float(difficulty_str)
                        logger.info(f"Fetched mining difficulty from blockchain: {config.mining_difficulty}")
            except Exception as e:
                # If blockchain query fails, use default
                logger.debug(f"Could not fetch mining difficulty from blockchain: {e}, using default")
        
        if os.getenv("GPU_MEMORY_LIMIT_MB"):
            try:
                config.gpu_memory_limit_mb = int(os.getenv("GPU_MEMORY_LIMIT_MB"))
            except ValueError:
                logger.warning("Invalid GPU_MEMORY_LIMIT_MB, using default")
        
        if os.getenv("P2P_PORT"):
            try:
                config.p2p_port = int(os.getenv("P2P_PORT"))
            except ValueError:
                logger.warning("Invalid P2P_PORT, using default")
        
        # Rate limiting settings (support both old RATE_LIMIT_* and new BACKEND_RATE_LIMIT_* for backward compatibility)
        config.rate_limit_chat = os.getenv("BACKEND_RATE_LIMIT_CHAT") or os.getenv("RATE_LIMIT_CHAT", config.rate_limit_chat)
        config.rate_limit_get = os.getenv("BACKEND_RATE_LIMIT_GET") or os.getenv("RATE_LIMIT_GET", config.rate_limit_get)
        config.rate_limit_post = os.getenv("BACKEND_RATE_LIMIT_POST") or os.getenv("RATE_LIMIT_POST", config.rate_limit_post)
        
        # Network settings (already handled above in env_mode check, but keep for backward compatibility)
        # Only override if not already set by production/development logic above
        if env_mode != "production":
            config.blockchain_rpc_url = os.getenv("BLOCKCHAIN_RPC_URL", config.blockchain_rpc_url)
            config.blockchain_grpc_url = os.getenv("BLOCKCHAIN_GRPC_URL", config.blockchain_grpc_url)
        
        # Feature flags
        config.auto_start_mining = os.getenv("AUTO_START_MINING", "false").lower() == "true"
        config.enable_notifications = os.getenv("ENABLE_NOTIFICATIONS", "true").lower() != "false"
        
        self._config = config
        return config
    
    async def load_async(self) -> AppConfig:
        """Load configuration asynchronously with Vault support."""
        # Load basic configuration first
        config = self.load()
        
        # Load sensitive configuration from Vault if available
        config = await self._load_from_vault(config)
        
        self._config = config
        return config
    
    def save(self, config: AppConfig) -> None:
        """Save configuration to file."""
        if self.config_file is None:
            logger.warning("Cannot save config file: config file path not set (production mode)")
            return
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
            logger.info(f"Saved configuration to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save config file: {e}")
            raise
    
    def get(self) -> AppConfig:
        """Get current configuration."""
        if self._config is None:
            self.load()
        return self._config
    
    def update(self, updates: Dict[str, Any]) -> AppConfig:
        """Update configuration with new values."""
        config = self.get()
        config_dict = config.to_dict()
        config_dict.update(updates)
        new_config = AppConfig.from_dict(config_dict)
        self.save(new_config)
        self._config = new_config
        return new_config


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get or create global config manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


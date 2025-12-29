"""
Environment Variable Constants and Standardization

This module defines standard environment variable names used across all R3MES components.
All components should import from this module to ensure consistency.

NAMING CONVENTION:
- All R3MES-specific variables start with R3MES_
- Blockchain-related variables: R3MES_BLOCKCHAIN_*
- Backend-related variables: R3MES_BACKEND_*
- Miner-related variables: R3MES_MINER_*
- Frontend-related variables: NEXT_PUBLIC_R3MES_* (for Next.js)

MIGRATION GUIDE:
Old Name                    -> New Name
BLOCKCHAIN_RPC_URL          -> R3MES_BLOCKCHAIN_RPC_URL
BLOCKCHAIN_GRPC_URL         -> R3MES_BLOCKCHAIN_GRPC_URL
BLOCKCHAIN_REST_URL         -> R3MES_BLOCKCHAIN_REST_URL
R3MES_NODE_GRPC_URL         -> R3MES_BLOCKCHAIN_GRPC_URL
"""

import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EnvVarMapping:
    """Mapping between old and new environment variable names."""
    new_name: str
    old_names: list
    description: str
    required_in_production: bool = False
    default_value: Optional[str] = None


# Standard environment variable mappings
ENV_VAR_MAPPINGS = {
    # Blockchain Configuration
    "blockchain_rpc_url": EnvVarMapping(
        new_name="R3MES_BLOCKCHAIN_RPC_URL",
        old_names=["BLOCKCHAIN_RPC_URL", "R3MES_TENDERMINT_RPC_ADDR"],
        description="Tendermint RPC endpoint URL",
        required_in_production=True,
        default_value="http://localhost:26657"
    ),
    "blockchain_grpc_url": EnvVarMapping(
        new_name="R3MES_BLOCKCHAIN_GRPC_URL",
        old_names=["BLOCKCHAIN_GRPC_URL", "R3MES_NODE_GRPC_URL", "R3MES_GRPC_ADDR"],
        description="Cosmos SDK gRPC endpoint URL",
        required_in_production=True,
        default_value="localhost:9090"
    ),
    "blockchain_rest_url": EnvVarMapping(
        new_name="R3MES_BLOCKCHAIN_REST_URL",
        old_names=["BLOCKCHAIN_REST_URL", "R3MES_REST_URL"],
        description="Cosmos SDK REST endpoint URL",
        required_in_production=True,
        default_value="http://localhost:1317"
    ),
    
    # Backend Configuration
    "backend_url": EnvVarMapping(
        new_name="R3MES_BACKEND_URL",
        old_names=["BACKEND_URL", "API_URL"],
        description="Backend API URL",
        required_in_production=True,
        default_value="http://localhost:8000"
    ),
    "database_url": EnvVarMapping(
        new_name="R3MES_DATABASE_URL",
        old_names=["DATABASE_URL", "DB_URL"],
        description="Database connection URL",
        required_in_production=True,
        default_value="sqlite:///r3mes.db"
    ),
    "redis_url": EnvVarMapping(
        new_name="R3MES_REDIS_URL",
        old_names=["REDIS_URL", "CACHE_URL"],
        description="Redis connection URL",
        required_in_production=True,
        default_value="redis://localhost:6379"
    ),
    
    # Environment Mode
    "environment": EnvVarMapping(
        new_name="R3MES_ENV",
        old_names=["NODE_ENV", "ENVIRONMENT"],
        description="Environment mode (development, staging, production)",
        required_in_production=False,
        default_value="development"
    ),
    "test_mode": EnvVarMapping(
        new_name="R3MES_TEST_MODE",
        old_names=["TEST_MODE"],
        description="Enable test mode (bypasses some validations)",
        required_in_production=False,
        default_value="false"
    ),
    
    # Chain Configuration
    "chain_id": EnvVarMapping(
        new_name="R3MES_CHAIN_ID",
        old_names=["CHAIN_ID"],
        description="Blockchain chain ID",
        required_in_production=True,
        default_value="remes-testnet-1"
    ),
}


def get_env_var(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get environment variable with fallback to old names.
    
    This function checks for the new standardized name first,
    then falls back to old names for backward compatibility.
    
    Args:
        key: The key in ENV_VAR_MAPPINGS (e.g., "blockchain_rpc_url")
        default: Default value if not found
        
    Returns:
        Environment variable value or default
    """
    if key not in ENV_VAR_MAPPINGS:
        # Not a mapped variable, use standard os.getenv
        return os.getenv(key, default)
    
    mapping = ENV_VAR_MAPPINGS[key]
    
    # Try new name first
    value = os.getenv(mapping.new_name)
    if value:
        return value
    
    # Try old names for backward compatibility
    for old_name in mapping.old_names:
        value = os.getenv(old_name)
        if value:
            logger.warning(
                f"Using deprecated environment variable '{old_name}'. "
                f"Please migrate to '{mapping.new_name}'."
            )
            return value
    
    # Return default
    return default if default is not None else mapping.default_value


def get_blockchain_rpc_url(default: Optional[str] = None) -> str:
    """Get blockchain RPC URL with backward compatibility."""
    return get_env_var("blockchain_rpc_url", default) or "http://localhost:26657"


def get_blockchain_grpc_url(default: Optional[str] = None) -> str:
    """Get blockchain gRPC URL with backward compatibility."""
    return get_env_var("blockchain_grpc_url", default) or "localhost:9090"


def get_blockchain_rest_url(default: Optional[str] = None) -> str:
    """Get blockchain REST URL with backward compatibility."""
    return get_env_var("blockchain_rest_url", default) or "http://localhost:1317"


def is_production() -> bool:
    """Check if running in production mode."""
    env = get_env_var("environment", "development")
    return env.lower() in ("production", "prod")


def is_test_mode() -> bool:
    """Check if test mode is enabled."""
    return get_env_var("test_mode", "false").lower() == "true"


def validate_production_env_vars() -> Dict[str, Any]:
    """
    Validate that all required production environment variables are set.
    
    Returns:
        Dict with validation results
    """
    if not is_production():
        return {"valid": True, "missing": [], "warnings": []}
    
    missing = []
    warnings = []
    
    for key, mapping in ENV_VAR_MAPPINGS.items():
        if mapping.required_in_production:
            value = get_env_var(key)
            if not value:
                missing.append(mapping.new_name)
            elif "localhost" in value.lower() or "127.0.0.1" in value:
                warnings.append(f"{mapping.new_name} uses localhost in production")
    
    return {
        "valid": len(missing) == 0,
        "missing": missing,
        "warnings": warnings
    }


def print_env_migration_guide():
    """Print environment variable migration guide."""
    print("\n" + "=" * 60)
    print("R3MES Environment Variable Migration Guide")
    print("=" * 60 + "\n")
    
    for key, mapping in ENV_VAR_MAPPINGS.items():
        print(f"📌 {mapping.description}")
        print(f"   New Name: {mapping.new_name}")
        print(f"   Old Names: {', '.join(mapping.old_names)}")
        print(f"   Required in Production: {'Yes' if mapping.required_in_production else 'No'}")
        print(f"   Default: {mapping.default_value or 'None'}")
        print()
    
    print("=" * 60 + "\n")

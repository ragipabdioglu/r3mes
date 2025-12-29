"""
Configuration Management Endpoints

Provides API endpoints for managing application settings.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from pathlib import Path
import logging

from .config_manager import get_config_manager, AppConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/config", tags=["configuration"])


class ConfigResponse(BaseModel):
    """Configuration response model."""
    config: Dict[str, Any]
    message: str


class ConfigUpdateRequest(BaseModel):
    """Configuration update request model."""
    base_model_path: Optional[str] = None
    model_download_dir: Optional[str] = None
    database_path: Optional[str] = None
    chain_json_path: Optional[str] = None
    mining_difficulty: Optional[float] = None
    gpu_memory_limit_mb: Optional[int] = None
    p2p_port: Optional[int] = None
    rate_limit_chat: Optional[str] = None
    rate_limit_get: Optional[str] = None
    blockchain_rpc_url: Optional[str] = None
    blockchain_grpc_url: Optional[str] = None
    auto_start_mining: Optional[bool] = None
    enable_notifications: Optional[bool] = None


@router.get("", response_model=ConfigResponse)
async def get_config():
    """
    Get current application configuration.
    
    Returns:
        Current configuration (sensitive values may be masked)
    """
    config_manager = get_config_manager()
    config = config_manager.get()
    
    # Return config as dict (can mask sensitive values if needed)
    return ConfigResponse(
        config=config.to_dict(),
        message="Configuration retrieved successfully"
    )


@router.put("", response_model=ConfigResponse)
async def update_config(update_request: ConfigUpdateRequest):
    """
    Update application configuration.
    
    Args:
        update_request: Configuration updates (only provided fields will be updated)
    
    Returns:
        Updated configuration
    """
    config_manager = get_config_manager()
    current_config = config_manager.get()
    
    # Build update dict (only include non-None values)
    updates = {}
    for field, value in update_request.dict().items():
        if value is not None:
            updates[field] = value
    
    if not updates:
        raise HTTPException(status_code=400, detail="No configuration updates provided")
    
    # Validate updates
    try:
        updated_config = config_manager.update(updates)
        logger.info(f"Configuration updated: {list(updates.keys())}")
        
        return ConfigResponse(
            config=updated_config.to_dict(),
            message="Configuration updated successfully. Some changes may require restart."
        )
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")


@router.get("/validate")
async def validate_config():
    """
    Validate current configuration.
    
    Returns:
        Validation results
    """
    config_manager = get_config_manager()
    config = config_manager.get()
    
    errors = []
    warnings = []
    
    # Validate paths
    if not Path(config.base_model_path).exists():
        warnings.append(f"Base model path does not exist: {config.base_model_path}")
    
    if config.gpu_memory_limit_mb is not None and config.gpu_memory_limit_mb < 0:
        errors.append("GPU memory limit cannot be negative")
    
    if config.p2p_port < 1024 or config.p2p_port > 65535:
        errors.append(f"Invalid P2P port: {config.p2p_port} (must be 1024-65535)")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }


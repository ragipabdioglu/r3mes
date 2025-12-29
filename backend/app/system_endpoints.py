"""
System API Endpoints

Provides system-level endpoints:
- Version check
- Time synchronization
"""

import os
import time
from datetime import datetime
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/system", tags=["system"])


class VersionResponse(BaseModel):
    """Response model for version check."""
    min_version: str
    current_version: str
    update_required: bool
    update_url: Optional[str] = None
    critical: bool = False


class TimeResponse(BaseModel):
    """Response model for time synchronization."""
    server_time: int  # Unix timestamp
    server_time_iso: str  # ISO format
    timezone: str = "UTC"


def compare_versions(current: str, minimum: str) -> int:
    """
    Compare version strings.
    
    Returns:
        -1 if current < minimum
        0 if current == minimum
        1 if current > minimum
    """
    def version_tuple(v: str) -> tuple:
        """Convert version string to tuple for comparison."""
        parts = v.split('.')
        return tuple(int(part) for part in parts)
    
    try:
        current_tuple = version_tuple(current)
        minimum_tuple = version_tuple(minimum)
        
        if current_tuple < minimum_tuple:
            return -1
        elif current_tuple == minimum_tuple:
            return 0
        else:
            return 1
    except (ValueError, AttributeError):
        # Invalid version format, assume update required
        return -1


@router.get("/version", response_model=VersionResponse)
async def get_version():
    """
    Get system version information for client version checking.
    
    Clients should check their version against min_version and prompt
    for update if update_required is true.
    """
    # Get version configuration from environment
    min_version = os.getenv("R3MES_MIN_VERSION", "0.1.0")
    current_version = os.getenv("R3MES_VERSION", "1.0.0")
    
    # Check if update is required
    version_comparison = compare_versions(current_version, min_version)
    update_required = version_comparison < 0
    
    # Determine if update is critical (e.g., security issue)
    critical = os.getenv("R3MES_CRITICAL_UPDATE", "false").lower() == "true"
    
    # Update URL (if available)
    update_url = os.getenv("R3MES_UPDATE_URL", "https://github.com/r3mes/r3mes/releases/latest")
    
    return VersionResponse(
        min_version=min_version,
        current_version=current_version,
        update_required=update_required,
        update_url=update_url,
        critical=critical,
    )


@router.get("/time", response_model=TimeResponse)
async def get_time():
    """
    Get server time for client time synchronization checking.
    
    Clients should compare their system time with server_time and
    warn user if drift > 5 seconds.
    """
    # Get current server time
    server_time = int(time.time())
    server_time_iso = datetime.utcnow().isoformat() + "Z"
    
    return TimeResponse(
        server_time=server_time,
        server_time_iso=server_time_iso,
        timezone="UTC",
    )


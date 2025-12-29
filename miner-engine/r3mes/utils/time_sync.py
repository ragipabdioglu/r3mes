"""
Time Synchronization Utility

Checks system time synchronization with server to prevent timestamp-related
blockchain transaction rejections.
"""

import os
import time
import logging
from typing import Tuple, Optional
import requests

logger = logging.getLogger(__name__)


def check_time_sync(
    api_url: str,
    max_drift_seconds: int = 5,
    timeout: int = 5
) -> Tuple[bool, Optional[str]]:
    """
    Check if system time is synchronized with server.
    
    Args:
        api_url: Backend API URL (e.g., "http://localhost:8000")
        max_drift_seconds: Maximum allowed time drift in seconds (default: 5)
        timeout: Request timeout in seconds (default: 5)
    
    Returns:
        Tuple of (is_synced, error_message)
        - is_synced: True if time is synchronized, False otherwise
        - error_message: Error message if sync check failed or drift too large
    """
    try:
        # Get server time
        response = requests.get(
            f"{api_url}/system/time",
            timeout=timeout
        )
        response.raise_for_status()
        data = response.json()
        
        server_time = data.get("server_time")
        if server_time is None:
            return False, "Server time response missing 'server_time' field"
        
        # Get local system time
        local_time = int(time.time())
        
        # Calculate drift
        drift = abs(local_time - server_time)
        
        if drift > max_drift_seconds:
            return False, (
                f"Time drift too large: {drift} seconds. "
                f"Please sync your system clock. "
                f"Server time: {data.get('server_time_iso')}, "
                f"Local time: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(local_time))}"
            )
        
        logger.debug(f"Time sync check passed (drift: {drift}s)")
        return True, None
        
    except requests.exceptions.Timeout:
        return False, f"Time sync check timeout after {timeout}s"
    except requests.exceptions.ConnectionError:
        return False, f"Failed to connect to {api_url}/system/time"
    except requests.exceptions.HTTPError as e:
        return False, f"HTTP error during time sync check: {e}"
    except Exception as e:
        logger.error(f"Time sync check error: {e}", exc_info=True)
        return False, f"Time sync check failed: {e}"


def check_time_sync_or_warn(
    api_url: Optional[str] = None,
    max_drift_seconds: int = 5,
    critical: bool = False
) -> bool:
    """
    Check time sync and log warning/error if not synced.
    
    Args:
        api_url: Backend API URL (default: from BACKEND_URL env var)
        max_drift_seconds: Maximum allowed drift (default: 5)
        critical: If True, raise exception on sync failure; if False, just warn
    
    Returns:
        True if synced, False otherwise (or raises exception if critical=True)
    """
    if api_url is None:
        api_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    
    is_synced, error_message = check_time_sync(
        api_url=api_url,
        max_drift_seconds=max_drift_seconds
    )
    
    if not is_synced:
        message = f"⚠️  Time synchronization check failed: {error_message}"
        
        if critical:
            logger.error(message)
            raise RuntimeError(f"Critical time sync failure: {error_message}")
        else:
            logger.warning(message)
            print(message)
    
    return is_synced


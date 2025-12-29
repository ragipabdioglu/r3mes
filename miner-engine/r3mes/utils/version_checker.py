"""
Version Check Utility

Checks client version against server minimum version to prompt users
for updates when necessary.
"""

import os
import logging
from typing import Tuple, Optional
import requests

logger = logging.getLogger(__name__)


def compare_versions(current: str, minimum: str) -> int:
    """
    Compare version strings.
    
    Returns:
        -1 if current < minimum (update required)
        0 if current == minimum
        1 if current > minimum
    """
    def version_tuple(v: str) -> tuple:
        """Convert version string to tuple for comparison."""
        # Remove 'v' prefix if present
        v = v.lstrip('v')
        parts = v.split('.')
        try:
            return tuple(int(part) for part in parts)
        except ValueError:
            # Invalid version format, assume needs update
            return (0, 0, 0)
    
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


def check_version(
    api_url: str,
    current_version: str,
    timeout: int = 5
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Check if client version meets minimum server requirement.
    
    Args:
        api_url: Backend API URL (e.g., "http://localhost:8000")
        current_version: Current client version (e.g., "0.1.0")
        timeout: Request timeout in seconds (default: 5)
    
    Returns:
        Tuple of (update_required, message, update_url)
        - update_required: True if update is required, False otherwise
        - message: Human-readable message about version status
        - update_url: URL to download update (if available)
    """
    try:
        # Get server version info
        response = requests.get(
            f"{api_url}/system/version",
            timeout=timeout
        )
        response.raise_for_status()
        data = response.json()
        
        min_version = data.get("min_version")
        server_version = data.get("current_version")
        update_required = data.get("update_required", False)
        critical = data.get("critical", False)
        update_url = data.get("update_url")
        
        if min_version is None:
            return False, None, None
        
        # Compare versions
        version_comparison = compare_versions(current_version, min_version)
        needs_update = version_comparison < 0 or update_required
        
        if needs_update:
            if critical:
                message = (
                    f"⚠️  CRITICAL UPDATE REQUIRED: "
                    f"Your version ({current_version}) is below minimum required ({min_version}). "
                    f"Please update immediately."
                )
            else:
                message = (
                    f"⚠️  Update available: "
                    f"Your version ({current_version}) is below minimum ({min_version}). "
                    f"Current server version: {server_version}. "
                    f"Please update when possible."
                )
            return True, message, update_url
        else:
            logger.debug(f"Version check passed: {current_version} >= {min_version}")
            return False, None, None
        
    except requests.exceptions.Timeout:
        logger.warning(f"Version check timeout after {timeout}s")
        return False, None, None
    except requests.exceptions.ConnectionError:
        logger.warning(f"Failed to connect to {api_url}/system/version")
        return False, None, None
    except requests.exceptions.HTTPError as e:
        logger.warning(f"HTTP error during version check: {e}")
        return False, None, None
    except Exception as e:
        logger.error(f"Version check error: {e}", exc_info=True)
        return False, None, None


def check_version_or_exit(
    api_url: Optional[str] = None,
    current_version: Optional[str] = None,
    critical_only: bool = False
) -> bool:
    """
    Check version and exit if critical update required.
    
    Args:
        api_url: Backend API URL (default: from BACKEND_URL env var)
        current_version: Current client version (default: from __version__)
        critical_only: If True, only exit on critical updates; if False, warn on any update
    
    Returns:
        True if version OK, False if update required (or exits if critical)
    """
    if api_url is None:
        api_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    
    if current_version is None:
        try:
            from r3mes import __version__
            current_version = __version__
        except ImportError:
            # Fallback version
            current_version = "0.1.0"
    
    update_required, message, update_url = check_version(
        api_url=api_url,
        current_version=current_version
    )
    
    if update_required and message:
        if critical_only:
            # Only exit on critical updates
            if "CRITICAL" in message:
                logger.error(message)
                if update_url:
                    print(f"\nUpdate URL: {update_url}\n")
                import sys
                sys.exit(1)
        else:
            # Warn on any update
            logger.warning(message)
            print(message)
            if update_url:
                print(f"Update URL: {update_url}")
    
    return not update_required


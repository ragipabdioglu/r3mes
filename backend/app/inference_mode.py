"""
Inference Mode Configuration

Controls how the Backend API handles AI inference requests.
Enables GPU-less deployment by supporting multiple inference modes.
"""

from enum import Enum
from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)


class InferenceMode(Enum):
    """
    Inference mode determines how the Backend API handles /chat requests.
    
    DISABLED: No inference available, returns 503 error
    MOCK: Returns mock responses for testing/development
    REMOTE: Proxies requests to registered Serving Nodes
    LOCAL: Runs inference locally (requires GPU)
    """
    DISABLED = "disabled"
    MOCK = "mock"
    REMOTE = "remote"
    LOCAL = "local"


# Cache the inference mode to avoid repeated env lookups
_cached_inference_mode: Optional[InferenceMode] = None


def get_inference_mode() -> InferenceMode:
    """
    Get the current inference mode from environment variable.
    
    Environment Variable: R3MES_INFERENCE_MODE
    Default: disabled (safe for GPU-less deployment)
    
    Returns:
        InferenceMode enum value
    """
    global _cached_inference_mode
    
    if _cached_inference_mode is not None:
        return _cached_inference_mode
    
    mode_str = os.getenv("R3MES_INFERENCE_MODE", "disabled").lower().strip()
    
    try:
        _cached_inference_mode = InferenceMode(mode_str)
        logger.info(f"Inference mode: {_cached_inference_mode.value}")
        return _cached_inference_mode
    except ValueError:
        logger.warning(
            f"Invalid R3MES_INFERENCE_MODE '{mode_str}'. "
            f"Valid values: {[m.value for m in InferenceMode]}. "
            f"Defaulting to 'disabled'."
        )
        _cached_inference_mode = InferenceMode.DISABLED
        return _cached_inference_mode


def should_load_ai_libraries() -> bool:
    """
    Check if AI/ML libraries (torch, transformers, etc.) should be loaded.
    
    Only returns True when inference mode is LOCAL, which requires GPU.
    All other modes (disabled, mock, remote) do not need AI libraries.
    
    Returns:
        True if AI libraries should be loaded, False otherwise
    """
    return get_inference_mode() == InferenceMode.LOCAL


def is_inference_available() -> bool:
    """
    Check if inference is available in any form.
    
    Returns:
        True if inference can be performed (mock, remote, or local)
        False if inference is disabled
    """
    return get_inference_mode() != InferenceMode.DISABLED


def get_inference_mode_description() -> str:
    """
    Get a human-readable description of the current inference mode.
    
    Returns:
        Description string for API responses
    """
    mode = get_inference_mode()
    descriptions = {
        InferenceMode.DISABLED: "Inference is disabled. No AI services available.",
        InferenceMode.MOCK: "Mock mode. Returns simulated responses for testing.",
        InferenceMode.REMOTE: "Remote mode. Requests are proxied to Serving Nodes.",
        InferenceMode.LOCAL: "Local mode. Inference runs on this server's GPU.",
    }
    return descriptions.get(mode, "Unknown inference mode")


def reset_inference_mode_cache():
    """
    Reset the cached inference mode.
    Useful for testing or when environment variables change.
    """
    global _cached_inference_mode
    _cached_inference_mode = None


def validate_inference_mode_for_startup() -> tuple[bool, str]:
    """
    Validate inference mode configuration at startup.
    
    Returns:
        Tuple of (is_valid, message)
    """
    mode = get_inference_mode()
    
    if mode == InferenceMode.LOCAL:
        # Check if GPU libraries can be imported
        try:
            import torch
            if not torch.cuda.is_available():
                return False, (
                    "R3MES_INFERENCE_MODE=local requires GPU, but no CUDA device found. "
                    "Set R3MES_INFERENCE_MODE=remote or R3MES_INFERENCE_MODE=disabled."
                )
            return True, f"Local inference mode validated. CUDA device: {torch.cuda.get_device_name(0)}"
        except ImportError:
            return False, (
                "R3MES_INFERENCE_MODE=local requires torch, but it's not installed. "
                "Install GPU dependencies or set R3MES_INFERENCE_MODE=remote."
            )
    
    return True, f"Inference mode '{mode.value}' is valid for GPU-less deployment."

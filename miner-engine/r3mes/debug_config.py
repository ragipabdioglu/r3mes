"""
Debug Configuration System for R3MES Miner Engine

Provides centralized debug configuration management with environment variable support.
"""

import os
from typing import Dict, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class DebugLevel(str, Enum):
    """Debug verbosity level"""
    MINIMAL = "minimal"  # Only critical errors and performance metrics
    STANDARD = "standard"  # Detailed logging, state inspection, performance profiling
    VERBOSE = "verbose"  # All system internals, trace logs, internal state dumps


@dataclass
class DebugConfig:
    """Debug configuration for the miner engine component"""
    
    # Global debug mode enabled
    enabled: bool = False
    
    # Debug verbosity level
    level: DebugLevel = DebugLevel.STANDARD
    
    # Enabled components (comma-separated: blockchain,backend,miner,launcher,frontend)
    # Empty or "*" means all components
    components: Set[str] = field(default_factory=set)
    
    # Feature flags
    logging: bool = True
    profiling: bool = True
    state_inspection: bool = True
    trace: bool = True
    
    # Logging configuration
    log_level: str = "TRACE"  # TRACE, DEBUG, INFO, WARN, ERROR
    log_format: str = "json"  # json, text
    log_file: Optional[str] = None
    
    # Performance profiling configuration
    profile_output: str = "~/.r3mes/profiles"
    profile_interval: int = 60  # seconds
    
    # Trace configuration
    trace_enabled: bool = True
    trace_buffer_size: int = 10000
    trace_export_path: str = "~/.r3mes/traces"
    
    # Miner-specific debug flags
    training_loop_debug: bool = True  # Debug training loop iterations
    gradient_debug: bool = True  # Debug gradient computations
    blockchain_interaction_debug: bool = True  # Debug blockchain interactions
    ipfs_debug: bool = True  # Debug IPFS operations
    
    def is_component_enabled(self, component: str) -> bool:
        """Check if debug is enabled for a specific component"""
        if not self.enabled:
            return False
        # If no components specified or "*" is present, enable all
        if not self.components or "*" in self.components:
            return True
        return component.lower() in self.components
    
    def is_miner_enabled(self) -> bool:
        """Check if debug is enabled for miner component"""
        return self.is_component_enabled("miner")


def _expand_path(path: str) -> str:
    """Expand user home directory in path"""
    if path.startswith("~/"):
        home_dir = os.path.expanduser("~")
        return path.replace("~/", f"{home_dir}/", 1)
    return path


def _get_env_bool(key: str, default: bool = False) -> bool:
    """Read a boolean environment variable"""
    value = os.getenv(key, "")
    if not value:
        return default
    return value.lower() == "true"


def _get_env_int(key: str, default: int) -> int:
    """Read an integer environment variable"""
    value = os.getenv(key, "")
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def load_debug_config() -> DebugConfig:
    """
    Load debug configuration from environment variables.
    
    Returns:
        DebugConfig instance with loaded configuration
    """
    config = DebugConfig()
    
    # Check if debug mode is enabled
    debug_mode = os.getenv("R3MES_DEBUG_MODE", "").lower()
    config.enabled = debug_mode == "true"
    
    if not config.enabled:
        # Return default (disabled) config
        return config
    
    # Load debug level
    level_str = os.getenv("R3MES_DEBUG_LEVEL", "").lower()
    try:
        config.level = DebugLevel(level_str) if level_str else DebugLevel.VERBOSE
    except ValueError:
        # Default to verbose if level is not valid
        config.level = DebugLevel.VERBOSE
    
    # Load enabled components
    components_str = os.getenv("R3MES_DEBUG_COMPONENTS", "")
    if not components_str:
        # Default to all components if not specified
        config.components = {"*"}
    else:
        components = [c.strip().lower() for c in components_str.split(",") if c.strip()]
        config.components = set(components)
    
    # Load feature flags (default to true if debug mode is enabled)
    config.logging = _get_env_bool("R3MES_DEBUG_LOGGING", True)
    config.profiling = _get_env_bool("R3MES_DEBUG_PROFILING", True)
    config.state_inspection = _get_env_bool("R3MES_DEBUG_STATE_INSPECTION", True)
    config.trace = _get_env_bool("R3MES_DEBUG_TRACE", True)
    
    # Load logging configuration
    log_level = os.getenv("R3MES_DEBUG_LOG_LEVEL", "TRACE").upper()
    valid_log_levels = {"TRACE", "DEBUG", "INFO", "WARN", "ERROR"}
    if log_level in valid_log_levels:
        config.log_level = log_level
    else:
        config.log_level = "TRACE"  # Default
    
    log_format = os.getenv("R3MES_DEBUG_LOG_FORMAT", "json").lower()
    if log_format in ("json", "text"):
        config.log_format = log_format
    else:
        config.log_format = "json"  # Default
    
    log_file = os.getenv("R3MES_DEBUG_LOG_FILE", "")
    if log_file:
        config.log_file = _expand_path(log_file)
    else:
        config.log_file = _expand_path("~/.r3mes/debug.log")
    
    # Load performance profiling configuration
    profile_output = os.getenv("R3MES_DEBUG_PROFILE_OUTPUT", "")
    if profile_output:
        config.profile_output = _expand_path(profile_output)
    else:
        config.profile_output = _expand_path("~/.r3mes/profiles")
    
    config.profile_interval = _get_env_int("R3MES_DEBUG_PROFILE_INTERVAL", 60)
    
    # Load trace configuration
    config.trace_enabled = _get_env_bool("R3MES_DEBUG_TRACE_ENABLED", True)
    config.trace_buffer_size = _get_env_int("R3MES_DEBUG_TRACE_BUFFER_SIZE", 10000)
    
    trace_export_path = os.getenv("R3MES_DEBUG_TRACE_EXPORT_PATH", "")
    if trace_export_path:
        config.trace_export_path = _expand_path(trace_export_path)
    else:
        config.trace_export_path = _expand_path("~/.r3mes/traces")
    
    # Load miner-specific debug flags
    config.training_loop_debug = _get_env_bool("R3MES_DEBUG_TRAINING_LOOP", True)
    config.gradient_debug = _get_env_bool("R3MES_DEBUG_GRADIENT", True)
    config.blockchain_interaction_debug = _get_env_bool("R3MES_DEBUG_BLOCKCHAIN", True)
    config.ipfs_debug = _get_env_bool("R3MES_DEBUG_IPFS", True)
    
    return config


def validate_debug_config(is_production: bool = False) -> Optional[str]:
    """
    Validate debug configuration for security.
    
    Args:
        is_production: Whether running in production environment
        
    Returns:
        Error message if validation fails, None otherwise
    """
    if not is_production:
        # In development/testing, debug mode is allowed
        return None
    
    # In production, check if debug mode is enabled
    debug_mode = os.getenv("R3MES_DEBUG_MODE", "").lower()
    if debug_mode == "true":
        return (
            "SECURITY ERROR: R3MES_DEBUG_MODE=true is set in production environment. "
            "Debug mode should only be used in development/testing. "
            "Please unset R3MES_DEBUG_MODE environment variable before running in production. "
            "If you need production debugging, use R3MES_DEBUG_MODE=minimal with explicit component flags."
        )
    
    # Check for verbose debug level in production (even if explicitly enabled)
    debug_level = os.getenv("R3MES_DEBUG_LEVEL", "").lower()
    if debug_level == "verbose":
        return (
            "SECURITY WARNING: R3MES_DEBUG_LEVEL=verbose is not recommended in production. "
            "Consider using 'standard' or 'minimal' level instead."
        )
    
    return None


# Global debug config instance (lazy-loaded)
_global_debug_config: Optional[DebugConfig] = None


def get_debug_config() -> DebugConfig:
    """
    Get the global debug configuration (cached).
    
    Returns:
        DebugConfig instance
    """
    global _global_debug_config
    if _global_debug_config is None:
        _global_debug_config = load_debug_config()
    return _global_debug_config

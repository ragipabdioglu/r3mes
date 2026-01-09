#!/usr/bin/env python3
"""
R3MES Advanced Configuration Management

Comprehensive configuration system with validation, hot-reloading, and environment-specific settings.
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field
import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


@dataclass
class ConfigValidationRule:
    """Configuration validation rule."""
    key: str
    required: bool = False
    type_check: Optional[type] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[Callable[[Any], bool]] = None
    description: str = ""


class ConfigFileWatcher(FileSystemEventHandler):
    """File system watcher for configuration hot-reloading."""
    
    def __init__(self, config_manager: 'AdvancedConfigManager'):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
    
    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory:
            file_path = Path(event.src_path)
            if file_path.name in self.config_manager.watched_files:
                self.logger.info(f"Configuration file changed: {file_path}")
                self.config_manager.reload_config(str(file_path))


class AdvancedConfigManager:
    """Advanced configuration manager with validation and hot-reloading."""
    
    def __init__(
        self,
        config_paths: Optional[List[str]] = None,
        enable_hot_reload: bool = True,
        enable_env_override: bool = True,
        validation_rules: Optional[List[ConfigValidationRule]] = None,
    ):
        """
        Initialize advanced configuration manager.
        
        Args:
            config_paths: List of configuration file paths
            enable_hot_reload: Enable hot-reloading of configuration files
            enable_env_override: Enable environment variable overrides
            validation_rules: List of validation rules
        """
        self.config_paths = config_paths or []
        self.enable_hot_reload = enable_hot_reload
        self.enable_env_override = enable_env_override
        self.validation_rules = validation_rules or []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Configuration storage
        self.config = {}
        self.config_sources = {}  # Track where each config value came from
        self.watched_files = set()
        
        # Hot-reload setup
        self.observer = None
        self.reload_callbacks = []
        self.lock = threading.Lock()
        
        # Load initial configuration
        self.load_all_configs()
        
        # Setup hot-reloading
        if enable_hot_reload:
            self.setup_hot_reload()
        
        self.logger.info("Advanced configuration manager initialized")
    
    def add_validation_rule(self, rule: ConfigValidationRule):
        """Add a validation rule."""
        self.validation_rules.append(rule)
    
    def add_reload_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback to be called when configuration is reloaded."""
        self.reload_callbacks.append(callback)
    
    def load_all_configs(self):
        """Load all configuration files."""
        with self.lock:
            self.config.clear()
            self.config_sources.clear()
            
            # Load from files
            for config_path in self.config_paths:
                self.load_config_file(config_path)
            
            # Apply environment variable overrides
            if self.enable_env_override:
                self.apply_env_overrides()
            
            # Validate configuration
            self.validate_config()
    
    def load_config_file(self, config_path: str):
        """Load configuration from a file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            self.logger.warning(f"Configuration file not found: {config_path}")
            return
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yml', '.yaml']:
                    file_config = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    file_config = json.load(f)
                else:
                    self.logger.error(f"Unsupported config file format: {config_path}")
                    return
            
            if file_config:
                self._merge_config(file_config, str(config_path))
                self.watched_files.add(config_path.name)
                self.logger.info(f"Loaded configuration from: {config_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading config file {config_path}: {e}")
    
    def _merge_config(self, new_config: Dict[str, Any], source: str):
        """Merge new configuration with existing configuration."""
        def merge_dict(target: Dict[str, Any], source_dict: Dict[str, Any], path: str = ""):
            for key, value in source_dict.items():
                full_key = f"{path}.{key}" if path else key
                
                if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                    merge_dict(target[key], value, full_key)
                else:
                    target[key] = value
                    self.config_sources[full_key] = source
        
        merge_dict(self.config, new_config)
    
    def apply_env_overrides(self):
        """Apply environment variable overrides."""
        env_prefix = "R3MES_"
        
        for env_key, env_value in os.environ.items():
            if env_key.startswith(env_prefix):
                # Convert environment variable name to config key
                config_key = env_key[len(env_prefix):].lower().replace('_', '.')
                
                # Try to parse value as JSON, fallback to string
                try:
                    parsed_value = json.loads(env_value)
                except (json.JSONDecodeError, ValueError):
                    parsed_value = env_value
                
                # Set nested configuration
                self._set_nested_config(config_key, parsed_value)
                self.config_sources[config_key] = f"env:{env_key}"
                
                self.logger.debug(f"Applied environment override: {config_key} = {parsed_value}")
    
    def _set_nested_config(self, key: str, value: Any):
        """Set nested configuration value using dot notation."""
        keys = key.split('.')
        current = self.config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def validate_config(self):
        """Validate configuration against rules."""
        errors = []
        
        for rule in self.validation_rules:
            try:
                value = self.get(rule.key)
                
                # Check if required
                if rule.required and value is None:
                    errors.append(f"Required configuration missing: {rule.key}")
                    continue
                
                if value is None:
                    continue  # Skip validation for optional missing values
                
                # Type check
                if rule.type_check and not isinstance(value, rule.type_check):
                    errors.append(f"Invalid type for {rule.key}: expected {rule.type_check.__name__}, got {type(value).__name__}")
                
                # Range checks
                if rule.min_value is not None and value < rule.min_value:
                    errors.append(f"Value for {rule.key} below minimum: {value} < {rule.min_value}")
                
                if rule.max_value is not None and value > rule.max_value:
                    errors.append(f"Value for {rule.key} above maximum: {value} > {rule.max_value}")
                
                # Allowed values check
                if rule.allowed_values and value not in rule.allowed_values:
                    errors.append(f"Invalid value for {rule.key}: {value} not in {rule.allowed_values}")
                
                # Custom validation
                if rule.custom_validator and not rule.custom_validator(value):
                    errors.append(f"Custom validation failed for {rule.key}: {value}")
                    
            except Exception as e:
                errors.append(f"Error validating {rule.key}: {e}")
        
        if errors:
            error_msg = "Configuration validation errors:\n" + "\n".join(f"  - {error}" for error in errors)
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.info("Configuration validation passed")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        current = self.config
        
        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any, source: str = "runtime"):
        """Set configuration value using dot notation."""
        with self.lock:
            self._set_nested_config(key, value)
            self.config_sources[key] = source
            self.logger.debug(f"Configuration updated: {key} = {value} (source: {source})")
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values."""
        with self.lock:
            return self.config.copy()
    
    def get_source(self, key: str) -> Optional[str]:
        """Get the source of a configuration value."""
        return self.config_sources.get(key)
    
    def reload_config(self, config_path: Optional[str] = None):
        """Reload configuration from files."""
        self.logger.info("Reloading configuration...")
        
        try:
            if config_path:
                # Reload specific file
                self.load_config_file(config_path)
            else:
                # Reload all files
                self.load_all_configs()
            
            # Notify callbacks
            for callback in self.reload_callbacks:
                try:
                    callback(self.config.copy())
                except Exception as e:
                    self.logger.error(f"Error in reload callback: {e}")
            
            self.logger.info("Configuration reloaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error reloading configuration: {e}")
    
    def setup_hot_reload(self):
        """Setup hot-reloading of configuration files."""
        if not self.config_paths:
            return
        
        try:
            self.observer = Observer()
            event_handler = ConfigFileWatcher(self)
            
            # Watch directories containing config files
            watched_dirs = set()
            for config_path in self.config_paths:
                config_dir = Path(config_path).parent
                if config_dir not in watched_dirs:
                    self.observer.schedule(event_handler, str(config_dir), recursive=False)
                    watched_dirs.add(config_dir)
            
            self.observer.start()
            self.logger.info("Configuration hot-reloading enabled")
            
        except Exception as e:
            self.logger.error(f"Failed to setup hot-reloading: {e}")
    
    def export_config(self, output_path: str, format: str = "yaml", include_sources: bool = False):
        """Export current configuration to file."""
        output_path = Path(output_path)
        
        export_data = self.config.copy()
        
        if include_sources:
            export_data["_sources"] = self.config_sources.copy()
        
        try:
            with open(output_path, 'w') as f:
                if format.lower() == "yaml":
                    yaml.dump(export_data, f, default_flow_style=False, indent=2)
                elif format.lower() == "json":
                    json.dump(export_data, f, indent=2)
                else:
                    raise ValueError(f"Unsupported export format: {format}")
            
            self.logger.info(f"Configuration exported to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting configuration: {e}")
            raise
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary with metadata."""
        return {
            "total_keys": len(self.config_sources),
            "config_files": len(self.config_paths),
            "watched_files": list(self.watched_files),
            "hot_reload_enabled": self.enable_hot_reload,
            "env_override_enabled": self.enable_env_override,
            "validation_rules": len(self.validation_rules),
            "sources": {
                source: len([k for k, s in self.config_sources.items() if s == source])
                for source in set(self.config_sources.values())
            }
        }
    
    def validate_key(self, key: str, value: Any) -> List[str]:
        """Validate a specific key-value pair."""
        errors = []
        
        for rule in self.validation_rules:
            if rule.key == key:
                try:
                    # Type check
                    if rule.type_check and not isinstance(value, rule.type_check):
                        errors.append(f"Invalid type: expected {rule.type_check.__name__}, got {type(value).__name__}")
                    
                    # Range checks
                    if rule.min_value is not None and value < rule.min_value:
                        errors.append(f"Value below minimum: {value} < {rule.min_value}")
                    
                    if rule.max_value is not None and value > rule.max_value:
                        errors.append(f"Value above maximum: {value} > {rule.max_value}")
                    
                    # Allowed values check
                    if rule.allowed_values and value not in rule.allowed_values:
                        errors.append(f"Invalid value: {value} not in {rule.allowed_values}")
                    
                    # Custom validation
                    if rule.custom_validator and not rule.custom_validator(value):
                        errors.append(f"Custom validation failed: {value}")
                        
                except Exception as e:
                    errors.append(f"Validation error: {e}")
        
        return errors
    
    def stop(self):
        """Stop the configuration manager."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
        self.logger.info("Configuration manager stopped")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def create_default_validation_rules() -> List[ConfigValidationRule]:
    """Create default validation rules for R3MES configuration."""
    return [
        # Blockchain configuration
        ConfigValidationRule(
            key="blockchain.url",
            required=True,
            type_check=str,
            description="Blockchain gRPC URL"
        ),
        ConfigValidationRule(
            key="blockchain.chain_id",
            required=True,
            type_check=str,
            description="Blockchain chain ID"
        ),
        ConfigValidationRule(
            key="blockchain.private_key",
            required=True,
            type_check=str,
            custom_validator=lambda x: len(x) == 64,  # 32 bytes hex
            description="Private key (64 hex characters)"
        ),
        
        # Mining configuration
        ConfigValidationRule(
            key="mining.batch_size",
            type_check=int,
            min_value=1,
            max_value=128,
            description="Training batch size"
        ),
        ConfigValidationRule(
            key="mining.learning_rate",
            type_check=float,
            min_value=1e-6,
            max_value=1.0,
            description="Learning rate"
        ),
        ConfigValidationRule(
            key="mining.lora_rank",
            type_check=int,
            min_value=1,
            max_value=256,
            description="LoRA rank"
        ),
        
        # Performance configuration
        ConfigValidationRule(
            key="performance.max_memory_mb",
            type_check=int,
            min_value=512,
            description="Maximum memory usage in MB"
        ),
        ConfigValidationRule(
            key="performance.gpu_memory_fraction",
            type_check=float,
            min_value=0.1,
            max_value=1.0,
            description="GPU memory fraction to use"
        ),
        
        # Logging configuration
        ConfigValidationRule(
            key="logging.level",
            type_check=str,
            allowed_values=["DEBUG", "INFO", "WARNING", "ERROR"],
            description="Logging level"
        ),
        
        # Network configuration
        ConfigValidationRule(
            key="network.timeout_seconds",
            type_check=int,
            min_value=1,
            max_value=300,
            description="Network timeout in seconds"
        ),
    ]


def create_config_manager(
    config_files: Optional[List[str]] = None,
    enable_hot_reload: bool = True,
    enable_validation: bool = True,
) -> AdvancedConfigManager:
    """
    Create advanced configuration manager with default settings.
    
    Args:
        config_files: List of configuration files to load
        enable_hot_reload: Enable hot-reloading
        enable_validation: Enable configuration validation
        
    Returns:
        AdvancedConfigManager instance
    """
    # Default config files
    if config_files is None:
        config_files = [
            "config/default.yaml",
            "config/local.yaml",
            os.path.expanduser("~/.r3mes/config.yaml"),
        ]
    
    # Validation rules
    validation_rules = create_default_validation_rules() if enable_validation else []
    
    return AdvancedConfigManager(
        config_paths=config_files,
        enable_hot_reload=enable_hot_reload,
        enable_env_override=True,
        validation_rules=validation_rules,
    )
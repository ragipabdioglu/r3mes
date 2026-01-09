"""
Configuration Management for R3MES Miner

Handles configuration file loading, saving, and validation.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """Manages miner configuration files."""
    
    def __init__(self):
        self.config_dir = self._get_config_directory()
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_config_directory(self) -> Path:
        """Get configuration directory path."""
        # Use ~/.r3mes/config on Linux/macOS, %APPDATA%\R3MES\config on Windows
        if os.name == 'nt':  # Windows
            base_dir = Path(os.getenv('APPDATA', Path.home() / 'AppData' / 'Roaming'))
            return base_dir / 'R3MES' / 'config'
        else:  # Linux/macOS
            return Path.home() / '.r3mes' / 'config'
    
    def get_default_config_path(self) -> Path:
        """Get default configuration file path."""
        return self.config_dir / 'config.json'
    
    def get_wallet_path(self) -> Path:
        """Get default wallet path (for status command)."""
        if os.name == 'nt':  # Windows
            base_dir = Path(os.getenv('APPDATA', Path.home() / 'AppData' / 'Roaming'))
            return base_dir / 'R3MES' / 'wallets' / 'default_wallet.json'
        else:  # Linux/macOS
            return Path.home() / '.r3mes' / 'wallets' / 'default_wallet.json'
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # LoRA-Enforced Architecture: Block full fine-tuning
        from r3mes.miner.model_loader import check_full_finetune_config
        check_full_finetune_config(config)
        
        # Validate required fields
        required_fields = ['private_key', 'blockchain_url', 'chain_id']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required configuration field: {field}")
        
        return config
    
    def save_config(self, config: Dict[str, Any], config_name: str = 'config') -> str:
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary
            config_name: Name for the config file (without extension)
            
        Returns:
            Path to the saved configuration file
        """
        config_path = self.config_dir / f'{config_name}.json'
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Set secure permissions (Unix only)
        if os.name != 'nt':
            os.chmod(config_path, 0o600)
        
        return str(config_path)


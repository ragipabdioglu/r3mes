"""
Wallet Management for R3MES Miner

Handles wallet creation, key generation, and address derivation.
"""

import os
import json
from pathlib import Path
from typing import Optional
from ecdsa import SigningKey, SECP256k1
import hashlib
import base64


class WalletManager:
    """Manages wallet creation and key operations."""
    
    def __init__(self):
        self.wallet_dir = self._get_wallet_directory()
        self.wallet_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_wallet_directory(self) -> Path:
        """Get wallet directory path."""
        # Use ~/.r3mes/wallets on Linux/macOS, %APPDATA%\R3MES\wallets on Windows
        if os.name == 'nt':  # Windows
            base_dir = Path(os.getenv('APPDATA', Path.home() / 'AppData' / 'Roaming'))
            return base_dir / 'R3MES' / 'wallets'
        else:  # Linux/macOS
            return Path.home() / '.r3mes' / 'wallets'
    
    def get_default_wallet_path(self) -> Path:
        """Get default wallet path."""
        return self.wallet_dir / 'default_wallet.json'
    
    def create_wallet(self, wallet_name: str = 'default_wallet') -> str:
        """
        Create a new wallet.
        
        Args:
            wallet_name: Name for the wallet file (without extension)
            
        Returns:
            Path to the created wallet file
        """
        # Generate new keypair
        sk = SigningKey.generate(curve=SECP256k1)
        vk = sk.get_verifying_key()
        
        private_key_bytes = sk.to_string()
        public_key_bytes = vk.to_string()
        
        # Derive address from public key (Cosmos SDK compatible)
        address = self._derive_address_from_public_key(public_key_bytes)
        
        # Create wallet data
        wallet_data = {
            'address': address,
            'private_key': private_key_bytes.hex(),
            'public_key': public_key_bytes.hex(),
        }
        
        # Save wallet file
        wallet_path = self.wallet_dir / f'{wallet_name}.json'
        with open(wallet_path, 'w') as f:
            json.dump(wallet_data, f, indent=2)
        
        # Set secure permissions (Unix only)
        if os.name != 'nt':
            os.chmod(wallet_path, 0o600)
        
        return str(wallet_path)
    
    def get_address(self, wallet_path: str) -> str:
        """Get address from wallet file."""
        wallet_data = self._load_wallet(wallet_path)
        return wallet_data['address']
    
    def get_private_key(self, wallet_path: str) -> str:
        """Get private key from wallet file."""
        wallet_data = self._load_wallet(wallet_path)
        return wallet_data['private_key']
    
    def _load_wallet(self, wallet_path: str) -> dict:
        """Load wallet data from file."""
        if not os.path.exists(wallet_path):
            raise FileNotFoundError(f"Wallet not found: {wallet_path}")
        
        with open(wallet_path, 'r') as f:
            wallet_data = json.load(f)
        
        # Validate wallet structure
        required_keys = ['address', 'private_key', 'public_key']
        for key in required_keys:
            if key not in wallet_data:
                raise ValueError(f"Invalid wallet format: missing {key}")
        
        return wallet_data
    
    def _derive_address_from_public_key(self, public_key_bytes: bytes) -> str:
        """
        Derive Cosmos SDK compatible address from public key.
        
        Uses SHA256 hash of public key, then takes first 20 bytes and encodes as bech32.
        For simplicity, we'll use hex encoding here (can be upgraded to bech32 later).
        """
        # SHA256 hash of public key
        hash_obj = hashlib.sha256(public_key_bytes)
        hash_bytes = hash_obj.digest()
        
        # Take first 20 bytes (standard Ethereum/Cosmos address length)
        address_bytes = hash_bytes[:20]
        
        # Encode as hex (for now - can be upgraded to bech32)
        address = address_bytes.hex()
        
        return address


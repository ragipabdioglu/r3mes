"""
Transaction Builder for Cosmos SDK

Builds and signs Cosmos SDK transactions for broadcasting.
"""

import hashlib
import logging
from typing import Dict, Any, Optional
import struct

from bridge.crypto import (
    sign_message,
    derive_address_from_public_key,
    hex_to_private_key,
    private_key_to_public_key,
)

logger = logging.getLogger(__name__)


class TransactionBuilder:
    """
    Builds Cosmos SDK transactions.
    
    Note: This is a simplified implementation. For production,
    consider using a library like `cosmpy` for full Cosmos SDK support.
    """
    
    def __init__(self, chain_id: str, private_key: str):
        """
        Initialize transaction builder.
        
        Args:
            chain_id: Chain ID
            private_key: Private key (hex string)
        """
        self.chain_id = chain_id
        self.private_key_hex = private_key
        self.private_key_bytes = hex_to_private_key(private_key)
        self.public_key_bytes = private_key_to_public_key(self.private_key_bytes)
        self.address = derive_address_from_public_key(self.public_key_bytes)
    
    def build_submit_gradient_tx(
        self,
        msg: Dict[str, Any],
        sequence: int,
        account_number: int,
        fee: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        """
        Build and sign a SubmitGradient transaction.
        
        This is a simplified implementation. For production, use proper
        Cosmos SDK transaction encoding (Amino or Protobuf).
        
        Args:
            msg: Message dictionary with gradient submission data
            sequence: Account sequence number
            account_number: Account number
            fee: Transaction fee (optional)
            
        Returns:
            Signed transaction bytes (simplified format)
            
        Note:
            This is a placeholder implementation. Full Cosmos SDK transaction
            building requires proper encoding of:
            - Transaction body (messages, memo, timeout_height)
            - AuthInfo (signer_infos, fee)
            - Signatures
            - Transaction encoding (Amino or Protobuf)
            
            For production, use a library like `cosmpy` or implement full
            Cosmos SDK transaction encoding.
        """
        # Simplified transaction building
        # In production, this should use proper Cosmos SDK transaction encoding
        
        # Create transaction body hash
        tx_body_hash = self._create_tx_body_hash(msg, sequence, account_number, fee)
        
        # Sign transaction body hash
        signature = sign_message(tx_body_hash, self.private_key_bytes)
        
        # Create simplified transaction (for demonstration)
        # In production, use proper Cosmos SDK transaction encoding
        tx_data = {
            "body": {
                "messages": [msg],
                "memo": "",
                "timeout_height": 0,
            },
            "auth_info": {
                "signer_infos": [{
                    "public_key": self.public_key_bytes.hex(),
                    "mode_info": {"single": {"mode": "SIGN_MODE_DIRECT"}},
                    "sequence": sequence,
                }],
                "fee": fee or {"amount": [], "gas_limit": 200000},
            },
            "signatures": [signature.hex()],
        }
        
        # Serialize to bytes (simplified - use proper encoding in production)
        tx_bytes = self._serialize_tx(tx_data)
        
        return tx_bytes
    
    def _create_tx_body_hash(
        self,
        msg: Dict[str, Any],
        sequence: int,
        account_number: int,
        fee: Optional[Dict[str, Any]],
    ) -> bytes:
        """Create hash of transaction body for signing."""
        # Simplified hash creation
        # In production, use proper Cosmos SDK transaction body encoding
        data = f"{self.chain_id}|{account_number}|{sequence}|{msg}"
        return hashlib.sha256(data.encode()).digest()
    
    def _serialize_tx(self, tx_data: Dict[str, Any]) -> bytes:
        """
        Serialize transaction to bytes.
        
        Note: This is a simplified serialization. For production,
        use proper Cosmos SDK transaction encoding (Protobuf).
        """
        # Simplified serialization (JSON for now)
        # In production, use Protobuf encoding
        import json
        return json.dumps(tx_data).encode("utf-8")


def private_key_to_public_key(private_key: bytes) -> bytes:
    """
    Derive public key from private key.
    
    Args:
        private_key: Private key bytes
        
    Returns:
        Public key bytes
    """
    from ecdsa import SigningKey, SECP256k1
    
    sk = SigningKey.from_string(private_key, curve=SECP256k1)
    vk = sk.get_verifying_key()
    return vk.to_string()


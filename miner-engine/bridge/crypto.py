"""
Cryptographic utilities for message signing and authentication.

Implements Secp256k1 signing compatible with Cosmos SDK.
"""

import hashlib
from typing import Tuple, Optional
from Crypto.Hash import SHA256
from ecdsa import SigningKey, SECP256k1, VerifyingKey
from ecdsa.util import sigencode_der_canonize
import base64


def generate_keypair() -> Tuple[bytes, bytes]:
    """
    Generate a new Secp256k1 keypair.
    
    Returns:
        Tuple of (private_key_bytes, public_key_bytes)
    """
    sk = SigningKey.generate(curve=SECP256k1)
    vk = sk.get_verifying_key()
    return sk.to_string(), vk.to_string()


def private_key_to_hex(private_key: bytes) -> str:
    """Convert private key bytes to hex string."""
    return private_key.hex()


def hex_to_private_key(hex_key: str) -> bytes:
    """Convert hex string to private key bytes."""
    return bytes.fromhex(hex_key)


def sign_message(message_bytes: bytes, private_key: bytes) -> bytes:
    """
    Sign a message with Secp256k1 private key.
    
    Args:
        message_bytes: Message to sign
        private_key: Private key bytes
        
    Returns:
        Signature bytes (DER format)
    """
    sk = SigningKey.from_string(private_key, curve=SECP256k1)
    hash_obj = SHA256.new(message_bytes)
    signature = sk.sign_digest(hash_obj.digest(), sigencode=sigencode_der_canonize)
    return signature


def verify_signature(message_bytes: bytes, signature: bytes, public_key: bytes) -> bool:
    """
    Verify a message signature.
    
    Args:
        message_bytes: Original message
        signature: Signature bytes
        public_key: Public key bytes
        
    Returns:
        True if signature is valid
    """
    try:
        vk = VerifyingKey.from_string(public_key, curve=SECP256k1)
        hash_obj = SHA256.new(message_bytes)
        vk.verify_digest(signature, hash_obj.digest())
        return True
    except Exception:
        return False


def derive_address_from_public_key(public_key: bytes) -> str:
    """
    Derive Cosmos address from public key.
    
    This is a simplified version. In production, use Cosmos SDK's address derivation.
    
    Args:
        public_key: Public key bytes
        
    Returns:
        Cosmos address string (simplified)
    """
    # Simplified: use hash of public key as address
    # In production, use bech32 encoding with proper prefix
    hash_obj = hashlib.sha256(public_key)
    address_hex = hash_obj.hexdigest()[:40]  # 20 bytes = 40 hex chars
    return f"cosmos1{address_hex}"


def create_message_hash(
    miner: str,
    ipfs_hash: str,
    model_version: str,
    training_round_id: int,
    shard_id: int,
    gradient_hash: str,
    gpu_architecture: str,
    nonce: int,
    chain_id: str,
) -> bytes:
    """
    Create deterministic message hash for signing.
    
    Args:
        miner: Miner address
        ipfs_hash: IPFS hash
        model_version: Model version
        training_round_id: Training round ID
        shard_id: Shard ID
        gradient_hash: Gradient hash
        gpu_architecture: GPU architecture
        nonce: Nonce value
        chain_id: Chain ID
        
    Returns:
        Message hash bytes
    """
    # Create deterministic message string
    message_str = (
        f"{chain_id}|{miner}|{ipfs_hash}|{model_version}|"
        f"{training_round_id}|{shard_id}|{gradient_hash}|{gpu_architecture}|{nonce}"
    )
    message_bytes = message_str.encode('utf-8')
    
    # Hash the message
    hash_obj = hashlib.sha256(message_bytes)
    return hash_obj.digest()


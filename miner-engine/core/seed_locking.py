"""
Global Seed Locking for Deterministic Execution

Ensures that all miners use the same global seed derived from blockchain state.
"""

from typing import Optional
import hashlib
import struct


def derive_seed_from_block_hash(block_hash: bytes, training_round_id: int) -> int:
    """
    Derive deterministic seed from block hash and training round ID.
    
    This matches the Go implementation in seed_locking.go.
    
    Args:
        block_hash: Block hash from blockchain (32 bytes)
        training_round_id: Training round ID
        
    Returns:
        Deterministic seed as integer (uint64)
    """
    # Combine block hash and training round ID
    combined = block_hash + struct.pack('>Q', training_round_id)  # Big-endian uint64
    
    # Hash to get deterministic seed
    seed_hash = hashlib.sha256(combined).digest()
    
    # Convert first 8 bytes to uint64
    seed = struct.unpack('>Q', seed_hash[:8])[0]  # Big-endian uint64
    
    return seed


def lock_seed_for_training(block_hash_hex: str, training_round_id: int) -> int:
    """
    Lock seed for training using block hash and training round.
    
    Args:
        block_hash_hex: Block hash as hex string
        training_round_id: Training round ID
        
    Returns:
        Locked seed as integer
    """
    # Convert hex string to bytes
    try:
        block_hash = bytes.fromhex(block_hash_hex)
    except ValueError:
        # If invalid hex, use hash of the string
        block_hash = hashlib.sha256(block_hash_hex.encode()).digest()
    
    # Ensure block hash is 32 bytes
    if len(block_hash) != 32:
        # Pad or truncate to 32 bytes
        if len(block_hash) < 32:
            block_hash = block_hash + b'\x00' * (32 - len(block_hash))
        else:
            block_hash = block_hash[:32]
    
    # Derive seed
    seed = derive_seed_from_block_hash(block_hash, training_round_id)
    
    return seed


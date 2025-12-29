"""
Proof-of-Work implementation for anti-spam protection.

Implements SHA256-based proof-of-work: find nonce such that
SHA256(message_hash + nonce) has N leading zeros.
"""

import hashlib
import struct
from typing import Optional


def calculate_proof_of_work(message_hash: bytes, difficulty: int = 4) -> Optional[int]:
    """
    Calculate proof-of-work for a message hash.
    
    Args:
        message_hash: SHA256 hash of the message
        difficulty: Number of leading zeros required (default: 4)
        
    Returns:
        Nonce value that satisfies proof-of-work, or None if not found
    """
    max_attempts = 10000
    
    for nonce in range(max_attempts):
        # Create proof-of-work input: message_hash + nonce
        nonce_bytes = struct.pack('>Q', nonce)  # Big-endian uint64
        pow_input = message_hash + nonce_bytes
        
        # Compute hash
        hash_obj = hashlib.sha256(pow_input)
        hash_bytes = hash_obj.digest()
        
        # Check leading zeros
        required_zeros = difficulty
        zero_bytes = required_zeros // 8
        zero_bits = required_zeros % 8
        
        # Check full zero bytes
        valid = True
        for i in range(zero_bytes):
            if hash_bytes[i] != 0:
                valid = False
                break
        
        if not valid:
            continue
        
        # Check partial zero byte
        if zero_bits > 0:
            mask = (0xFF << (8 - zero_bits)) & 0xFF
            if hash_bytes[zero_bytes] & mask != 0:
                valid = False
        
        if valid:
            return nonce
    
    return None


def verify_proof_of_work(message_hash: bytes, nonce: int, difficulty: int = 4) -> bool:
    """
    Verify proof-of-work for a message hash and nonce.
    
    Args:
        message_hash: SHA256 hash of the message
        nonce: Nonce value to verify
        difficulty: Number of leading zeros required (default: 4)
        
    Returns:
        True if proof-of-work is valid
    """
    # Create proof-of-work input
    nonce_bytes = struct.pack('>Q', nonce)
    pow_input = message_hash + nonce_bytes
    
    # Compute hash
    hash_obj = hashlib.sha256(pow_input)
    hash_bytes = hash_obj.digest()
    
    # Check leading zeros
    required_zeros = difficulty
    zero_bytes = required_zeros // 8
    zero_bits = required_zeros % 8
    
    # Check full zero bytes
    for i in range(zero_bytes):
        if hash_bytes[i] != 0:
            return False
    
    # Check partial zero byte
    if zero_bits > 0:
        mask = (0xFF << (8 - zero_bits)) & 0xFF
        if hash_bytes[zero_bytes] & mask != 0:
            return False
    
    return True


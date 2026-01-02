"""
Proof-of-Work implementation for anti-spam protection.

Implements SHA256-based proof-of-work: find nonce such that
SHA256(message_hash + nonce) has N leading zeros.

SECURITY: Default difficulty increased from 4 to 20 bits for production security.
"""

import hashlib
import struct
import os
import time
from typing import Optional, Tuple


def get_difficulty() -> int:
    """Get PoW difficulty from environment or use secure default."""
    env_difficulty = os.getenv("R3MES_POW_DIFFICULTY")
    if env_difficulty:
        try:
            difficulty = int(env_difficulty)
            # Validate difficulty range
            if difficulty < 16:
                raise ValueError(f"Minimum difficulty is 16 bits, got {difficulty}")
            if difficulty > 32:
                raise ValueError(f"Maximum difficulty is 32 bits, got {difficulty}")
            return difficulty
        except ValueError as e:
            raise ValueError(f"Invalid R3MES_POW_DIFFICULTY: {e}")
    
    # Production default: 20 bits (was 4 bits - SECURITY FIX)
    return 20


def calculate_proof_of_work(
    message_hash: bytes, 
    difficulty: Optional[int] = None,
    max_attempts: int = 1000000,
    timeout_seconds: int = 30,
) -> Optional[Tuple[int, float]]:
    """
    Calculate proof-of-work for a message hash.
    
    Args:
        message_hash: SHA256 hash of the message
        difficulty: Number of leading zeros required (default: from environment or 20)
        max_attempts: Maximum number of attempts (default: 1M)
        timeout_seconds: Maximum time to spend (default: 30s)
        
    Returns:
        (nonce, computation_time) tuple if found, None if not found
    """
    if difficulty is None:
        difficulty = get_difficulty()
    
    start_time = time.time()
    
    # Validate inputs
    if len(message_hash) != 32:
        raise ValueError("message_hash must be 32 bytes (SHA256)")
    if difficulty < 1 or difficulty > 32:
        raise ValueError("difficulty must be between 1 and 32 bits")
    
    for nonce in range(max_attempts):
        # Check timeout
        if time.time() - start_time > timeout_seconds:
            return None
        
        # Create proof-of-work input: message_hash + nonce
        nonce_bytes = struct.pack('>Q', nonce)  # Big-endian uint64
        pow_input = message_hash + nonce_bytes
        
        # Compute hash
        hash_obj = hashlib.sha256(pow_input)
        hash_bytes = hash_obj.digest()
        
        # Check leading zeros
        if _has_leading_zeros(hash_bytes, difficulty):
            computation_time = time.time() - start_time
            return nonce, computation_time
    
    return None


def verify_proof_of_work(
    message_hash: bytes, 
    nonce: int, 
    difficulty: Optional[int] = None
) -> bool:
    """
    Verify proof-of-work for a message hash and nonce.
    
    Args:
        message_hash: SHA256 hash of the message
        nonce: Nonce value to verify
        difficulty: Number of leading zeros required (default: from environment or 20)
        
    Returns:
        True if proof-of-work is valid
    """
    if difficulty is None:
        difficulty = get_difficulty()
    
    # Validate inputs
    if len(message_hash) != 32:
        raise ValueError("message_hash must be 32 bytes (SHA256)")
    if difficulty < 1 or difficulty > 32:
        raise ValueError("difficulty must be between 1 and 32 bits")
    if nonce < 0 or nonce >= 2**64:
        raise ValueError("nonce must be a valid uint64")
    
    # Create proof-of-work input
    nonce_bytes = struct.pack('>Q', nonce)
    pow_input = message_hash + nonce_bytes
    
    # Compute hash
    hash_obj = hashlib.sha256(pow_input)
    hash_bytes = hash_obj.digest()
    
    # Check leading zeros
    return _has_leading_zeros(hash_bytes, difficulty)


def _has_leading_zeros(hash_bytes: bytes, difficulty: int) -> bool:
    """Check if hash has required number of leading zeros."""
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


def estimate_computation_time(difficulty: int) -> float:
    """
    Estimate average computation time for given difficulty.
    
    Args:
        difficulty: Number of leading zeros required
        
    Returns:
        Estimated computation time in seconds
    """
    # Rough estimate based on average hash rate
    # Assumes ~1M hashes per second on typical hardware
    expected_attempts = 2 ** difficulty
    hash_rate = 1_000_000  # hashes per second
    return expected_attempts / hash_rate


def get_difficulty_info() -> dict:
    """Get information about current PoW difficulty."""
    difficulty = get_difficulty()
    return {
        "difficulty": difficulty,
        "expected_attempts": 2 ** difficulty,
        "estimated_time_seconds": estimate_computation_time(difficulty),
        "security_level": "high" if difficulty >= 20 else "medium" if difficulty >= 16 else "low",
    }


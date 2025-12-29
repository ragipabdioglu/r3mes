#!/usr/bin/env python3
"""
R3MES Core Validation

Input validation utilities for the miner-engine.
"""

import re
from typing import Optional, Tuple, Any
import torch

from core.constants import (
    CHUNK_SIZE_TOKENS,
    MIN_CHUNK_SIZE_TOKENS,
    MAX_SEQUENCE_LENGTH,
    COSINE_SIMILARITY_THRESHOLD,
    MIN_BOND_AMOUNT,
)
from core.exceptions import (
    ValidationError,
    InvalidAddressError,
    InvalidHashError,
    InvalidConfigError,
    ChunkSizeError,
)


# =============================================================================
# ADDRESS VALIDATION
# =============================================================================

# Cosmos SDK address pattern (bech32)
COSMOS_ADDRESS_PATTERN = re.compile(r'^remes1[a-z0-9]{38}$')

# Ethereum-style address pattern (hex)
ETH_ADDRESS_PATTERN = re.compile(r'^0x[a-fA-F0-9]{40}$')


def validate_address(address: str, address_type: str = "cosmos") -> bool:
    """
    Validate blockchain address format.
    
    Args:
        address: Address string to validate
        address_type: "cosmos" or "eth"
    
    Returns:
        True if valid
    
    Raises:
        InvalidAddressError: If address is invalid
    """
    if not address or not isinstance(address, str):
        raise InvalidAddressError(str(address), "empty or not a string")
    
    if address_type == "cosmos":
        if not COSMOS_ADDRESS_PATTERN.match(address):
            raise InvalidAddressError(address, "invalid Cosmos address format")
    elif address_type == "eth":
        if not ETH_ADDRESS_PATTERN.match(address):
            raise InvalidAddressError(address, "invalid Ethereum address format")
    else:
        raise InvalidAddressError(address, f"unknown address type: {address_type}")
    
    return True


# =============================================================================
# HASH VALIDATION
# =============================================================================

# SHA256 hash pattern (64 hex chars)
SHA256_PATTERN = re.compile(r'^[a-fA-F0-9]{64}$')

# IPFS CID v0 pattern (Qm...)
IPFS_CID_V0_PATTERN = re.compile(r'^Qm[1-9A-HJ-NP-Za-km-z]{44}$')

# IPFS CID v1 pattern (bafy...)
IPFS_CID_V1_PATTERN = re.compile(r'^b[a-z2-7]{58}$')


def validate_hash(hash_value: str, hash_type: str = "sha256") -> bool:
    """
    Validate hash format.
    
    Args:
        hash_value: Hash string to validate
        hash_type: "sha256" or "ipfs"
    
    Returns:
        True if valid
    
    Raises:
        InvalidHashError: If hash is invalid
    """
    if not hash_value or not isinstance(hash_value, str):
        raise InvalidHashError(str(hash_value), hash_type)
    
    if hash_type == "sha256":
        if not SHA256_PATTERN.match(hash_value):
            raise InvalidHashError(hash_value, "SHA256 (64 hex chars)")
    elif hash_type == "ipfs":
        if not (IPFS_CID_V0_PATTERN.match(hash_value) or 
                IPFS_CID_V1_PATTERN.match(hash_value)):
            raise InvalidHashError(hash_value, "IPFS CID (Qm... or bafy...)")
    else:
        raise InvalidHashError(hash_value, f"unknown hash type: {hash_type}")
    
    return True


def is_valid_ipfs_hash(hash_value: str) -> bool:
    """Check if string is a valid IPFS hash (non-throwing)."""
    if not hash_value or not isinstance(hash_value, str):
        return False
    return bool(IPFS_CID_V0_PATTERN.match(hash_value) or 
                IPFS_CID_V1_PATTERN.match(hash_value))


# =============================================================================
# CHUNK VALIDATION
# =============================================================================

def validate_chunk_size(token_count: int, strict: bool = True) -> bool:
    """
    Validate chunk token count.
    
    Args:
        token_count: Number of tokens in chunk
        strict: If True, require exact CHUNK_SIZE_TOKENS
    
    Returns:
        True if valid
    
    Raises:
        ChunkSizeError: If chunk size is invalid
    """
    if strict:
        if token_count != CHUNK_SIZE_TOKENS:
            raise ChunkSizeError(token_count, CHUNK_SIZE_TOKENS)
    else:
        if token_count < MIN_CHUNK_SIZE_TOKENS:
            raise ChunkSizeError(token_count, MIN_CHUNK_SIZE_TOKENS)
        if token_count > MAX_SEQUENCE_LENGTH:
            raise ChunkSizeError(token_count, MAX_SEQUENCE_LENGTH)
    
    return True


def validate_tensor_shape(
    tensor: torch.Tensor,
    expected_dims: int,
    name: str = "tensor",
) -> bool:
    """
    Validate tensor dimensions.
    
    Args:
        tensor: Tensor to validate
        expected_dims: Expected number of dimensions
        name: Name for error messages
    
    Returns:
        True if valid
    
    Raises:
        ValidationError: If tensor shape is invalid
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValidationError(f"{name} must be a torch.Tensor")
    
    if tensor.dim() != expected_dims:
        raise ValidationError(
            f"{name} has {tensor.dim()} dims, expected {expected_dims}"
        )
    
    return True


# =============================================================================
# CONFIG VALIDATION
# =============================================================================

def validate_learning_rate(lr: float) -> bool:
    """Validate learning rate is in reasonable range."""
    if not isinstance(lr, (int, float)):
        raise InvalidConfigError("learning_rate", lr, "must be a number")
    if lr <= 0:
        raise InvalidConfigError("learning_rate", lr, "must be positive")
    if lr > 1.0:
        raise InvalidConfigError("learning_rate", lr, "must be <= 1.0")
    return True


def validate_lora_rank(rank: int) -> bool:
    """Validate LoRA rank."""
    if not isinstance(rank, int):
        raise InvalidConfigError("lora_rank", rank, "must be an integer")
    if rank < 1:
        raise InvalidConfigError("lora_rank", rank, "must be >= 1")
    if rank > 256:
        raise InvalidConfigError("lora_rank", rank, "must be <= 256")
    return True


def validate_compression_ratio(ratio: float) -> bool:
    """Validate compression ratio (0.0 to 1.0)."""
    if not isinstance(ratio, (int, float)):
        raise InvalidConfigError("compression_ratio", ratio, "must be a number")
    if ratio <= 0 or ratio > 1.0:
        raise InvalidConfigError("compression_ratio", ratio, "must be in (0, 1]")
    return True


def validate_bond_amount(amount: int) -> bool:
    """Validate bond amount meets minimum."""
    if not isinstance(amount, int):
        raise InvalidConfigError("bond_amount", amount, "must be an integer")
    if amount < MIN_BOND_AMOUNT:
        raise InvalidConfigError(
            "bond_amount", amount, f"must be >= {MIN_BOND_AMOUNT}"
        )
    return True


def validate_similarity_threshold(threshold: float) -> bool:
    """Validate similarity threshold."""
    if not isinstance(threshold, (int, float)):
        raise InvalidConfigError("similarity_threshold", threshold, "must be a number")
    if threshold < 0 or threshold > 1.0:
        raise InvalidConfigError("similarity_threshold", threshold, "must be in [0, 1]")
    return True


# =============================================================================
# GRADIENT VALIDATION
# =============================================================================

def validate_gradient_dict(
    gradients: dict,
    require_non_empty: bool = True,
) -> Tuple[bool, Optional[str]]:
    """
    Validate gradient dictionary.
    
    Args:
        gradients: Dictionary of gradients
        require_non_empty: Require at least one gradient
    
    Returns:
        (is_valid, error_message) tuple
    """
    if not isinstance(gradients, dict):
        return False, "gradients must be a dictionary"
    
    if require_non_empty and len(gradients) == 0:
        return False, "gradients dictionary is empty"
    
    for name, grad in gradients.items():
        if not isinstance(name, str):
            return False, f"gradient key must be string, got {type(name)}"
        if not isinstance(grad, torch.Tensor):
            return False, f"gradient '{name}' must be torch.Tensor"
        if grad.isnan().any():
            return False, f"gradient '{name}' contains NaN values"
        if grad.isinf().any():
            return False, f"gradient '{name}' contains Inf values"
    
    return True, None


def validate_private_key(key: str) -> bool:
    """Validate private key format (hex string)."""
    if not key or not isinstance(key, str):
        raise InvalidConfigError("private_key", "***", "must be a non-empty string")
    
    # Remove 0x prefix if present
    clean_key = key[2:] if key.startswith("0x") else key
    
    # Check hex format and length (32 bytes = 64 hex chars)
    if not re.match(r'^[a-fA-F0-9]{64}$', clean_key):
        raise InvalidConfigError("private_key", "***", "must be 64 hex characters")
    
    return True

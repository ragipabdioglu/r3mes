"""
Input Validation Module for R3MES Backend

Provides comprehensive input validation for all API endpoints.
Includes validators for wallet addresses, IPFS hashes, pagination, and more.
"""

import re
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator
from .exceptions import InvalidInputError, InvalidWalletAddressError


# Regex patterns for validation
WALLET_ADDRESS_PATTERN = re.compile(r'^remes1[a-z0-9]{38,58}$')  # More flexible length
IPFS_HASH_PATTERN = re.compile(r'^(Qm[1-9A-HJ-NP-Za-km-z]{44}|bafy[a-z0-9]{50,60})$')
TX_HASH_PATTERN = re.compile(r'^[A-Fa-f0-9]{64}$')
HEX_PATTERN = re.compile(r'^[A-Fa-f0-9]+$')


def validate_wallet_address(address: str) -> str:
    """
    Validate Cosmos/R3MES wallet address format.
    
    Args:
        address: Wallet address to validate
        
    Returns:
        Validated and sanitized address
        
    Raises:
        InvalidWalletAddressError: If address format is invalid
    """
    if not address:
        raise InvalidWalletAddressError("Wallet address cannot be empty")
    
    address = address.strip().lower()
    
    if not address.startswith("remes1"):
        raise InvalidWalletAddressError("Invalid address format: must start with 'remes1'")
    
    # More flexible length validation (Cosmos addresses can vary)
    if len(address) < 44 or len(address) > 64:  # remes1 (6) + 38-58 characters
        raise InvalidWalletAddressError(f"Invalid address length: {len(address)} (expected 44-64 characters)")
    
    if not WALLET_ADDRESS_PATTERN.match(address):
        raise InvalidWalletAddressError("Invalid address format: contains invalid characters")
    
    return address


def validate_ipfs_hash(ipfs_hash: str) -> str:
    """
    Validate IPFS hash format (CIDv0 or CIDv1).
    
    Args:
        ipfs_hash: IPFS hash to validate
        
    Returns:
        Validated IPFS hash
        
    Raises:
        InvalidInputError: If hash format is invalid
    """
    if not ipfs_hash:
        raise InvalidInputError("IPFS hash cannot be empty")
    
    ipfs_hash = ipfs_hash.strip()
    
    if not IPFS_HASH_PATTERN.match(ipfs_hash):
        raise InvalidInputError(f"Invalid IPFS hash format: {ipfs_hash[:20]}...")
    
    return ipfs_hash


def validate_tx_hash(tx_hash: str) -> str:
    """
    Validate transaction hash format.
    
    Args:
        tx_hash: Transaction hash to validate
        
    Returns:
        Validated transaction hash
        
    Raises:
        InvalidInputError: If hash format is invalid
    """
    if not tx_hash:
        raise InvalidInputError("Transaction hash cannot be empty")
    
    tx_hash = tx_hash.strip().upper()
    
    if not TX_HASH_PATTERN.match(tx_hash):
        raise InvalidInputError(f"Invalid transaction hash format: {tx_hash[:20]}...")
    
    return tx_hash


def validate_pagination(limit: int, offset: int, max_limit: int = 1000) -> tuple:
    """
    Validate pagination parameters.
    
    Args:
        limit: Number of items to return
        offset: Pagination offset
        max_limit: Maximum allowed limit
        
    Returns:
        Tuple of (validated_limit, validated_offset)
        
    Raises:
        InvalidInputError: If parameters are invalid
    """
    if limit < 1:
        raise InvalidInputError("Limit must be at least 1")
    
    if limit > max_limit:
        raise InvalidInputError(f"Limit cannot exceed {max_limit}")
    
    if offset < 0:
        raise InvalidInputError("Offset cannot be negative")
    
    return (limit, offset)


def validate_status_filter(status: Optional[str], allowed_values: List[str]) -> Optional[str]:
    """
    Validate status filter parameter.
    
    Args:
        status: Status value to validate
        allowed_values: List of allowed status values
        
    Returns:
        Validated status or None
        
    Raises:
        InvalidInputError: If status is not in allowed values
    """
    if status is None:
        return None
    
    status = status.strip().lower()
    
    if status not in [v.lower() for v in allowed_values]:
        raise InvalidInputError(f"Invalid status: {status}. Allowed values: {', '.join(allowed_values)}")
    
    return status


def sanitize_string(value: str, max_length: int = 1000) -> str:
    """
    Sanitize string input by removing dangerous characters.
    
    Args:
        value: String to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
        
    Raises:
        InvalidInputError: If string is too long
    """
    if not value:
        return ""
    
    # Remove null bytes and control characters
    value = value.replace('\x00', '')
    value = ''.join(char for char in value if ord(char) >= 32 or char in '\n\r\t')
    
    # Trim whitespace
    value = value.strip()
    
    # Check length
    if len(value) > max_length:
        raise InvalidInputError(f"Input too long: {len(value)} characters (max {max_length})")
    
    return value


def validate_role_id(role_id: int) -> int:
    """
    Validate node role ID.
    
    Args:
        role_id: Role ID to validate
        
    Returns:
        Validated role ID
        
    Raises:
        InvalidInputError: If role ID is invalid
    """
    valid_roles = [1, 2, 3, 4]  # MINER, SERVING, VALIDATOR, PROPOSER
    
    if role_id not in valid_roles:
        raise InvalidInputError(f"Invalid role ID: {role_id}. Valid roles: {valid_roles}")
    
    return role_id


def validate_amount(amount: str) -> str:
    """
    Validate token amount format.
    
    Args:
        amount: Amount string (e.g., "1000uremes")
        
    Returns:
        Validated amount string
        
    Raises:
        InvalidInputError: If amount format is invalid
    """
    if not amount:
        raise InvalidInputError("Amount cannot be empty")
    
    amount = amount.strip().lower()
    
    # Check for valid denomination
    if not amount.endswith("uremes") and not amount.endswith("remes"):
        raise InvalidInputError("Amount must end with 'uremes' or 'remes'")
    
    # Extract numeric part
    numeric_part = amount.replace("uremes", "").replace("remes", "")
    
    try:
        value = int(numeric_part)
        if value < 0:
            raise InvalidInputError("Amount cannot be negative")
    except ValueError:
        raise InvalidInputError(f"Invalid amount format: {amount}")
    
    return amount


# Pydantic models with validation

class PaginationParams(BaseModel):
    """Pagination parameters with validation."""
    limit: int = Field(default=50, ge=1, le=1000, description="Number of items to return")
    offset: int = Field(default=0, ge=0, description="Pagination offset")


class WalletAddressParam(BaseModel):
    """Wallet address parameter with validation."""
    address: str = Field(..., min_length=40, max_length=65, description="Wallet address")
    
    @field_validator("address")
    @classmethod
    def validate_address(cls, v: str) -> str:
        return validate_wallet_address(v)


class IPFSHashParam(BaseModel):
    """IPFS hash parameter with validation."""
    ipfs_hash: str = Field(..., min_length=46, max_length=70, description="IPFS hash")
    
    @field_validator("ipfs_hash")
    @classmethod
    def validate_hash(cls, v: str) -> str:
        return validate_ipfs_hash(v)

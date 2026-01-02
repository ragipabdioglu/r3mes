"""
Input Validation Middleware for R3MES Backend

Production-ready validation with Pydantic models and custom validators.
"""

import re
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator, root_validator
from fastapi import HTTPException, status

from .exceptions import InvalidInputError, ValidationError, validate_wallet_address


class WalletAddressValidator:
    """Validator for R3MES wallet addresses."""
    
    WALLET_REGEX = re.compile(r'^remes1[a-z0-9]{38}$')
    
    @classmethod
    def validate(cls, address: str) -> str:
        """Validate wallet address format."""
        if not address:
            raise InvalidInputError("Wallet address cannot be empty", field="wallet_address")
        
        if not cls.WALLET_REGEX.match(address):
            raise InvalidInputError(
                "Invalid wallet address format. Must start with 'remes1' followed by 38 characters",
                field="wallet_address",
                value=address
            )
        
        return address


class IPFSHashValidator:
    """Validator for IPFS hashes."""
    
    # IPFS hash patterns (CIDv0 and CIDv1)
    CIDV0_REGEX = re.compile(r'^Qm[1-9A-HJ-NP-Za-km-z]{44}$')
    CIDV1_REGEX = re.compile(r'^b[a-z2-7]{58}$')
    
    @classmethod
    def validate(cls, ipfs_hash: str) -> str:
        """Validate IPFS hash format."""
        if not ipfs_hash:
            raise InvalidInputError("IPFS hash cannot be empty", field="ipfs_hash")
        
        if not (cls.CIDV0_REGEX.match(ipfs_hash) or cls.CIDV1_REGEX.match(ipfs_hash)):
            raise InvalidInputError(
                "Invalid IPFS hash format",
                field="ipfs_hash",
                value=ipfs_hash
            )
        
        return ipfs_hash


# Pydantic Models for Request Validation

class ChatRequest(BaseModel):
    """Chat request validation model."""
    
    message: str = Field(..., min_length=1, max_length=10000, description="Chat message")
    wallet_address: Optional[str] = Field(None, description="User wallet address")
    model_version: Optional[str] = Field("default", max_length=100, description="Model version")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Generation temperature")
    max_tokens: Optional[int] = Field(1000, ge=1, le=4000, description="Maximum tokens to generate")
    
    @validator('wallet_address')
    def validate_wallet_address(cls, v):
        if v is not None:
            return WalletAddressValidator.validate(v)
        return v
    
    @validator('message')
    def validate_message(cls, v):
        if not v.strip():
            raise ValueError("Message cannot be empty or whitespace only")
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Hello, how can I help you?",
                "wallet_address": "remes1abcdefghijklmnopqrstuvwxyz123456789abc",
                "model_version": "llama-3-8b",
                "temperature": 0.7,
                "max_tokens": 1000
            }
        }


class GradientSubmissionRequest(BaseModel):
    """Gradient submission validation model."""
    
    miner_address: str = Field(..., description="Miner wallet address")
    ipfs_hash: str = Field(..., description="IPFS hash of gradient data")
    model_version: str = Field(..., max_length=100, description="Model version")
    training_round_id: int = Field(..., ge=0, description="Training round ID")
    shard_id: int = Field(..., ge=0, description="Shard ID")
    gradient_hash: str = Field(..., min_length=64, max_length=64, description="SHA256 hash of gradient")
    gpu_architecture: str = Field(..., max_length=50, description="GPU architecture")
    nonce: int = Field(..., ge=1, description="Unique nonce for replay protection")
    signature: str = Field(..., description="Cryptographic signature")
    
    @validator('miner_address')
    def validate_miner_address(cls, v):
        return WalletAddressValidator.validate(v)
    
    @validator('ipfs_hash')
    def validate_ipfs_hash(cls, v):
        return IPFSHashValidator.validate(v)
    
    @validator('gradient_hash')
    def validate_gradient_hash(cls, v):
        if not re.match(r'^[a-fA-F0-9]{64}$', v):
            raise ValueError("Gradient hash must be a 64-character hexadecimal string")
        return v.lower()
    
    @validator('signature')
    def validate_signature(cls, v):
        # Basic signature format validation (base64 or hex)
        if not re.match(r'^[A-Za-z0-9+/=]+$', v) and not re.match(r'^[a-fA-F0-9]+$', v):
            raise ValueError("Invalid signature format")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "miner_address": "remes1abcdefghijklmnopqrstuvwxyz123456789abc",
                "ipfs_hash": "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG",
                "model_version": "llama-3-8b",
                "training_round_id": 1,
                "shard_id": 0,
                "gradient_hash": "a1b2c3d4e5f6789012345678901234567890123456789012345678901234567890",
                "gpu_architecture": "RTX4090",
                "nonce": 12345,
                "signature": "base64_encoded_signature_here"
            }
        }


class UserInfoRequest(BaseModel):
    """User info request validation model."""
    
    wallet_address: str = Field(..., description="User wallet address")
    
    @validator('wallet_address')
    def validate_wallet_address(cls, v):
        return WalletAddressValidator.validate(v)


class CreditOperationRequest(BaseModel):
    """Credit operation validation model."""
    
    wallet_address: str = Field(..., description="User wallet address")
    amount: float = Field(..., gt=0, le=1000000, description="Credit amount")
    operation_type: str = Field(..., regex=r'^(add|deduct)$', description="Operation type")
    reason: Optional[str] = Field(None, max_length=500, description="Operation reason")
    
    @validator('wallet_address')
    def validate_wallet_address(cls, v):
        return WalletAddressValidator.validate(v)
    
    @validator('amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError("Amount must be positive")
        if v > 1000000:
            raise ValueError("Amount too large (max: 1,000,000)")
        return round(v, 6)  # Limit to 6 decimal places


class APIKeyCreateRequest(BaseModel):
    """API key creation validation model."""
    
    wallet_address: str = Field(..., description="Owner wallet address")
    name: str = Field(..., min_length=1, max_length=100, description="API key name")
    expires_in_days: Optional[int] = Field(None, ge=1, le=365, description="Expiration in days")
    
    @validator('wallet_address')
    def validate_wallet_address(cls, v):
        return WalletAddressValidator.validate(v)
    
    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError("Name cannot be empty")
        # Remove potentially dangerous characters
        cleaned_name = re.sub(r'[<>"\']', '', v.strip())
        return cleaned_name


class MinerStatsRequest(BaseModel):
    """Miner stats request validation model."""
    
    wallet_address: str = Field(..., description="Miner wallet address")
    days: Optional[int] = Field(7, ge=1, le=365, description="Number of days for history")
    
    @validator('wallet_address')
    def validate_wallet_address(cls, v):
        return WalletAddressValidator.validate(v)


class NetworkStatsRequest(BaseModel):
    """Network stats request validation model."""
    
    include_history: Optional[bool] = Field(False, description="Include historical data")
    limit: Optional[int] = Field(100, ge=1, le=1000, description="Limit for results")


class PaginationParams(BaseModel):
    """Pagination parameters validation model."""
    
    page: int = Field(1, ge=1, le=10000, description="Page number")
    limit: int = Field(20, ge=1, le=100, description="Items per page")
    
    @property
    def offset(self) -> int:
        """Calculate offset from page and limit."""
        return (self.page - 1) * self.limit


class DateRangeParams(BaseModel):
    """Date range parameters validation model."""
    
    start_date: Optional[datetime] = Field(None, description="Start date (ISO format)")
    end_date: Optional[datetime] = Field(None, description="End date (ISO format)")
    
    @root_validator
    def validate_date_range(cls, values):
        start_date = values.get('start_date')
        end_date = values.get('end_date')
        
        if start_date and end_date:
            if start_date >= end_date:
                raise ValueError("Start date must be before end date")
            
            # Limit date range to prevent excessive queries
            if (end_date - start_date).days > 365:
                raise ValueError("Date range cannot exceed 365 days")
        
        return values


# Validation Middleware

class ValidationMiddleware:
    """Middleware for request validation and sanitization."""
    
    @staticmethod
    def validate_request_size(content_length: Optional[int], max_size: int = 10 * 1024 * 1024):
        """Validate request content length."""
        if content_length and content_length > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Request too large. Maximum size: {max_size} bytes"
            )
    
    @staticmethod
    def sanitize_string_input(value: str, max_length: int = 1000) -> str:
        """Sanitize string input to prevent injection attacks."""
        if not isinstance(value, str):
            raise InvalidInputError("Value must be a string")
        
        # Remove null bytes and control characters
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', value)
        
        # Limit length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized.strip()
    
    @staticmethod
    def validate_json_structure(data: Dict[str, Any], required_fields: List[str]) -> Dict[str, Any]:
        """Validate JSON structure and required fields."""
        if not isinstance(data, dict):
            raise ValidationError("Request body must be a JSON object")
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValidationError(
                f"Missing required fields: {', '.join(missing_fields)}",
                validation_errors={field: "Field is required" for field in missing_fields}
            )
        
        return data


# Custom Validators

def validate_model_version(version: str) -> str:
    """Validate model version string."""
    if not version:
        raise InvalidInputError("Model version cannot be empty")
    
    # Allow alphanumeric, hyphens, underscores, and dots
    if not re.match(r'^[a-zA-Z0-9._-]+$', version):
        raise InvalidInputError("Invalid model version format")
    
    if len(version) > 100:
        raise InvalidInputError("Model version too long (max 100 characters)")
    
    return version


def validate_gpu_architecture(architecture: str) -> str:
    """Validate GPU architecture string."""
    if not architecture:
        raise InvalidInputError("GPU architecture cannot be empty")
    
    # Common GPU architectures
    valid_architectures = {
        'RTX4090', 'RTX4080', 'RTX4070', 'RTX3090', 'RTX3080', 'RTX3070',
        'RTX2080', 'RTX2070', 'GTX1080', 'GTX1070', 'V100', 'A100', 'H100',
        'T4', 'P100', 'K80', 'M40', 'TITAN', 'QUADRO'
    }
    
    # Allow exact matches or partial matches for custom architectures
    architecture_upper = architecture.upper()
    if not any(arch in architecture_upper for arch in valid_architectures):
        # Still allow it but log a warning
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Unknown GPU architecture: {architecture}")
    
    return architecture


def validate_training_round_id(round_id: int) -> int:
    """Validate training round ID."""
    if round_id < 0:
        raise InvalidInputError("Training round ID cannot be negative")
    
    if round_id > 1000000:  # Reasonable upper limit
        raise InvalidInputError("Training round ID too large")
    
    return round_id


def validate_shard_id(shard_id: int) -> int:
    """Validate shard ID."""
    if shard_id < 0:
        raise InvalidInputError("Shard ID cannot be negative")
    
    if shard_id > 10000:  # Reasonable upper limit for shards
        raise InvalidInputError("Shard ID too large")
    
    return shard_id


# Rate Limiting Validation

class RateLimitValidator:
    """Validator for rate limiting parameters."""
    
    @staticmethod
    def validate_rate_limit_key(key: str) -> str:
        """Validate rate limit key format."""
        if not key:
            raise InvalidInputError("Rate limit key cannot be empty")
        
        # Allow alphanumeric, colons, and hyphens for Redis keys
        if not re.match(r'^[a-zA-Z0-9:_-]+$', key):
            raise InvalidInputError("Invalid rate limit key format")
        
        if len(key) > 200:
            raise InvalidInputError("Rate limit key too long")
        
        return key
    
    @staticmethod
    def validate_rate_limit_window(window_seconds: int) -> int:
        """Validate rate limit window."""
        if window_seconds <= 0:
            raise InvalidInputError("Rate limit window must be positive")
        
        if window_seconds > 86400:  # Max 24 hours
            raise InvalidInputError("Rate limit window too large (max 24 hours)")
        
        return window_seconds
    
    @staticmethod
    def validate_rate_limit_count(count: int) -> int:
        """Validate rate limit count."""
        if count <= 0:
            raise InvalidInputError("Rate limit count must be positive")
        
        if count > 100000:  # Reasonable upper limit
            raise InvalidInputError("Rate limit count too large")
        
        return count
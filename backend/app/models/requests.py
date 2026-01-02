"""
Request Models for R3MES Backend API

Centralized Pydantic models for all API request validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime

from ..input_validation import (
    validate_wallet_address,
    validate_ipfs_hash,
    sanitize_string
)


class ChatRequest(BaseModel):
    """Request model for chat inference."""
    
    message: str = Field(
        ..., 
        min_length=1, 
        max_length=10000,
        description="User message for AI inference"
    )
    wallet_address: Optional[str] = Field(
        None, 
        description="Wallet address (optional if API key is provided)"
    )
    adapter_name: Optional[str] = Field(
        None,
        max_length=100,
        description="LoRA adapter name (optional)"
    )
    temperature: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0-2.0)"
    )
    max_tokens: Optional[int] = Field(
        None,
        ge=1,
        le=4096,
        description="Maximum tokens to generate"
    )
    
    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Message cannot be empty")
        # Remove null bytes and control characters
        v = v.replace('\x00', '')
        # Sanitize string
        v = sanitize_string(v, max_length=10000)
        return v.strip()
    
    @field_validator("wallet_address")
    @classmethod
    def validate_wallet(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return validate_wallet_address(v)
    
    @field_validator("adapter_name")
    @classmethod
    def validate_adapter(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        # Basic sanitization
        v = sanitize_string(v, max_length=100)
        return v


class CreateAPIKeyRequest(BaseModel):
    """Request model for API key creation."""
    
    wallet_address: str = Field(
        ..., 
        description="Wallet address for API key"
    )
    name: Optional[str] = Field(
        None, 
        max_length=100, 
        description="API key name"
    )
    expires_days: Optional[int] = Field(
        None, 
        ge=1, 
        le=365,
        description="Expiration in days (1-365)"
    )
    
    @field_validator("wallet_address")
    @classmethod
    def validate_wallet(cls, v: str) -> str:
        return validate_wallet_address(v)
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        # XSS prevention
        dangerous_chars = ['<', '>', '"', "'", '&', '{', '}', '[', ']']
        if any(char in v for char in dangerous_chars):
            raise ValueError("Name contains invalid characters")
        # SQL injection prevention
        if any(keyword in v.lower() for keyword in ['select', 'insert', 'delete', 'drop', 'union']):
            raise ValueError("Name contains invalid keywords")
        return sanitize_string(v, max_length=100)


class RevokeAPIKeyRequest(BaseModel):
    """Request model for API key revocation."""
    
    api_key_id: int = Field(
        ..., 
        ge=1,
        description="API key ID to revoke"
    )
    wallet_address: str = Field(
        ..., 
        description="Wallet address (for authorization)"
    )
    
    @field_validator("wallet_address")
    @classmethod
    def validate_wallet(cls, v: str) -> str:
        return validate_wallet_address(v)


class DeleteAPIKeyRequest(BaseModel):
    """Request model for API key deletion."""
    
    api_key_id: int = Field(
        ..., 
        ge=1,
        description="API key ID to delete"
    )
    wallet_address: str = Field(
        ..., 
        description="Wallet address (for authorization)"
    )
    
    @field_validator("wallet_address")
    @classmethod
    def validate_wallet(cls, v: str) -> str:
        return validate_wallet_address(v)


class LoRARegisterRequest(BaseModel):
    """Request model for LoRA adapter registration."""
    
    name: str = Field(
        ..., 
        min_length=1,
        max_length=100,
        description="LoRA adapter name"
    )
    ipfs_hash: str = Field(
        ..., 
        description="IPFS hash of the adapter"
    )
    wallet_address: str = Field(
        ..., 
        description="Wallet address of the registrant"
    )
    description: Optional[str] = Field(
        None,
        max_length=500,
        description="Adapter description"
    )
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Name cannot be empty")
        # Alphanumeric and basic chars only
        if not v.replace('-', '').replace('_', '').replace('.', '').isalnum():
            raise ValueError("Name can only contain alphanumeric characters, hyphens, underscores, and dots")
        return sanitize_string(v, max_length=100)
    
    @field_validator("ipfs_hash")
    @classmethod
    def validate_ipfs(cls, v: str) -> str:
        return validate_ipfs_hash(v)
    
    @field_validator("wallet_address")
    @classmethod
    def validate_wallet(cls, v: str) -> str:
        return validate_wallet_address(v)
    
    @field_validator("description")
    @classmethod
    def validate_description(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        return sanitize_string(v, max_length=500)


class ServingNodeRegisterRequest(BaseModel):
    """Request model for serving node registration."""
    
    endpoint_url: str = Field(
        ..., 
        description="Serving node endpoint URL"
    )
    wallet_address: str = Field(
        ..., 
        description="Wallet address of the node operator"
    )
    supported_adapters: List[str] = Field(
        ..., 
        description="List of supported LoRA adapters"
    )
    max_concurrent_requests: Optional[int] = Field(
        None,
        ge=1,
        le=100,
        description="Maximum concurrent requests"
    )
    
    @field_validator("endpoint_url")
    @classmethod
    def validate_endpoint(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Endpoint URL cannot be empty")
        
        # Basic URL validation
        if not v.startswith(('http://', 'https://')):
            raise ValueError("Endpoint URL must start with http:// or https://")
        
        # SSRF protection - basic checks
        if any(blocked in v.lower() for blocked in ['localhost', '127.0.0.1', '0.0.0.0', '::1']):
            raise ValueError("Localhost endpoints are not allowed")
        
        return v
    
    @field_validator("wallet_address")
    @classmethod
    def validate_wallet(cls, v: str) -> str:
        return validate_wallet_address(v)
    
    @field_validator("supported_adapters")
    @classmethod
    def validate_adapters(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("At least one supported adapter is required")
        
        validated_adapters = []
        for adapter in v:
            adapter = adapter.strip()
            if adapter:
                # Basic sanitization
                adapter = sanitize_string(adapter, max_length=100)
                validated_adapters.append(adapter)
        
        if not validated_adapters:
            raise ValueError("At least one valid adapter is required")
        
        return validated_adapters


class ServingNodeHeartbeatRequest(BaseModel):
    """Request model for serving node heartbeat."""
    
    endpoint_url: str = Field(
        ..., 
        description="Serving node endpoint URL"
    )
    wallet_address: str = Field(
        ..., 
        description="Wallet address of the node operator"
    )
    current_load: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Current load (0.0-1.0)"
    )
    active_requests: Optional[int] = Field(
        None,
        ge=0,
        description="Number of active requests"
    )
    
    @field_validator("endpoint_url")
    @classmethod
    def validate_endpoint(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Endpoint URL cannot be empty")
        
        # Basic URL validation
        if not v.startswith(('http://', 'https://')):
            raise ValueError("Endpoint URL must start with http:// or https://")
        
        return v
    
    @field_validator("wallet_address")
    @classmethod
    def validate_wallet(cls, v: str) -> str:
        return validate_wallet_address(v)


class PaginationRequest(BaseModel):
    """Request model for pagination parameters."""
    
    limit: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Number of items to return (1-1000)"
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Number of items to skip"
    )


class DateRangeRequest(BaseModel):
    """Request model for date range queries."""
    
    start_date: Optional[datetime] = Field(
        None,
        description="Start date (ISO format)"
    )
    end_date: Optional[datetime] = Field(
        None,
        description="End date (ISO format)"
    )
    days: Optional[int] = Field(
        None,
        ge=1,
        le=365,
        description="Number of days from now (alternative to date range)"
    )
    
    @field_validator("end_date")
    @classmethod
    def validate_date_range(cls, v: Optional[datetime], info) -> Optional[datetime]:
        if v is None:
            return None
        
        start_date = info.data.get('start_date')
        if start_date and v <= start_date:
            raise ValueError("End date must be after start date")
        
        return v
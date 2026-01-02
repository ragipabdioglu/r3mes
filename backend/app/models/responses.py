"""
Response Models for R3MES Backend API

Centralized Pydantic models for all API responses.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class UserInfoResponse(BaseModel):
    """Response model for user information."""
    
    wallet_address: str = Field(..., description="User wallet address")
    credits: float = Field(..., description="Available credits")
    is_miner: bool = Field(..., description="Whether user is a miner")
    last_mining_time: Optional[datetime] = Field(None, description="Last mining timestamp")
    created_at: Optional[datetime] = Field(None, description="Account creation timestamp")


class NetworkStatsResponse(BaseModel):
    """Response model for network statistics."""
    
    active_miners: int = Field(..., description="Number of active miners")
    total_users: int = Field(..., description="Total number of users")
    total_credits: float = Field(..., description="Total credits in circulation")
    block_height: Optional[int] = Field(None, description="Current block height")
    network_hashrate: Optional[float] = Field(None, description="Network hashrate")
    difficulty: Optional[float] = Field(None, description="Mining difficulty")


class BlockResponse(BaseModel):
    """Response model for blockchain block information."""
    
    height: int = Field(..., description="Block height")
    miner: Optional[str] = Field(None, description="Miner wallet address")
    timestamp: Optional[str] = Field(None, description="Block timestamp")
    hash: Optional[str] = Field(None, description="Block hash")
    transactions: Optional[int] = Field(None, description="Number of transactions")
    reward: Optional[float] = Field(None, description="Block reward")


class BlocksResponse(BaseModel):
    """Response model for multiple blocks."""
    
    blocks: List[BlockResponse] = Field(..., description="List of blocks")
    limit: int = Field(..., description="Requested limit")
    total: int = Field(..., description="Total number of blocks")
    has_more: bool = Field(..., description="Whether more blocks are available")


class MinerStatsResponse(BaseModel):
    """Response model for miner statistics."""
    
    wallet_address: str = Field(..., description="Miner wallet address")
    total_earnings: float = Field(..., description="Total earnings")
    hashrate: float = Field(..., description="Current hashrate")
    gpu_temperature: Optional[float] = Field(None, description="GPU temperature")
    blocks_found: int = Field(..., description="Number of blocks found")
    uptime_percentage: float = Field(..., description="Uptime percentage")
    network_difficulty: Optional[float] = Field(None, description="Network difficulty")
    last_block_time: Optional[datetime] = Field(None, description="Last block found timestamp")


class EarningsRecord(BaseModel):
    """Model for individual earnings record."""
    
    timestamp: datetime = Field(..., description="Earnings timestamp")
    amount: float = Field(..., description="Earnings amount")
    block_height: Optional[int] = Field(None, description="Block height")
    transaction_hash: Optional[str] = Field(None, description="Transaction hash")


class EarningsHistoryResponse(BaseModel):
    """Response model for earnings history."""
    
    wallet_address: str = Field(..., description="Wallet address")
    earnings: List[EarningsRecord] = Field(..., description="List of earnings records")
    total_earnings: float = Field(..., description="Total earnings in period")
    period_days: int = Field(..., description="Period in days")


class HashrateRecord(BaseModel):
    """Model for individual hashrate record."""
    
    timestamp: datetime = Field(..., description="Hashrate timestamp")
    hashrate: float = Field(..., description="Hashrate value")
    gpu_temperature: Optional[float] = Field(None, description="GPU temperature")
    power_consumption: Optional[float] = Field(None, description="Power consumption")


class HashrateHistoryResponse(BaseModel):
    """Response model for hashrate history."""
    
    wallet_address: str = Field(..., description="Wallet address")
    hashrate_data: List[HashrateRecord] = Field(..., description="List of hashrate records")
    average_hashrate: float = Field(..., description="Average hashrate in period")
    peak_hashrate: float = Field(..., description="Peak hashrate in period")
    period_days: int = Field(..., description="Period in days")


class APIKeyInfo(BaseModel):
    """Model for API key information."""
    
    key_id: str = Field(..., description="API key identifier")
    name: Optional[str] = Field(None, description="API key name")
    is_active: bool = Field(..., description="Whether key is active")
    created_at: datetime = Field(..., description="Creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    last_used_at: Optional[datetime] = Field(None, description="Last used timestamp")


class CreateAPIKeyResponse(BaseModel):
    """Response model for API key creation."""
    
    api_key: str = Field(..., description="Generated API key (shown only once)")
    key_id: str = Field(..., description="API key identifier")
    name: Optional[str] = Field(None, description="API key name")
    created_at: datetime = Field(..., description="Creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    wallet_address: str = Field(..., description="Owner wallet address")


class ListAPIKeysResponse(BaseModel):
    """Response model for API key listing."""
    
    wallet_address: str = Field(..., description="Wallet address")
    api_keys: List[APIKeyInfo] = Field(..., description="List of API keys")
    total_keys: int = Field(..., description="Total number of keys")
    active_keys: int = Field(..., description="Number of active keys")


class APIKeyActionResponse(BaseModel):
    """Response model for API key actions (revoke, delete)."""
    
    success: bool = Field(..., description="Whether action was successful")
    message: str = Field(..., description="Action result message")
    key_id: Optional[str] = Field(None, description="API key identifier")


class ChatResponse(BaseModel):
    """Response model for chat inference."""
    
    response: str = Field(..., description="AI response")
    model_used: Optional[str] = Field(None, description="Model used for inference")
    adapter_used: Optional[str] = Field(None, description="LoRA adapter used")
    tokens_used: Optional[int] = Field(None, description="Number of tokens used")
    credits_deducted: Optional[float] = Field(None, description="Credits deducted")
    inference_time: Optional[float] = Field(None, description="Inference time in seconds")
    timestamp: datetime = Field(..., description="Response timestamp")


class LoRAAdapterInfo(BaseModel):
    """Model for LoRA adapter information."""
    
    name: str = Field(..., description="Adapter name")
    ipfs_hash: str = Field(..., description="IPFS hash")
    owner: str = Field(..., description="Owner wallet address")
    description: Optional[str] = Field(None, description="Adapter description")
    created_at: datetime = Field(..., description="Creation timestamp")
    downloads: int = Field(..., description="Number of downloads")
    rating: Optional[float] = Field(None, description="Average rating")


class RegisterLoRAResponse(BaseModel):
    """Response model for LoRA adapter registration."""
    
    success: bool = Field(..., description="Whether registration was successful")
    message: str = Field(..., description="Registration result message")
    adapter_info: Optional[LoRAAdapterInfo] = Field(None, description="Registered adapter info")


class ServingNodeInfo(BaseModel):
    """Model for serving node information."""
    
    endpoint_url: str = Field(..., description="Node endpoint URL")
    owner: str = Field(..., description="Owner wallet address")
    supported_adapters: List[str] = Field(..., description="Supported adapters")
    max_concurrent_requests: int = Field(..., description="Max concurrent requests")
    current_load: Optional[float] = Field(None, description="Current load")
    active_requests: Optional[int] = Field(None, description="Active requests")
    last_heartbeat: Optional[datetime] = Field(None, description="Last heartbeat timestamp")
    status: str = Field(..., description="Node status")


class RegisterServingNodeResponse(BaseModel):
    """Response model for serving node registration."""
    
    success: bool = Field(..., description="Whether registration was successful")
    message: str = Field(..., description="Registration result message")
    node_info: Optional[ServingNodeInfo] = Field(None, description="Registered node info")


class ServingNodeHeartbeatResponse(BaseModel):
    """Response model for serving node heartbeat."""
    
    success: bool = Field(..., description="Whether heartbeat was successful")
    message: str = Field(..., description="Heartbeat result message")
    next_heartbeat_in: Optional[int] = Field(None, description="Next heartbeat in seconds")


class HealthCheckResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="Service version")
    uptime: float = Field(..., description="Uptime in seconds")
    components: Dict[str, str] = Field(..., description="Component status")


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    error: bool = Field(True, description="Error flag")
    error_code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


class SuccessResponse(BaseModel):
    """Generic success response model."""
    
    success: bool = Field(True, description="Success flag")
    message: str = Field(..., description="Success message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    timestamp: datetime = Field(..., description="Response timestamp")


class PaginatedResponse(BaseModel):
    """Base model for paginated responses."""
    
    items: List[Any] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items")
    limit: int = Field(..., description="Items per page")
    offset: int = Field(..., description="Number of items skipped")
    has_more: bool = Field(..., description="Whether more items are available")


class MetricsResponse(BaseModel):
    """Response model for metrics."""
    
    metrics: Dict[str, Any] = Field(..., description="Metrics data")
    timestamp: datetime = Field(..., description="Metrics timestamp")
    period: str = Field(..., description="Metrics period")


class SystemStatusResponse(BaseModel):
    """Response model for system status."""
    
    status: str = Field(..., description="System status")
    components: Dict[str, Dict[str, Any]] = Field(..., description="Component statuses")
    performance: Dict[str, float] = Field(..., description="Performance metrics")
    timestamp: datetime = Field(..., description="Status timestamp")
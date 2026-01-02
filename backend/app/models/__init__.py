"""
Models Package for R3MES Backend

Centralized Pydantic models for requests and responses.
"""

from .requests import (
    ChatRequest,
    CreateAPIKeyRequest,
    RevokeAPIKeyRequest,
    DeleteAPIKeyRequest,
    LoRARegisterRequest,
    ServingNodeRegisterRequest,
    ServingNodeHeartbeatRequest,
    PaginationRequest,
    DateRangeRequest
)

from .responses import (
    UserInfoResponse,
    NetworkStatsResponse,
    BlockResponse,
    BlocksResponse,
    MinerStatsResponse,
    EarningsRecord,
    EarningsHistoryResponse,
    HashrateRecord,
    HashrateHistoryResponse,
    APIKeyInfo,
    CreateAPIKeyResponse,
    ListAPIKeysResponse,
    APIKeyActionResponse,
    ChatResponse,
    LoRAAdapterInfo,
    RegisterLoRAResponse,
    ServingNodeInfo,
    RegisterServingNodeResponse,
    ServingNodeHeartbeatResponse,
    HealthCheckResponse,
    ErrorResponse,
    SuccessResponse,
    PaginatedResponse,
    MetricsResponse,
    SystemStatusResponse
)

__all__ = [
    # Request models
    'ChatRequest',
    'CreateAPIKeyRequest',
    'RevokeAPIKeyRequest',
    'DeleteAPIKeyRequest',
    'LoRARegisterRequest',
    'ServingNodeRegisterRequest',
    'ServingNodeHeartbeatRequest',
    'PaginationRequest',
    'DateRangeRequest',
    
    # Response models
    'UserInfoResponse',
    'NetworkStatsResponse',
    'BlockResponse',
    'BlocksResponse',
    'MinerStatsResponse',
    'EarningsRecord',
    'EarningsHistoryResponse',
    'HashrateRecord',
    'HashrateHistoryResponse',
    'APIKeyInfo',
    'CreateAPIKeyResponse',
    'ListAPIKeysResponse',
    'APIKeyActionResponse',
    'ChatResponse',
    'LoRAAdapterInfo',
    'RegisterLoRAResponse',
    'ServingNodeInfo',
    'RegisterServingNodeResponse',
    'ServingNodeHeartbeatResponse',
    'HealthCheckResponse',
    'ErrorResponse',
    'SuccessResponse',
    'PaginatedResponse',
    'MetricsResponse',
    'SystemStatusResponse'
]
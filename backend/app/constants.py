"""
Constants Module for R3MES Backend

Centralizes all magic numbers and configuration constants.
All values can be overridden via environment variables.
"""

import os
from typing import Final


# =============================================================================
# RATE LIMITING
# =============================================================================

# Default rate limits (requests per minute)
RATE_LIMIT_DEFAULT: Final[int] = int(os.getenv("R3MES_RATE_LIMIT_DEFAULT", "100"))
RATE_LIMIT_CHAT: Final[str] = os.getenv("R3MES_RATE_LIMIT_CHAT", "10/minute")
RATE_LIMIT_GET: Final[str] = os.getenv("R3MES_RATE_LIMIT_GET", "100/minute")
RATE_LIMIT_POST: Final[str] = os.getenv("R3MES_RATE_LIMIT_POST", "30/minute")


# =============================================================================
# PAGINATION
# =============================================================================

# Default pagination values
PAGINATION_DEFAULT_LIMIT: Final[int] = int(os.getenv("R3MES_PAGINATION_DEFAULT_LIMIT", "50"))
PAGINATION_MAX_LIMIT: Final[int] = int(os.getenv("R3MES_PAGINATION_MAX_LIMIT", "1000"))
PAGINATION_DEFAULT_OFFSET: Final[int] = 0


# =============================================================================
# WEBSOCKET
# =============================================================================

# WebSocket configuration
WEBSOCKET_HEARTBEAT_INTERVAL_MS: Final[int] = int(os.getenv("R3MES_WS_HEARTBEAT_INTERVAL", "30000"))
WEBSOCKET_TIMEOUT_MS: Final[int] = int(os.getenv("R3MES_WS_TIMEOUT", "60000"))
WEBSOCKET_MAX_MESSAGE_SIZE: Final[int] = int(os.getenv("R3MES_WS_MAX_MESSAGE_SIZE", "1048576"))  # 1MB
WEBSOCKET_RECONNECT_DELAY_MS: Final[int] = int(os.getenv("R3MES_WS_RECONNECT_DELAY", "5000"))
WEBSOCKET_MAX_RECONNECT_ATTEMPTS: Final[int] = int(os.getenv("R3MES_WS_MAX_RECONNECT_ATTEMPTS", "10"))


# =============================================================================
# CACHE
# =============================================================================

# Cache TTL values (in seconds)
CACHE_TTL_SHORT: Final[int] = int(os.getenv("R3MES_CACHE_TTL_SHORT", "60"))  # 1 minute
CACHE_TTL_MEDIUM: Final[int] = int(os.getenv("R3MES_CACHE_TTL_MEDIUM", "300"))  # 5 minutes
CACHE_TTL_LONG: Final[int] = int(os.getenv("R3MES_CACHE_TTL_LONG", "3600"))  # 1 hour
CACHE_TTL_VERY_LONG: Final[int] = int(os.getenv("R3MES_CACHE_TTL_VERY_LONG", "86400"))  # 24 hours

# Cache key prefixes
CACHE_PREFIX_USER: Final[str] = "user:"
CACHE_PREFIX_BLOCK: Final[str] = "block:"
CACHE_PREFIX_VALIDATOR: Final[str] = "validator:"
CACHE_PREFIX_STATS: Final[str] = "stats:"


# =============================================================================
# INFERENCE
# =============================================================================

# Inference configuration
INFERENCE_MAX_TOKENS: Final[int] = int(os.getenv("R3MES_INFERENCE_MAX_TOKENS", "2048"))
INFERENCE_DEFAULT_TEMPERATURE: Final[float] = float(os.getenv("R3MES_INFERENCE_TEMPERATURE", "0.7"))
INFERENCE_TIMEOUT_SECONDS: Final[int] = int(os.getenv("R3MES_INFERENCE_TIMEOUT", "120"))
INFERENCE_MAX_CONCURRENT: Final[int] = int(os.getenv("R3MES_INFERENCE_MAX_CONCURRENT", "4"))


# =============================================================================
# CREDITS
# =============================================================================

# Credit system configuration
CREDIT_COST_PER_INFERENCE: Final[float] = float(os.getenv("R3MES_CREDIT_COST_INFERENCE", "1.0"))
CREDIT_REWARD_PER_BLOCK: Final[float] = float(os.getenv("R3MES_CREDIT_REWARD_BLOCK", "10.0"))
CREDIT_MIN_BALANCE: Final[float] = float(os.getenv("R3MES_CREDIT_MIN_BALANCE", "0.0"))


# =============================================================================
# BLOCKCHAIN
# =============================================================================

# Blockchain configuration
BLOCKCHAIN_BLOCK_TIME_SECONDS: Final[int] = int(os.getenv("R3MES_BLOCK_TIME", "6"))
BLOCKCHAIN_CONFIRMATION_BLOCKS: Final[int] = int(os.getenv("R3MES_CONFIRMATION_BLOCKS", "1"))
BLOCKCHAIN_MAX_RETRIES: Final[int] = int(os.getenv("R3MES_BLOCKCHAIN_MAX_RETRIES", "3"))
BLOCKCHAIN_RETRY_DELAY_MS: Final[int] = int(os.getenv("R3MES_BLOCKCHAIN_RETRY_DELAY", "1000"))


# =============================================================================
# MINER ENGINE
# =============================================================================

# Task pool configuration
TASK_POOL_MAX_PREFETCH: Final[int] = int(os.getenv("R3MES_MAX_PREFETCH_TASKS", "5"))
TASK_POOL_CLAIM_LIMIT: Final[int] = int(os.getenv("R3MES_TASK_POOL_CLAIM_LIMIT", "1"))
TASK_POOL_QUERY_LIMIT: Final[int] = int(os.getenv("R3MES_TASK_POOL_QUERY_LIMIT", "1000"))

# GPU configuration
GPU_TEMP_WARNING_CELSIUS: Final[int] = int(os.getenv("R3MES_GPU_TEMP_WARNING", "80"))
GPU_TEMP_CRITICAL_CELSIUS: Final[int] = int(os.getenv("R3MES_GPU_TEMP_CRITICAL", "90"))
GPU_VRAM_WARNING_PERCENT: Final[int] = int(os.getenv("R3MES_GPU_VRAM_WARNING", "90"))


# =============================================================================
# SERVING NODE
# =============================================================================

# Serving node configuration
SERVING_NODE_HEARTBEAT_INTERVAL_SECONDS: Final[int] = int(os.getenv("R3MES_SERVING_HEARTBEAT", "30"))
SERVING_NODE_STALE_THRESHOLD_SECONDS: Final[int] = int(os.getenv("R3MES_SERVING_STALE_THRESHOLD", "120"))
SERVING_NODE_MAX_LATENCY_MS: Final[int] = int(os.getenv("R3MES_SERVING_MAX_LATENCY", "5000"))


# =============================================================================
# SECURITY
# =============================================================================

# Security configuration
API_KEY_LENGTH: Final[int] = int(os.getenv("R3MES_API_KEY_LENGTH", "32"))
API_KEY_EXPIRY_DAYS: Final[int] = int(os.getenv("R3MES_API_KEY_EXPIRY_DAYS", "365"))
MAX_LOGIN_ATTEMPTS: Final[int] = int(os.getenv("R3MES_MAX_LOGIN_ATTEMPTS", "5"))
LOGIN_LOCKOUT_MINUTES: Final[int] = int(os.getenv("R3MES_LOGIN_LOCKOUT_MINUTES", "15"))


# =============================================================================
# VALIDATION
# =============================================================================

# Input validation limits
MAX_MESSAGE_LENGTH: Final[int] = int(os.getenv("R3MES_MAX_MESSAGE_LENGTH", "10000"))
MAX_WALLET_ADDRESS_LENGTH: Final[int] = 65
MIN_WALLET_ADDRESS_LENGTH: Final[int] = 40
MAX_IPFS_HASH_LENGTH: Final[int] = 70
MIN_IPFS_HASH_LENGTH: Final[int] = 46


# =============================================================================
# NODE ROLES
# =============================================================================

# Node role IDs (matching blockchain proto definitions)
NODE_ROLE_MINER: Final[int] = 1
NODE_ROLE_SERVING: Final[int] = 2
NODE_ROLE_VALIDATOR: Final[int] = 3
NODE_ROLE_PROPOSER: Final[int] = 4

NODE_ROLE_NAMES: Final[dict] = {
    NODE_ROLE_MINER: "MINER",
    NODE_ROLE_SERVING: "SERVING",
    NODE_ROLE_VALIDATOR: "VALIDATOR",
    NODE_ROLE_PROPOSER: "PROPOSER",
}


# =============================================================================
# STATUS VALUES
# =============================================================================

# Common status values
STATUS_PENDING: Final[str] = "pending"
STATUS_ACTIVE: Final[str] = "active"
STATUS_COMPLETED: Final[str] = "completed"
STATUS_FAILED: Final[str] = "failed"
STATUS_CANCELLED: Final[str] = "cancelled"


# =============================================================================
# HTTP STATUS CODES (for reference)
# =============================================================================

HTTP_OK: Final[int] = 200
HTTP_CREATED: Final[int] = 201
HTTP_BAD_REQUEST: Final[int] = 400
HTTP_UNAUTHORIZED: Final[int] = 401
HTTP_PAYMENT_REQUIRED: Final[int] = 402
HTTP_FORBIDDEN: Final[int] = 403
HTTP_NOT_FOUND: Final[int] = 404
HTTP_RATE_LIMITED: Final[int] = 429
HTTP_INTERNAL_ERROR: Final[int] = 500
HTTP_SERVICE_UNAVAILABLE: Final[int] = 503

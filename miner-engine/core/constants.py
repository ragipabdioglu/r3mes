#!/usr/bin/env python3
"""
R3MES Core Constants

Centralized constants for the entire miner-engine.
All magic numbers and configuration defaults are defined here.
"""

from typing import Final

# =============================================================================
# CHUNK & TOKEN CONSTANTS
# =============================================================================

# Fixed chunk size in tokens (as per documentation)
CHUNK_SIZE_TOKENS: Final[int] = 2048

# Maximum sequence length for model input
MAX_SEQUENCE_LENGTH: Final[int] = 4096

# Minimum chunk size (smaller chunks are padded)
MIN_CHUNK_SIZE_TOKENS: Final[int] = 256

# =============================================================================
# VERIFICATION THRESHOLDS
# =============================================================================

# Cosine similarity threshold for gradient verification
COSINE_SIMILARITY_THRESHOLD: Final[float] = 0.95

# Top-K fingerprint size for similarity comparison
TOP_K_FINGERPRINT_SIZE: Final[int] = 100

# Loss tolerance for Layer 2 verification (percentage)
LOSS_TOLERANCE_PERCENT: Final[float] = 0.01  # 1%

# Maximum allowed loss difference for verification
MAX_LOSS_DIFFERENCE: Final[float] = 0.1

# =============================================================================
# TRAP JOB CONSTANTS
# =============================================================================

# Trap job injection ratio (10% of chunks are traps)
TRAP_JOB_RATIO: Final[float] = 0.10

# Minimum trap jobs per batch
MIN_TRAP_JOBS_PER_BATCH: Final[int] = 1

# Genesis vault minimum entries
GENESIS_VAULT_MIN_ENTRIES: Final[int] = 100

# Trap verification similarity threshold
TRAP_VERIFICATION_THRESHOLD: Final[float] = 0.98

# =============================================================================
# GRADIENT COMPRESSION
# =============================================================================

# Default top-k compression ratio (keep top 10% of values)
DEFAULT_TOP_K_COMPRESSION: Final[float] = 0.10

# Gradient accumulation steps (default)
DEFAULT_GRADIENT_ACCUMULATION_STEPS: Final[int] = 4

# Maximum gradient norm for clipping
MAX_GRADIENT_NORM: Final[float] = 1.0

# =============================================================================
# LORA CONFIGURATION
# =============================================================================

# Default LoRA rank
DEFAULT_LORA_RANK: Final[int] = 8

# Default LoRA alpha
DEFAULT_LORA_ALPHA: Final[int] = 16

# Default LoRA dropout
DEFAULT_LORA_DROPOUT: Final[float] = 0.1

# =============================================================================
# NETWORK & TIMEOUT CONSTANTS
# =============================================================================

# gRPC connection timeout (seconds)
GRPC_TIMEOUT_SECONDS: Final[int] = 30

# IPFS upload timeout (seconds)
IPFS_UPLOAD_TIMEOUT_SECONDS: Final[int] = 60

# IPFS download timeout (seconds)
IPFS_DOWNLOAD_TIMEOUT_SECONDS: Final[int] = 120

# Task claim timeout (seconds)
TASK_CLAIM_TIMEOUT_SECONDS: Final[int] = 10

# Heartbeat interval (seconds)
HEARTBEAT_INTERVAL_SECONDS: Final[int] = 30

# =============================================================================
# TASK POOL CONSTANTS
# =============================================================================

# Maximum prefetch tasks
DEFAULT_MAX_PREFETCH: Final[int] = 5

# Task pool query limit
DEFAULT_TASK_POOL_QUERY_LIMIT: Final[int] = 1000

# Task pool claim limit per request
DEFAULT_TASK_POOL_CLAIM_LIMIT: Final[int] = 1

# =============================================================================
# VRAM PROFILES (in GB)
# =============================================================================

# VRAM thresholds for profile detection
VRAM_THRESHOLD_LOW: Final[int] = 4  # < 4GB
VRAM_THRESHOLD_MEDIUM: Final[int] = 8  # 4-8GB
VRAM_THRESHOLD_HIGH: Final[int] = 16  # 8-16GB
VRAM_THRESHOLD_ULTRA: Final[int] = 24  # > 24GB

# Batch sizes per VRAM profile
BATCH_SIZE_LOW_VRAM: Final[int] = 1
BATCH_SIZE_MEDIUM_VRAM: Final[int] = 2
BATCH_SIZE_HIGH_VRAM: Final[int] = 4
BATCH_SIZE_ULTRA_VRAM: Final[int] = 8

# =============================================================================
# BLOCKCHAIN CONSTANTS
# =============================================================================

# Default chain ID
DEFAULT_CHAIN_ID: Final[str] = "remes-test"

# Default gRPC port
DEFAULT_GRPC_PORT: Final[int] = 9090

# Default REST port
DEFAULT_REST_PORT: Final[int] = 1317

# Default Arrow Flight port
DEFAULT_ARROW_FLIGHT_PORT: Final[int] = 8815

# =============================================================================
# QUANTIZATION CONSTANTS
# =============================================================================

# BitNet quantization bits
BITNET_QUANTIZATION_BITS: Final[int] = 1  # 1.58-bit

# Default quantization bits for model loading
DEFAULT_QUANTIZATION_BITS: Final[int] = 4

# =============================================================================
# VERIFICATION LAYERS
# =============================================================================

# Layer 1: GPU hash verification (fast, same architecture)
VERIFICATION_LAYER_1_GPU: Final[int] = 1

# Layer 2: Loss-based spot check (medium, cross-architecture)
VERIFICATION_LAYER_2_LOSS: Final[int] = 2

# Layer 3: CPU sandbox verification (slow, dispute resolution)
VERIFICATION_LAYER_3_CPU: Final[int] = 3

# =============================================================================
# BOND & STAKING CONSTANTS
# =============================================================================

# Minimum bond amount (in smallest unit)
MIN_BOND_AMOUNT: Final[int] = 1000

# Bond multiplier for disputes
DISPUTE_BOND_MULTIPLIER: Final[float] = 2.0

# Slash percentage for failed verification
SLASH_PERCENTAGE: Final[float] = 0.10  # 10%

# =============================================================================
# LOGGING CONSTANTS
# =============================================================================

# Default log level
DEFAULT_LOG_LEVEL: Final[str] = "INFO"

# Log rotation size (bytes)
LOG_ROTATION_SIZE_BYTES: Final[int] = 10 * 1024 * 1024  # 10MB

# Log retention days
LOG_RETENTION_DAYS: Final[int] = 7

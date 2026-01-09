#!/usr/bin/env python3
"""
R3MES Core Types

Shared type definitions, dataclasses, and enums for the miner-engine.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, List, Any, Union
import torch


# =============================================================================
# ENUMS
# =============================================================================

class VerificationLayer(Enum):
    """Three-layer verification system layers."""
    LAYER_1_GPU = auto()      # Fast GPU hash verification
    LAYER_2_LOSS = auto()     # Loss-based spot check
    LAYER_3_CPU = auto()      # CPU sandbox verification


class VerificationStatus(Enum):
    """Verification status for gradient submissions."""
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    ESCALATED = "escalated"
    DISPUTED = "disputed"


class TaskStatus(Enum):
    """Task/chunk processing status."""
    AVAILABLE = "available"
    CLAIMED = "claimed"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


class MinerRole(Enum):
    """Miner node roles."""
    MINER = "miner"
    SERVING = "serving"
    PROPOSER = "proposer"
    MULTI_ROLE = "multi_role"


class GPUArchitecture(Enum):
    """NVIDIA GPU architectures."""
    AMPERE = "Ampere"      # RTX 30xx, A100
    ADA = "Ada"            # RTX 40xx
    HOPPER = "Hopper"      # H100
    TURING = "Turing"      # RTX 20xx
    VOLTA = "Volta"        # V100
    PASCAL = "Pascal"      # GTX 10xx
    UNKNOWN = "Unknown"


class CompressionType(Enum):
    """Gradient compression types."""
    NONE = "none"
    TOP_K = "top_k"
    RANDOM_K = "random_k"
    QUANTIZED = "quantized"


# =============================================================================
# DATACLASSES - Verification
# =============================================================================

@dataclass
class VerificationResult:
    """Result of gradient verification."""
    is_valid: bool
    layer: VerificationLayer
    status: VerificationStatus
    similarity_score: Optional[float] = None
    loss_difference: Optional[float] = None
    hash_match: bool = False
    reason: str = ""
    requires_escalation: bool = False
    bond_required: Optional[int] = None


@dataclass
class ChallengeRecord:
    """Record of a verification challenge."""
    challenge_id: str
    challenger_address: str
    miner_address: str
    gradient_hash: str
    layer: VerificationLayer
    status: VerificationStatus
    created_at: int  # Unix timestamp
    resolved_at: Optional[int] = None
    resolution: Optional[str] = None
    bond_amount: int = 0


@dataclass
class CPUVerificationRequest:
    """Request for CPU sandbox verification."""
    challenge_id: str
    gradient_ipfs_hash: str
    expected_hash: str
    seed: int
    chunk_id: int
    miner_address: str
    execution_mode: str = "CPU"


# =============================================================================
# DATACLASSES - Task & Chunk
# =============================================================================

@dataclass
class TaskChunk:
    """A single task chunk for processing."""
    chunk_id: int
    pool_id: int
    data_hash: str
    shard_id: int
    token_count: int = 2048
    is_trap: bool = False  # Hidden from miner
    status: TaskStatus = TaskStatus.AVAILABLE
    claimed_by: Optional[str] = None
    claimed_at: Optional[int] = None
    completed_at: Optional[int] = None
    gradient_hash: Optional[str] = None


@dataclass
class TaskPool:
    """Task pool containing chunks for a training round."""
    pool_id: int
    training_round_id: int
    total_chunks: int
    available_chunks: int
    completed_chunks: int
    trap_chunks: int
    created_at: int
    expires_at: int
    status: str = "active"


@dataclass
class ChunkData:
    """Chunk data downloaded from IPFS."""
    chunk_id: int
    input_ids: torch.Tensor
    labels: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# DATACLASSES - Trap Jobs
# =============================================================================

@dataclass
class GenesisVaultEntry:
    """Entry in the Genesis Vault for trap verification."""
    entry_id: str
    chunk_hash: str
    expected_gradient_hash: str
    expected_fingerprint: List[float]
    seed: int
    created_at: int
    verified_count: int = 0
    last_verified_at: Optional[int] = None


@dataclass
class TrapVerificationResult:
    """Result of trap job verification."""
    chunk_id: int
    is_valid: bool
    similarity_score: float
    expected_hash: str
    actual_hash: str
    miner_address: str
    verified_at: int


@dataclass
class BlindDeliveryBatch:
    """Batch of chunks with blind trap injection."""
    batch_id: str
    real_chunks: List[TaskChunk]
    trap_chunks: List[TaskChunk]
    mixed_chunks: List[TaskChunk]  # Sanitized for miner
    trap_ratio: float


# =============================================================================
# DATACLASSES - Gradient
# =============================================================================

@dataclass
class GradientFingerprint:
    """Top-K fingerprint of gradient for similarity comparison."""
    indices: List[int]
    values: List[float]
    layer_name: str
    k: int = 100


@dataclass
class CompressedGradient:
    """Compressed gradient data."""
    indices: torch.Tensor
    values: torch.Tensor
    shape: tuple
    compression_type: CompressionType
    compression_ratio: float
    original_size: int
    compressed_size: int


@dataclass
class GradientSubmission:
    """Gradient submission to blockchain."""
    submission_id: str
    miner_address: str
    chunk_id: int
    pool_id: int
    gradient_hash: str
    ipfs_hash: str
    loss: float
    submitted_at: int
    verification_status: VerificationStatus = VerificationStatus.PENDING


# =============================================================================
# DATACLASSES - Miner State
# =============================================================================

@dataclass
class MinerStats:
    """Miner statistics."""
    address: str
    total_submissions: int = 0
    successful_submissions: int = 0
    failed_submissions: int = 0
    total_rewards: int = 0
    current_streak: int = 0
    best_streak: int = 0
    average_loss: float = 0.0
    gpu_architecture: Optional[str] = None
    vram_gb: Optional[float] = None


@dataclass
class TrainingRound:
    """Training round information."""
    round_id: int
    global_seed: int
    start_block: int
    end_block: int
    total_chunks: int
    completed_chunks: int
    participating_miners: int
    status: str = "active"


# =============================================================================
# DATACLASSES - Configuration
# =============================================================================

@dataclass
class MinerConfig:
    """Miner configuration."""
    private_key: str
    blockchain_url: str
    chain_id: str = "remes-test"
    role: MinerRole = MinerRole.MINER
    lora_rank: int = 8
    learning_rate: float = 1e-4
    gradient_accumulation_steps: int = 4
    top_k_compression: float = 0.1
    deterministic: bool = True
    use_tls: bool = False
    tls_cert_file: Optional[str] = None
    tls_key_file: Optional[str] = None
    tls_ca_file: Optional[str] = None


@dataclass
class ServingConfig:
    """Serving node configuration."""
    private_key: str
    blockchain_url: str
    chain_id: str = "remes-test"
    inference_port: int = 8080
    max_concurrent_requests: int = 10
    lora_cache_size: int = 5
    heartbeat_interval: int = 30


@dataclass
class ProposerConfig:
    """Proposer node configuration."""
    private_key: str
    blockchain_url: str
    chain_id: str = "remes-test"
    aggregation_threshold: int = 100
    commit_interval: int = 60
    min_gradients_per_commit: int = 10


# =============================================================================
# TYPE ALIASES
# =============================================================================

# Gradient dictionary type
GradientDict = Dict[str, torch.Tensor]

# Address type (hex string)
Address = str

# Hash type (hex string)
Hash = str

# IPFS CID type
IPFSHash = str

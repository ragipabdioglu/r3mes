"""
Core training modules for R3MES miner engine.

This module provides:
- BitLinear: BitNet 1.58-bit linear layer with LoRA
- BitLinearDoRA: DoRA (Weight-Decomposed Low-Rank Adaptation) layer
- DoRAAdapter: Serializable DoRA adapter container
- DoRAExpertRegistry: Registry for managing DoRA experts
- LoRATrainer: Training utilities for LoRA adapters
- Constants: Centralized configuration constants
- Types: Shared dataclasses and enums
- Exceptions: Custom exception classes
- Validation: Input validation utilities
- Verification: Gradient verification system
"""

from core.bitlinear import BitLinear
from core.trainer import LoRATrainer

# DoRA (Weight-Decomposed Low-Rank Adaptation)
from core.dora import (
    BitLinearDoRA,
    DoRAAdapter,
    DoRAExpertRegistry,
    create_dora_from_bitlinear,
    merge_dora_experts,
)

# Inference Backend
from core.inference_backend import (
    InferenceBackend,
    BackendType,
    BackendCapabilities,
    InferenceResult,
    BackendRegistry,
    get_backend_registry,
    get_best_backend,
)

# Constants
from core.constants import (
    CHUNK_SIZE_TOKENS,
    COSINE_SIMILARITY_THRESHOLD,
    TOP_K_FINGERPRINT_SIZE,
    TRAP_JOB_RATIO,
    DEFAULT_LORA_RANK,
    DEFAULT_LORA_ALPHA,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_TOP_K_COMPRESSION,
)

# Types
from core.types import (
    VerificationLayer,
    VerificationStatus,
    TaskStatus,
    MinerRole,
    GPUArchitecture,
    VerificationResult,
    TaskChunk,
    GenesisVaultEntry,
    GradientFingerprint,
    MinerConfig,
)

# Exceptions
from core.exceptions import (
    R3MESError,
    VerificationError,
    HashMismatchError,
    TaskError,
    TrapJobError,
    GradientError,
    BlockchainError,
    ValidationError,
)

# Validation
from core.validation import (
    validate_address,
    validate_hash,
    validate_chunk_size,
    validate_gradient_dict,
)

__all__ = [
    # Core classes
    'BitLinear',
    'LoRATrainer',
    # DoRA
    'BitLinearDoRA',
    'DoRAAdapter',
    'DoRAExpertRegistry',
    'create_dora_from_bitlinear',
    'merge_dora_experts',
    # Inference Backend
    'InferenceBackend',
    'BackendType',
    'BackendCapabilities',
    'InferenceResult',
    'BackendRegistry',
    'get_backend_registry',
    'get_best_backend',
    # Constants
    'CHUNK_SIZE_TOKENS',
    'COSINE_SIMILARITY_THRESHOLD',
    'TOP_K_FINGERPRINT_SIZE',
    'TRAP_JOB_RATIO',
    'DEFAULT_LORA_RANK',
    'DEFAULT_LORA_ALPHA',
    'DEFAULT_GRADIENT_ACCUMULATION_STEPS',
    'DEFAULT_TOP_K_COMPRESSION',
    # Types
    'VerificationLayer',
    'VerificationStatus',
    'TaskStatus',
    'MinerRole',
    'GPUArchitecture',
    'VerificationResult',
    'TaskChunk',
    'GenesisVaultEntry',
    'GradientFingerprint',
    'MinerConfig',
    # Exceptions
    'R3MESError',
    'VerificationError',
    'HashMismatchError',
    'TaskError',
    'TrapJobError',
    'GradientError',
    'BlockchainError',
    'ValidationError',
    # Validation
    'validate_address',
    'validate_hash',
    'validate_chunk_size',
    'validate_gradient_dict',
]


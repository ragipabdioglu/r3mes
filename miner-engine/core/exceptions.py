#!/usr/bin/env python3
"""
R3MES Core Exceptions

Custom exception classes for the miner-engine.
Provides specific error types for better error handling and debugging.
"""

from typing import Optional, Any


# =============================================================================
# BASE EXCEPTIONS
# =============================================================================

class R3MESError(Exception):
    """Base exception for all R3MES errors."""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message)
        self.message = message
        self.details = details
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class R3MESConfigError(R3MESError):
    """Configuration-related errors."""
    pass


class R3MESNetworkError(R3MESError):
    """Network and communication errors."""
    pass


# =============================================================================
# VERIFICATION EXCEPTIONS
# =============================================================================

class VerificationError(R3MESError):
    """Base exception for verification errors."""
    pass


class HashMismatchError(VerificationError):
    """Gradient hash mismatch error."""
    
    def __init__(
        self,
        expected_hash: str,
        actual_hash: str,
        layer: Optional[str] = None,
    ):
        message = f"Hash mismatch: expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
        super().__init__(message, {"expected": expected_hash, "actual": actual_hash, "layer": layer})
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash
        self.layer = layer


class SimilarityThresholdError(VerificationError):
    """Similarity score below threshold error."""
    
    def __init__(
        self,
        similarity_score: float,
        threshold: float,
        chunk_id: Optional[int] = None,
    ):
        message = f"Similarity {similarity_score:.4f} below threshold {threshold:.4f}"
        super().__init__(message, {"score": similarity_score, "threshold": threshold, "chunk_id": chunk_id})
        self.similarity_score = similarity_score
        self.threshold = threshold
        self.chunk_id = chunk_id


class CPUVerificationRequired(VerificationError):
    """CPU sandbox verification required for dispute resolution."""
    
    def __init__(self, reason: str, challenge_id: Optional[str] = None):
        message = f"CPU verification required: {reason}"
        super().__init__(message, {"reason": reason, "challenge_id": challenge_id})
        self.reason = reason
        self.challenge_id = challenge_id


class CrossArchitectureError(VerificationError):
    """Cross-architecture verification error."""
    
    def __init__(
        self,
        miner_arch: str,
        validator_arch: str,
    ):
        message = f"Cross-architecture verification required: {miner_arch} vs {validator_arch}"
        super().__init__(message, {"miner_arch": miner_arch, "validator_arch": validator_arch})
        self.miner_arch = miner_arch
        self.validator_arch = validator_arch


# =============================================================================
# TASK & CHUNK EXCEPTIONS
# =============================================================================

class TaskError(R3MESError):
    """Base exception for task-related errors."""
    pass


class TaskNotFoundError(TaskError):
    """Task not found in pool."""
    
    def __init__(self, chunk_id: int, pool_id: Optional[int] = None):
        message = f"Task not found: chunk_id={chunk_id}"
        if pool_id:
            message += f", pool_id={pool_id}"
        super().__init__(message, {"chunk_id": chunk_id, "pool_id": pool_id})
        self.chunk_id = chunk_id
        self.pool_id = pool_id


class TaskClaimError(TaskError):
    """Failed to claim task."""
    
    def __init__(self, chunk_id: int, reason: str):
        message = f"Failed to claim task {chunk_id}: {reason}"
        super().__init__(message, {"chunk_id": chunk_id, "reason": reason})
        self.chunk_id = chunk_id
        self.reason = reason


class TaskExpiredError(TaskError):
    """Task has expired."""
    
    def __init__(self, chunk_id: int, expired_at: int):
        message = f"Task {chunk_id} expired at {expired_at}"
        super().__init__(message, {"chunk_id": chunk_id, "expired_at": expired_at})
        self.chunk_id = chunk_id
        self.expired_at = expired_at


class ChunkSizeError(TaskError):
    """Invalid chunk size."""
    
    def __init__(self, actual_size: int, expected_size: int):
        message = f"Invalid chunk size: {actual_size} tokens (expected {expected_size})"
        super().__init__(message, {"actual": actual_size, "expected": expected_size})
        self.actual_size = actual_size
        self.expected_size = expected_size


class NoAvailableTasksError(TaskError):
    """No available tasks in pool."""
    
    def __init__(self, pool_id: int):
        message = f"No available tasks in pool {pool_id}"
        super().__init__(message, {"pool_id": pool_id})
        self.pool_id = pool_id


# =============================================================================
# TRAP JOB EXCEPTIONS
# =============================================================================

class TrapJobError(R3MESError):
    """Base exception for trap job errors."""
    pass


class TrapVerificationFailed(TrapJobError):
    """Trap job verification failed - potential fraud."""
    
    def __init__(
        self,
        chunk_id: int,
        miner_address: str,
        similarity_score: float,
    ):
        message = f"Trap verification failed for chunk {chunk_id} by {miner_address[:16]}..."
        super().__init__(message, {
            "chunk_id": chunk_id,
            "miner": miner_address,
            "similarity": similarity_score,
        })
        self.chunk_id = chunk_id
        self.miner_address = miner_address
        self.similarity_score = similarity_score


class GenesisVaultError(TrapJobError):
    """Genesis vault access error."""
    
    def __init__(self, message: str, entry_id: Optional[str] = None):
        super().__init__(message, {"entry_id": entry_id})
        self.entry_id = entry_id


# =============================================================================
# GRADIENT EXCEPTIONS
# =============================================================================

class GradientError(R3MESError):
    """Base exception for gradient errors."""
    pass


class GradientCompressionError(GradientError):
    """Gradient compression failed."""
    
    def __init__(self, reason: str, layer_name: Optional[str] = None):
        message = f"Gradient compression failed: {reason}"
        super().__init__(message, {"reason": reason, "layer": layer_name})
        self.reason = reason
        self.layer_name = layer_name


class GradientUploadError(GradientError):
    """Failed to upload gradient to IPFS."""
    
    def __init__(self, reason: str, ipfs_error: Optional[str] = None):
        message = f"Gradient upload failed: {reason}"
        super().__init__(message, {"reason": reason, "ipfs_error": ipfs_error})
        self.reason = reason
        self.ipfs_error = ipfs_error


class GradientDownloadError(GradientError):
    """Failed to download gradient from IPFS."""
    
    def __init__(self, ipfs_hash: str, reason: str):
        message = f"Failed to download gradient {ipfs_hash[:16]}...: {reason}"
        super().__init__(message, {"ipfs_hash": ipfs_hash, "reason": reason})
        self.ipfs_hash = ipfs_hash
        self.reason = reason


# =============================================================================
# BLOCKCHAIN EXCEPTIONS
# =============================================================================

class BlockchainError(R3MESError):
    """Base exception for blockchain errors."""
    pass


class TransactionError(BlockchainError):
    """Transaction submission failed."""
    
    def __init__(self, tx_type: str, reason: str, tx_hash: Optional[str] = None):
        message = f"Transaction {tx_type} failed: {reason}"
        super().__init__(message, {"tx_type": tx_type, "reason": reason, "tx_hash": tx_hash})
        self.tx_type = tx_type
        self.reason = reason
        self.tx_hash = tx_hash


class GlobalSeedError(BlockchainError):
    """Failed to retrieve global seed from blockchain."""
    
    def __init__(self, training_round_id: int, reason: str):
        message = f"Failed to get global seed for round {training_round_id}: {reason}"
        super().__init__(message, {"round_id": training_round_id, "reason": reason})
        self.training_round_id = training_round_id
        self.reason = reason


class InsufficientBondError(BlockchainError):
    """Insufficient bond for operation."""
    
    def __init__(self, required: int, available: int):
        message = f"Insufficient bond: required {required}, available {available}"
        super().__init__(message, {"required": required, "available": available})
        self.required = required
        self.available = available


# =============================================================================
# MODEL & TRAINING EXCEPTIONS
# =============================================================================

class ModelError(R3MESError):
    """Base exception for model errors."""
    pass


class ModelLoadError(ModelError):
    """Failed to load model."""
    
    def __init__(self, model_name: str, reason: str):
        message = f"Failed to load model {model_name}: {reason}"
        super().__init__(message, {"model": model_name, "reason": reason})
        self.model_name = model_name
        self.reason = reason


class LoRAError(ModelError):
    """LoRA-related error."""
    
    def __init__(self, message: str, layer_name: Optional[str] = None):
        super().__init__(message, {"layer": layer_name})
        self.layer_name = layer_name


class DeterministicExecutionError(ModelError):
    """Deterministic execution requirement not met."""
    
    def __init__(self, reason: str):
        message = f"Deterministic execution failed: {reason}"
        super().__init__(message, {"reason": reason})
        self.reason = reason


# =============================================================================
# IPFS EXCEPTIONS
# =============================================================================

class IPFSError(R3MESNetworkError):
    """Base exception for IPFS errors."""
    pass


class IPFSConnectionError(IPFSError):
    """IPFS daemon not connected."""
    
    def __init__(self, endpoint: Optional[str] = None):
        message = "IPFS daemon not connected"
        if endpoint:
            message += f" at {endpoint}"
        super().__init__(message, {"endpoint": endpoint})
        self.endpoint = endpoint


class IPFSTimeoutError(IPFSError):
    """IPFS operation timed out."""
    
    def __init__(self, operation: str, timeout: int):
        message = f"IPFS {operation} timed out after {timeout}s"
        super().__init__(message, {"operation": operation, "timeout": timeout})
        self.operation = operation
        self.timeout = timeout


# =============================================================================
# VALIDATION EXCEPTIONS
# =============================================================================

class ValidationError(R3MESError):
    """Input validation error."""
    pass


class InvalidAddressError(ValidationError):
    """Invalid blockchain address."""
    
    def __init__(self, address: str, reason: str = "invalid format"):
        message = f"Invalid address {address[:16]}...: {reason}"
        super().__init__(message, {"address": address, "reason": reason})
        self.address = address
        self.reason = reason


class InvalidHashError(ValidationError):
    """Invalid hash format."""
    
    def __init__(self, hash_value: str, expected_format: str = "hex"):
        message = f"Invalid hash format: expected {expected_format}"
        super().__init__(message, {"hash": hash_value, "expected": expected_format})
        self.hash_value = hash_value
        self.expected_format = expected_format


class InvalidConfigError(ValidationError):
    """Invalid configuration value."""
    
    def __init__(self, field: str, value: Any, reason: str):
        message = f"Invalid config {field}={value}: {reason}"
        super().__init__(message, {"field": field, "value": value, "reason": reason})
        self.field = field
        self.value = value
        self.reason = reason

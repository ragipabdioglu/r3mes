"""
Utility modules for R3MES miner engine.

Security Features:
- MANDATORY checksum verification (no bypass)
- Atomic downloads with rollback
- Retry with exponential backoff
- Multiple download source fallback
"""

from utils.gpu_detection import GPUArchitectureDetector
from utils.dataset_registry import DatasetRegistry, DatasetInfo, DatasetStatus
from utils.adapter_registry import AdapterRegistry, AdapterInfo, AdapterStatus, AdapterType
from utils.model_registry import ModelRegistry, ModelInfo, ModelStatus
from utils.unified_registry import UnifiedRegistry, SystemStatus, SystemState, AtomicLoadResult
from utils.verification import (
    VerificationPolicy,
    VerificationLevel,
    VerificationResult,
    VerificationChain,
    RetryConfig,
    AtomicDownload,
    calculate_checksum,
    verify_checksum,
    VerificationError,
    ChecksumMismatchError,
)
from utils.download_manager import (
    DownloadManager,
    DownloadSpec,
    DownloadResult,
    DownloadProgress,
    DownloadSource,
    DownloadStatus,
)

__all__ = [
    'GPUArchitectureDetector',
    # Dataset Registry
    'DatasetRegistry',
    'DatasetInfo', 
    'DatasetStatus',
    # Adapter Registry
    'AdapterRegistry',
    'AdapterInfo',
    'AdapterStatus',
    'AdapterType',
    # Model Registry
    'ModelRegistry',
    'ModelInfo',
    'ModelStatus',
    # Unified Registry
    'UnifiedRegistry',
    'SystemStatus',
    'SystemState',
    'AtomicLoadResult',
    # Verification
    'VerificationPolicy',
    'VerificationLevel',
    'VerificationResult',
    'VerificationChain',
    'RetryConfig',
    'AtomicDownload',
    'calculate_checksum',
    'verify_checksum',
    'VerificationError',
    'ChecksumMismatchError',
    # Download Manager
    'DownloadManager',
    'DownloadSpec',
    'DownloadResult',
    'DownloadProgress',
    'DownloadSource',
    'DownloadStatus',
]


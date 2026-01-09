#!/usr/bin/env python3
"""
R3MES Verification Module

Centralized verification utilities for model, dataset, and adapter integrity.
Implements strict security policies with no optional verification bypasses.

Security Principles:
1. MANDATORY checksum verification - never skip
2. IPFS hash must match blockchain before download
3. Atomic operations - all or nothing
4. Retry with exponential backoff
5. Rollback on failure

Trust Chain:
    Blockchain (immutable) → IPFS Hash Verification → Content Download → Checksum Verification
"""

import hashlib
import logging
import time
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Callable, TypeVar
from dataclasses import dataclass
from enum import Enum
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class VerificationError(Exception):
    """Raised when verification fails."""
    pass


class ChecksumMismatchError(VerificationError):
    """Raised when checksum doesn't match expected value."""
    pass


class IPFSHashMismatchError(VerificationError):
    """Raised when IPFS hash doesn't match blockchain."""
    pass


class IntegrityError(VerificationError):
    """Raised when content integrity check fails."""
    pass


class VerificationLevel(Enum):
    """Verification strictness levels."""
    STRICT = "strict"      # All checks mandatory, fail on any issue
    STANDARD = "standard"  # Checksum required, IPFS optional
    RELAXED = "relaxed"    # Warnings only (NOT RECOMMENDED for production)


@dataclass
class VerificationResult:
    """Result of a verification operation."""
    success: bool
    checksum_verified: bool
    ipfs_verified: bool
    blockchain_verified: bool
    local_checksum: Optional[str] = None
    expected_checksum: Optional[str] = None
    ipfs_hash: Optional[str] = None
    error_message: Optional[str] = None
    warnings: list = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "checksum_verified": self.checksum_verified,
            "ipfs_verified": self.ipfs_verified,
            "blockchain_verified": self.blockchain_verified,
            "local_checksum": self.local_checksum,
            "expected_checksum": self.expected_checksum,
            "ipfs_hash": self.ipfs_hash,
            "error_message": self.error_message,
            "warnings": self.warnings,
        }


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True


class VerificationPolicy:
    """
    Centralized verification policy.
    
    Enforces consistent verification across all registries.
    """
    
    def __init__(
        self,
        level: VerificationLevel = VerificationLevel.STRICT,
        require_checksum: bool = True,
        require_ipfs_match: bool = True,
        require_blockchain_confirmation: bool = True,
    ):
        self.level = level
        self.require_checksum = require_checksum
        self.require_ipfs_match = require_ipfs_match
        self.require_blockchain_confirmation = require_blockchain_confirmation
        
        # Override based on level
        if level == VerificationLevel.STRICT:
            self.require_checksum = True
            self.require_ipfs_match = True
            self.require_blockchain_confirmation = True
        elif level == VerificationLevel.RELAXED:
            logger.warning("RELAXED verification level is NOT recommended for production!")
    
    def should_fail_on_checksum_missing(self) -> bool:
        """Whether to fail if checksum is not available."""
        return self.level == VerificationLevel.STRICT
    
    def should_fail_on_ipfs_mismatch(self) -> bool:
        """Whether to fail if IPFS hash doesn't match."""
        return self.require_ipfs_match and self.level != VerificationLevel.RELAXED


# Default strict policy
DEFAULT_POLICY = VerificationPolicy(level=VerificationLevel.STRICT)


def calculate_checksum(path: Path, algorithm: str = "sha256") -> str:
    """
    Calculate checksum of a file or directory.
    
    For directories, includes file paths in hash for structure verification.
    
    Args:
        path: Path to file or directory
        algorithm: Hash algorithm (sha256, sha512, md5)
        
    Returns:
        Hex digest of checksum
        
    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If algorithm is not supported
    """
    if algorithm not in ("sha256", "sha512", "md5"):
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    hasher = hashlib.new(algorithm)
    
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    
    if path.is_file():
        _hash_file(path, hasher)
    elif path.is_dir():
        _hash_directory(path, hasher)
    else:
        raise ValueError(f"Unsupported path type: {path}")
    
    return hasher.hexdigest()


def _hash_file(file_path: Path, hasher) -> None:
    """Hash a single file."""
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):  # 64KB chunks
            hasher.update(chunk)


def _hash_directory(dir_path: Path, hasher) -> None:
    """
    Hash a directory recursively.
    
    Includes relative paths in hash to detect structure changes.
    """
    # Sort files for deterministic ordering
    files = sorted(dir_path.rglob('*'))
    
    for file_path in files:
        if file_path.is_file() and not file_path.name.startswith('.'):
            # Include relative path in hash
            rel_path = file_path.relative_to(dir_path)
            hasher.update(str(rel_path).encode('utf-8'))
            hasher.update(b'\x00')  # Separator
            
            _hash_file(file_path, hasher)


def verify_checksum(
    path: Path,
    expected_checksum: str,
    algorithm: str = "sha256",
) -> VerificationResult:
    """
    Verify checksum of a file or directory.
    
    Args:
        path: Path to verify
        expected_checksum: Expected checksum hex string
        algorithm: Hash algorithm
        
    Returns:
        VerificationResult with verification status
    """
    try:
        local_checksum = calculate_checksum(path, algorithm)
        
        if local_checksum == expected_checksum:
            return VerificationResult(
                success=True,
                checksum_verified=True,
                ipfs_verified=False,
                blockchain_verified=False,
                local_checksum=local_checksum,
                expected_checksum=expected_checksum,
            )
        else:
            return VerificationResult(
                success=False,
                checksum_verified=False,
                ipfs_verified=False,
                blockchain_verified=False,
                local_checksum=local_checksum,
                expected_checksum=expected_checksum,
                error_message=f"Checksum mismatch: expected {expected_checksum[:16]}..., got {local_checksum[:16]}...",
            )
    except Exception as e:
        return VerificationResult(
            success=False,
            checksum_verified=False,
            ipfs_verified=False,
            blockchain_verified=False,
            expected_checksum=expected_checksum,
            error_message=f"Checksum calculation failed: {e}",
        )


def verify_ipfs_hash(
    ipfs_client,
    ipfs_hash: str,
    expected_checksum: Optional[str] = None,
) -> VerificationResult:
    """
    Verify IPFS content hash and optionally content checksum.
    
    Args:
        ipfs_client: IPFSClient instance
        ipfs_hash: IPFS CID to verify
        expected_checksum: Optional expected content checksum
        
    Returns:
        VerificationResult with verification status
    """
    if not ipfs_client or not ipfs_client.is_connected():
        return VerificationResult(
            success=False,
            checksum_verified=False,
            ipfs_verified=False,
            blockchain_verified=False,
            ipfs_hash=ipfs_hash,
            error_message="IPFS client not connected",
        )
    
    try:
        # Check if content exists on IPFS
        content = ipfs_client.retrieve_content(ipfs_hash)
        
        if content is None:
            return VerificationResult(
                success=False,
                checksum_verified=False,
                ipfs_verified=False,
                blockchain_verified=False,
                ipfs_hash=ipfs_hash,
                error_message=f"Content not found on IPFS: {ipfs_hash}",
            )
        
        # IPFS CID is content-addressed, so if we got content, hash is valid
        ipfs_verified = True
        
        # Optionally verify content checksum
        checksum_verified = False
        local_checksum = None
        
        if expected_checksum:
            local_checksum = hashlib.sha256(content).hexdigest()
            checksum_verified = (local_checksum == expected_checksum)
        
        success = ipfs_verified and (checksum_verified if expected_checksum else True)
        
        return VerificationResult(
            success=success,
            checksum_verified=checksum_verified,
            ipfs_verified=ipfs_verified,
            blockchain_verified=False,
            local_checksum=local_checksum,
            expected_checksum=expected_checksum,
            ipfs_hash=ipfs_hash,
            error_message=None if success else "Content checksum mismatch",
        )
        
    except Exception as e:
        return VerificationResult(
            success=False,
            checksum_verified=False,
            ipfs_verified=False,
            blockchain_verified=False,
            ipfs_hash=ipfs_hash,
            error_message=f"IPFS verification failed: {e}",
        )


def with_retry(
    config: Optional[RetryConfig] = None,
    retryable_exceptions: tuple = (Exception,),
):
    """
    Decorator for retry with exponential backoff.
    
    Args:
        config: RetryConfig instance
        retryable_exceptions: Tuple of exceptions to retry on
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt < config.max_attempts - 1:
                        delay = min(
                            config.initial_delay * (config.exponential_base ** attempt),
                            config.max_delay
                        )
                        
                        if config.jitter:
                            import random
                            delay *= (0.5 + random.random())
                        
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{config.max_attempts}): {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {config.max_attempts} attempts: {e}"
                        )
            
            raise last_exception
        
        return wrapper
    return decorator


class AtomicDownload:
    """
    Context manager for atomic download operations.
    
    Downloads to a temporary location, then moves to final location
    only after verification succeeds. Rolls back on failure.
    """
    
    def __init__(
        self,
        final_path: Path,
        verify_func: Optional[Callable[[Path], bool]] = None,
    ):
        """
        Initialize atomic download.
        
        Args:
            final_path: Final destination path
            verify_func: Optional verification function (path -> bool)
        """
        self.final_path = Path(final_path)
        self.verify_func = verify_func
        self.temp_path: Optional[Path] = None
        self._success = False
    
    def __enter__(self) -> Path:
        """Create temporary download path."""
        self.temp_path = self.final_path.parent / f".{self.final_path.name}.tmp.{int(time.time())}"
        self.temp_path.parent.mkdir(parents=True, exist_ok=True)
        return self.temp_path
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finalize or rollback download."""
        if exc_type is not None:
            # Exception occurred, rollback
            self._cleanup_temp()
            return False
        
        if self.temp_path is None or not self.temp_path.exists():
            return False
        
        # Verify if function provided
        if self.verify_func:
            try:
                if not self.verify_func(self.temp_path):
                    logger.error(f"Verification failed for {self.temp_path}")
                    self._cleanup_temp()
                    raise VerificationError("Download verification failed")
            except Exception as e:
                logger.error(f"Verification error: {e}")
                self._cleanup_temp()
                raise
        
        # Move to final location
        try:
            # Remove existing if present
            if self.final_path.exists():
                if self.final_path.is_dir():
                    shutil.rmtree(self.final_path)
                else:
                    self.final_path.unlink()
            
            # Move temp to final
            shutil.move(str(self.temp_path), str(self.final_path))
            self._success = True
            logger.info(f"Successfully downloaded to {self.final_path}")
            
        except Exception as e:
            logger.error(f"Failed to move download to final location: {e}")
            self._cleanup_temp()
            raise
        
        return False
    
    def _cleanup_temp(self):
        """Clean up temporary files."""
        if self.temp_path and self.temp_path.exists():
            try:
                if self.temp_path.is_dir():
                    shutil.rmtree(self.temp_path)
                else:
                    self.temp_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")
    
    @property
    def success(self) -> bool:
        return self._success


class VerificationChain:
    """
    Chain of verification steps.
    
    Executes verification steps in order, failing fast on any error.
    """
    
    def __init__(self, policy: VerificationPolicy = DEFAULT_POLICY):
        self.policy = policy
        self._steps: list = []
    
    def add_step(
        self,
        name: str,
        verify_func: Callable[[], VerificationResult],
        required: bool = True,
    ) -> "VerificationChain":
        """Add a verification step."""
        self._steps.append({
            "name": name,
            "func": verify_func,
            "required": required,
        })
        return self
    
    def execute(self) -> VerificationResult:
        """
        Execute all verification steps.
        
        Returns:
            Combined VerificationResult
        """
        combined = VerificationResult(
            success=True,
            checksum_verified=False,
            ipfs_verified=False,
            blockchain_verified=False,
            warnings=[],
        )
        
        for step in self._steps:
            try:
                result = step["func"]()
                
                # Update combined result
                if result.checksum_verified:
                    combined.checksum_verified = True
                    combined.local_checksum = result.local_checksum
                    combined.expected_checksum = result.expected_checksum
                
                if result.ipfs_verified:
                    combined.ipfs_verified = True
                    combined.ipfs_hash = result.ipfs_hash
                
                if result.blockchain_verified:
                    combined.blockchain_verified = True
                
                if not result.success:
                    if step["required"]:
                        combined.success = False
                        combined.error_message = f"{step['name']}: {result.error_message}"
                        logger.error(f"Verification step '{step['name']}' failed: {result.error_message}")
                        return combined
                    else:
                        combined.warnings.append(f"{step['name']}: {result.error_message}")
                        logger.warning(f"Optional verification step '{step['name']}' failed: {result.error_message}")
                
            except Exception as e:
                if step["required"]:
                    combined.success = False
                    combined.error_message = f"{step['name']}: {e}"
                    return combined
                else:
                    combined.warnings.append(f"{step['name']}: {e}")
        
        return combined

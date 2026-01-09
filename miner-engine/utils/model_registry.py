#!/usr/bin/env python3
"""
R3MES Model Registry

Production-ready model management that:
1. Tracks approved models from blockchain
2. Verifies model integrity via IPFS hash (MANDATORY)
3. Downloads models from HuggingFace/IPFS with fallback
4. Ensures all nodes use the same model version
5. Handles model updates and migrations

Trust Chain:
    Blockchain (immutable hash) → IPFS (content verification) → Local Model

Security:
    - Checksum verification is MANDATORY (no bypass)
    - IPFS hash must match blockchain before download
    - Atomic downloads with rollback on failure
    - Retry with exponential backoff
"""

import os
import json
import logging
import time
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from utils.verification import (
    VerificationPolicy,
    VerificationLevel,
    VerificationResult,
    VerificationChain,
    calculate_checksum,
    verify_checksum,
    AtomicDownload,
    RetryConfig,
    ChecksumMismatchError,
)
from utils.download_manager import (
    DownloadManager,
    DownloadSpec,
    DownloadResult,
    DownloadProgress,
)

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model status enum."""
    NOT_FOUND = "not_found"
    DOWNLOADING = "downloading"
    VERIFYING = "verifying"
    VALID = "valid"
    INVALID = "invalid"
    OUTDATED = "outdated"
    CHECKSUM_MISMATCH = "checksum_mismatch"


@dataclass
class ModelInfo:
    """Information about a model version."""
    model_id: str
    version: str
    ipfs_hash: str
    checksum: str
    size_bytes: int
    architecture: str
    description: str
    huggingface_repo: Optional[str] = None
    huggingface_revision: Optional[str] = None
    approved_at: int = 0
    approval_height: int = 0
    is_active: bool = False
    min_vram_gb: float = 0.0
    min_ram_gb: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "ipfs_hash": self.ipfs_hash,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
            "architecture": self.architecture,
            "description": self.description,
            "huggingface_repo": self.huggingface_repo,
            "huggingface_revision": self.huggingface_revision,
            "approved_at": self.approved_at,
            "approval_height": self.approval_height,
            "is_active": self.is_active,
            "min_vram_gb": self.min_vram_gb,
            "min_ram_gb": self.min_ram_gb,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelInfo":
        return cls(
            model_id=data.get("model_id", ""),
            version=data.get("version", data.get("model_version", "1.0.0")),
            ipfs_hash=data.get("ipfs_hash", data.get("model_hash", "")),
            checksum=data.get("checksum", ""),
            size_bytes=data.get("size_bytes", 0),
            architecture=data.get("architecture", "bitnet"),
            description=data.get("description", ""),
            huggingface_repo=data.get("huggingface_repo"),
            huggingface_revision=data.get("huggingface_revision"),
            approved_at=data.get("approved_at", 0),
            approval_height=data.get("approval_height", data.get("update_height", 0)),
            is_active=data.get("is_active", False),
            min_vram_gb=data.get("min_vram_gb", 0.0),
            min_ram_gb=data.get("min_ram_gb", 0.0),
            metadata=data.get("metadata", {}),
        )
    
    def has_valid_checksum(self) -> bool:
        """Check if model has a valid checksum defined."""
        return bool(self.checksum and len(self.checksum) >= 32)


@dataclass
class LocalModelStatus:
    """Status of a local model."""
    model_id: str
    version: str
    status: ModelStatus
    local_path: Optional[str] = None
    local_checksum: Optional[str] = None
    expected_checksum: Optional[str] = None
    expected_ipfs_hash: Optional[str] = None
    size_bytes: int = 0
    last_verified: Optional[int] = None
    verification_result: Optional[VerificationResult] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "status": self.status.value,
            "local_path": self.local_path,
            "local_checksum": self.local_checksum,
            "expected_checksum": self.expected_checksum,
            "expected_ipfs_hash": self.expected_ipfs_hash,
            "size_bytes": self.size_bytes,
            "last_verified": self.last_verified,
            "error_message": self.error_message,
        }


class ModelRegistry:
    """
    Model registry for managing AI models.
    
    Implements trust chain: Blockchain → IPFS → Local
    
    Security Features:
    - MANDATORY checksum verification
    - Atomic downloads with rollback
    - Retry with exponential backoff
    - Multiple download source fallback
    """
    
    def __init__(
        self,
        blockchain_client=None,
        ipfs_client=None,
        models_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        auto_download: bool = True,
        verify_on_load: bool = True,
        verification_policy: Optional[VerificationPolicy] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        self.blockchain_client = blockchain_client
        self.ipfs_client = ipfs_client
        self.auto_download = auto_download
        self.verify_on_load = verify_on_load
        
        # Security: Default to STRICT verification
        self.verification_policy = verification_policy or VerificationPolicy(
            level=VerificationLevel.STRICT
        )
        self.retry_config = retry_config or RetryConfig()
        
        # Setup directories
        home = os.environ.get("HOME", os.environ.get("USERPROFILE", "."))
        self.models_dir = Path(models_dir or os.path.join(home, ".r3mes", "models"))
        self.cache_dir = Path(cache_dir or os.path.join(home, ".r3mes", "cache", "models"))
        
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Registry data
        self._approved_models: Dict[str, ModelInfo] = {}
        self._local_status: Dict[str, LocalModelStatus] = {}
        self._active_model_version: Optional[str] = None
        
        # Download manager with fallback support
        self._download_manager = DownloadManager(
            ipfs_client=ipfs_client,
            retry_config=self.retry_config,
            prefer_huggingface=True,
        )
        
        self._load_cache()
        logger.info(f"ModelRegistry initialized (models_dir: {self.models_dir})")
    
    def _load_cache(self):
        """Load cached model metadata."""
        try:
            cache_file = self.cache_dir / "model_registry.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                for model_data in data.get("approved_models", []):
                    info = ModelInfo.from_dict(model_data)
                    self._approved_models[info.version] = info
                self._active_model_version = data.get("active_model_version")
                logger.info(f"Loaded {len(self._approved_models)} models from cache")
        except Exception as e:
            logger.warning(f"Failed to load model cache: {e}")
    
    def _save_cache(self):
        """Save model metadata to cache."""
        try:
            cache_file = self.cache_dir / "model_registry.json"
            data = {
                "approved_models": [m.to_dict() for m in self._approved_models.values()],
                "active_model_version": self._active_model_version,
                "last_updated": int(time.time()),
            }
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save model cache: {e}")
    
    def sync_from_blockchain(self) -> bool:
        """Sync approved models from blockchain."""
        if not self.blockchain_client:
            logger.warning("No blockchain client configured, using cached data")
            return False
        
        try:
            logger.info("Syncing models from blockchain...")
            
            if hasattr(self.blockchain_client, 'query_global_model_state'):
                response = self.blockchain_client.query_global_model_state()
            else:
                response = self.blockchain_client.get_model_params()
            
            if response.get("success", True):
                checksum = response.get("checksum", "")
                if not checksum:
                    logger.warning("Blockchain returned model without checksum!")
                
                model_info = ModelInfo(
                    model_id="global",
                    version=response.get("model_version", "1.0.0"),
                    ipfs_hash=response.get("model_ipfs_hash", response.get("ipfs_hash", "")),
                    checksum=checksum,
                    size_bytes=response.get("size_bytes", 0),
                    architecture="bitnet",
                    description="Global model",
                    approval_height=response.get("last_updated_height", response.get("update_height", 0)),
                    is_active=True,
                )
                self._approved_models[model_info.version] = model_info
                self._active_model_version = model_info.version
            
            if hasattr(self.blockchain_client, 'query_model_versions'):
                versions_response = self.blockchain_client.query_model_versions()
                if versions_response.get("success"):
                    for version_data in versions_response.get("versions", []):
                        info = ModelInfo.from_dict(version_data)
                        self._approved_models[info.version] = info
                    if versions_response.get("active_version"):
                        self._active_model_version = versions_response["active_version"]
            
            self._save_cache()
            logger.info(f"Synced {len(self._approved_models)} models from blockchain")
            return True
        except Exception as e:
            logger.error(f"Failed to sync from blockchain: {e}")
            return False
    
    def get_active_model_hash(self) -> Optional[str]:
        """Get the IPFS hash of the currently active model."""
        if self.blockchain_client:
            try:
                response = self.blockchain_client.get_model_params()
                ipfs_hash = response.get("model_ipfs_hash")
                if ipfs_hash:
                    return ipfs_hash
            except Exception as e:
                logger.warning(f"Failed to query active model: {e}")
        
        if self._active_model_version and self._active_model_version in self._approved_models:
            return self._approved_models[self._active_model_version].ipfs_hash
        return None
    
    def get_active_model_version(self) -> Optional[str]:
        """Get the currently active model version."""
        return self._active_model_version

    
    def verify_model(
        self,
        local_path: str,
        model_version: Optional[str] = None,
        strict: bool = True,
    ) -> LocalModelStatus:
        """
        Verify a local model against blockchain-approved hash.
        
        SECURITY: Checksum verification is MANDATORY in strict mode.
        """
        version = model_version or self._active_model_version
        
        if not version:
            self.sync_from_blockchain()
            version = self._active_model_version
        
        if not version or version not in self._approved_models:
            return LocalModelStatus(
                model_id="unknown",
                version=version or "unknown",
                status=ModelStatus.NOT_FOUND,
                error_message="Model version not found in approved list",
            )
        
        model_info = self._approved_models[version]
        path = Path(local_path)
        
        if not path.exists():
            return LocalModelStatus(
                model_id=model_info.model_id,
                version=version,
                status=ModelStatus.NOT_FOUND,
                expected_checksum=model_info.checksum,
                expected_ipfs_hash=model_info.ipfs_hash,
                error_message=f"Model not found at {local_path}",
            )
        
        # SECURITY: Require valid checksum in strict mode
        if strict and not model_info.has_valid_checksum():
            logger.error(f"Model {version} has no valid checksum - security risk")
            return LocalModelStatus(
                model_id=model_info.model_id,
                version=version,
                status=ModelStatus.INVALID,
                local_path=local_path,
                error_message="Model has no valid checksum in blockchain",
            )
        
        # Calculate local checksum
        try:
            local_checksum = calculate_checksum(path)
        except Exception as e:
            return LocalModelStatus(
                model_id=model_info.model_id,
                version=version,
                status=ModelStatus.INVALID,
                local_path=local_path,
                expected_checksum=model_info.checksum,
                error_message=f"Failed to calculate checksum: {e}",
            )
        
        # Get size
        if path.is_file():
            size_bytes = path.stat().st_size
        else:
            size_bytes = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        
        # MANDATORY checksum verification
        if local_checksum == model_info.checksum:
            status = LocalModelStatus(
                model_id=model_info.model_id,
                version=version,
                status=ModelStatus.VALID,
                local_path=local_path,
                local_checksum=local_checksum,
                expected_checksum=model_info.checksum,
                expected_ipfs_hash=model_info.ipfs_hash,
                size_bytes=size_bytes,
                last_verified=int(time.time()),
            )
        else:
            status = LocalModelStatus(
                model_id=model_info.model_id,
                version=version,
                status=ModelStatus.CHECKSUM_MISMATCH,
                local_path=local_path,
                local_checksum=local_checksum,
                expected_checksum=model_info.checksum,
                expected_ipfs_hash=model_info.ipfs_hash,
                size_bytes=size_bytes,
                error_message=f"Checksum mismatch: expected {model_info.checksum[:16]}..., got {local_checksum[:16]}...",
            )
        
        self._local_status[version] = status
        return status
    
    def verify_model_against_blockchain(self, local_path: str, model_version: Optional[str] = None) -> LocalModelStatus:
        """Alias for verify_model for backward compatibility."""
        return self.verify_model(local_path, model_version)
    
    def calculate_model_checksum(self, model_path: str) -> str:
        """Calculate SHA256 checksum of a model."""
        return calculate_checksum(Path(model_path))
    
    def download_model(
        self,
        model_version: Optional[str] = None,
        force: bool = False,
        progress_callback=None,
    ) -> Tuple[bool, str]:
        """Download a model with automatic fallback and verification."""
        version = model_version or self._active_model_version
        
        if not version:
            self.sync_from_blockchain()
            version = self._active_model_version
        
        if not version or version not in self._approved_models:
            return False, f"Model version {version} not found in approved list"
        
        model_info = self._approved_models[version]
        local_path = self.models_dir / version
        
        # SECURITY: Require checksum for download
        if not model_info.has_valid_checksum():
            return False, "Model has no valid checksum - refusing to download"
        
        # Check if already exists and valid
        if not force and local_path.exists():
            status = self.verify_model(str(local_path), version)
            if status.status == ModelStatus.VALID:
                logger.info(f"Model {version} already exists and is valid")
                return True, str(local_path)
        
        # Create download spec
        spec = DownloadSpec(
            name=f"model-{version}",
            version=version,
            ipfs_hash=model_info.ipfs_hash,
            huggingface_repo=model_info.huggingface_repo,
            huggingface_revision=model_info.huggingface_revision,
            expected_checksum=model_info.checksum,
            expected_size_bytes=model_info.size_bytes,
            verify_after_download=True,
        )
        
        # Download with atomic operation
        try:
            def verify_func(path: Path) -> bool:
                status = self.verify_model(str(path), version, strict=True)
                return status.status == ModelStatus.VALID
            
            with AtomicDownload(local_path, verify_func) as temp_path:
                result = self._download_manager.download(
                    spec=spec,
                    dest_path=temp_path,
                    progress_callback=progress_callback,
                    verify_checksum=True,
                )
                if not result.success:
                    raise Exception(result.error_message)
            
            status = self.verify_model(str(local_path), version)
            if status.status == ModelStatus.VALID:
                logger.info(f"Model {version} downloaded and verified")
                return True, str(local_path)
            else:
                return False, f"Post-download verification failed: {status.error_message}"
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False, str(e)
    
    def get_approved_models(self) -> List[ModelInfo]:
        """Get list of all approved models."""
        return list(self._approved_models.values())
    
    def get_model_info(self, version: str) -> Optional[ModelInfo]:
        """Get info for a specific model version."""
        return self._approved_models.get(version)
    
    def get_local_status(self, version: str) -> Optional[LocalModelStatus]:
        """Get local status for a model version."""
        return self._local_status.get(version)
    
    def ensure_model_available(self, model_version: Optional[str] = None) -> Tuple[bool, str]:
        """Ensure a model is available locally, downloading if necessary."""
        version = model_version or self._active_model_version
        
        if not version:
            self.sync_from_blockchain()
            version = self._active_model_version
        
        if not version:
            return False, "No active model version configured"
        
        local_path = self.models_dir / version
        
        if local_path.exists():
            status = self.verify_model(str(local_path), version)
            if status.status == ModelStatus.VALID:
                return True, str(local_path)
        
        if self.auto_download:
            return self.download_model(version)
        
        return False, f"Model {version} not available locally"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        local_valid = sum(1 for s in self._local_status.values() if s.status == ModelStatus.VALID)
        models_with_checksum = sum(1 for m in self._approved_models.values() if m.has_valid_checksum())
        
        return {
            "approved_models": len(self._approved_models),
            "models_with_checksum": models_with_checksum,
            "local_models": len(self._local_status),
            "local_valid": local_valid,
            "active_model_version": self._active_model_version,
            "models_dir": str(self.models_dir),
            "auto_download": self.auto_download,
            "verify_on_load": self.verify_on_load,
        }

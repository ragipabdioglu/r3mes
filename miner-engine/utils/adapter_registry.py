#!/usr/bin/env python3
"""
R3MES Adapter Registry

Production-ready DoRA/LoRA adapter management that:
1. Checks compatibility BEFORE download (not after)
2. Tracks adapter-model compatibility
3. Verifies adapter integrity via IPFS/blockchain (MANDATORY)
4. Handles adapter updates and migrations
5. Detects adapter conflicts

Security:
    - Checksum verification is MANDATORY
    - Compatibility check BEFORE download
    - Atomic downloads with rollback
    - Conflict detection
"""

import os
import json
import logging
import time
from typing import Optional, Dict, Any, List, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from utils.verification import (
    VerificationPolicy,
    VerificationLevel,
    calculate_checksum,
    verify_checksum,
    AtomicDownload,
    RetryConfig,
)
from utils.download_manager import (
    DownloadManager,
    DownloadSpec,
)

logger = logging.getLogger(__name__)


class AdapterType(Enum):
    """Adapter type enum."""
    LORA = "lora"
    DORA = "dora"
    QLORA = "qlora"


class AdapterStatus(Enum):
    """Adapter status enum."""
    NOT_FOUND = "not_found"
    DOWNLOADING = "downloading"
    VALID = "valid"
    INVALID = "invalid"
    INCOMPATIBLE = "incompatible"
    OUTDATED = "outdated"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    CONFLICT = "conflict"


@dataclass
class AdapterInfo:
    """Information about a DoRA/LoRA adapter."""
    adapter_id: str
    name: str
    adapter_type: AdapterType
    version: str
    ipfs_hash: str
    checksum: str  # REQUIRED
    size_bytes: int
    compatible_model_versions: List[str]
    min_model_version: str
    max_model_version: Optional[str]
    domain: str
    description: str
    lora_rank: int
    lora_alpha: float
    target_modules: List[str]
    approved_at: int
    approval_tx_hash: str
    proposer: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "name": self.name,
            "adapter_type": self.adapter_type.value,
            "version": self.version,
            "ipfs_hash": self.ipfs_hash,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
            "compatible_model_versions": self.compatible_model_versions,
            "min_model_version": self.min_model_version,
            "max_model_version": self.max_model_version,
            "domain": self.domain,
            "description": self.description,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "approved_at": self.approved_at,
            "approval_tx_hash": self.approval_tx_hash,
            "proposer": self.proposer,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdapterInfo":
        adapter_type_str = data.get("adapter_type", "dora")
        try:
            adapter_type = AdapterType(adapter_type_str)
        except ValueError:
            adapter_type = AdapterType.DORA
        
        return cls(
            adapter_id=data.get("adapter_id", ""),
            name=data.get("name", ""),
            adapter_type=adapter_type,
            version=data.get("version", "1.0.0"),
            ipfs_hash=data.get("ipfs_hash", ""),
            checksum=data.get("checksum", ""),
            size_bytes=data.get("size_bytes", 0),
            compatible_model_versions=data.get("compatible_model_versions", []),
            min_model_version=data.get("min_model_version", "1.0.0"),
            max_model_version=data.get("max_model_version"),
            domain=data.get("domain", "general"),
            description=data.get("description", ""),
            lora_rank=data.get("lora_rank", 8),
            lora_alpha=data.get("lora_alpha", 16.0),
            target_modules=data.get("target_modules", []),
            approved_at=data.get("approved_at", 0),
            approval_tx_hash=data.get("approval_tx_hash", ""),
            proposer=data.get("proposer", ""),
            metadata=data.get("metadata", {}),
        )
    
    def has_valid_checksum(self) -> bool:
        """Check if adapter has a valid checksum defined."""
        return bool(self.checksum and len(self.checksum) >= 32)
    
    def is_compatible_with(self, model_version: str) -> bool:
        """Check if adapter is compatible with a model version."""
        if self.compatible_model_versions:
            return model_version in self.compatible_model_versions
        
        if self.min_model_version and model_version < self.min_model_version:
            return False
        if self.max_model_version and model_version > self.max_model_version:
            return False
        
        return True


@dataclass
class LocalAdapterStatus:
    """Status of a local adapter."""
    adapter_id: str
    status: AdapterStatus
    local_path: Optional[str] = None
    local_checksum: Optional[str] = None
    expected_checksum: Optional[str] = None
    size_bytes: int = 0
    last_verified: Optional[int] = None
    compatibility_status: Optional[str] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "status": self.status.value,
            "local_path": self.local_path,
            "local_checksum": self.local_checksum,
            "expected_checksum": self.expected_checksum,
            "size_bytes": self.size_bytes,
            "last_verified": self.last_verified,
            "compatibility_status": self.compatibility_status,
            "error_message": self.error_message,
        }


class CompatibilityMatrix:
    """Manages adapter-model compatibility relationships."""
    
    def __init__(self):
        self._matrix: Dict[str, Set[str]] = {}
        self._adapter_models: Dict[str, Set[str]] = {}
    
    def add_compatibility(self, adapter_id: str, model_version: str):
        if model_version not in self._matrix:
            self._matrix[model_version] = set()
        self._matrix[model_version].add(adapter_id)
        
        if adapter_id not in self._adapter_models:
            self._adapter_models[adapter_id] = set()
        self._adapter_models[adapter_id].add(model_version)
    
    def is_compatible(self, adapter_id: str, model_version: str) -> bool:
        if adapter_id in self._adapter_models:
            return model_version in self._adapter_models[adapter_id]
        return False
    
    def get_compatible_adapters(self, model_version: str) -> Set[str]:
        return self._matrix.get(model_version, set())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "matrix": {k: list(v) for k, v in self._matrix.items()},
            "adapter_models": {k: list(v) for k, v in self._adapter_models.items()},
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompatibilityMatrix":
        matrix = cls()
        for model_version, adapters in data.get("matrix", {}).items():
            for adapter_id in adapters:
                matrix.add_compatibility(adapter_id, model_version)
        return matrix


class AdapterConflictDetector:
    """
    Detects conflicts between adapters.
    
    Conflicts occur when multiple adapters target the same modules.
    """
    
    def __init__(self):
        self._loaded_adapters: Dict[str, AdapterInfo] = {}
        self._module_usage: Dict[str, List[str]] = {}  # module -> list of adapter_ids
    
    def register_adapter(self, adapter_info: AdapterInfo) -> List[str]:
        """
        Register an adapter and check for conflicts.
        
        Returns list of conflicting adapter IDs.
        """
        conflicts = []
        
        for module in adapter_info.target_modules:
            if module in self._module_usage:
                conflicts.extend(self._module_usage[module])
            
            if module not in self._module_usage:
                self._module_usage[module] = []
            self._module_usage[module].append(adapter_info.adapter_id)
        
        self._loaded_adapters[adapter_info.adapter_id] = adapter_info
        return list(set(conflicts))
    
    def unregister_adapter(self, adapter_id: str):
        """Remove an adapter from conflict tracking."""
        if adapter_id in self._loaded_adapters:
            adapter_info = self._loaded_adapters[adapter_id]
            for module in adapter_info.target_modules:
                if module in self._module_usage:
                    self._module_usage[module] = [
                        aid for aid in self._module_usage[module]
                        if aid != adapter_id
                    ]
            del self._loaded_adapters[adapter_id]
    
    def check_conflicts(self, adapter_info: AdapterInfo) -> List[str]:
        """Check if adapter would conflict with loaded adapters."""
        conflicts = []
        for module in adapter_info.target_modules:
            if module in self._module_usage:
                conflicts.extend(self._module_usage[module])
        return list(set(conflicts))
    
    def get_loaded_adapters(self) -> List[str]:
        """Get list of loaded adapter IDs."""
        return list(self._loaded_adapters.keys())


class AdapterRegistry:
    """
    Adapter registry for managing DoRA/LoRA adapters.
    
    Security Features:
    - Compatibility check BEFORE download
    - MANDATORY checksum verification
    - Conflict detection
    - Atomic downloads with rollback
    """
    
    def __init__(
        self,
        blockchain_client=None,
        ipfs_client=None,
        adapters_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        current_model_version: str = "1.0.0",
        auto_download: bool = True,
        verify_on_load: bool = True,
        verification_policy: Optional[VerificationPolicy] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        self.blockchain_client = blockchain_client
        self.ipfs_client = ipfs_client
        self.current_model_version = current_model_version
        self.auto_download = auto_download
        self.verify_on_load = verify_on_load
        
        self.verification_policy = verification_policy or VerificationPolicy(
            level=VerificationLevel.STRICT
        )
        self.retry_config = retry_config or RetryConfig()
        
        home = os.environ.get("HOME", os.environ.get("USERPROFILE", "."))
        self.adapters_dir = Path(adapters_dir or os.path.join(home, ".r3mes", "adapters"))
        self.cache_dir = Path(cache_dir or os.path.join(home, ".r3mes", "cache", "adapters"))
        
        self.adapters_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._approved_adapters: Dict[str, AdapterInfo] = {}
        self._local_status: Dict[str, LocalAdapterStatus] = {}
        self._compatibility_matrix = CompatibilityMatrix()
        self._conflict_detector = AdapterConflictDetector()
        self._domain_index: Dict[str, List[str]] = {}
        
        self._download_manager = DownloadManager(
            ipfs_client=ipfs_client,
            retry_config=self.retry_config,
            prefer_huggingface=False,
        )
        
        self._load_cache()
        logger.info(f"AdapterRegistry initialized (adapters_dir: {self.adapters_dir})")
    
    def _load_cache(self):
        """Load cached adapter metadata."""
        try:
            cache_file = self.cache_dir / "adapter_registry.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                for adapter_data in data.get("approved_adapters", []):
                    info = AdapterInfo.from_dict(adapter_data)
                    self._approved_adapters[info.adapter_id] = info
                    self._index_adapter(info)
                if "compatibility_matrix" in data:
                    self._compatibility_matrix = CompatibilityMatrix.from_dict(
                        data["compatibility_matrix"]
                    )
                logger.info(f"Loaded {len(self._approved_adapters)} adapters from cache")
        except Exception as e:
            logger.warning(f"Failed to load adapter cache: {e}")
    
    def _save_cache(self):
        """Save adapter metadata to cache."""
        try:
            cache_file = self.cache_dir / "adapter_registry.json"
            data = {
                "approved_adapters": [a.to_dict() for a in self._approved_adapters.values()],
                "compatibility_matrix": self._compatibility_matrix.to_dict(),
                "last_updated": int(time.time()),
            }
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save adapter cache: {e}")
    
    def _index_adapter(self, adapter_info: AdapterInfo):
        """Index adapter for quick lookup."""
        domain = adapter_info.domain
        if domain not in self._domain_index:
            self._domain_index[domain] = []
        if adapter_info.adapter_id not in self._domain_index[domain]:
            self._domain_index[domain].append(adapter_info.adapter_id)
        
        for model_version in adapter_info.compatible_model_versions:
            self._compatibility_matrix.add_compatibility(
                adapter_info.adapter_id, model_version
            )
    
    def sync_from_blockchain(self) -> bool:
        """Sync approved adapters from blockchain."""
        if not self.blockchain_client:
            logger.warning("No blockchain client configured, using cached data")
            return False
        
        try:
            logger.info("Syncing adapters from blockchain...")
            
            response = self.blockchain_client.query_approved_adapters()
            
            if not response.get("success", False):
                logger.error(f"Failed to query adapters: {response.get('error')}")
                return False
            
            adapters = response.get("adapters", [])
            
            for adapter_data in adapters:
                info = AdapterInfo.from_dict(adapter_data)
                if not info.has_valid_checksum():
                    logger.warning(f"Adapter {info.adapter_id} has no valid checksum!")
                self._approved_adapters[info.adapter_id] = info
                self._index_adapter(info)
            
            self._save_cache()
            logger.info(f"Synced {len(adapters)} adapters from blockchain")
            return True
        except Exception as e:
            logger.error(f"Failed to sync from blockchain: {e}")
            return False

    
    def check_compatibility(
        self,
        adapter_id: str,
        model_version: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Check if an adapter is compatible with a model version.
        
        IMPORTANT: Call this BEFORE downloading!
        """
        model_version = model_version or self.current_model_version
        
        if adapter_id not in self._approved_adapters:
            return False, f"Adapter {adapter_id} not found in approved list"
        
        adapter_info = self._approved_adapters[adapter_id]
        
        if adapter_info.is_compatible_with(model_version):
            return True, "Compatible"
        
        return False, (
            f"Adapter {adapter_id} (v{adapter_info.version}) is not compatible "
            f"with model version {model_version}. "
            f"Compatible versions: {adapter_info.compatible_model_versions}"
        )
    
    def check_conflicts(self, adapter_id: str) -> Tuple[bool, List[str]]:
        """
        Check if adapter would conflict with loaded adapters.
        
        Returns (has_conflicts, list_of_conflicting_adapter_ids)
        """
        if adapter_id not in self._approved_adapters:
            return False, []
        
        adapter_info = self._approved_adapters[adapter_id]
        conflicts = self._conflict_detector.check_conflicts(adapter_info)
        
        return len(conflicts) > 0, conflicts
    
    def calculate_adapter_checksum(self, adapter_path: str) -> str:
        """Calculate SHA256 checksum of an adapter file."""
        return calculate_checksum(Path(adapter_path))
    
    def verify_adapter(
        self,
        adapter_id: str,
        local_path: Optional[str] = None,
        strict: bool = True,
    ) -> LocalAdapterStatus:
        """
        Verify a local adapter against blockchain-approved hash.
        
        SECURITY: Checksum verification is MANDATORY in strict mode.
        """
        if adapter_id not in self._approved_adapters:
            self.sync_from_blockchain()
        
        if adapter_id not in self._approved_adapters:
            return LocalAdapterStatus(
                adapter_id=adapter_id,
                status=AdapterStatus.NOT_FOUND,
                error_message="Adapter not found in approved list",
            )
        
        adapter_info = self._approved_adapters[adapter_id]
        
        if local_path is None:
            local_path = str(self.adapters_dir / f"{adapter_id}.pt")
        
        path = Path(local_path)
        
        if not path.exists():
            return LocalAdapterStatus(
                adapter_id=adapter_id,
                status=AdapterStatus.NOT_FOUND,
                expected_checksum=adapter_info.checksum,
                error_message=f"Adapter not found at {local_path}",
            )
        
        # Check compatibility FIRST
        is_compatible, compat_reason = self.check_compatibility(adapter_id)
        if not is_compatible:
            return LocalAdapterStatus(
                adapter_id=adapter_id,
                status=AdapterStatus.INCOMPATIBLE,
                local_path=local_path,
                compatibility_status=compat_reason,
                error_message=compat_reason,
            )
        
        # SECURITY: Require valid checksum in strict mode
        if strict and not adapter_info.has_valid_checksum():
            logger.error(f"Adapter {adapter_id} has no valid checksum - security risk")
            return LocalAdapterStatus(
                adapter_id=adapter_id,
                status=AdapterStatus.INVALID,
                local_path=local_path,
                error_message="Adapter has no valid checksum in blockchain",
            )
        
        # Calculate checksum
        try:
            local_checksum = calculate_checksum(path)
        except Exception as e:
            return LocalAdapterStatus(
                adapter_id=adapter_id,
                status=AdapterStatus.INVALID,
                local_path=local_path,
                expected_checksum=adapter_info.checksum,
                error_message=f"Failed to calculate checksum: {e}",
            )
        
        # MANDATORY checksum verification
        if local_checksum == adapter_info.checksum:
            status = LocalAdapterStatus(
                adapter_id=adapter_id,
                status=AdapterStatus.VALID,
                local_path=local_path,
                local_checksum=local_checksum,
                expected_checksum=adapter_info.checksum,
                size_bytes=path.stat().st_size,
                last_verified=int(time.time()),
                compatibility_status="Compatible",
            )
        else:
            status = LocalAdapterStatus(
                adapter_id=adapter_id,
                status=AdapterStatus.CHECKSUM_MISMATCH,
                local_path=local_path,
                local_checksum=local_checksum,
                expected_checksum=adapter_info.checksum,
                error_message=f"Checksum mismatch: expected {adapter_info.checksum[:16]}..., got {local_checksum[:16]}...",
            )
        
        self._local_status[adapter_id] = status
        return status
    
    def download_adapter(
        self,
        adapter_id: str,
        force: bool = False,
        check_conflicts: bool = True,
        progress_callback=None,
    ) -> Tuple[bool, str]:
        """
        Download an adapter from IPFS.
        
        IMPORTANT: Checks compatibility BEFORE downloading!
        """
        if adapter_id not in self._approved_adapters:
            self.sync_from_blockchain()
        
        if adapter_id not in self._approved_adapters:
            return False, f"Adapter {adapter_id} not found in approved list"
        
        adapter_info = self._approved_adapters[adapter_id]
        local_path = self.adapters_dir / f"{adapter_id}.pt"
        
        # SECURITY: Check compatibility BEFORE download
        is_compatible, reason = self.check_compatibility(adapter_id)
        if not is_compatible:
            logger.warning(f"Adapter {adapter_id} is not compatible: {reason}")
            return False, reason
        
        # Check for conflicts
        if check_conflicts:
            has_conflicts, conflicts = self.check_conflicts(adapter_id)
            if has_conflicts:
                logger.warning(f"Adapter {adapter_id} conflicts with: {conflicts}")
                return False, f"Adapter conflicts with loaded adapters: {conflicts}"
        
        # SECURITY: Require checksum for download
        if not adapter_info.has_valid_checksum():
            return False, "Adapter has no valid checksum - refusing to download"
        
        # Check if already exists and valid
        if not force and local_path.exists():
            status = self.verify_adapter(adapter_id, str(local_path))
            if status.status == AdapterStatus.VALID:
                logger.info(f"Adapter {adapter_id} already exists and is valid")
                return True, str(local_path)
        
        # Update status
        self._local_status[adapter_id] = LocalAdapterStatus(
            adapter_id=adapter_id,
            status=AdapterStatus.DOWNLOADING,
        )
        
        spec = DownloadSpec(
            name=adapter_id,
            version=adapter_info.version,
            ipfs_hash=adapter_info.ipfs_hash,
            expected_checksum=adapter_info.checksum,
            expected_size_bytes=adapter_info.size_bytes,
            extract_archive=False,  # Adapters are single files
            verify_after_download=True,
        )
        
        try:
            def verify_func(path: Path) -> bool:
                status = self.verify_adapter(adapter_id, str(path), strict=True)
                return status.status == AdapterStatus.VALID
            
            with AtomicDownload(local_path, verify_func) as temp_path:
                result = self._download_manager.download(
                    spec=spec,
                    dest_path=temp_path,
                    progress_callback=progress_callback,
                    verify_checksum=True,
                )
                if not result.success:
                    raise Exception(result.error_message)
            
            status = self.verify_adapter(adapter_id, str(local_path))
            if status.status == AdapterStatus.VALID:
                logger.info(f"Adapter {adapter_id} downloaded and verified")
                return True, str(local_path)
            else:
                return False, f"Post-download verification failed: {status.error_message}"
        except Exception as e:
            logger.error(f"Download failed: {e}")
            self._local_status[adapter_id] = LocalAdapterStatus(
                adapter_id=adapter_id,
                status=AdapterStatus.INVALID,
                error_message=str(e),
            )
            return False, str(e)

    
    def ensure_adapter_available(
        self,
        adapter_id: str,
        check_conflicts: bool = True,
    ) -> Tuple[bool, str]:
        """Ensure an adapter is available locally, downloading if necessary."""
        local_path = self.adapters_dir / f"{adapter_id}.pt"
        
        # Check compatibility FIRST
        is_compatible, reason = self.check_compatibility(adapter_id)
        if not is_compatible:
            return False, reason
        
        # Check conflicts
        if check_conflicts:
            has_conflicts, conflicts = self.check_conflicts(adapter_id)
            if has_conflicts:
                return False, f"Adapter conflicts with: {conflicts}"
        
        if local_path.exists():
            status = self.verify_adapter(adapter_id, str(local_path))
            if status.status == AdapterStatus.VALID:
                return True, str(local_path)
        
        if self.auto_download:
            return self.download_adapter(adapter_id, check_conflicts=check_conflicts)
        
        return False, f"Adapter {adapter_id} not available locally"
    
    def load_adapter(self, adapter_id: str) -> Tuple[bool, str]:
        """
        Load an adapter and register it for conflict tracking.
        
        Returns (success, local_path or error_message)
        """
        success, result = self.ensure_adapter_available(adapter_id)
        if not success:
            return False, result
        
        # Register for conflict tracking
        adapter_info = self._approved_adapters[adapter_id]
        conflicts = self._conflict_detector.register_adapter(adapter_info)
        
        if conflicts:
            logger.warning(f"Adapter {adapter_id} loaded with conflicts: {conflicts}")
        
        return True, result
    
    def unload_adapter(self, adapter_id: str):
        """Unload an adapter from conflict tracking."""
        self._conflict_detector.unregister_adapter(adapter_id)
    
    def get_compatible_adapters(
        self,
        model_version: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> List[AdapterInfo]:
        """Get all adapters compatible with a model version."""
        model_version = model_version or self.current_model_version
        
        compatible = []
        for adapter_info in self._approved_adapters.values():
            if adapter_info.is_compatible_with(model_version):
                if domain is None or adapter_info.domain == domain:
                    compatible.append(adapter_info)
        
        return compatible
    
    def get_adapters_by_domain(self, domain: str) -> List[AdapterInfo]:
        """Get all adapters for a specific domain."""
        adapter_ids = self._domain_index.get(domain, [])
        return [
            self._approved_adapters[aid]
            for aid in adapter_ids
            if aid in self._approved_adapters
        ]
    
    def get_approved_adapters(self) -> List[AdapterInfo]:
        """Get list of all approved adapters."""
        return list(self._approved_adapters.values())
    
    def get_adapter_info(self, adapter_id: str) -> Optional[AdapterInfo]:
        """Get info for a specific adapter."""
        return self._approved_adapters.get(adapter_id)
    
    def get_local_status(self, adapter_id: str) -> Optional[LocalAdapterStatus]:
        """Get local status for an adapter."""
        return self._local_status.get(adapter_id)
    
    def get_domains(self) -> List[str]:
        """Get list of all adapter domains."""
        return list(self._domain_index.keys())
    
    def set_model_version(self, model_version: str):
        """Update the current model version for compatibility checks."""
        self.current_model_version = model_version
        logger.info(f"Model version set to: {model_version}")
    
    def get_loaded_adapters(self) -> List[str]:
        """Get list of currently loaded adapter IDs."""
        return self._conflict_detector.get_loaded_adapters()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        local_valid = sum(1 for s in self._local_status.values() if s.status == AdapterStatus.VALID)
        adapters_with_checksum = sum(1 for a in self._approved_adapters.values() if a.has_valid_checksum())
        compatible_count = len(self.get_compatible_adapters())
        
        return {
            "approved_adapters": len(self._approved_adapters),
            "adapters_with_checksum": adapters_with_checksum,
            "local_adapters": len(self._local_status),
            "local_valid": local_valid,
            "compatible_with_current_model": compatible_count,
            "loaded_adapters": len(self.get_loaded_adapters()),
            "current_model_version": self.current_model_version,
            "domains": list(self._domain_index.keys()),
            "adapters_dir": str(self.adapters_dir),
            "auto_download": self.auto_download,
            "verify_on_load": self.verify_on_load,
        }

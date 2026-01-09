#!/usr/bin/env python3
"""
R3MES Dataset Registry

Production-ready dataset management that:
1. Tracks approved datasets from blockchain
2. Verifies local dataset integrity against blockchain hash (MANDATORY)
3. Downloads datasets from IPFS with HuggingFace fallback
4. Ensures all miners use the same dataset version
5. Handles dataset versioning and updates

Security:
    - Checksum verification is MANDATORY
    - Multiple download sources with fallback
    - Atomic downloads with rollback
    - Retry with exponential backoff
"""

import os
import json
import logging
import time
from typing import Optional, Dict, Any, List, Tuple, Iterator, Generator
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
    DownloadProgress,
)

logger = logging.getLogger(__name__)


class DatasetStatus(Enum):
    """Dataset status enum."""
    NOT_FOUND = "not_found"
    DOWNLOADING = "downloading"
    VERIFYING = "verifying"
    VALID = "valid"
    INVALID = "invalid"
    OUTDATED = "outdated"
    CHECKSUM_MISMATCH = "checksum_mismatch"


@dataclass
class DatasetInfo:
    """Information about a dataset."""
    dataset_id: str
    ipfs_hash: str
    name: str
    version: str
    size_bytes: int
    checksum: str  # REQUIRED
    category: str
    description: str
    approved_at: int
    approval_tx_hash: str
    # HuggingFace fallback (NEW)
    huggingface_repo: Optional[str] = None
    huggingface_subset: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "ipfs_hash": self.ipfs_hash,
            "name": self.name,
            "version": self.version,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "category": self.category,
            "description": self.description,
            "approved_at": self.approved_at,
            "approval_tx_hash": self.approval_tx_hash,
            "huggingface_repo": self.huggingface_repo,
            "huggingface_subset": self.huggingface_subset,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetInfo":
        return cls(
            dataset_id=data.get("dataset_id", ""),
            ipfs_hash=data.get("ipfs_hash", data.get("dataset_ipfs_hash", "")),
            name=data.get("name", data.get("dataset_name", "")),
            version=data.get("version", "1.0.0"),
            size_bytes=data.get("size_bytes", 0),
            checksum=data.get("checksum", ""),
            category=data.get("category", ""),
            description=data.get("description", ""),
            approved_at=data.get("approved_at", 0),
            approval_tx_hash=data.get("approval_tx_hash", ""),
            huggingface_repo=data.get("huggingface_repo"),
            huggingface_subset=data.get("huggingface_subset"),
            metadata=data.get("metadata", {}),
        )
    
    def has_valid_checksum(self) -> bool:
        """Check if dataset has a valid checksum defined."""
        return bool(self.checksum and len(self.checksum) >= 32)


@dataclass
class LocalDatasetStatus:
    """Status of a local dataset."""
    dataset_id: str
    status: DatasetStatus
    local_path: Optional[str] = None
    local_checksum: Optional[str] = None
    expected_checksum: Optional[str] = None
    size_bytes: int = 0
    last_verified: Optional[int] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "status": self.status.value,
            "local_path": self.local_path,
            "local_checksum": self.local_checksum,
            "expected_checksum": self.expected_checksum,
            "size_bytes": self.size_bytes,
            "last_verified": self.last_verified,
            "error_message": self.error_message,
        }


class DatasetRegistry:
    """
    Dataset registry for managing training datasets.
    
    Security Features:
    - MANDATORY checksum verification
    - HuggingFace fallback for reliability
    - Streaming support for large datasets
    - Atomic downloads with rollback
    """
    
    def __init__(
        self,
        blockchain_client=None,
        ipfs_client=None,
        datasets_dir: Optional[str] = None,
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
        self.datasets_dir = Path(datasets_dir or os.path.join(home, ".r3mes", "datasets"))
        self.cache_dir = Path(cache_dir or os.path.join(home, ".r3mes", "cache", "datasets"))
        
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Registry data
        self._approved_datasets: Dict[str, DatasetInfo] = {}
        self._local_status: Dict[str, LocalDatasetStatus] = {}
        self._active_dataset_id: Optional[str] = None
        
        # Download manager with fallback
        self._download_manager = DownloadManager(
            ipfs_client=ipfs_client,
            retry_config=self.retry_config,
            prefer_huggingface=False,  # Prefer IPFS for datasets
        )
        
        self._load_cache()
        logger.info(f"DatasetRegistry initialized (datasets_dir: {self.datasets_dir})")
    
    def _load_cache(self):
        """Load cached dataset metadata."""
        try:
            cache_file = self.cache_dir / "dataset_registry.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                for dataset_data in data.get("approved_datasets", []):
                    info = DatasetInfo.from_dict(dataset_data)
                    self._approved_datasets[info.dataset_id] = info
                self._active_dataset_id = data.get("active_dataset_id")
                logger.info(f"Loaded {len(self._approved_datasets)} datasets from cache")
        except Exception as e:
            logger.warning(f"Failed to load dataset cache: {e}")
    
    def _save_cache(self):
        """Save dataset metadata to cache."""
        try:
            cache_file = self.cache_dir / "dataset_registry.json"
            data = {
                "approved_datasets": [d.to_dict() for d in self._approved_datasets.values()],
                "active_dataset_id": self._active_dataset_id,
                "last_updated": int(time.time()),
            }
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save dataset cache: {e}")
    
    def sync_from_blockchain(self) -> bool:
        """Sync approved datasets from blockchain."""
        if not self.blockchain_client:
            logger.warning("No blockchain client configured, using cached data")
            return False
        
        try:
            logger.info("Syncing datasets from blockchain...")
            
            response = self.blockchain_client.query_approved_datasets()
            
            if not response.get("success", False):
                logger.error(f"Failed to query datasets: {response.get('error')}")
                return False
            
            datasets = response.get("datasets", [])
            
            for dataset_data in datasets:
                info = DatasetInfo.from_dict(dataset_data)
                if not info.has_valid_checksum():
                    logger.warning(f"Dataset {info.dataset_id} has no valid checksum!")
                self._approved_datasets[info.dataset_id] = info
            
            active_response = self.blockchain_client.query_active_dataset()
            if active_response.get("success"):
                self._active_dataset_id = active_response.get("dataset_id")
            
            self._save_cache()
            logger.info(f"Synced {len(datasets)} datasets from blockchain")
            return True
        except Exception as e:
            logger.error(f"Failed to sync from blockchain: {e}")
            return False
    
    def get_active_dataset_hash(self) -> Optional[str]:
        """Get the IPFS hash of the currently active dataset."""
        if self.blockchain_client:
            try:
                response = self.blockchain_client.query_active_dataset()
                if response.get("success"):
                    return response.get("ipfs_hash")
            except Exception as e:
                logger.warning(f"Failed to query active dataset: {e}")
        
        if self._active_dataset_id and self._active_dataset_id in self._approved_datasets:
            return self._approved_datasets[self._active_dataset_id].ipfs_hash
        return None
    
    def get_active_dataset_id(self) -> Optional[str]:
        """Get the active dataset ID."""
        return self._active_dataset_id

    
    def calculate_dataset_checksum(self, dataset_path: str) -> str:
        """Calculate SHA256 checksum of a dataset."""
        return calculate_checksum(Path(dataset_path))
    
    def verify_local_dataset(
        self,
        dataset_id: str,
        local_path: Optional[str] = None,
        strict: bool = True,
    ) -> LocalDatasetStatus:
        """
        Verify a local dataset against blockchain-approved hash.
        
        SECURITY: Checksum verification is MANDATORY in strict mode.
        """
        if dataset_id not in self._approved_datasets:
            self.sync_from_blockchain()
        
        if dataset_id not in self._approved_datasets:
            return LocalDatasetStatus(
                dataset_id=dataset_id,
                status=DatasetStatus.NOT_FOUND,
                error_message="Dataset not found in approved list",
            )
        
        dataset_info = self._approved_datasets[dataset_id]
        
        if local_path is None:
            local_path = str(self.datasets_dir / dataset_id)
        
        path = Path(local_path)
        
        if not path.exists():
            return LocalDatasetStatus(
                dataset_id=dataset_id,
                status=DatasetStatus.NOT_FOUND,
                expected_checksum=dataset_info.checksum,
                error_message=f"Dataset not found at {local_path}",
            )
        
        # SECURITY: Require valid checksum in strict mode
        if strict and not dataset_info.has_valid_checksum():
            logger.error(f"Dataset {dataset_id} has no valid checksum - security risk")
            return LocalDatasetStatus(
                dataset_id=dataset_id,
                status=DatasetStatus.INVALID,
                local_path=local_path,
                error_message="Dataset has no valid checksum in blockchain",
            )
        
        # Calculate checksum
        try:
            local_checksum = calculate_checksum(path)
        except Exception as e:
            return LocalDatasetStatus(
                dataset_id=dataset_id,
                status=DatasetStatus.INVALID,
                local_path=local_path,
                expected_checksum=dataset_info.checksum,
                error_message=f"Failed to calculate checksum: {e}",
            )
        
        # Get size
        if path.is_file():
            size_bytes = path.stat().st_size
        else:
            size_bytes = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        
        # MANDATORY checksum verification
        if local_checksum == dataset_info.checksum:
            status = LocalDatasetStatus(
                dataset_id=dataset_id,
                status=DatasetStatus.VALID,
                local_path=local_path,
                local_checksum=local_checksum,
                expected_checksum=dataset_info.checksum,
                size_bytes=size_bytes,
                last_verified=int(time.time()),
            )
        else:
            status = LocalDatasetStatus(
                dataset_id=dataset_id,
                status=DatasetStatus.CHECKSUM_MISMATCH,
                local_path=local_path,
                local_checksum=local_checksum,
                expected_checksum=dataset_info.checksum,
                size_bytes=size_bytes,
                error_message=f"Checksum mismatch: expected {dataset_info.checksum[:16]}..., got {local_checksum[:16]}...",
            )
        
        self._local_status[dataset_id] = status
        return status
    
    def download_dataset(
        self,
        dataset_id: str,
        force: bool = False,
        progress_callback=None,
    ) -> Tuple[bool, str]:
        """
        Download a dataset with automatic fallback (IPFS → HuggingFace).
        """
        if dataset_id not in self._approved_datasets:
            self.sync_from_blockchain()
        
        if dataset_id not in self._approved_datasets:
            return False, f"Dataset {dataset_id} not found in approved list"
        
        dataset_info = self._approved_datasets[dataset_id]
        local_path = self.datasets_dir / dataset_id
        
        # SECURITY: Require checksum for download
        if not dataset_info.has_valid_checksum():
            return False, "Dataset has no valid checksum - refusing to download"
        
        # Check if already exists and valid
        if not force and local_path.exists():
            status = self.verify_local_dataset(dataset_id, str(local_path))
            if status.status == DatasetStatus.VALID:
                logger.info(f"Dataset {dataset_id} already exists and is valid")
                return True, str(local_path)
        
        # Update status
        self._local_status[dataset_id] = LocalDatasetStatus(
            dataset_id=dataset_id,
            status=DatasetStatus.DOWNLOADING,
        )
        
        # Create download spec with fallback sources
        spec = DownloadSpec(
            name=dataset_id,
            version=dataset_info.version,
            ipfs_hash=dataset_info.ipfs_hash,
            huggingface_repo=dataset_info.huggingface_repo,
            expected_checksum=dataset_info.checksum,
            expected_size_bytes=dataset_info.size_bytes,
            verify_after_download=True,
        )
        
        try:
            def verify_func(path: Path) -> bool:
                status = self.verify_local_dataset(dataset_id, str(path), strict=True)
                return status.status == DatasetStatus.VALID
            
            with AtomicDownload(local_path, verify_func) as temp_path:
                result = self._download_manager.download(
                    spec=spec,
                    dest_path=temp_path,
                    progress_callback=progress_callback,
                    verify_checksum=True,
                )
                if not result.success:
                    raise Exception(result.error_message)
            
            status = self.verify_local_dataset(dataset_id, str(local_path))
            if status.status == DatasetStatus.VALID:
                logger.info(f"Dataset {dataset_id} downloaded and verified")
                return True, str(local_path)
            else:
                return False, f"Post-download verification failed: {status.error_message}"
        except Exception as e:
            logger.error(f"Download failed: {e}")
            self._local_status[dataset_id] = LocalDatasetStatus(
                dataset_id=dataset_id,
                status=DatasetStatus.INVALID,
                error_message=str(e),
            )
            return False, str(e)
    
    def ensure_dataset_available(
        self,
        dataset_id: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Ensure a dataset is available locally, downloading if necessary."""
        if dataset_id is None:
            if self._active_dataset_id:
                dataset_id = self._active_dataset_id
            else:
                self.sync_from_blockchain()
                if self._active_dataset_id:
                    dataset_id = self._active_dataset_id
                else:
                    return False, "No active dataset configured"
        
        local_path = self.datasets_dir / dataset_id
        
        if local_path.exists():
            status = self.verify_local_dataset(dataset_id, str(local_path))
            if status.status == DatasetStatus.VALID:
                return True, str(local_path)
        
        if self.auto_download:
            return self.download_dataset(dataset_id)
        
        return False, f"Dataset {dataset_id} not available locally"

    
    def load_dataset(
        self,
        dataset_id: Optional[str] = None,
        verify: Optional[bool] = None,
        streaming: bool = False,
    ) -> Tuple[Optional[Any], Optional[str]]:
        """
        Load a dataset for training.
        
        Args:
            dataset_id: Dataset ID (uses active if not provided)
            verify: Verify before loading (default: self.verify_on_load)
            streaming: Use streaming mode for large datasets (NEW)
        """
        verify = verify if verify is not None else self.verify_on_load
        
        success, result = self.ensure_dataset_available(dataset_id)
        if not success:
            return None, result
        
        local_path = Path(result)
        
        if verify:
            status = self.verify_local_dataset(
                dataset_id or self._active_dataset_id,
                str(local_path),
            )
            if status.status != DatasetStatus.VALID:
                return None, f"Dataset verification failed: {status.error_message}"
        
        try:
            if streaming:
                return self._load_streaming(local_path), None
            else:
                return self._load_full(local_path), None
        except Exception as e:
            return None, f"Failed to load dataset: {e}"
    
    def _load_full(self, local_path: Path) -> List[Dict[str, Any]]:
        """Load entire dataset into memory."""
        data = []
        
        # Try JSONL files
        jsonl_files = list(local_path.glob("*.jsonl"))
        if jsonl_files:
            for jsonl_file in jsonl_files:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
            return data
        
        # Try JSON files
        json_files = list(local_path.glob("*.json"))
        if json_files:
            with open(json_files[0], 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    return loaded
                return [loaded]
        
        # Try data.jsonl in subdirectory
        data_file = local_path / "data.jsonl"
        if data_file.exists():
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            return data
        
        raise FileNotFoundError(f"No supported dataset format found in {local_path}")
    
    def _load_streaming(self, local_path: Path) -> Generator[Dict[str, Any], None, None]:
        """
        Load dataset in streaming mode (memory efficient for large datasets).
        
        Yields one record at a time instead of loading all into memory.
        """
        jsonl_files = list(local_path.glob("*.jsonl"))
        if not jsonl_files:
            data_file = local_path / "data.jsonl"
            if data_file.exists():
                jsonl_files = [data_file]
        
        if jsonl_files:
            for jsonl_file in jsonl_files:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            yield json.loads(line)
            return
        
        # Fallback to JSON (loads all, then yields)
        json_files = list(local_path.glob("*.json"))
        if json_files:
            with open(json_files[0], 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    for item in loaded:
                        yield item
                else:
                    yield loaded
            return
        
        raise FileNotFoundError(f"No supported dataset format found in {local_path}")
    
    def get_approved_datasets(self) -> List[DatasetInfo]:
        """Get list of all approved datasets."""
        return list(self._approved_datasets.values())
    
    def get_dataset_info(self, dataset_id: str) -> Optional[DatasetInfo]:
        """Get info for a specific dataset."""
        return self._approved_datasets.get(dataset_id)
    
    def get_local_status(self, dataset_id: str) -> Optional[LocalDatasetStatus]:
        """Get local status for a dataset."""
        return self._local_status.get(dataset_id)
    
    def verify_active_dataset(self, local_path: str) -> bool:
        """Verify that local dataset matches blockchain-approved active dataset."""
        if not self._active_dataset_id:
            self.sync_from_blockchain()
        
        if not self._active_dataset_id:
            logger.warning("No active dataset configured")
            return False
        
        status = self.verify_local_dataset(self._active_dataset_id, local_path)
        return status.status == DatasetStatus.VALID
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        local_valid = sum(1 for s in self._local_status.values() if s.status == DatasetStatus.VALID)
        datasets_with_checksum = sum(1 for d in self._approved_datasets.values() if d.has_valid_checksum())
        
        return {
            "approved_datasets": len(self._approved_datasets),
            "datasets_with_checksum": datasets_with_checksum,
            "local_datasets": len(self._local_status),
            "local_valid": local_valid,
            "active_dataset_id": self._active_dataset_id,
            "datasets_dir": str(self.datasets_dir),
            "auto_download": self.auto_download,
            "verify_on_load": self.verify_on_load,
        }

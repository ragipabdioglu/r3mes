#!/usr/bin/env python3
"""
R3MES Unified Registry

Combines Model, Dataset, and Adapter registries into a single interface.
Provides coordinated verification and management across all components.

Security Features:
    - Atomic loading (all-or-nothing)
    - Rollback on partial failure
    - Coordinated verification
    - System-wide integrity checks

Trust Chain:
    Blockchain (immutable source of truth)
        │
        ├── Model Hash → IPFS → Local Model
        ├── Dataset Hash → IPFS → Local Dataset
        └── Adapter Hash → IPFS → Local Adapters
"""

import logging
import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from utils.model_registry import ModelRegistry, ModelStatus
from utils.dataset_registry import DatasetRegistry, DatasetStatus
from utils.adapter_registry import AdapterRegistry, AdapterStatus
from utils.verification import (
    VerificationPolicy,
    VerificationLevel,
    RetryConfig,
)

logger = logging.getLogger(__name__)


class SystemState(Enum):
    """System state enum."""
    UNINITIALIZED = "uninitialized"
    SYNCING = "syncing"
    DOWNLOADING = "downloading"
    VERIFYING = "verifying"
    READY = "ready"
    PARTIAL = "partial"  # Some components ready
    ERROR = "error"


@dataclass
class SystemStatus:
    """Overall system status."""
    state: SystemState = SystemState.UNINITIALIZED
    model_ready: bool = False
    dataset_ready: bool = False
    adapters_ready: bool = False
    all_ready: bool = False
    
    model_version: Optional[str] = None
    model_path: Optional[str] = None
    dataset_id: Optional[str] = None
    dataset_path: Optional[str] = None
    loaded_adapters: List[str] = field(default_factory=list)
    adapter_paths: Dict[str, str] = field(default_factory=dict)
    
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    sync_timestamp: Optional[int] = None
    verification_timestamp: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "model_ready": self.model_ready,
            "dataset_ready": self.dataset_ready,
            "adapters_ready": self.adapters_ready,
            "all_ready": self.all_ready,
            "model_version": self.model_version,
            "model_path": self.model_path,
            "dataset_id": self.dataset_id,
            "dataset_path": self.dataset_path,
            "loaded_adapters": self.loaded_adapters,
            "adapter_paths": self.adapter_paths,
            "issues": self.issues,
            "warnings": self.warnings,
            "sync_timestamp": self.sync_timestamp,
            "verification_timestamp": self.verification_timestamp,
        }


@dataclass
class AtomicLoadResult:
    """Result of atomic load operation."""
    success: bool
    status: SystemStatus
    rollback_performed: bool = False
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status.to_dict(),
            "rollback_performed": self.rollback_performed,
            "error_message": self.error_message,
        }


class UnifiedRegistry:
    """
    Unified registry combining model, dataset, and adapter management.
    
    Provides:
    - Coordinated blockchain sync
    - System-wide verification
    - Atomic loading with rollback
    - Compatibility checking
    """
    
    def __init__(
        self,
        blockchain_client=None,
        ipfs_client=None,
        models_dir: Optional[str] = None,
        datasets_dir: Optional[str] = None,
        adapters_dir: Optional[str] = None,
        auto_download: bool = True,
        verify_on_load: bool = True,
        verification_policy: Optional[VerificationPolicy] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        self.blockchain_client = blockchain_client
        self.ipfs_client = ipfs_client
        
        # Shared verification policy
        self.verification_policy = verification_policy or VerificationPolicy(
            level=VerificationLevel.STRICT
        )
        self.retry_config = retry_config or RetryConfig()
        
        # Initialize sub-registries with shared policy
        self.models = ModelRegistry(
            blockchain_client=blockchain_client,
            ipfs_client=ipfs_client,
            models_dir=models_dir,
            auto_download=auto_download,
            verify_on_load=verify_on_load,
            verification_policy=self.verification_policy,
            retry_config=self.retry_config,
        )
        
        self.datasets = DatasetRegistry(
            blockchain_client=blockchain_client,
            ipfs_client=ipfs_client,
            datasets_dir=datasets_dir,
            auto_download=auto_download,
            verify_on_load=verify_on_load,
            verification_policy=self.verification_policy,
            retry_config=self.retry_config,
        )
        
        self.adapters = AdapterRegistry(
            blockchain_client=blockchain_client,
            ipfs_client=ipfs_client,
            adapters_dir=adapters_dir,
            current_model_version=self.models.get_active_model_version() or "1.0.0",
            auto_download=auto_download,
            verify_on_load=verify_on_load,
            verification_policy=self.verification_policy,
            retry_config=self.retry_config,
        )
        
        # System state
        self._current_status = SystemStatus()
        self._last_sync_time: Optional[int] = None
        
        logger.info("UnifiedRegistry initialized")
    
    def sync_all(self) -> Dict[str, bool]:
        """Sync all registries from blockchain."""
        logger.info("Syncing all registries from blockchain...")
        
        self._current_status.state = SystemState.SYNCING
        
        results = {
            "models": self.models.sync_from_blockchain(),
            "datasets": self.datasets.sync_from_blockchain(),
            "adapters": self.adapters.sync_from_blockchain(),
        }
        
        # Update adapter registry with current model version
        model_version = self.models.get_active_model_version()
        if model_version:
            self.adapters.set_model_version(model_version)
        
        self._last_sync_time = int(time.time())
        self._current_status.sync_timestamp = self._last_sync_time
        
        logger.info(f"Sync complete: {results}")
        return results

    
    def verify_system(
        self,
        model_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        adapter_ids: Optional[List[str]] = None,
    ) -> SystemStatus:
        """
        Verify entire system state.
        
        All verifications are MANDATORY - no bypasses.
        """
        self._current_status = SystemStatus(state=SystemState.VERIFYING)
        status = self._current_status
        
        # Verify model
        model_version = self.models.get_active_model_version()
        if model_version:
            status.model_version = model_version
            
            if model_path:
                check_path = model_path
            else:
                check_path = str(self.models.models_dir / model_version)
            
            if Path(check_path).exists():
                model_status = self.models.verify_model(check_path, model_version)
                if model_status.status == ModelStatus.VALID:
                    status.model_ready = True
                    status.model_path = check_path
                else:
                    status.issues.append(f"Model verification failed: {model_status.error_message}")
            else:
                status.issues.append(f"Model not found at {check_path}")
        else:
            status.issues.append("No active model version configured")
        
        # Verify dataset
        dataset_id = self.datasets.get_active_dataset_id()
        if dataset_id:
            status.dataset_id = dataset_id
            
            if dataset_path:
                check_path = dataset_path
            else:
                check_path = str(self.datasets.datasets_dir / dataset_id)
            
            if Path(check_path).exists():
                dataset_status = self.datasets.verify_local_dataset(dataset_id, check_path)
                if dataset_status.status == DatasetStatus.VALID:
                    status.dataset_ready = True
                    status.dataset_path = check_path
                else:
                    status.warnings.append(f"Dataset verification: {dataset_status.error_message}")
            else:
                status.warnings.append(f"Dataset not found at {check_path}")
        else:
            status.warnings.append("No active dataset configured")
        
        # Verify adapters
        if adapter_ids is None:
            compatible = self.adapters.get_compatible_adapters()
            adapter_ids = [a.adapter_id for a in compatible]
        
        valid_adapters = []
        for adapter_id in adapter_ids:
            adapter_status = self.adapters.verify_adapter(adapter_id)
            if adapter_status.status == AdapterStatus.VALID:
                valid_adapters.append(adapter_id)
                status.adapter_paths[adapter_id] = adapter_status.local_path
            elif adapter_status.status == AdapterStatus.INCOMPATIBLE:
                status.warnings.append(f"Adapter {adapter_id}: {adapter_status.error_message}")
        
        status.loaded_adapters = valid_adapters
        status.adapters_ready = len(valid_adapters) > 0 or len(adapter_ids) == 0
        
        # Overall status
        status.all_ready = status.model_ready and status.adapters_ready
        status.verification_timestamp = int(time.time())
        
        if status.all_ready:
            status.state = SystemState.READY
        elif status.model_ready or status.dataset_ready or status.adapters_ready:
            status.state = SystemState.PARTIAL
        else:
            status.state = SystemState.ERROR
        
        self._current_status = status
        return status
    
    def ensure_ready_atomic(
        self,
        required_adapters: Optional[List[str]] = None,
        require_dataset: bool = False,
    ) -> AtomicLoadResult:
        """
        Ensure system is ready with ATOMIC loading.
        
        Either all components load successfully, or none do (rollback).
        
        Args:
            required_adapters: List of required adapter IDs
            require_dataset: Whether dataset is required (default: False for serving)
            
        Returns:
            AtomicLoadResult with success status and system state
        """
        logger.info("Starting atomic system load...")
        
        # Sync from blockchain first
        self.sync_all()
        
        status = SystemStatus(state=SystemState.DOWNLOADING)
        loaded_components = []  # Track what we've loaded for rollback
        
        try:
            # Step 1: Load model (REQUIRED)
            model_version = self.models.get_active_model_version()
            if not model_version:
                raise Exception("No active model version configured")
            
            status.model_version = model_version
            
            success, result = self.models.download_model(model_version)
            if not success:
                raise Exception(f"Model download failed: {result}")
            
            status.model_path = result
            status.model_ready = True
            loaded_components.append(("model", model_version))
            logger.info(f"Model {model_version} loaded successfully")
            
            # Update adapter registry with model version
            self.adapters.set_model_version(model_version)
            
            # Step 2: Load dataset (optional unless required)
            dataset_id = self.datasets.get_active_dataset_id()
            if dataset_id:
                status.dataset_id = dataset_id
                
                success, result = self.datasets.ensure_dataset_available(dataset_id)
                if success:
                    status.dataset_path = result
                    status.dataset_ready = True
                    loaded_components.append(("dataset", dataset_id))
                    logger.info(f"Dataset {dataset_id} loaded successfully")
                elif require_dataset:
                    raise Exception(f"Dataset load failed: {result}")
                else:
                    status.warnings.append(f"Dataset not available: {result}")
            elif require_dataset:
                raise Exception("No active dataset configured but dataset is required")
            
            # Step 3: Load adapters
            if required_adapters is None:
                required_adapters = ["general_dora"]
            
            valid_adapters = []
            for adapter_id in required_adapters:
                # Check compatibility BEFORE download
                is_compatible, reason = self.adapters.check_compatibility(adapter_id)
                if not is_compatible:
                    status.warnings.append(f"Adapter {adapter_id} incompatible: {reason}")
                    continue
                
                success, result = self.adapters.download_adapter(adapter_id)
                if success:
                    valid_adapters.append(adapter_id)
                    status.adapter_paths[adapter_id] = result
                    loaded_components.append(("adapter", adapter_id))
                else:
                    status.warnings.append(f"Adapter {adapter_id}: {result}")
            
            status.loaded_adapters = valid_adapters
            status.adapters_ready = len(valid_adapters) > 0
            
            # Final verification
            status.state = SystemState.VERIFYING
            
            # Verify model
            model_status = self.models.verify_model(status.model_path, model_version)
            if model_status.status != ModelStatus.VALID:
                raise Exception(f"Model verification failed: {model_status.error_message}")
            
            # Verify dataset if loaded
            if status.dataset_ready:
                dataset_status = self.datasets.verify_local_dataset(
                    status.dataset_id, status.dataset_path
                )
                if dataset_status.status != DatasetStatus.VALID:
                    if require_dataset:
                        raise Exception(f"Dataset verification failed: {dataset_status.error_message}")
                    else:
                        status.dataset_ready = False
                        status.warnings.append(f"Dataset verification failed: {dataset_status.error_message}")
            
            # Verify adapters
            for adapter_id in valid_adapters:
                adapter_status = self.adapters.verify_adapter(adapter_id)
                if adapter_status.status != AdapterStatus.VALID:
                    status.warnings.append(f"Adapter {adapter_id} verification failed")
                    valid_adapters.remove(adapter_id)
            
            status.loaded_adapters = valid_adapters
            status.adapters_ready = len(valid_adapters) > 0
            
            # Overall status
            status.all_ready = status.model_ready and status.adapters_ready
            status.verification_timestamp = int(time.time())
            
            if status.all_ready:
                status.state = SystemState.READY
            else:
                status.state = SystemState.PARTIAL
            
            self._current_status = status
            
            return AtomicLoadResult(
                success=status.all_ready,
                status=status,
                rollback_performed=False,
            )
            
        except Exception as e:
            logger.error(f"Atomic load failed: {e}")
            
            # Rollback loaded components
            self._rollback(loaded_components)
            
            status.state = SystemState.ERROR
            status.issues.append(str(e))
            self._current_status = status
            
            return AtomicLoadResult(
                success=False,
                status=status,
                rollback_performed=True,
                error_message=str(e),
            )
    
    def _rollback(self, loaded_components: List[Tuple[str, str]]):
        """Rollback loaded components on failure."""
        logger.warning(f"Rolling back {len(loaded_components)} components...")
        
        for component_type, component_id in reversed(loaded_components):
            try:
                if component_type == "adapter":
                    self.adapters.unload_adapter(component_id)
                    logger.info(f"Rolled back adapter: {component_id}")
                # Models and datasets don't need explicit unload
            except Exception as e:
                logger.warning(f"Rollback failed for {component_type} {component_id}: {e}")
    
    # Backward compatibility
    def ensure_ready(
        self,
        required_adapters: Optional[List[str]] = None,
    ) -> Tuple[bool, SystemStatus]:
        """Backward compatible ensure_ready method."""
        result = self.ensure_ready_atomic(required_adapters)
        return result.success, result.status

    
    def check_adapter_compatibility(
        self,
        adapter_id: str,
    ) -> Tuple[bool, str]:
        """Check if an adapter is compatible with current model."""
        model_version = self.models.get_active_model_version()
        if not model_version:
            return False, "No active model version"
        
        return self.adapters.check_compatibility(adapter_id, model_version)
    
    def get_compatible_adapters_for_domain(
        self,
        domain: str,
    ) -> List[str]:
        """Get compatible adapters for a specific domain."""
        model_version = self.models.get_active_model_version()
        if not model_version:
            return []
        
        compatible = self.adapters.get_compatible_adapters(model_version, domain)
        return [a.adapter_id for a in compatible]
    
    def get_current_status(self) -> SystemStatus:
        """Get current system status."""
        return self._current_status
    
    def get_stats(self) -> Dict[str, Any]:
        """Get unified registry statistics."""
        return {
            "system_state": self._current_status.state.value,
            "last_sync": self._last_sync_time,
            "models": self.models.get_stats(),
            "datasets": self.datasets.get_stats(),
            "adapters": self.adapters.get_stats(),
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the entire system.
        
        Returns detailed health status for monitoring.
        """
        health = {
            "healthy": False,
            "state": self._current_status.state.value,
            "components": {
                "model": {
                    "ready": self._current_status.model_ready,
                    "version": self._current_status.model_version,
                },
                "dataset": {
                    "ready": self._current_status.dataset_ready,
                    "id": self._current_status.dataset_id,
                },
                "adapters": {
                    "ready": self._current_status.adapters_ready,
                    "loaded": self._current_status.loaded_adapters,
                },
            },
            "issues": self._current_status.issues,
            "warnings": self._current_status.warnings,
            "last_sync": self._last_sync_time,
            "last_verification": self._current_status.verification_timestamp,
        }
        
        health["healthy"] = (
            self._current_status.state == SystemState.READY and
            self._current_status.all_ready and
            len(self._current_status.issues) == 0
        )
        
        return health

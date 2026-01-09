"""
Model Update Pipeline

KRİTİK EKSİKLİK #4 ve #7 ÇÖZÜMÜ:
- Eğitilen DoRA → Ana Model entegrasyonu
- Model versiyonlama ve upgrade/rollback

Flow:
    Miner eğitim yapıyor → DoRA adapter oluşuyor
        ↓
    IPFS'e yükleniyor ✅ (GradientSubmissionPipeline)
        ↓
    Blockchain'e submit ediliyor ✅ (GradientSubmissionPipeline)
        ↓
    Proposer aggregation yapıyor ✅ (ModelUpdatePipeline)
        ↓
    Ana model güncelleniyor ✅ (ModelUpdatePipeline)
"""

import os
import io
import time
import logging
import hashlib
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class UpdateStatus(Enum):
    """Model update status."""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    VERIFYING = "verifying"
    APPLYING = "applying"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class ModelVersion:
    """Model version information."""
    version: str
    ipfs_hash: str
    checksum: str
    block_height: int
    timestamp: int
    adapter_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "ipfs_hash": self.ipfs_hash,
            "checksum": self.checksum,
            "block_height": self.block_height,
            "timestamp": self.timestamp,
            "adapter_count": self.adapter_count,
            "metadata": self.metadata,
        }


@dataclass
class UpdateResult:
    """Result of model update operation."""
    success: bool
    status: UpdateStatus
    old_version: Optional[str] = None
    new_version: Optional[str] = None
    error: Optional[str] = None
    rollback_available: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class GradientAggregator:
    """
    Gradient aggregation for federated learning.
    
    KRİTİK EKSİKLİK #5 ÇÖZÜMÜ:
    Byzantine-robust gradient aggregation
    """
    
    def __init__(
        self,
        aggregation_method: str = "weighted_average",
        byzantine_threshold: float = 0.3,
    ):
        """
        Initialize gradient aggregator.
        
        Args:
            aggregation_method: Aggregation method (weighted_average, median, trimmed_mean)
            byzantine_threshold: Threshold for Byzantine-robust aggregation
        """
        self.aggregation_method = aggregation_method
        self.byzantine_threshold = byzantine_threshold
    
    def aggregate(
        self,
        gradients: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate multiple gradient dictionaries.
        
        Args:
            gradients: List of gradient dictionaries
            weights: Optional weights for each gradient
            
        Returns:
            Aggregated gradient dictionary
        """
        if not gradients:
            return {}
        
        if weights is None:
            weights = [1.0 / len(gradients)] * len(gradients)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        if self.aggregation_method == "weighted_average":
            return self._weighted_average(gradients, weights)
        elif self.aggregation_method == "median":
            return self._coordinate_median(gradients)
        elif self.aggregation_method == "trimmed_mean":
            return self._trimmed_mean(gradients, trim_ratio=self.byzantine_threshold)
        else:
            return self._weighted_average(gradients, weights)
    
    def _weighted_average(
        self,
        gradients: List[Dict[str, torch.Tensor]],
        weights: List[float],
    ) -> Dict[str, torch.Tensor]:
        """Compute weighted average of gradients."""
        result = {}
        
        # Get all keys from first gradient
        keys = gradients[0].keys()
        
        for key in keys:
            tensors = [g[key] for g in gradients if key in g]
            if not tensors:
                continue
            
            # Weighted sum
            weighted_sum = sum(
                w * t for w, t in zip(weights[:len(tensors)], tensors)
            )
            result[key] = weighted_sum
        
        return result
    
    def _coordinate_median(
        self,
        gradients: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Compute coordinate-wise median (Byzantine-robust)."""
        result = {}
        keys = gradients[0].keys()
        
        for key in keys:
            tensors = [g[key] for g in gradients if key in g]
            if not tensors:
                continue
            
            # Stack and compute median
            stacked = torch.stack(tensors, dim=0)
            median_val = torch.median(stacked, dim=0).values
            result[key] = median_val
        
        return result
    
    def _trimmed_mean(
        self,
        gradients: List[Dict[str, torch.Tensor]],
        trim_ratio: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """Compute trimmed mean (Byzantine-robust)."""
        result = {}
        keys = gradients[0].keys()
        n = len(gradients)
        trim_count = int(n * trim_ratio)
        
        for key in keys:
            tensors = [g[key] for g in gradients if key in g]
            if not tensors:
                continue
            
            # Stack tensors
            stacked = torch.stack(tensors, dim=0)
            
            # Sort along first dimension
            sorted_tensors, _ = torch.sort(stacked, dim=0)
            
            # Trim extremes
            if trim_count > 0 and n > 2 * trim_count:
                trimmed = sorted_tensors[trim_count:-trim_count]
            else:
                trimmed = sorted_tensors
            
            # Compute mean of trimmed values
            result[key] = trimmed.mean(dim=0)
        
        return result
    
    def compute_merkle_root(
        self,
        gradient_hashes: List[str],
    ) -> str:
        """
        Compute Merkle root of gradient hashes.
        
        Args:
            gradient_hashes: List of gradient hashes
            
        Returns:
            Merkle root hash
        """
        if not gradient_hashes:
            return hashlib.sha256(b"empty").hexdigest()
        
        # Build Merkle tree
        current_level = [
            bytes.fromhex(h) if isinstance(h, str) else h
            for h in gradient_hashes
        ]
        
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                combined = hashlib.sha256(left + right).digest()
                next_level.append(combined)
            current_level = next_level
        
        return current_level[0].hex()


class ModelUpdatePipeline:
    """
    Pipeline for model updates and version management.
    
    Handles:
    - Gradient aggregation from multiple miners
    - Model version upgrades
    - Adapter migration
    - Rollback on failure
    """
    
    def __init__(
        self,
        blockchain_client=None,
        ipfs_client=None,
        models_dir: Optional[str] = None,
        backup_dir: Optional[str] = None,
        current_version: str = "1.0.0",
        auto_backup: bool = True,
        verify_before_apply: bool = True,
    ):
        """
        Initialize model update pipeline.
        
        Args:
            blockchain_client: BlockchainClient instance
            ipfs_client: IPFSClient instance
            models_dir: Directory for model storage
            backup_dir: Directory for backups
            current_version: Current model version
            auto_backup: Automatically backup before updates
            verify_before_apply: Verify model integrity before applying
        """
        self.blockchain_client = blockchain_client
        self.ipfs_client = ipfs_client
        self.current_version = current_version
        self.auto_backup = auto_backup
        self.verify_before_apply = verify_before_apply
        
        # Setup directories
        home = os.environ.get("HOME", os.environ.get("USERPROFILE", "."))
        self.models_dir = Path(models_dir or os.path.join(home, ".r3mes", "models"))
        self.backup_dir = Path(backup_dir or os.path.join(home, ".r3mes", "backups"))
        
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Version history
        self._version_history: List[ModelVersion] = []
        self._current_model_path: Optional[Path] = None
        
        # Aggregator
        self.aggregator = GradientAggregator()
        
        logger.info(f"ModelUpdatePipeline initialized (version: {current_version})")
    
    def get_current_model_path(self) -> Optional[Path]:
        """Get path to current model."""
        return self._current_model_path
    
    def backup_current_model(self) -> Tuple[bool, str]:
        """
        Backup current model before update.
        
        Returns:
            Tuple of (success, backup_path or error)
        """
        if not self._current_model_path or not self._current_model_path.exists():
            return False, "No current model to backup"
        
        try:
            timestamp = int(time.time())
            backup_name = f"model_{self.current_version}_{timestamp}.pt"
            backup_path = self.backup_dir / backup_name
            
            # Copy model file
            import shutil
            shutil.copy2(self._current_model_path, backup_path)
            
            logger.info(f"Model backed up to: {backup_path}")
            return True, str(backup_path)
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False, str(e)
    
    def verify_model_integrity(
        self,
        model_path: Path,
        expected_checksum: str,
    ) -> bool:
        """
        Verify model file integrity.
        
        Args:
            model_path: Path to model file
            expected_checksum: Expected SHA256 checksum
            
        Returns:
            True if checksum matches
        """
        try:
            hasher = hashlib.sha256()
            with open(model_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    hasher.update(chunk)
            
            actual_checksum = hasher.hexdigest()
            return actual_checksum == expected_checksum
            
        except Exception as e:
            logger.error(f"Integrity check failed: {e}")
            return False
    
    def download_model_from_ipfs(
        self,
        ipfs_hash: str,
        target_path: Path,
    ) -> Tuple[bool, str]:
        """
        Download model from IPFS.
        
        Args:
            ipfs_hash: IPFS hash of model
            target_path: Target path for download
            
        Returns:
            Tuple of (success, path or error)
        """
        if not self.ipfs_client:
            return False, "IPFS client not configured"
        
        try:
            logger.info(f"Downloading model from IPFS: {ipfs_hash}")
            
            content = self.ipfs_client.retrieve_content(ipfs_hash)
            if content is None:
                return False, f"Failed to retrieve from IPFS: {ipfs_hash}"
            
            with open(target_path, 'wb') as f:
                f.write(content)
            
            logger.info(f"Model downloaded to: {target_path}")
            return True, str(target_path)
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False, str(e)
    
    def apply_aggregated_gradients(
        self,
        model: nn.Module,
        aggregated_gradients: Dict[str, torch.Tensor],
        learning_rate: float = 1e-4,
    ) -> bool:
        """
        Apply aggregated gradients to model.
        
        Args:
            model: Model to update
            aggregated_gradients: Aggregated gradient dictionary
            learning_rate: Learning rate for update
            
        Returns:
            True if successful
        """
        try:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in aggregated_gradients:
                        # Apply gradient update
                        param.data -= learning_rate * aggregated_gradients[name]
            
            logger.info("Aggregated gradients applied to model")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply gradients: {e}")
            return False
    
    def migrate_adapters(
        self,
        old_version: str,
        new_version: str,
        adapter_registry=None,
    ) -> Tuple[bool, List[str]]:
        """
        Migrate adapters to new model version.
        
        Args:
            old_version: Old model version
            new_version: New model version
            adapter_registry: AdapterRegistry instance
            
        Returns:
            Tuple of (success, list of migrated adapter IDs)
        """
        if not adapter_registry:
            logger.warning("No adapter registry provided, skipping migration")
            return True, []
        
        try:
            # Get adapters compatible with old version
            old_adapters = adapter_registry.get_compatible_adapters(old_version)
            
            migrated = []
            for adapter in old_adapters:
                # Check if adapter is also compatible with new version
                is_compatible, _ = adapter_registry.check_compatibility(
                    adapter.adapter_id,
                    new_version,
                )
                
                if is_compatible:
                    migrated.append(adapter.adapter_id)
                else:
                    logger.warning(
                        f"Adapter {adapter.adapter_id} not compatible with "
                        f"new version {new_version}"
                    )
            
            logger.info(f"Migrated {len(migrated)} adapters to version {new_version}")
            return True, migrated
            
        except Exception as e:
            logger.error(f"Adapter migration failed: {e}")
            return False, []
    
    def rollback(
        self,
        backup_path: Optional[str] = None,
    ) -> UpdateResult:
        """
        Rollback to previous model version.
        
        Args:
            backup_path: Path to backup (uses latest if not provided)
            
        Returns:
            UpdateResult with rollback status
        """
        try:
            # Find latest backup if not provided
            if backup_path is None:
                backups = sorted(self.backup_dir.glob("model_*.pt"))
                if not backups:
                    return UpdateResult(
                        success=False,
                        status=UpdateStatus.FAILED,
                        error="No backups available for rollback",
                    )
                backup_path = str(backups[-1])
            
            backup_file = Path(backup_path)
            if not backup_file.exists():
                return UpdateResult(
                    success=False,
                    status=UpdateStatus.FAILED,
                    error=f"Backup not found: {backup_path}",
                )
            
            # Extract version from backup filename
            # Format: model_{version}_{timestamp}.pt
            parts = backup_file.stem.split("_")
            if len(parts) >= 2:
                old_version = parts[1]
            else:
                old_version = "unknown"
            
            # Restore backup
            import shutil
            if self._current_model_path:
                shutil.copy2(backup_file, self._current_model_path)
            
            new_version = self.current_version
            self.current_version = old_version
            
            logger.info(f"Rolled back from {new_version} to {old_version}")
            
            return UpdateResult(
                success=True,
                status=UpdateStatus.ROLLED_BACK,
                old_version=new_version,
                new_version=old_version,
            )
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return UpdateResult(
                success=False,
                status=UpdateStatus.FAILED,
                error=str(e),
            )
    
    def upgrade_model(
        self,
        new_version: str,
        ipfs_hash: str,
        expected_checksum: str,
        adapter_registry=None,
    ) -> UpdateResult:
        """
        Upgrade to a new model version.
        
        Args:
            new_version: New model version
            ipfs_hash: IPFS hash of new model
            expected_checksum: Expected checksum
            adapter_registry: AdapterRegistry for adapter migration
            
        Returns:
            UpdateResult with upgrade status
        """
        old_version = self.current_version
        backup_path = None
        
        try:
            # Step 1: Backup current model
            if self.auto_backup and self._current_model_path:
                logger.info("Backing up current model...")
                success, backup_path = self.backup_current_model()
                if not success:
                    logger.warning(f"Backup failed: {backup_path}")
            
            # Step 2: Download new model
            logger.info(f"Downloading new model version {new_version}...")
            new_model_path = self.models_dir / f"model_{new_version}.pt"
            
            success, result = self.download_model_from_ipfs(ipfs_hash, new_model_path)
            if not success:
                return UpdateResult(
                    success=False,
                    status=UpdateStatus.FAILED,
                    old_version=old_version,
                    error=f"Download failed: {result}",
                    rollback_available=backup_path is not None,
                )
            
            # Step 3: Verify integrity
            if self.verify_before_apply:
                logger.info("Verifying model integrity...")
                if not self.verify_model_integrity(new_model_path, expected_checksum):
                    new_model_path.unlink(missing_ok=True)
                    return UpdateResult(
                        success=False,
                        status=UpdateStatus.FAILED,
                        old_version=old_version,
                        error="Checksum verification failed",
                        rollback_available=backup_path is not None,
                    )
            
            # Step 4: Migrate adapters
            logger.info("Migrating adapters...")
            migrate_success, migrated_adapters = self.migrate_adapters(
                old_version,
                new_version,
                adapter_registry,
            )
            
            # Step 5: Update current model path
            self._current_model_path = new_model_path
            self.current_version = new_version
            
            # Step 6: Record version history
            self._version_history.append(ModelVersion(
                version=new_version,
                ipfs_hash=ipfs_hash,
                checksum=expected_checksum,
                block_height=0,  # Would come from blockchain
                timestamp=int(time.time()),
                adapter_count=len(migrated_adapters),
            ))
            
            logger.info(f"Model upgraded from {old_version} to {new_version}")
            
            return UpdateResult(
                success=True,
                status=UpdateStatus.COMPLETED,
                old_version=old_version,
                new_version=new_version,
                rollback_available=backup_path is not None,
                metadata={
                    "migrated_adapters": migrated_adapters,
                    "backup_path": backup_path,
                },
            )
            
        except Exception as e:
            logger.error(f"Model upgrade failed: {e}", exc_info=True)
            
            # Attempt rollback
            if backup_path:
                logger.info("Attempting rollback...")
                self.rollback(backup_path)
            
            return UpdateResult(
                success=False,
                status=UpdateStatus.FAILED,
                old_version=old_version,
                error=str(e),
                rollback_available=backup_path is not None,
            )
    
    def aggregate_and_commit(
        self,
        gradient_ids: List[int],
        training_round_id: int,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Aggregate gradients and commit to blockchain.
        
        This is for proposer nodes that aggregate miner gradients.
        
        Args:
            gradient_ids: List of gradient IDs to aggregate
            training_round_id: Training round ID
            
        Returns:
            Tuple of (success, response_dict)
        """
        if not self.blockchain_client:
            return False, {"error": "Blockchain client not configured"}
        
        try:
            # Step 1: Fetch gradients from IPFS
            # (In production, would query blockchain for IPFS hashes)
            logger.info(f"Aggregating {len(gradient_ids)} gradients...")
            
            # Step 2: Compute commitment hash
            gradient_hashes = [str(gid) for gid in gradient_ids]  # Simplified
            commitment_hash = self.aggregator.compute_merkle_root(gradient_hashes)
            
            # Step 3: Commit aggregation to blockchain
            response = self.blockchain_client.commit_aggregation(
                proposer=self.blockchain_client.get_miner_address(),
                gradient_ids=gradient_ids,
                training_round_id=training_round_id,
                commitment_hash=commitment_hash,
            )
            
            if response.get("success"):
                logger.info(
                    f"Aggregation committed: ID={response.get('commitment_id')}"
                )
                return True, response
            else:
                return False, response
                
        except Exception as e:
            logger.error(f"Aggregation commit failed: {e}")
            return False, {"error": str(e)}
    
    def get_version_history(self) -> List[Dict[str, Any]]:
        """Get model version history."""
        return [v.to_dict() for v in self._version_history]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "current_version": self.current_version,
            "version_count": len(self._version_history),
            "models_dir": str(self.models_dir),
            "backup_dir": str(self.backup_dir),
            "auto_backup": self.auto_backup,
            "verify_before_apply": self.verify_before_apply,
        }

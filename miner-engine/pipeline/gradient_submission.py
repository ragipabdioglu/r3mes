"""
Gradient Submission Pipeline

KRİTİK EKSİKLİK #1 ÇÖZÜMÜ:
DoRA Gradient → IPFS → Blockchain entegrasyonu

Flow:
    DoRATrainer.train_step()
        ↓
    compute_gradient() ✅
        ↓
    GradientSubmissionPipeline.submit_after_training()
        ↓
    IPFSClient.upload_gradient() ✅
        ↓
    BlockchainClient.submit_gradient() ✅
"""

import io
import time
import logging
import hashlib
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)


@dataclass
class SubmissionResult:
    """Result of gradient submission."""
    success: bool
    ipfs_hash: Optional[str] = None
    tx_hash: Optional[str] = None
    stored_gradient_id: Optional[int] = None
    gradient_hash: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "ipfs_hash": self.ipfs_hash,
            "tx_hash": self.tx_hash,
            "stored_gradient_id": self.stored_gradient_id,
            "gradient_hash": self.gradient_hash,
            "error": self.error,
            "metadata": self.metadata,
        }


class GradientSubmissionPipeline:
    """
    Pipeline for submitting gradients to IPFS and blockchain.
    
    Solves the critical gap where DoRA training produces gradients
    but they are not automatically uploaded to IPFS and submitted
    to the blockchain.
    
    Usage:
        pipeline = GradientSubmissionPipeline(
            ipfs_client=ipfs_client,
            blockchain_client=blockchain_client,
        )
        
        # After training step
        result = pipeline.submit_after_training(
            gradients=gradients_dict,
            metadata=training_metadata,
        )
    """
    
    def __init__(
        self,
        ipfs_client=None,
        blockchain_client=None,
        model_version: str = "1.0.0",
        auto_retry: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize gradient submission pipeline.
        
        Args:
            ipfs_client: IPFSClient instance
            blockchain_client: BlockchainClient instance
            model_version: Current model version
            auto_retry: Enable automatic retry on failure
            max_retries: Maximum retry attempts
            retry_delay: Initial delay between retries (exponential backoff)
        """
        self.ipfs_client = ipfs_client
        self.blockchain_client = blockchain_client
        self.model_version = model_version
        self.auto_retry = auto_retry
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Submission tracking
        self._submission_count = 0
        self._successful_submissions = 0
        self._failed_submissions = 0
        
        logger.info("GradientSubmissionPipeline initialized")
    
    def serialize_gradients(self, gradients: Dict[str, torch.Tensor]) -> bytes:
        """
        Serialize gradients to bytes for IPFS upload.
        
        Args:
            gradients: Dictionary of gradient tensors
            
        Returns:
            Serialized bytes
        """
        buffer = io.BytesIO()
        
        # Convert to CPU and save
        cpu_gradients = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in gradients.items()
        }
        
        torch.save(cpu_gradients, buffer)
        return buffer.getvalue()
    
    def compute_gradient_hash(
        self,
        gradients: Dict[str, torch.Tensor],
        precision: str = "int8",
    ) -> str:
        """
        Compute deterministic hash of gradients.
        
        Args:
            gradients: Dictionary of gradient tensors
            precision: Hash precision ("int8" for cross-platform)
            
        Returns:
            Hex string hash
        """
        hasher = hashlib.sha256()
        
        # Sort keys for determinism
        for key in sorted(gradients.keys()):
            tensor = gradients[key]
            
            if isinstance(tensor, torch.Tensor):
                # Quantize to int8 for cross-platform determinism
                if precision == "int8":
                    # Scale to int8 range
                    tensor_cpu = tensor.cpu().float()
                    max_val = tensor_cpu.abs().max()
                    if max_val > 0:
                        scaled = (tensor_cpu / max_val * 127).to(torch.int8)
                    else:
                        scaled = tensor_cpu.to(torch.int8)
                    hasher.update(scaled.numpy().tobytes())
                else:
                    hasher.update(tensor.cpu().numpy().tobytes())
            else:
                hasher.update(str(tensor).encode())
            
            hasher.update(key.encode())
        
        return hasher.hexdigest()
    
    def upload_to_ipfs(
        self,
        gradient_bytes: bytes,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """
        Upload gradient data to IPFS.
        
        Args:
            gradient_bytes: Serialized gradient bytes
            metadata: Optional metadata
            
        Returns:
            Tuple of (success, ipfs_hash or error_message)
        """
        if not self.ipfs_client:
            return False, "IPFS client not configured"
        
        for attempt in range(self.max_retries if self.auto_retry else 1):
            try:
                ipfs_hash = self.ipfs_client.upload_gradient(
                    gradient_bytes,
                    metadata=metadata,
                )
                
                if ipfs_hash:
                    logger.info(f"Gradient uploaded to IPFS: {ipfs_hash}")
                    return True, ipfs_hash
                else:
                    raise ValueError("Empty IPFS hash returned")
                    
            except Exception as e:
                if attempt < self.max_retries - 1 and self.auto_retry:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"IPFS upload failed (attempt {attempt + 1}): {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"IPFS upload failed after {attempt + 1} attempts: {e}")
                    return False, str(e)
        
        return False, "Max retries exceeded"
    
    def submit_to_blockchain(
        self,
        ipfs_hash: str,
        gradient_hash: str,
        training_round_id: int,
        shard_id: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Submit gradient reference to blockchain.
        
        Args:
            ipfs_hash: IPFS hash of gradient data
            gradient_hash: Deterministic gradient hash
            training_round_id: Training round ID
            shard_id: Shard assignment
            metadata: Additional metadata
            
        Returns:
            Tuple of (success, response_dict)
        """
        if not self.blockchain_client:
            return False, {"error": "Blockchain client not configured"}
        
        metadata = metadata or {}
        gpu_architecture = metadata.get("gpu_architecture", "unknown")
        claimed_loss = metadata.get("loss")
        
        for attempt in range(self.max_retries if self.auto_retry else 1):
            try:
                response = self.blockchain_client.submit_gradient(
                    miner_address=self.blockchain_client.get_miner_address(),
                    ipfs_hash=ipfs_hash,
                    model_version=self.model_version,
                    training_round_id=training_round_id,
                    shard_id=shard_id,
                    gradient_hash=gradient_hash,
                    gpu_architecture=gpu_architecture,
                    claimed_loss=str(claimed_loss) if claimed_loss else None,
                )
                
                if response.get("success"):
                    logger.info(
                        f"Gradient submitted to blockchain: "
                        f"ID={response.get('stored_gradient_id')}, "
                        f"TX={response.get('tx_hash')}"
                    )
                    return True, response
                else:
                    raise ValueError(response.get("error", "Unknown error"))
                    
            except Exception as e:
                if attempt < self.max_retries - 1 and self.auto_retry:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Blockchain submit failed (attempt {attempt + 1}): {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Blockchain submit failed after {attempt + 1} attempts: {e}")
                    return False, {"error": str(e)}
        
        return False, {"error": "Max retries exceeded"}
    
    def submit_after_training(
        self,
        gradients: Dict[str, torch.Tensor],
        training_round_id: int,
        shard_id: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SubmissionResult:
        """
        Submit gradients after training step.
        
        This is the main entry point that handles the full pipeline:
        1. Serialize gradients
        2. Compute deterministic hash
        3. Upload to IPFS
        4. Submit to blockchain
        
        Args:
            gradients: Dictionary of gradient tensors
            training_round_id: Training round ID
            shard_id: Shard assignment
            metadata: Training metadata (gpu_architecture, loss, etc.)
            
        Returns:
            SubmissionResult with success status and details
        """
        self._submission_count += 1
        metadata = metadata or {}
        
        try:
            # Step 1: Serialize gradients
            logger.debug("Serializing gradients...")
            gradient_bytes = self.serialize_gradients(gradients)
            
            # Step 2: Compute deterministic hash
            logger.debug("Computing gradient hash...")
            gradient_hash = self.compute_gradient_hash(gradients)
            
            # Step 3: Upload to IPFS
            logger.info("Uploading gradient to IPFS...")
            ipfs_success, ipfs_result = self.upload_to_ipfs(
                gradient_bytes,
                metadata={
                    "gradient_hash": gradient_hash,
                    "training_round_id": training_round_id,
                    "shard_id": shard_id,
                    **metadata,
                },
            )
            
            if not ipfs_success:
                self._failed_submissions += 1
                return SubmissionResult(
                    success=False,
                    gradient_hash=gradient_hash,
                    error=f"IPFS upload failed: {ipfs_result}",
                    metadata=metadata,
                )
            
            ipfs_hash = ipfs_result
            
            # Step 4: Submit to blockchain
            logger.info("Submitting gradient to blockchain...")
            bc_success, bc_response = self.submit_to_blockchain(
                ipfs_hash=ipfs_hash,
                gradient_hash=gradient_hash,
                training_round_id=training_round_id,
                shard_id=shard_id,
                metadata=metadata,
            )
            
            if not bc_success:
                self._failed_submissions += 1
                return SubmissionResult(
                    success=False,
                    ipfs_hash=ipfs_hash,
                    gradient_hash=gradient_hash,
                    error=f"Blockchain submit failed: {bc_response.get('error')}",
                    metadata=metadata,
                )
            
            # Success!
            self._successful_submissions += 1
            return SubmissionResult(
                success=True,
                ipfs_hash=ipfs_hash,
                tx_hash=bc_response.get("tx_hash"),
                stored_gradient_id=bc_response.get("stored_gradient_id"),
                gradient_hash=gradient_hash,
                metadata=metadata,
            )
            
        except Exception as e:
            self._failed_submissions += 1
            logger.error(f"Gradient submission failed: {e}", exc_info=True)
            return SubmissionResult(
                success=False,
                error=str(e),
                metadata=metadata,
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get submission statistics."""
        return {
            "total_submissions": self._submission_count,
            "successful_submissions": self._successful_submissions,
            "failed_submissions": self._failed_submissions,
            "success_rate": (
                self._successful_submissions / self._submission_count
                if self._submission_count > 0 else 0.0
            ),
            "model_version": self.model_version,
            "auto_retry": self.auto_retry,
            "max_retries": self.max_retries,
        }

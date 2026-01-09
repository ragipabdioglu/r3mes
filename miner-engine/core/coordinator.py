#!/usr/bin/env python3
"""
Off-Chain Distributed Training Coordinator

Manages distributed training coordination:
- Global model distribution to participating miners
- Gradient collection and metadata tracking
- IPFS integration for model and gradient storage
- Training round management
"""

import hashlib
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from utils.ipfs_client import IPFSClient
from bridge.blockchain_client import BlockchainClient
from typing import Dict


class TrainingRoundStatus(Enum):
    """Training round status."""
    INITIALIZING = "initializing"
    COLLECTING_GRADIENTS = "collecting_gradients"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GradientMetadata:
    """Metadata for a gradient submission."""
    gradient_id: str
    miner_address: str
    ipfs_hash: str
    gradient_hash: str
    model_version: str
    training_round_id: int
    shard_id: int
    gpu_architecture: str
    submitted_at: float
    status: str = "pending"
    porep_proof_ipfs_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gradient_id": self.gradient_id,
            "miner_address": self.miner_address,
            "ipfs_hash": self.ipfs_hash,
            "gradient_hash": self.gradient_hash,
            "model_version": self.model_version,
            "training_round_id": self.training_round_id,
            "shard_id": self.shard_id,
            "gpu_architecture": self.gpu_architecture,
            "submitted_at": self.submitted_at,
            "status": self.status,
            "porep_proof_ipfs_hash": self.porep_proof_ipfs_hash,
        }


@dataclass
class TrainingRound:
    """Represents a training round."""
    round_id: int
    model_version: str
    model_ipfs_hash: str
    status: TrainingRoundStatus
    start_time: float
    end_time: Optional[float] = None
    gradients: List[GradientMetadata] = field(default_factory=list)
    aggregated_gradient_ipfs_hash: Optional[str] = None
    merkle_root: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "round_id": self.round_id,
            "model_version": self.model_version,
            "model_ipfs_hash": self.model_ipfs_hash,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "gradient_count": len(self.gradients),
            "aggregated_gradient_ipfs_hash": self.aggregated_gradient_ipfs_hash,
            "merkle_root": self.merkle_root,
        }


class OffChainDistributedCoordinator:
    """
    Off-chain distributed training coordinator.
    
    Manages:
    - Global model distribution to participating miners
    - Gradient collection and metadata tracking
    - IPFS integration for model and gradient storage
    - Training round management
    """
    
    def __init__(
        self,
        ipfs_client: IPFSClient,
        blockchain_client: BlockchainClient,
        model_version: str = "1",
    ):
        """
        Initialize the coordinator.
        
        Args:
            ipfs_client: IPFS client for storage
            blockchain_client: Blockchain client for queries
            model_version: Current model version
        """
        self.ipfs_client = ipfs_client
        self.blockchain_client = blockchain_client
        self.model_version = model_version
        
        # Training rounds tracking
        self.training_rounds: Dict[int, TrainingRound] = {}
        self.current_round_id: int = 0
        
        # Gradient metadata tracking
        self.gradient_metadata: Dict[str, GradientMetadata] = {}
        
    def get_current_model_params(self) -> Optional[Dict[str, Any]]:
        """
        Get current model parameters from blockchain.
        
        Returns:
            Model parameters dict with ipfs_hash and version, or None
        """
        try:
            params = self.blockchain_client.get_model_params()
            return params
        except Exception as e:
            print(f"Error getting model params: {e}")
            return None
    
    def start_training_round(
        self,
        round_id: Optional[int] = None,
        model_ipfs_hash: Optional[str] = None,
    ) -> TrainingRound:
        """
        Start a new training round.
        
        Args:
            round_id: Optional round ID (auto-increments if not provided)
            model_ipfs_hash: Optional model IPFS hash (fetches from blockchain if not provided)
        
        Returns:
            TrainingRound instance
        """
        if round_id is None:
            self.current_round_id += 1
            round_id = self.current_round_id
        
        # Get model IPFS hash if not provided
        if model_ipfs_hash is None:
            model_params = self.get_current_model_params()
            if model_params:
                model_ipfs_hash = model_params.get("model_ipfs_hash")
            else:
                raise ValueError("Could not fetch model IPFS hash from blockchain")
        
        training_round = TrainingRound(
            round_id=round_id,
            model_version=self.model_version,
            model_ipfs_hash=model_ipfs_hash,
            status=TrainingRoundStatus.INITIALIZING,
            start_time=time.time(),
        )
        
        self.training_rounds[round_id] = training_round
        print(f"Started training round {round_id} with model {model_ipfs_hash}")
        
        return training_round
    
    def register_gradient(
        self,
        round_id: int,
        miner_address: str,
        ipfs_hash: str,
        gradient_hash: str,
        shard_id: int,
        gpu_architecture: str,
        porep_proof_ipfs_hash: Optional[str] = None,
    ) -> GradientMetadata:
        """
        Register a gradient submission.
        
        Args:
            round_id: Training round ID
            miner_address: Miner address
            ipfs_hash: IPFS hash of gradient
            gradient_hash: Deterministic hash of gradient
            shard_id: Shard ID
            gpu_architecture: GPU architecture
        
        Returns:
            GradientMetadata instance
        """
        if round_id not in self.training_rounds:
            raise ValueError(f"Training round {round_id} not found")
        
        training_round = self.training_rounds[round_id]
        
        # Generate gradient ID
        gradient_id = hashlib.sha256(
            f"{round_id}:{miner_address}:{ipfs_hash}:{time.time()}".encode()
        ).hexdigest()[:16]
        
        metadata = GradientMetadata(
            gradient_id=gradient_id,
            miner_address=miner_address,
            ipfs_hash=ipfs_hash,
            gradient_hash=gradient_hash,
            model_version=self.model_version,
            training_round_id=round_id,
            shard_id=shard_id,
            gpu_architecture=gpu_architecture,
            submitted_at=time.time(),
            status="registered",
            porep_proof_ipfs_hash=porep_proof_ipfs_hash,
        )
        
        training_round.gradients.append(metadata)
        self.gradient_metadata[gradient_id] = metadata
        
        # Update round status
        if training_round.status == TrainingRoundStatus.INITIALIZING:
            training_round.status = TrainingRoundStatus.COLLECTING_GRADIENTS
        
        print(f"Registered gradient {gradient_id} from {miner_address} in round {round_id}")
        
        return metadata
    
    def get_round_gradients(self, round_id: int) -> List[GradientMetadata]:
        """
        Get all gradients for a training round.
        
        Args:
            round_id: Training round ID
        
        Returns:
            List of GradientMetadata
        """
        if round_id not in self.training_rounds:
            return []
        
        return self.training_rounds[round_id].gradients
    
    def get_round_status(self, round_id: int) -> Optional[TrainingRoundStatus]:
        """
        Get training round status.
        
        Args:
            round_id: Training round ID
        
        Returns:
            TrainingRoundStatus or None
        """
        if round_id not in self.training_rounds:
            return None
        
        return self.training_rounds[round_id].status
    
    def mark_round_aggregating(self, round_id: int) -> None:
        """Mark training round as aggregating."""
        if round_id not in self.training_rounds:
            raise ValueError(f"Training round {round_id} not found")
        
        self.training_rounds[round_id].status = TrainingRoundStatus.AGGREGATING
    
    def complete_training_round(
        self,
        round_id: int,
        aggregated_gradient_ipfs_hash: str,
        merkle_root: str,
    ) -> None:
        """
        Complete a training round with aggregated gradient.
        
        Args:
            round_id: Training round ID
            aggregated_gradient_ipfs_hash: IPFS hash of aggregated gradient
            merkle_root: Merkle root of gradient hashes
        """
        if round_id not in self.training_rounds:
            raise ValueError(f"Training round {round_id} not found")
        
        training_round = self.training_rounds[round_id]
        training_round.status = TrainingRoundStatus.COMPLETED
        training_round.end_time = time.time()
        training_round.aggregated_gradient_ipfs_hash = aggregated_gradient_ipfs_hash
        training_round.merkle_root = merkle_root
        
        print(f"Completed training round {round_id} with {len(training_round.gradients)} gradients")
    
    def get_training_round(self, round_id: int) -> Optional[TrainingRound]:
        """
        Get training round by ID.
        
        Args:
            round_id: Training round ID
        
        Returns:
            TrainingRound or None
        """
        return self.training_rounds.get(round_id)
    
    def list_training_rounds(self) -> List[TrainingRound]:
        """
        List all training rounds.
        
        Returns:
            List of TrainingRound instances
        """
        return list(self.training_rounds.values())
    
    def get_gradient_metadata(self, gradient_id: str) -> Optional[GradientMetadata]:
        """
        Get gradient metadata by ID.
        
        Args:
            gradient_id: Gradient ID
        
        Returns:
            GradientMetadata or None
        """
        return self.gradient_metadata.get(gradient_id)


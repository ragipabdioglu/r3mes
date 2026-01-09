#!/usr/bin/env python3
"""
Multi-Proposer Aggregation System

Implements:
- Proposer rotation with VRF-based selection
- Commit-reveal scheme for aggregation results
- Robust combination using median/trimmed mean
- Merkle tree construction for gradient verification
"""

import hashlib
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import statistics

from core.coordinator import TrainingRound, GradientMetadata


class ProposerStatus(Enum):
    """Proposer status."""
    SELECTED = "selected"
    COMMITTED = "committed"
    REVEALED = "revealed"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AggregationCommit:
    """Commit phase data for commit-reveal scheme."""
    proposer_address: str
    commit_hash: str  # Hash of (aggregated_gradient_ipfs_hash + merkle_root + secret)
    block_height: int
    timestamp: float
    status: ProposerStatus = ProposerStatus.COMMITTED


@dataclass
class AggregationReveal:
    """Reveal phase data for commit-reveal scheme."""
    proposer_address: str
    aggregated_gradient_ipfs_hash: str
    merkle_root: str
    secret: str  # Random secret used in commit
    participant_gradient_ids: List[str]
    timestamp: float
    status: ProposerStatus = ProposerStatus.REVEALED


@dataclass
class ProposerCandidate:
    """Proposer candidate for VRF selection."""
    address: str
    stake: float
    vrf_output: Optional[bytes] = None
    vrf_proof: Optional[bytes] = None
    selected: bool = False


class VRFSelector:
    """
    VRF-based proposer selection.
    
    Simplified VRF implementation using hash-based selection.
    For production, use a proper VRF library.
    """
    
    @staticmethod
    def compute_vrf(seed: bytes, address: str) -> Tuple[bytes, bytes]:
        """
        Compute VRF output and proof.
        
        Args:
            seed: Random seed (e.g., block hash)
            address: Proposer address
        
        Returns:
            (vrf_output, vrf_proof) tuple
        """
        # Simplified VRF: hash(seed + address)
        # In production, use proper VRF (e.g., ECVRF)
        combined = seed + address.encode()
        vrf_output = hashlib.sha256(combined).digest()
        vrf_proof = hashlib.sha256(vrf_output + b"proof").digest()
        return vrf_output, vrf_proof
    
    @staticmethod
    def select_proposers(
        candidates: List[ProposerCandidate],
        seed: bytes,
        num_proposers: int = 3,
    ) -> List[ProposerCandidate]:
        """
        Select proposers using VRF.
        
        Args:
            candidates: List of proposer candidates
            seed: Random seed for selection
            num_proposers: Number of proposers to select
        
        Returns:
            List of selected proposers
        """
        # Compute VRF for all candidates
        for candidate in candidates:
            vrf_output, vrf_proof = VRFSelector.compute_vrf(seed, candidate.address)
            candidate.vrf_output = vrf_output
            candidate.vrf_proof = vrf_proof
        
        # Sort by VRF output (lower is better for selection)
        candidates_sorted = sorted(candidates, key=lambda c: c.vrf_output)
        
        # Select top N proposers
        selected = candidates_sorted[:num_proposers]
        for candidate in selected:
            candidate.selected = True
        
        return selected


class CommitRevealAggregator:
    """
    Commit-reveal scheme for aggregation.
    
    Prevents proposers from seeing each other's results before committing,
    reducing collusion and ensuring fair aggregation.
    """
    
    @staticmethod
    def create_commit(
        aggregated_gradient_ipfs_hash: str,
        merkle_root: str,
        secret: str,
    ) -> str:
        """
        Create commit hash.
        
        Args:
            aggregated_gradient_ipfs_hash: IPFS hash of aggregated gradient
            merkle_root: Merkle root of included gradients
            secret: Random secret
        
        Returns:
            Commit hash
        """
        combined = f"{aggregated_gradient_ipfs_hash}:{merkle_root}:{secret}".encode()
        return hashlib.sha256(combined).hexdigest()
    
    @staticmethod
    def verify_commit(
        commit_hash: str,
        aggregated_gradient_ipfs_hash: str,
        merkle_root: str,
        secret: str,
    ) -> bool:
        """
        Verify commit against reveal.
        
        Args:
            commit_hash: Original commit hash
            aggregated_gradient_ipfs_hash: Revealed IPFS hash
            merkle_root: Revealed Merkle root
            secret: Revealed secret
        
        Returns:
            True if commit matches reveal
        """
        expected_commit = CommitRevealAggregator.create_commit(
            aggregated_gradient_ipfs_hash,
            merkle_root,
            secret,
        )
        return commit_hash == expected_commit


class RobustAggregator:
    """
    Robust aggregation using median/trimmed mean.
    
    Handles outliers and malicious gradients by using robust statistics.
    """
    
    @staticmethod
    def aggregate_gradients(
        gradient_hashes: List[str],
        aggregation_method: str = "median",
        trim_percentage: float = 0.1,
    ) -> str:
        """
        Aggregate gradients using robust method.
        
        Args:
            gradient_hashes: List of gradient IPFS hashes
            aggregation_method: "median", "trimmed_mean", or "mean"
            trim_percentage: Percentage to trim for trimmed mean
        
        Returns:
            Aggregated gradient IPFS hash (simulated for now)
        """
        if not gradient_hashes:
            raise ValueError("No gradients to aggregate")
        
        if aggregation_method == "median":
            # Use median gradient (most robust to outliers)
            mid = len(gradient_hashes) // 2
            return gradient_hashes[mid]
        
        elif aggregation_method == "trimmed_mean":
            # Trim outliers and use mean
            trim_count = int(len(gradient_hashes) * trim_percentage)
            trimmed = gradient_hashes[trim_count:-trim_count] if trim_count > 0 else gradient_hashes
            # For now, return first trimmed gradient (in production, compute actual mean)
            return trimmed[0] if trimmed else gradient_hashes[0]
        
        elif aggregation_method == "mean":
            # Simple mean (least robust)
            return gradient_hashes[0]  # Placeholder
        
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    
    @staticmethod
    def compute_merkle_root(gradient_hashes: List[str]) -> str:
        """
        Compute Merkle root of gradient hashes.
        
        Args:
            gradient_hashes: List of gradient hashes
        
        Returns:
            Merkle root hash
        """
        from core.merkle import MerkleTree
        
        if not gradient_hashes:
            return ""
        
        tree = MerkleTree(gradient_hashes)
        return tree.get_root()


class MultiProposerAggregator:
    """
    Multi-proposer aggregation system.
    
    Coordinates multiple proposers to aggregate gradients with:
    - VRF-based proposer selection
    - Commit-reveal scheme
    - Robust aggregation (median/trimmed mean)
    """
    
    def __init__(
        self,
        num_proposers: int = 3,
        aggregation_method: str = "median",
    ):
        """
        Initialize multi-proposer aggregator.
        
        Args:
            num_proposers: Number of proposers to select
            aggregation_method: Aggregation method ("median", "trimmed_mean", "mean")
        """
        self.num_proposers = num_proposers
        self.aggregation_method = aggregation_method
        
        # Proposer tracking
        self.commits: Dict[str, AggregationCommit] = {}
        self.reveals: Dict[str, AggregationReveal] = {}
        self.selected_proposers: List[ProposerCandidate] = []
    
    def select_proposers(
        self,
        candidates: List[ProposerCandidate],
        seed: bytes,
    ) -> List[ProposerCandidate]:
        """
        Select proposers using VRF.
        
        Args:
            candidates: List of proposer candidates
            seed: Random seed (e.g., block hash)
        
        Returns:
            List of selected proposers
        """
        selected = VRFSelector.select_proposers(
            candidates,
            seed,
            num_proposers=self.num_proposers,
        )
        self.selected_proposers = selected
        return selected
    
    def register_commit(
        self,
        proposer_address: str,
        commit_hash: str,
        block_height: int,
    ) -> AggregationCommit:
        """
        Register a commit from a proposer.
        
        Args:
            proposer_address: Proposer address
            commit_hash: Commit hash
            block_height: Block height at commit time
        
        Returns:
            AggregationCommit instance
        """
        commit = AggregationCommit(
            proposer_address=proposer_address,
            commit_hash=commit_hash,
            block_height=block_height,
            timestamp=time.time(),
        )
        self.commits[proposer_address] = commit
        return commit
    
    def register_reveal(
        self,
        proposer_address: str,
        aggregated_gradient_ipfs_hash: str,
        merkle_root: str,
        secret: str,
        participant_gradient_ids: List[str],
    ) -> AggregationReveal:
        """
        Register a reveal from a proposer.
        
        Args:
            proposer_address: Proposer address
            aggregated_gradient_ipfs_hash: IPFS hash of aggregated gradient
            merkle_root: Merkle root of included gradients
            secret: Secret used in commit
            participant_gradient_ids: List of gradient IDs included
        
        Returns:
            AggregationReveal instance
        
        Raises:
            ValueError: If commit doesn't match reveal
        """
        # Verify commit matches reveal
        if proposer_address not in self.commits:
            raise ValueError(f"No commit found for proposer {proposer_address}")
        
        commit = self.commits[proposer_address]
        if not CommitRevealAggregator.verify_commit(
            commit.commit_hash,
            aggregated_gradient_ipfs_hash,
            merkle_root,
            secret,
        ):
            raise ValueError(f"Commit verification failed for proposer {proposer_address}")
        
        reveal = AggregationReveal(
            proposer_address=proposer_address,
            aggregated_gradient_ipfs_hash=aggregated_gradient_ipfs_hash,
            merkle_root=merkle_root,
            secret=secret,
            participant_gradient_ids=participant_gradient_ids,
            timestamp=time.time(),
        )
        self.reveals[proposer_address] = reveal
        return reveal
    
    def aggregate_proposals(
        self,
        training_round: TrainingRound,
    ) -> Tuple[str, str]:
        """
        Aggregate proposals using robust method.
        
        Args:
            training_round: Training round with gradients
        
        Returns:
            (aggregated_gradient_ipfs_hash, merkle_root) tuple
        """
        if not self.reveals:
            raise ValueError("No reveals to aggregate")
        
        # Collect all gradient IPFS hashes from reveals
        all_gradient_hashes = []
        for reveal in self.reveals.values():
            # In production, retrieve actual gradients from IPFS
            # For now, use gradient IDs as placeholders
            all_gradient_hashes.extend(reveal.participant_gradient_ids)
        
        # Remove duplicates
        unique_hashes = list(set(all_gradient_hashes))
        
        # Aggregate using robust method
        aggregated_hash = RobustAggregator.aggregate_gradients(
            unique_hashes,
            aggregation_method=self.aggregation_method,
        )
        
        # Compute Merkle root
        merkle_root = RobustAggregator.compute_merkle_root(unique_hashes)
        
        return aggregated_hash, merkle_root
    
    def get_aggregation_summary(self) -> Dict[str, Any]:
        """
        Get aggregation summary.
        
        Returns:
            Dictionary with aggregation statistics
        """
        return {
            "num_proposers": self.num_proposers,
            "selected_proposers": len(self.selected_proposers),
            "commits": len(self.commits),
            "reveals": len(self.reveals),
            "aggregation_method": self.aggregation_method,
        }


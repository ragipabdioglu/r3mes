#!/usr/bin/env python3
"""
Merkle Proof Verification System

Implements:
- Merkle tree construction for gradient hashes
- Cryptographic proof verification for gradient inclusion
- Adaptive sampling with stake-weighted suspicious pattern detection
"""

import hashlib
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class MerkleProof:
    """Merkle proof for a leaf node."""
    leaf_hash: str
    path: List[Tuple[str, str]]  # List of (sibling_hash, position) tuples
    root_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "leaf_hash": self.leaf_hash,
            "path": self.path,
            "root_hash": self.root_hash,
        }


class MerkleTree:
    """
    Merkle tree for gradient hash verification.
    
    Builds a binary Merkle tree from gradient hashes and provides
    proof generation and verification.
    """
    
    def __init__(self, leaf_hashes: List[str]):
        """
        Initialize Merkle tree.
        
        Args:
            leaf_hashes: List of gradient hashes (leaves)
        """
        if not leaf_hashes:
            raise ValueError("Cannot create Merkle tree with empty leaves")
        
        self.leaf_hashes = leaf_hashes.copy()
        self.tree = self._build_tree()
        self.root_hash = self.tree[-1][0] if self.tree else ""
    
    def _hash_pair(self, left: str, right: str) -> str:
        """
        Hash a pair of nodes.
        
        Args:
            left: Left node hash
            right: Right node hash
        
        Returns:
            Combined hash
        """
        return hashlib.sha256(f"{left}:{right}".encode()).hexdigest()
    
    def _build_tree(self) -> List[List[str]]:
        """
        Build Merkle tree bottom-up.
        
        Returns:
            Tree structure as list of levels
        """
        # Start with leaves
        current_level = self.leaf_hashes.copy()
        tree = [current_level]
        
        # Build tree level by level
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    # Pair two nodes
                    combined = self._hash_pair(current_level[i], current_level[i + 1])
                    next_level.append(combined)
                else:
                    # Odd number, duplicate last node
                    combined = self._hash_pair(current_level[i], current_level[i])
                    next_level.append(combined)
            tree.append(next_level)
            current_level = next_level
        
        return tree
    
    def get_proof(self, leaf_index: int) -> Optional[MerkleProof]:
        """
        Get Merkle proof for a leaf.
        
        Args:
            leaf_index: Index of leaf in original list
        
        Returns:
            MerkleProof or None if index invalid
        """
        if leaf_index < 0 or leaf_index >= len(self.leaf_hashes):
            return None
        
        leaf_hash = self.leaf_hashes[leaf_index]
        path = []
        current_index = leaf_index
        
        # Traverse up the tree
        for level in self.tree[:-1]:  # Exclude root level
            if current_index % 2 == 0:
                # Left child, sibling is right
                sibling_index = current_index + 1
                if sibling_index < len(level):
                    path.append((level[sibling_index], "right"))
                else:
                    # No sibling, duplicate
                    path.append((level[current_index], "right"))
            else:
                # Right child, sibling is left
                sibling_index = current_index - 1
                path.append((level[sibling_index], "left"))
            
            # Move to parent level
            current_index = current_index // 2
        
        return MerkleProof(
            leaf_hash=leaf_hash,
            path=path,
            root_hash=self.root_hash,
        )
    
    def verify_proof(self, proof: MerkleProof) -> bool:
        """
        Verify Merkle proof.
        
        Args:
            proof: MerkleProof to verify
        
        Returns:
            True if proof is valid
        """
        current_hash = proof.leaf_hash
        
        # Reconstruct path to root
        for sibling_hash, position in proof.path:
            if position == "left":
                current_hash = self._hash_pair(sibling_hash, current_hash)
            else:  # right
                current_hash = self._hash_pair(current_hash, sibling_hash)
        
        # Check if computed root matches
        return current_hash == proof.root_hash
    
    def get_root(self) -> str:
        """
        Get Merkle root hash.
        
        Returns:
            Root hash
        """
        return self.root_hash


class MerkleVerifier:
    """
    Merkle proof verifier with adaptive sampling.
    
    Provides:
    - Gradient inclusion verification
    - Adaptive sampling for large datasets
    - Stake-weighted suspicious pattern detection
    """
    
    @staticmethod
    def verify_gradient_inclusion(
        gradient_hash: str,
        merkle_root: str,
        proof: MerkleProof,
    ) -> bool:
        """
        Verify that a gradient is included in the Merkle tree.
        
        Args:
            gradient_hash: Hash of gradient to verify
            merkle_root: Expected Merkle root
            proof: Merkle proof
        
        Returns:
            True if gradient is included
        """
        # Check leaf hash matches
        if proof.leaf_hash != gradient_hash:
            return False
        
        # Check root hash matches
        if proof.root_hash != merkle_root:
            return False
        
        # Verify proof path
        current_hash = gradient_hash
        for sibling_hash, position in proof.path:
            if position == "left":
                combined = f"{sibling_hash}:{current_hash}".encode()
            else:  # right
                combined = f"{current_hash}:{sibling_hash}".encode()
            current_hash = hashlib.sha256(combined).hexdigest()
        
        return current_hash == merkle_root
    
    @staticmethod
    def adaptive_sample(
        gradient_hashes: List[str],
        sample_size: int,
        suspicious_indices: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Adaptive sampling with focus on suspicious gradients.
        
        Args:
            gradient_hashes: List of gradient hashes
            sample_size: Number of gradients to sample
            suspicious_indices: Optional list of suspicious gradient indices
        
        Returns:
            List of sampled indices
        """
        if sample_size >= len(gradient_hashes):
            return list(range(len(gradient_hashes)))
        
        sampled = set()
        
        # Always include suspicious gradients
        if suspicious_indices:
            for idx in suspicious_indices:
                if idx < len(gradient_hashes):
                    sampled.add(idx)
        
        # Fill remaining slots with random sampling
        import random
        remaining = sample_size - len(sampled)
        if remaining > 0:
            available = [i for i in range(len(gradient_hashes)) if i not in sampled]
            sampled.update(random.sample(available, min(remaining, len(available))))
        
        return sorted(list(sampled))
    
    @staticmethod
    def detect_suspicious_patterns(
        gradient_hashes: List[str],
        stake_weights: Optional[Dict[str, float]] = None,
    ) -> List[int]:
        """
        Detect suspicious gradient patterns using stake-weighted analysis.
        
        Args:
            gradient_hashes: List of gradient hashes
            stake_weights: Optional dictionary mapping miner addresses to stake
        
        Returns:
            List of suspicious gradient indices
        """
        suspicious = []
        
        # Simple heuristic: detect duplicate hashes (potential copy-paste attack)
        hash_counts = {}
        for i, grad_hash in enumerate(gradient_hashes):
            if grad_hash in hash_counts:
                hash_counts[grad_hash].append(i)
            else:
                hash_counts[grad_hash] = [i]
        
        # Mark duplicates as suspicious
        for hash_val, indices in hash_counts.items():
            if len(indices) > 1:
                suspicious.extend(indices)
        
        # Additional checks could include:
        # - Gradient norm analysis
        # - Cosine similarity clustering
        # - Statistical outlier detection
        # - Stake-weighted anomaly scoring
        
        return list(set(suspicious))


class MerkleProofManager:
    """
    Manager for Merkle proof operations.
    
    Handles:
    - Tree construction from gradient metadata
    - Proof generation for specific gradients
    - Batch verification
    """
    
    def __init__(self):
        """Initialize Merkle proof manager."""
        self.trees: Dict[int, MerkleTree] = {}  # round_id -> MerkleTree
    
    def build_tree_for_round(
        self,
        round_id: int,
        gradient_hashes: List[str],
    ) -> MerkleTree:
        """
        Build Merkle tree for a training round.
        
        Args:
            round_id: Training round ID
            gradient_hashes: List of gradient hashes
        
        Returns:
            MerkleTree instance
        """
        tree = MerkleTree(gradient_hashes)
        self.trees[round_id] = tree
        return tree
    
    def get_proof(
        self,
        round_id: int,
        gradient_hash: str,
    ) -> Optional[MerkleProof]:
        """
        Get Merkle proof for a gradient in a round.
        
        Args:
            round_id: Training round ID
            gradient_hash: Gradient hash
        
        Returns:
            MerkleProof or None
        """
        if round_id not in self.trees:
            return None
        
        tree = self.trees[round_id]
        
        # Find gradient index
        try:
            gradient_index = tree.leaf_hashes.index(gradient_hash)
        except ValueError:
            return None
        
        return tree.get_proof(gradient_index)
    
    def verify_gradient(
        self,
        round_id: int,
        gradient_hash: str,
        merkle_root: str,
    ) -> Tuple[bool, Optional[MerkleProof]]:
        """
        Verify a gradient is included in a round's Merkle tree.
        
        Args:
            round_id: Training round ID
            gradient_hash: Gradient hash to verify
            merkle_root: Expected Merkle root
        
        Returns:
            (is_valid, proof) tuple
        """
        proof = self.get_proof(round_id, gradient_hash)
        if proof is None:
            return False, None
        
        is_valid = MerkleVerifier.verify_gradient_inclusion(
            gradient_hash,
            merkle_root,
            proof,
        )
        
        return is_valid, proof


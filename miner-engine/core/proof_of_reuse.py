#!/usr/bin/env python3
"""
R3MES Proof of Reuse Module

Implements the Proof of Reuse mechanism:
- Verified jobs become future trap candidates
- Genesis Vault grows organically
- Reduces need for pre-computation
"""

import hashlib
import time
import json
from typing import Optional, Dict, List, Any
from dataclasses import asdict
import logging

from core.types import (
    GenesisVaultEntry,
    GradientFingerprint,
    GradientSubmission,
    VerificationStatus,
)
from core.similarity import FingerprintExtractor
from core.trap_jobs import GenesisVaultManager

logger = logging.getLogger(__name__)


class ProofOfReuseManager:
    """
    Manages the Proof of Reuse system.
    
    When a gradient submission passes verification, it can be
    added to the Genesis Vault as a future trap candidate.
    This creates a self-sustaining verification system.
    """
    
    def __init__(
        self,
        vault_manager: GenesisVaultManager,
        min_verifications: int = 3,
        max_vault_size: int = 10000,
        fingerprint_k: int = 100,
    ):
        """
        Initialize Proof of Reuse manager.
        
        Args:
            vault_manager: Genesis Vault manager
            min_verifications: Min verifications before adding to vault
            max_vault_size: Maximum vault entries
            fingerprint_k: Top-K for fingerprint extraction
        """
        self.vault = vault_manager
        self.min_verifications = min_verifications
        self.max_vault_size = max_vault_size
        self.fingerprint_k = fingerprint_k
        
        # Pending candidates (need multiple verifications)
        self.pending_candidates: Dict[str, Dict[str, Any]] = {}
    
    def submit_verified_job(
        self,
        submission: GradientSubmission,
        gradient_tensor: 'torch.Tensor',
        chunk_hash: str,
        seed: int,
    ) -> bool:
        """
        Submit a verified job as potential vault candidate.
        
        Args:
            submission: Verified gradient submission
            gradient_tensor: The gradient tensor
            chunk_hash: Hash of the input chunk
            seed: Deterministic seed used
        
        Returns:
            True if added to vault, False if pending
        """
        if submission.verification_status != VerificationStatus.PASSED:
            logger.warning("Cannot add unverified submission to vault")
            return False
        
        candidate_id = f"{chunk_hash}_{seed}"
        
        # Check if already in pending
        if candidate_id in self.pending_candidates:
            self.pending_candidates[candidate_id]['verification_count'] += 1
            
            # Check if ready for vault
            if self.pending_candidates[candidate_id]['verification_count'] >= self.min_verifications:
                return self._promote_to_vault(candidate_id, gradient_tensor)
            return False
        
        # Add as new candidate
        fingerprint = FingerprintExtractor.extract_top_k(
            gradient_tensor,
            k=self.fingerprint_k,
            layer_name="aggregated",
        )
        
        self.pending_candidates[candidate_id] = {
            'chunk_hash': chunk_hash,
            'gradient_hash': submission.gradient_hash,
            'fingerprint': fingerprint,
            'seed': seed,
            'verification_count': 1,
            'first_verified_at': int(time.time()),
            'last_verified_at': int(time.time()),
        }
        
        logger.debug(f"Added candidate {candidate_id} (1/{self.min_verifications})")
        return False

    def _promote_to_vault(
        self,
        candidate_id: str,
        gradient_tensor: 'torch.Tensor',
    ) -> bool:
        """
        Promote candidate to Genesis Vault.
        
        Args:
            candidate_id: Candidate ID
            gradient_tensor: Gradient tensor for fingerprint
        
        Returns:
            True if promoted successfully
        """
        if len(self.vault.entries) >= self.max_vault_size:
            # Prune old entries first
            self.prune_old_entries(keep_count=self.max_vault_size - 100)
        
        candidate = self.pending_candidates.get(candidate_id)
        if not candidate:
            return False
        
        # Create vault entry
        entry = GenesisVaultEntry(
            entry_id=f"por_{candidate_id}",
            chunk_hash=candidate['chunk_hash'],
            expected_gradient_hash=candidate['gradient_hash'],
            expected_fingerprint=candidate['fingerprint'].values,
            seed=candidate['seed'],
            created_at=int(time.time()),
            verified_count=candidate['verification_count'],
        )
        
        self.vault.add_entry(entry)
        del self.pending_candidates[candidate_id]
        
        logger.info(f"Promoted {candidate_id} to Genesis Vault (total: {len(self.vault.entries)})")
        return True
    
    def prune_old_entries(self, keep_count: int = 5000) -> int:
        """
        Prune old vault entries to maintain size limit.
        
        Keeps entries with highest verification counts.
        
        Args:
            keep_count: Number of entries to keep
        
        Returns:
            Number of entries pruned
        """
        if len(self.vault.entries) <= keep_count:
            return 0
        
        # Sort by verification count (descending)
        sorted_entries = sorted(
            self.vault.entries.items(),
            key=lambda x: x[1].verified_count,
            reverse=True,
        )
        
        # Keep top entries
        entries_to_keep = dict(sorted_entries[:keep_count])
        pruned_count = len(self.vault.entries) - len(entries_to_keep)
        
        self.vault.entries = entries_to_keep
        
        logger.info(f"Pruned {pruned_count} old vault entries")
        return pruned_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get Proof of Reuse statistics."""
        vault_stats = self.vault.get_vault_statistics()
        
        return {
            **vault_stats,
            "pending_candidates": len(self.pending_candidates),
            "min_verifications_required": self.min_verifications,
            "max_vault_size": self.max_vault_size,
        }
    
    def cleanup_stale_candidates(self, max_age_seconds: int = 86400) -> int:
        """
        Remove candidates that haven't been verified recently.
        
        Args:
            max_age_seconds: Maximum age in seconds (default 24h)
        
        Returns:
            Number of candidates removed
        """
        now = int(time.time())
        stale_ids = []
        
        for cid, candidate in self.pending_candidates.items():
            age = now - candidate['last_verified_at']
            if age > max_age_seconds:
                stale_ids.append(cid)
        
        for cid in stale_ids:
            del self.pending_candidates[cid]
        
        if stale_ids:
            logger.info(f"Cleaned up {len(stale_ids)} stale candidates")
        
        return len(stale_ids)


class VaultBootstrapper:
    """
    Bootstrap Genesis Vault with initial entries.
    
    Used for initial network setup before Proof of Reuse
    can generate enough entries organically.
    """
    
    @staticmethod
    def generate_bootstrap_entries(
        chunks: List[Dict[str, Any]],
        compute_gradient_fn,
        seed: int,
        fingerprint_k: int = 100,
    ) -> List[GenesisVaultEntry]:
        """
        Generate bootstrap vault entries from chunks.
        
        Args:
            chunks: List of chunk data dicts
            compute_gradient_fn: Function to compute gradient
            seed: Deterministic seed
            fingerprint_k: Top-K for fingerprint
        
        Returns:
            List of GenesisVaultEntry
        """
        import torch
        
        entries = []
        
        for i, chunk in enumerate(chunks):
            # Compute gradient deterministically
            gradient = compute_gradient_fn(chunk, seed)
            
            # Compute hash
            gradient_bytes = gradient.detach().cpu().numpy().tobytes()
            gradient_hash = hashlib.sha256(gradient_bytes).hexdigest()
            
            # Extract fingerprint
            fingerprint = FingerprintExtractor.extract_top_k(
                gradient, k=fingerprint_k, layer_name="bootstrap"
            )
            
            # Compute chunk hash
            chunk_bytes = json.dumps(chunk, sort_keys=True).encode()
            chunk_hash = hashlib.sha256(chunk_bytes).hexdigest()
            
            entry = GenesisVaultEntry(
                entry_id=f"bootstrap_{i}",
                chunk_hash=chunk_hash,
                expected_gradient_hash=gradient_hash,
                expected_fingerprint=fingerprint.values,
                seed=seed,
                created_at=int(time.time()),
            )
            entries.append(entry)
        
        logger.info(f"Generated {len(entries)} bootstrap vault entries")
        return entries
    
    @staticmethod
    def save_bootstrap_vault(
        entries: List[GenesisVaultEntry],
        output_path: str,
    ) -> None:
        """Save bootstrap entries to file."""
        data = [asdict(e) for e in entries]
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved bootstrap vault to {output_path}")

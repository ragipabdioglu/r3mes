#!/usr/bin/env python3
"""
R3MES Trap Job System

Implements the trap job mechanism for fraud detection:
- Genesis Vault management (pre-computed trap results)
- Blind delivery mixing (inject traps without miner knowledge)
- Trap verification (compare miner results against vault)
"""

import json
import random
import hashlib
import time
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import logging

from core.constants import (
    TRAP_JOB_RATIO,
    MIN_TRAP_JOBS_PER_BATCH,
    GENESIS_VAULT_MIN_ENTRIES,
    TRAP_VERIFICATION_THRESHOLD,
)
from core.types import (
    GenesisVaultEntry,
    TaskChunk,
    TrapVerificationResult,
    BlindDeliveryBatch,
    GradientFingerprint,
    TaskStatus,
)
from core.exceptions import (
    TrapJobError,
    TrapVerificationFailed,
    GenesisVaultError,
)
from core.similarity import CosineSimilarityCalculator, FingerprintExtractor

logger = logging.getLogger(__name__)


class GenesisVaultManager:
    """
    Manages the Genesis Vault - pre-computed trap job results.
    
    The Genesis Vault contains entries with:
    - Known input chunks
    - Pre-computed expected gradient hashes
    - Top-K fingerprints for similarity verification
    
    These are used to verify miner honesty without re-computation.
    """
    
    def __init__(self, vault_path: Optional[str] = None):
        """
        Initialize Genesis Vault manager.
        
        Args:
            vault_path: Path to vault JSON file (optional)
        """
        self.entries: Dict[str, GenesisVaultEntry] = {}
        self.vault_path = vault_path
        
        if vault_path:
            self.load_from_file(vault_path)
    
    def load_from_file(self, path: str) -> int:
        """
        Load vault entries from JSON file.
        
        Args:
            path: Path to JSON file
        
        Returns:
            Number of entries loaded
        """
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            for entry_data in data:
                entry = GenesisVaultEntry(
                    entry_id=entry_data['entry_id'],
                    chunk_hash=entry_data['chunk_hash'],
                    expected_gradient_hash=entry_data['expected_gradient_hash'],
                    expected_fingerprint=entry_data['expected_fingerprint'],
                    seed=entry_data['seed'],
                    created_at=entry_data['created_at'],
                    verified_count=entry_data.get('verified_count', 0),
                )
                self.entries[entry.entry_id] = entry
            
            logger.info(f"Loaded {len(self.entries)} vault entries from {path}")
            return len(self.entries)
        except FileNotFoundError:
            logger.warning(f"Vault file not found: {path}")
            return 0
        except Exception as e:
            raise GenesisVaultError(f"Failed to load vault: {e}")

    def save_to_file(self, path: Optional[str] = None) -> None:
        """Save vault entries to JSON file."""
        save_path = path or self.vault_path
        if not save_path:
            raise GenesisVaultError("No vault path specified")
        
        data = [asdict(entry) for entry in self.entries.values()]
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(self.entries)} vault entries to {save_path}")
    
    def add_entry(self, entry: GenesisVaultEntry) -> None:
        """Add entry to vault."""
        self.entries[entry.entry_id] = entry
    
    def get_entry(self, entry_id: str) -> Optional[GenesisVaultEntry]:
        """Get entry by ID."""
        return self.entries.get(entry_id)
    
    def get_random_trap(self) -> Optional[GenesisVaultEntry]:
        """Get random trap entry for injection."""
        if not self.entries:
            return None
        return random.choice(list(self.entries.values()))
    
    def get_random_traps(self, count: int) -> List[GenesisVaultEntry]:
        """Get multiple random trap entries."""
        if not self.entries:
            return []
        available = list(self.entries.values())
        count = min(count, len(available))
        return random.sample(available, count)
    
    def verify_against_vault(
        self,
        entry_id: str,
        miner_gradient_hash: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify miner result against vault entry (hash comparison).
        
        Args:
            entry_id: Vault entry ID
            miner_gradient_hash: Miner's submitted gradient hash
        
        Returns:
            (is_valid, reason)
        """
        entry = self.get_entry(entry_id)
        if not entry:
            return False, f"Entry {entry_id} not found in vault"
        
        if miner_gradient_hash == entry.expected_gradient_hash:
            # Update verification count
            entry.verified_count += 1
            entry.last_verified_at = int(time.time())
            return True, None
        
        return False, "Hash mismatch"
    
    def get_vault_statistics(self) -> Dict[str, Any]:
        """Get vault statistics."""
        if not self.entries:
            return {"total_entries": 0}
        
        total_verifications = sum(e.verified_count for e in self.entries.values())
        return {
            "total_entries": len(self.entries),
            "total_verifications": total_verifications,
            "avg_verifications_per_entry": total_verifications / len(self.entries),
        }


class BlindDeliveryMixer:
    """
    Mixes trap jobs with real tasks for blind delivery.
    
    Miners receive a batch of tasks without knowing which are traps.
    The is_trap flag is stripped before delivery.
    """
    
    def __init__(
        self,
        vault_manager: GenesisVaultManager,
        trap_ratio: float = TRAP_JOB_RATIO,
    ):
        """
        Initialize blind delivery mixer.
        
        Args:
            vault_manager: Genesis Vault manager
            trap_ratio: Ratio of trap jobs (default 10%)
        """
        self.vault = vault_manager
        self.trap_ratio = trap_ratio
    
    def mix_chunks(
        self,
        real_chunks: List[TaskChunk],
        pool_id: int,
    ) -> BlindDeliveryBatch:
        """
        Mix real chunks with trap chunks.
        
        Args:
            real_chunks: List of real task chunks
            pool_id: Task pool ID
        
        Returns:
            BlindDeliveryBatch with mixed chunks
        """
        # Calculate trap count
        trap_count = max(
            MIN_TRAP_JOBS_PER_BATCH,
            int(len(real_chunks) * self.trap_ratio)
        )
        
        # Get trap entries from vault
        trap_entries = self.vault.get_random_traps(trap_count)
        
        # Convert trap entries to TaskChunks
        trap_chunks = []
        for i, entry in enumerate(trap_entries):
            trap_chunk = TaskChunk(
                chunk_id=-1000 - i,  # Negative IDs for traps
                pool_id=pool_id,
                data_hash=entry.chunk_hash,
                shard_id=0,
                is_trap=True,
                status=TaskStatus.AVAILABLE,
            )
            trap_chunks.append(trap_chunk)
        
        # Mix and shuffle
        all_chunks = real_chunks + trap_chunks
        random.shuffle(all_chunks)
        
        # Sanitize for miner (remove is_trap flag)
        sanitized = [self._sanitize_chunk(c) for c in all_chunks]
        
        batch_id = hashlib.sha256(
            f"{pool_id}:{time.time()}".encode()
        ).hexdigest()[:16]
        
        return BlindDeliveryBatch(
            batch_id=batch_id,
            real_chunks=real_chunks,
            trap_chunks=trap_chunks,
            mixed_chunks=sanitized,
            trap_ratio=len(trap_chunks) / len(all_chunks) if all_chunks else 0,
        )

    def _sanitize_chunk(self, chunk: TaskChunk) -> TaskChunk:
        """
        Sanitize chunk for miner delivery.
        
        Removes is_trap flag so miner cannot distinguish traps.
        """
        return TaskChunk(
            chunk_id=chunk.chunk_id,
            pool_id=chunk.pool_id,
            data_hash=chunk.data_hash,
            shard_id=chunk.shard_id,
            token_count=chunk.token_count,
            is_trap=False,  # Always False for miner
            status=chunk.status,
            claimed_by=chunk.claimed_by,
            claimed_at=chunk.claimed_at,
        )
    
    def is_trap_chunk(self, chunk_id: int) -> bool:
        """Check if chunk ID is a trap (negative ID)."""
        return chunk_id < 0


class TrapJobVerifier:
    """
    Verifies miner results against trap job expectations.
    
    Uses both hash comparison and similarity verification.
    """
    
    def __init__(
        self,
        vault_manager: GenesisVaultManager,
        similarity_threshold: float = TRAP_VERIFICATION_THRESHOLD,
    ):
        """
        Initialize trap job verifier.
        
        Args:
            vault_manager: Genesis Vault manager
            similarity_threshold: Similarity threshold for verification
        """
        self.vault = vault_manager
        self.threshold = similarity_threshold
        self.similarity_calc = CosineSimilarityCalculator()
    
    def verify_trap_result(
        self,
        entry_id: str,
        miner_gradient_hash: str,
        miner_address: str,
    ) -> TrapVerificationResult:
        """
        Verify miner's trap job result.
        
        Args:
            entry_id: Vault entry ID
            miner_gradient_hash: Miner's gradient hash
            miner_address: Miner's address
        
        Returns:
            TrapVerificationResult
        """
        entry = self.vault.get_entry(entry_id)
        if not entry:
            raise GenesisVaultError(f"Entry {entry_id} not found", entry_id)
        
        # Hash comparison
        is_valid = miner_gradient_hash == entry.expected_gradient_hash
        
        # Calculate similarity score (1.0 if hash matches, 0.0 otherwise)
        similarity = 1.0 if is_valid else 0.0
        
        result = TrapVerificationResult(
            chunk_id=int(entry_id.split('_')[-1]) if '_' in entry_id else 0,
            is_valid=is_valid,
            similarity_score=similarity,
            expected_hash=entry.expected_gradient_hash,
            actual_hash=miner_gradient_hash,
            miner_address=miner_address,
            verified_at=int(time.time()),
        )
        
        if not is_valid:
            logger.warning(
                f"Trap verification FAILED for miner {miner_address[:16]}... "
                f"(entry={entry_id}, similarity={similarity:.4f})"
            )
        
        return result
    
    def verify_with_fingerprint(
        self,
        entry_id: str,
        miner_gradient: 'torch.Tensor',
        miner_address: str,
    ) -> TrapVerificationResult:
        """
        Verify using fingerprint similarity (more lenient).
        
        Args:
            entry_id: Vault entry ID
            miner_gradient: Miner's gradient tensor
            miner_address: Miner's address
        
        Returns:
            TrapVerificationResult
        """
        import torch
        
        entry = self.vault.get_entry(entry_id)
        if not entry:
            raise GenesisVaultError(f"Entry {entry_id} not found", entry_id)
        
        # Create fingerprint from entry
        fingerprint = GradientFingerprint(
            indices=list(range(len(entry.expected_fingerprint))),
            values=entry.expected_fingerprint,
            layer_name="trap",
            k=len(entry.expected_fingerprint),
        )
        
        # Calculate masked similarity
        similarity = self.similarity_calc.calculate_masked(
            fingerprint, miner_gradient
        )
        
        is_valid = similarity >= self.threshold
        
        result = TrapVerificationResult(
            chunk_id=int(entry_id.split('_')[-1]) if '_' in entry_id else 0,
            is_valid=is_valid,
            similarity_score=similarity,
            expected_hash=entry.expected_gradient_hash,
            actual_hash="fingerprint_comparison",
            miner_address=miner_address,
            verified_at=int(time.time()),
        )
        
        if not is_valid:
            logger.warning(
                f"Trap fingerprint verification FAILED for miner {miner_address[:16]}... "
                f"(entry={entry_id}, similarity={similarity:.4f}, threshold={self.threshold})"
            )
        
        return result
    
    def verify_or_raise(
        self,
        entry_id: str,
        miner_gradient_hash: str,
        miner_address: str,
    ) -> TrapVerificationResult:
        """
        Verify and raise exception on failure.
        
        Raises:
            TrapVerificationFailed: If verification fails
        """
        result = self.verify_trap_result(entry_id, miner_gradient_hash, miner_address)
        
        if not result.is_valid:
            raise TrapVerificationFailed(
                chunk_id=result.chunk_id,
                miner_address=miner_address,
                similarity_score=result.similarity_score,
            )
        
        return result

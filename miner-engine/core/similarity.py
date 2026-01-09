#!/usr/bin/env python3
"""
R3MES Cosine Similarity Module

Implements gradient similarity comparison for verification:
- Cosine similarity calculation
- Top-K fingerprint extraction
- Masked similarity for trap verification
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from core.constants import (
    COSINE_SIMILARITY_THRESHOLD,
    TOP_K_FINGERPRINT_SIZE,
    TRAP_VERIFICATION_THRESHOLD,
)
from core.types import GradientFingerprint
from core.exceptions import SimilarityThresholdError


@dataclass
class SimilarityResult:
    """Result of similarity comparison."""
    score: float
    is_valid: bool
    threshold: float
    method: str
    details: Optional[Dict] = None


class CosineSimilarityCalculator:
    """
    Cosine similarity calculator for gradient verification.
    
    Implements the similarity comparison described in documentation:
    - Standard cosine similarity for full gradient comparison
    - Masked cosine similarity for trap job verification
    - Top-K fingerprint extraction for efficient comparison
    """
    
    @staticmethod
    def calculate(
        vec1: torch.Tensor,
        vec2: torch.Tensor,
        eps: float = 1e-8,
    ) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            eps: Small value to avoid division by zero
        
        Returns:
            Cosine similarity score in [-1, 1]
        """
        # Flatten tensors
        v1 = vec1.flatten().float()
        v2 = vec2.flatten().float()
        
        # Ensure same size
        if v1.shape != v2.shape:
            raise ValueError(f"Shape mismatch: {v1.shape} vs {v2.shape}")
        
        # Calculate cosine similarity
        dot_product = torch.dot(v1, v2)
        norm1 = torch.norm(v1)
        norm2 = torch.norm(v2)
        
        similarity = dot_product / (norm1 * norm2 + eps)
        
        return similarity.item()

    @staticmethod
    def calculate_masked(
        expected_fingerprint: GradientFingerprint,
        miner_gradient: torch.Tensor,
    ) -> float:
        """
        Calculate masked cosine similarity for trap verification.
        
        Only compares values at the fingerprint indices (top-K positions).
        This is used for blind trap verification where we compare
        miner's gradient against pre-computed expected values.
        
        Args:
            expected_fingerprint: Pre-computed fingerprint from Genesis Vault
            miner_gradient: Miner's submitted gradient
        
        Returns:
            Masked cosine similarity score
        """
        # Flatten miner gradient
        miner_flat = miner_gradient.flatten().float()
        
        # Extract values at fingerprint indices
        indices = torch.tensor(expected_fingerprint.indices, dtype=torch.long)
        
        # Ensure indices are within bounds
        max_idx = miner_flat.shape[0]
        valid_mask = indices < max_idx
        valid_indices = indices[valid_mask]
        
        if len(valid_indices) == 0:
            return 0.0
        
        # Get miner values at fingerprint positions
        miner_values = miner_flat[valid_indices]
        
        # Get expected values (only valid ones)
        expected_values = torch.tensor(
            [expected_fingerprint.values[i] for i, v in enumerate(valid_mask) if v],
            dtype=torch.float32
        )
        
        # Calculate cosine similarity
        dot_product = torch.dot(miner_values, expected_values)
        norm1 = torch.norm(miner_values)
        norm2 = torch.norm(expected_values)
        
        eps = 1e-8
        similarity = dot_product / (norm1 * norm2 + eps)
        
        return similarity.item()
    
    @staticmethod
    def calculate_batch(
        gradients1: Dict[str, torch.Tensor],
        gradients2: Dict[str, torch.Tensor],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate average cosine similarity across all gradient layers.
        
        Args:
            gradients1: First gradient dictionary
            gradients2: Second gradient dictionary
        
        Returns:
            (average_similarity, per_layer_similarities)
        """
        per_layer = {}
        total_sim = 0.0
        count = 0
        
        for name in gradients1.keys():
            if name in gradients2:
                sim = CosineSimilarityCalculator.calculate(
                    gradients1[name],
                    gradients2[name]
                )
                per_layer[name] = sim
                total_sim += sim
                count += 1
        
        avg_sim = total_sim / count if count > 0 else 0.0
        return avg_sim, per_layer


class FingerprintExtractor:
    """
    Extract top-K fingerprint from gradients for efficient comparison.
    
    The fingerprint contains the indices and values of the K largest
    absolute values in the gradient, enabling efficient similarity
    comparison without storing full gradients.
    """
    
    @staticmethod
    def extract_top_k(
        gradient: torch.Tensor,
        k: int = TOP_K_FINGERPRINT_SIZE,
        layer_name: str = "",
    ) -> GradientFingerprint:
        """
        Extract top-K fingerprint from gradient.
        
        Args:
            gradient: Gradient tensor
            k: Number of top values to extract
            layer_name: Name of the layer
        
        Returns:
            GradientFingerprint with indices and values
        """
        # Flatten gradient
        flat = gradient.flatten().float()
        
        # Get absolute values for ranking
        abs_values = torch.abs(flat)
        
        # Limit k to tensor size
        k = min(k, flat.shape[0])
        
        # Get top-k indices by absolute value
        _, indices = torch.topk(abs_values, k)
        
        # Get actual values at those indices
        values = flat[indices]
        
        return GradientFingerprint(
            indices=indices.tolist(),
            values=values.tolist(),
            layer_name=layer_name,
            k=k,
        )

    @staticmethod
    def extract_from_dict(
        gradients: Dict[str, torch.Tensor],
        k: int = TOP_K_FINGERPRINT_SIZE,
    ) -> Dict[str, GradientFingerprint]:
        """
        Extract fingerprints from all gradients in dictionary.
        
        Args:
            gradients: Dictionary of gradients
            k: Number of top values per layer
        
        Returns:
            Dictionary of fingerprints
        """
        fingerprints = {}
        for name, grad in gradients.items():
            fingerprints[name] = FingerprintExtractor.extract_top_k(
                grad, k=k, layer_name=name
            )
        return fingerprints
    
    @staticmethod
    def compare_fingerprints(
        fp1: GradientFingerprint,
        fp2: GradientFingerprint,
    ) -> float:
        """
        Compare two fingerprints using cosine similarity.
        
        Args:
            fp1: First fingerprint
            fp2: Second fingerprint
        
        Returns:
            Similarity score
        """
        # Find common indices
        set1 = set(fp1.indices)
        set2 = set(fp2.indices)
        common = set1.intersection(set2)
        
        if len(common) == 0:
            return 0.0
        
        # Build value vectors for common indices
        idx_to_val1 = dict(zip(fp1.indices, fp1.values))
        idx_to_val2 = dict(zip(fp2.indices, fp2.values))
        
        vals1 = [idx_to_val1[i] for i in common]
        vals2 = [idx_to_val2[i] for i in common]
        
        v1 = torch.tensor(vals1, dtype=torch.float32)
        v2 = torch.tensor(vals2, dtype=torch.float32)
        
        return CosineSimilarityCalculator.calculate(v1, v2)


class SimilarityVerifier:
    """
    High-level similarity verification for gradient validation.
    """
    
    def __init__(
        self,
        threshold: float = COSINE_SIMILARITY_THRESHOLD,
        trap_threshold: float = TRAP_VERIFICATION_THRESHOLD,
    ):
        """
        Initialize similarity verifier.
        
        Args:
            threshold: Standard verification threshold
            trap_threshold: Trap job verification threshold (stricter)
        """
        self.threshold = threshold
        self.trap_threshold = trap_threshold
        self.calculator = CosineSimilarityCalculator()
        self.extractor = FingerprintExtractor()
    
    def verify_gradients(
        self,
        expected: Dict[str, torch.Tensor],
        actual: Dict[str, torch.Tensor],
        strict: bool = False,
    ) -> SimilarityResult:
        """
        Verify gradient similarity.
        
        Args:
            expected: Expected gradients
            actual: Actual gradients from miner
            strict: Use stricter trap threshold
        
        Returns:
            SimilarityResult
        """
        threshold = self.trap_threshold if strict else self.threshold
        
        avg_sim, per_layer = self.calculator.calculate_batch(expected, actual)
        
        is_valid = avg_sim >= threshold
        
        return SimilarityResult(
            score=avg_sim,
            is_valid=is_valid,
            threshold=threshold,
            method="cosine_batch",
            details={"per_layer": per_layer},
        )
    
    def verify_with_fingerprint(
        self,
        fingerprint: GradientFingerprint,
        gradient: torch.Tensor,
        strict: bool = True,
    ) -> SimilarityResult:
        """
        Verify gradient against pre-computed fingerprint.
        
        Used for trap job verification.
        
        Args:
            fingerprint: Expected fingerprint from Genesis Vault
            gradient: Miner's gradient
            strict: Use trap threshold (default True for traps)
        
        Returns:
            SimilarityResult
        """
        threshold = self.trap_threshold if strict else self.threshold
        
        score = self.calculator.calculate_masked(fingerprint, gradient)
        is_valid = score >= threshold
        
        return SimilarityResult(
            score=score,
            is_valid=is_valid,
            threshold=threshold,
            method="masked_cosine",
        )
    
    def verify_or_raise(
        self,
        expected: Dict[str, torch.Tensor],
        actual: Dict[str, torch.Tensor],
        chunk_id: Optional[int] = None,
    ) -> SimilarityResult:
        """
        Verify gradients and raise exception if below threshold.
        
        Args:
            expected: Expected gradients
            actual: Actual gradients
            chunk_id: Optional chunk ID for error context
        
        Returns:
            SimilarityResult if valid
        
        Raises:
            SimilarityThresholdError: If similarity below threshold
        """
        result = self.verify_gradients(expected, actual)
        
        if not result.is_valid:
            raise SimilarityThresholdError(
                similarity_score=result.score,
                threshold=result.threshold,
                chunk_id=chunk_id,
            )
        
        return result

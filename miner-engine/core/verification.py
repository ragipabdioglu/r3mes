#!/usr/bin/env python3
"""
Deterministic Gradient Hash Verification

Implements:
- Exact gradient hash matching with no tolerance thresholds
- Deterministic quantization-aware training (QAT) verification
- CPU Iron Sandbox fallback for hash mismatch disputes
- GPU architecture-aware verification
"""

import hashlib
import torch
import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from utils.gpu_detection import GPUArchitectureDetector


class VerificationResult(Enum):
    """Verification result status."""
    VALID = "valid"
    INVALID = "invalid"
    REQUIRES_CPU_VERIFICATION = "requires_cpu_verification"
    CROSS_ARCHITECTURE = "cross_architecture"
    HASH_MISMATCH = "hash_mismatch"


@dataclass
class GradientVerification:
    """Gradient verification result."""
    is_valid: bool
    result: VerificationResult
    reason: str
    requires_cpu_verification: bool = False
    hash_match: bool = False
    cross_architecture: bool = False
    miner_gpu_architecture: Optional[str] = None
    validator_gpu_architecture: Optional[str] = None


class DeterministicHashVerifier:
    """
    Deterministic gradient hash verifier.
    
    Implements exact hash matching with no tolerance thresholds.
    Uses GPU architecture-aware verification with CPU fallback.
    """
    
    @staticmethod
    def compute_deterministic_hash(
        gradients: Dict[str, torch.Tensor],
        precision: str = "float32",
    ) -> str:
        """
        Compute deterministic hash of gradients.
        
        Args:
            gradients: Dictionary of gradients
            precision: Precision for hashing ("float32", "float16", "int8")
        
        Returns:
            Hex string hash
        """
        # Sort gradients by name for deterministic hashing
        sorted_names = sorted(gradients.keys())
        hash_input = b""
        
        for name in sorted_names:
            grad = gradients[name]
            # Convert to numpy for consistent hashing
            grad_np = grad.detach().cpu().numpy()
            
            # Apply precision quantization
            if precision == "float32":
                grad_bytes = grad_np.astype(np.float32).tobytes()
            elif precision == "float16":
                grad_bytes = grad_np.astype(np.float16).tobytes()
            elif precision == "int8":
                # Quantize to int8 for QAT
                grad_quantized = np.clip(
                    np.round(grad_np * 127.0),
                    -128, 127
                ).astype(np.int8)
                grad_bytes = grad_quantized.tobytes()
            else:
                raise ValueError(f"Unknown precision: {precision}")
            
            hash_input += name.encode() + grad_bytes
        
        return hashlib.sha256(hash_input).hexdigest()
    
    @staticmethod
    def verify_hash_match(
        miner_hash: str,
        validator_hash: str,
    ) -> bool:
        """
        Verify exact hash match (no tolerance).
        
        Args:
            miner_hash: Miner's gradient hash
            validator_hash: Validator's gradient hash
        
        Returns:
            True if hashes match exactly
        """
        return miner_hash == validator_hash


class GPUArchitectureVerifier:
    """
    GPU architecture-aware gradient verification.
    
    Handles floating-point precision differences between GPU architectures
    (Ampere vs Ada, etc.) by requiring CPU verification for cross-architecture
    comparisons.
    """
    
    @staticmethod
    def verify_with_architecture(
        miner_hash: str,
        validator_hash: str,
        miner_gpu_architecture: str,
        validator_gpu_architecture: str,
    ) -> GradientVerification:
        """
        Verify gradient hash with GPU architecture awareness.
        
        Args:
            miner_hash: Miner's gradient hash
            validator_hash: Validator's gradient hash
            miner_gpu_architecture: Miner's GPU architecture (e.g., "Ampere", "Ada")
            validator_gpu_architecture: Validator's GPU architecture
        
        Returns:
            GradientVerification result
        """
        # Step 1: Check if same GPU architecture - direct hash comparison
        if miner_gpu_architecture == validator_gpu_architecture:
            # Same architecture: Require exact hash match (bit-exact)
            if miner_hash == validator_hash:
                return GradientVerification(
                    is_valid=True,
                    result=VerificationResult.VALID,
                    reason="exact_hash_match_same_architecture",
                    hash_match=True,
                    miner_gpu_architecture=miner_gpu_architecture,
                    validator_gpu_architecture=validator_gpu_architecture,
                )
            # Same architecture but hash mismatch - likely fraud or non-deterministic execution
            return GradientVerification(
                is_valid=False,
                result=VerificationResult.HASH_MISMATCH,
                reason="hash_mismatch_same_architecture",
                requires_cpu_verification=True,  # MANDATORY CPU fallback
                hash_match=False,
                miner_gpu_architecture=miner_gpu_architecture,
                validator_gpu_architecture=validator_gpu_architecture,
            )
        
        # Step 2: Different GPU architectures - MANDATORY CPU verification
        # Floating-point differences between architectures are expected
        # CPU Iron Sandbox provides bit-exact resolution
        return GradientVerification(
            is_valid=False,
            result=VerificationResult.CROSS_ARCHITECTURE,
            reason="cross_architecture_verification_required",
            requires_cpu_verification=True,  # MANDATORY for cross-architecture disputes
            hash_match=False,
            cross_architecture=True,
            miner_gpu_architecture=miner_gpu_architecture,
            validator_gpu_architecture=validator_gpu_architecture,
        )
    
    @staticmethod
    def should_compare_directly(
        miner_gpu_architecture: str,
        validator_gpu_architecture: str,
    ) -> bool:
        """
        Check if architectures should be compared directly.
        
        Args:
            miner_gpu_architecture: Miner's GPU architecture
            validator_gpu_architecture: Validator's GPU architecture
        
        Returns:
            True if same architecture (can compare directly)
        """
        return miner_gpu_architecture == validator_gpu_architecture


class CPUIronSandbox:
    """
    CPU Iron Sandbox for bit-exact gradient verification.
    
    Used as fallback when:
    - Hash mismatch occurs (same architecture)
    - Cross-architecture verification required
    - Dispute resolution needed
    """
    
    @staticmethod
    def verify_on_cpu(
        gradients: Dict[str, torch.Tensor],
        expected_hash: str,
        seed: int,
    ) -> Tuple[bool, str]:
        """
        Verify gradients on CPU for bit-exact results.
        
        Args:
            gradients: Dictionary of gradients
            expected_hash: Expected gradient hash
            seed: Deterministic seed for verification
        
        Returns:
            (is_valid, computed_hash) tuple
        """
        # Force CPU execution
        cpu_gradients = {}
        for name, grad in gradients.items():
            cpu_gradients[name] = grad.detach().cpu()
        
        # Compute hash on CPU with deterministic seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        computed_hash = DeterministicHashVerifier.compute_deterministic_hash(
            cpu_gradients,
            precision="float32",
        )
        
        is_valid = computed_hash == expected_hash
        
        return is_valid, computed_hash
    
    @staticmethod
    def create_verification_request(
        gradient_ipfs_hash: str,
        expected_hash: str,
        seed: int,
        challenge_id: str,
    ) -> Dict[str, Any]:
        """
        Create CPU verification request for dispute resolution.
        
        Args:
            gradient_ipfs_hash: IPFS hash of disputed gradient
            expected_hash: Expected gradient hash
            seed: Deterministic seed
            challenge_id: Challenge ID
        
        Returns:
            Verification request dictionary
        """
        return {
            "challenge_id": challenge_id,
            "disputed_gradient": gradient_ipfs_hash,
            "execution_mode": "CPU",  # MUST be CPU for disputes
            "expected_result": expected_hash,
            "seed": seed,
            "requires_deterministic_execution": True,
        }


class QuantizationAwareVerifier:
    """
    Quantization-Aware Training (QAT) verification.
    
    Verifies gradients with quantization-aware hashing for
    BitNet 1.58-bit model compatibility.
    """
    
    @staticmethod
    def compute_quantized_hash(
        gradients: Dict[str, torch.Tensor],
        quantization_bits: int = 1,
    ) -> str:
        """
        Compute hash with quantization-aware quantization.
        
        Args:
            gradients: Dictionary of gradients
            quantization_bits: Number of quantization bits (1 for BitNet)
        
        Returns:
            Hex string hash
        """
        sorted_names = sorted(gradients.keys())
        hash_input = b""
        
        for name in sorted_names:
            grad = gradients[name]
            grad_np = grad.detach().cpu().numpy()
            
            # Quantize to specified bits
            if quantization_bits == 1:
                # BitNet 1.58-bit: {-1, 0, +1}
                grad_quantized = np.sign(grad_np) * np.clip(
                    np.abs(grad_np),
                    0, 1
                )
                # Convert to int8 representation
                grad_bytes = grad_quantized.astype(np.int8).tobytes()
            else:
                # General quantization
                scale = (2 ** (quantization_bits - 1)) - 1
                grad_quantized = np.clip(
                    np.round(grad_np * scale),
                    -scale, scale
                ).astype(np.int8)
                grad_bytes = grad_quantized.tobytes()
            
            hash_input += name.encode() + grad_bytes
        
        return hashlib.sha256(hash_input).hexdigest()
    
    @staticmethod
    def verify_quantized_match(
        miner_hash: str,
        validator_hash: str,
    ) -> bool:
        """
        Verify quantized hash match.
        
        Args:
            miner_hash: Miner's quantized hash
            validator_hash: Validator's quantized hash
        
        Returns:
            True if hashes match
        """
        return miner_hash == validator_hash


class GradientHashVerifier:
    """
    Main gradient hash verifier.
    
    Coordinates all verification methods:
    - Exact hash matching
    - GPU architecture-aware verification
    - CPU Iron Sandbox fallback
    - Quantization-aware verification
    """
    
    def __init__(self):
        """Initialize gradient hash verifier."""
        self.gpu_detector = GPUArchitectureDetector()
    
    def verify_gradient(
        self,
        miner_hash: str,
        validator_hash: str,
        miner_gpu_architecture: Optional[str] = None,
        validator_gpu_architecture: Optional[str] = None,
        require_exact_match: bool = True,
    ) -> GradientVerification:
        """
        Verify gradient hash with full verification pipeline.
        
        Args:
            miner_hash: Miner's gradient hash
            validator_hash: Validator's gradient hash
            miner_gpu_architecture: Miner's GPU architecture
            validator_gpu_architecture: Validator's GPU architecture
            require_exact_match: Require exact match (no tolerance)
        
        Returns:
            GradientVerification result
        """
        # Auto-detect GPU architectures if not provided
        if miner_gpu_architecture is None:
            miner_gpu_architecture = self.gpu_detector.get_architecture()
        if validator_gpu_architecture is None:
            validator_gpu_architecture = self.gpu_detector.get_architecture()
        
        # Step 1: Direct hash comparison
        if miner_hash == validator_hash:
            return GradientVerification(
                is_valid=True,
                result=VerificationResult.VALID,
                reason="exact_hash_match",
                hash_match=True,
                miner_gpu_architecture=miner_gpu_architecture,
                validator_gpu_architecture=validator_gpu_architecture,
            )
        
        # Step 2: GPU architecture-aware verification
        if miner_gpu_architecture and validator_gpu_architecture:
            return GPUArchitectureVerifier.verify_with_architecture(
                miner_hash,
                validator_hash,
                miner_gpu_architecture,
                validator_gpu_architecture,
            )
        
        # Step 3: Hash mismatch - requires CPU verification
        return GradientVerification(
            is_valid=False,
            result=VerificationResult.REQUIRES_CPU_VERIFICATION,
            reason="hash_mismatch_requires_cpu_verification",
            requires_cpu_verification=True,
            hash_match=False,
            miner_gpu_architecture=miner_gpu_architecture,
            validator_gpu_architecture=validator_gpu_architecture,
        )
    
    def verify_with_cpu_fallback(
        self,
        gradients: Dict[str, torch.Tensor],
        expected_hash: str,
        seed: int,
    ) -> Tuple[bool, str]:
        """
        Verify gradients with CPU fallback.
        
        Args:
            gradients: Dictionary of gradients
            expected_hash: Expected gradient hash
            seed: Deterministic seed
        
        Returns:
            (is_valid, computed_hash) tuple
        """
        return CPUIronSandbox.verify_on_cpu(gradients, expected_hash, seed)


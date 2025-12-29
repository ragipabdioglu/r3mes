"""
Unit tests for Gradient Verification module.

Tests:
- Deterministic hash computation
- Hash verification
- GPU architecture-aware verification
- CPU Iron Sandbox fallback
- Quantization-aware verification
"""

import pytest
import torch
import numpy as np

from core.verification import (
    DeterministicHashVerifier,
    GPUArchitectureVerifier,
    CPUIronSandbox,
    QuantizationAwareVerifier,
    GradientHashVerifier,
    VerificationResult,
    GradientVerification,
)


class TestDeterministicHashVerifier:
    """Tests for DeterministicHashVerifier."""
    
    def test_compute_hash_float32(self):
        """Test hash computation with float32 precision."""
        gradients = {
            "layer1.weight": torch.randn(10, 10),
            "layer2.weight": torch.randn(5, 10),
        }
        
        hash_result = DeterministicHashVerifier.compute_deterministic_hash(
            gradients, precision="float32"
        )
        
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64  # SHA256 hex length
    
    def test_compute_hash_float16(self):
        """Test hash computation with float16 precision."""
        gradients = {
            "layer1.weight": torch.randn(10, 10),
        }
        
        hash_result = DeterministicHashVerifier.compute_deterministic_hash(
            gradients, precision="float16"
        )
        
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64
    
    def test_compute_hash_int8(self):
        """Test hash computation with int8 precision."""
        gradients = {
            "layer1.weight": torch.randn(10, 10),
        }
        
        hash_result = DeterministicHashVerifier.compute_deterministic_hash(
            gradients, precision="int8"
        )
        
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64
    
    def test_hash_determinism(self):
        """Test that hash computation is deterministic."""
        torch.manual_seed(42)
        gradients = {
            "layer1.weight": torch.randn(10, 10),
            "layer2.weight": torch.randn(5, 10),
        }
        
        hash1 = DeterministicHashVerifier.compute_deterministic_hash(gradients)
        hash2 = DeterministicHashVerifier.compute_deterministic_hash(gradients)
        
        assert hash1 == hash2
    
    def test_hash_order_independence(self):
        """Test that hash is independent of dictionary order."""
        grad1 = torch.randn(10, 10)
        grad2 = torch.randn(5, 10)
        
        gradients1 = {"a": grad1, "b": grad2}
        gradients2 = {"b": grad2, "a": grad1}
        
        hash1 = DeterministicHashVerifier.compute_deterministic_hash(gradients1)
        hash2 = DeterministicHashVerifier.compute_deterministic_hash(gradients2)
        
        assert hash1 == hash2
    
    def test_different_gradients_different_hash(self):
        """Test that different gradients produce different hashes."""
        gradients1 = {"layer1.weight": torch.randn(10, 10)}
        gradients2 = {"layer1.weight": torch.randn(10, 10)}
        
        hash1 = DeterministicHashVerifier.compute_deterministic_hash(gradients1)
        hash2 = DeterministicHashVerifier.compute_deterministic_hash(gradients2)
        
        assert hash1 != hash2
    
    def test_verify_hash_match_true(self):
        """Test hash match verification - matching hashes."""
        hash1 = "abc123"
        hash2 = "abc123"
        
        assert DeterministicHashVerifier.verify_hash_match(hash1, hash2) is True
    
    def test_verify_hash_match_false(self):
        """Test hash match verification - non-matching hashes."""
        hash1 = "abc123"
        hash2 = "def456"
        
        assert DeterministicHashVerifier.verify_hash_match(hash1, hash2) is False
    
    def test_invalid_precision_raises_error(self):
        """Test that invalid precision raises ValueError."""
        gradients = {"layer1.weight": torch.randn(10, 10)}
        
        with pytest.raises(ValueError, match="Unknown precision"):
            DeterministicHashVerifier.compute_deterministic_hash(
                gradients, precision="invalid"
            )


class TestGPUArchitectureVerifier:
    """Tests for GPUArchitectureVerifier."""
    
    def test_same_architecture_matching_hash(self):
        """Test verification with same architecture and matching hash."""
        result = GPUArchitectureVerifier.verify_with_architecture(
            miner_hash="abc123",
            validator_hash="abc123",
            miner_gpu_architecture="Ampere",
            validator_gpu_architecture="Ampere",
        )
        
        assert result.is_valid is True
        assert result.result == VerificationResult.VALID
        assert result.hash_match is True
        assert result.requires_cpu_verification is False
    
    def test_same_architecture_mismatching_hash(self):
        """Test verification with same architecture but mismatching hash."""
        result = GPUArchitectureVerifier.verify_with_architecture(
            miner_hash="abc123",
            validator_hash="def456",
            miner_gpu_architecture="Ampere",
            validator_gpu_architecture="Ampere",
        )
        
        assert result.is_valid is False
        assert result.result == VerificationResult.HASH_MISMATCH
        assert result.hash_match is False
        assert result.requires_cpu_verification is True
    
    def test_different_architecture(self):
        """Test verification with different architectures."""
        result = GPUArchitectureVerifier.verify_with_architecture(
            miner_hash="abc123",
            validator_hash="abc123",
            miner_gpu_architecture="Ampere",
            validator_gpu_architecture="Ada",
        )
        
        assert result.is_valid is False
        assert result.result == VerificationResult.CROSS_ARCHITECTURE
        assert result.cross_architecture is True
        assert result.requires_cpu_verification is True
    
    def test_should_compare_directly_same_arch(self):
        """Test should_compare_directly with same architecture."""
        result = GPUArchitectureVerifier.should_compare_directly("Ampere", "Ampere")
        assert result is True
    
    def test_should_compare_directly_different_arch(self):
        """Test should_compare_directly with different architectures."""
        result = GPUArchitectureVerifier.should_compare_directly("Ampere", "Ada")
        assert result is False


class TestCPUIronSandbox:
    """Tests for CPUIronSandbox."""
    
    def test_verify_on_cpu_matching(self):
        """Test CPU verification with matching hash."""
        torch.manual_seed(42)
        gradients = {"layer1.weight": torch.randn(10, 10)}
        
        # Compute expected hash
        expected_hash = DeterministicHashVerifier.compute_deterministic_hash(
            gradients, precision="float32"
        )
        
        is_valid, computed_hash = CPUIronSandbox.verify_on_cpu(
            gradients, expected_hash, seed=42
        )
        
        assert is_valid is True
        assert computed_hash == expected_hash
    
    def test_verify_on_cpu_mismatching(self):
        """Test CPU verification with mismatching hash."""
        gradients = {"layer1.weight": torch.randn(10, 10)}
        
        is_valid, computed_hash = CPUIronSandbox.verify_on_cpu(
            gradients, "wrong_hash", seed=42
        )
        
        assert is_valid is False
        assert computed_hash != "wrong_hash"
    
    def test_create_verification_request(self):
        """Test creation of verification request."""
        request = CPUIronSandbox.create_verification_request(
            gradient_ipfs_hash="QmTest123",
            expected_hash="abc123",
            seed=42,
            challenge_id="challenge_001",
        )
        
        assert request["challenge_id"] == "challenge_001"
        assert request["disputed_gradient"] == "QmTest123"
        assert request["execution_mode"] == "CPU"
        assert request["expected_result"] == "abc123"
        assert request["seed"] == 42
        assert request["requires_deterministic_execution"] is True


class TestQuantizationAwareVerifier:
    """Tests for QuantizationAwareVerifier."""
    
    def test_compute_quantized_hash_1bit(self):
        """Test quantized hash computation with 1-bit quantization."""
        gradients = {"layer1.weight": torch.randn(10, 10)}
        
        hash_result = QuantizationAwareVerifier.compute_quantized_hash(
            gradients, quantization_bits=1
        )
        
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64
    
    def test_compute_quantized_hash_8bit(self):
        """Test quantized hash computation with 8-bit quantization."""
        gradients = {"layer1.weight": torch.randn(10, 10)}
        
        hash_result = QuantizationAwareVerifier.compute_quantized_hash(
            gradients, quantization_bits=8
        )
        
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64
    
    def test_verify_quantized_match_true(self):
        """Test quantized hash match - matching."""
        assert QuantizationAwareVerifier.verify_quantized_match("abc", "abc") is True
    
    def test_verify_quantized_match_false(self):
        """Test quantized hash match - non-matching."""
        assert QuantizationAwareVerifier.verify_quantized_match("abc", "def") is False


class TestGradientHashVerifier:
    """Tests for main GradientHashVerifier class."""
    
    def test_verify_gradient_matching_hash(self):
        """Test gradient verification with matching hash."""
        verifier = GradientHashVerifier()
        
        result = verifier.verify_gradient(
            miner_hash="abc123",
            validator_hash="abc123",
        )
        
        assert result.is_valid is True
        assert result.result == VerificationResult.VALID
        assert result.hash_match is True
    
    def test_verify_gradient_mismatching_hash(self):
        """Test gradient verification with mismatching hash."""
        verifier = GradientHashVerifier()
        
        result = verifier.verify_gradient(
            miner_hash="abc123",
            validator_hash="def456",
        )
        
        assert result.is_valid is False
        assert result.requires_cpu_verification is True
    
    def test_verify_gradient_with_architecture(self):
        """Test gradient verification with GPU architecture info."""
        verifier = GradientHashVerifier()
        
        result = verifier.verify_gradient(
            miner_hash="abc123",
            validator_hash="abc123",
            miner_gpu_architecture="Ampere",
            validator_gpu_architecture="Ampere",
        )
        
        assert result.is_valid is True
        assert result.miner_gpu_architecture == "Ampere"
        assert result.validator_gpu_architecture == "Ampere"
    
    def test_verify_with_cpu_fallback(self):
        """Test verification with CPU fallback."""
        verifier = GradientHashVerifier()
        
        torch.manual_seed(42)
        gradients = {"layer1.weight": torch.randn(10, 10)}
        
        expected_hash = DeterministicHashVerifier.compute_deterministic_hash(
            gradients, precision="float32"
        )
        
        is_valid, computed_hash = verifier.verify_with_cpu_fallback(
            gradients, expected_hash, seed=42
        )
        
        assert is_valid is True


class TestGradientVerificationDataclass:
    """Tests for GradientVerification dataclass."""
    
    def test_dataclass_creation(self):
        """Test GradientVerification dataclass creation."""
        verification = GradientVerification(
            is_valid=True,
            result=VerificationResult.VALID,
            reason="test_reason",
        )
        
        assert verification.is_valid is True
        assert verification.result == VerificationResult.VALID
        assert verification.reason == "test_reason"
        assert verification.requires_cpu_verification is False
        assert verification.hash_match is False
    
    def test_dataclass_with_all_fields(self):
        """Test GradientVerification with all fields."""
        verification = GradientVerification(
            is_valid=False,
            result=VerificationResult.CROSS_ARCHITECTURE,
            reason="cross_arch",
            requires_cpu_verification=True,
            hash_match=False,
            cross_architecture=True,
            miner_gpu_architecture="Ampere",
            validator_gpu_architecture="Ada",
        )
        
        assert verification.cross_architecture is True
        assert verification.miner_gpu_architecture == "Ampere"
        assert verification.validator_gpu_architecture == "Ada"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

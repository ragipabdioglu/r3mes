#!/usr/bin/env python3
"""
Security tests for R3MES Registry System

Tests the security improvements:
1. MANDATORY checksum verification
2. Compatibility check BEFORE download
3. Atomic downloads with rollback
4. Retry with exponential backoff
5. Conflict detection
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.verification import (
    VerificationPolicy,
    VerificationLevel,
    calculate_checksum,
    verify_checksum,
    AtomicDownload,
    RetryConfig,
    VerificationError,
)
from utils.model_registry import ModelRegistry, ModelInfo, ModelStatus
from utils.dataset_registry import DatasetRegistry, DatasetInfo, DatasetStatus
from utils.adapter_registry import AdapterRegistry, AdapterInfo, AdapterStatus, AdapterType
from utils.unified_registry import UnifiedRegistry, SystemState


class TestVerificationPolicy:
    """Test verification policy enforcement."""
    
    def test_strict_policy_requires_checksum(self):
        """STRICT policy should require checksum."""
        policy = VerificationPolicy(level=VerificationLevel.STRICT)
        assert policy.require_checksum is True
        assert policy.require_ipfs_match is True
        assert policy.should_fail_on_checksum_missing() is True
    
    def test_standard_policy(self):
        """STANDARD policy should require checksum but not IPFS."""
        policy = VerificationPolicy(level=VerificationLevel.STANDARD)
        assert policy.require_checksum is True
    
    def test_relaxed_policy_warns(self):
        """RELAXED policy should log warning."""
        with patch('utils.verification.logger') as mock_logger:
            policy = VerificationPolicy(level=VerificationLevel.RELAXED)
            mock_logger.warning.assert_called()


class TestChecksumVerification:
    """Test checksum calculation and verification."""
    
    def test_calculate_file_checksum(self):
        """Should calculate correct checksum for file."""
        # Use TemporaryDirectory for Windows compatibility
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_bytes(b"test content")
            
            checksum = calculate_checksum(test_file)
            
            # SHA256 of "test content"
            expected = "6ae8a75555209fd6c44157c0aed8016e763ff435a19cf186f76863140143ff72"
            assert checksum == expected
    
    def test_calculate_directory_checksum(self):
        """Should calculate deterministic checksum for directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "file1.txt").write_text("content1")
            (Path(tmpdir) / "file2.txt").write_text("content2")
            
            checksum1 = calculate_checksum(Path(tmpdir))
            checksum2 = calculate_checksum(Path(tmpdir))
            
            # Should be deterministic
            assert checksum1 == checksum2
    
    def test_verify_checksum_success(self):
        """Should return success when checksum matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_bytes(b"test content")
            
            expected = "6ae8a75555209fd6c44157c0aed8016e763ff435a19cf186f76863140143ff72"
            result = verify_checksum(test_file, expected)
            
            assert result.success is True
            assert result.checksum_verified is True
    
    def test_verify_checksum_failure(self):
        """Should return failure when checksum doesn't match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_bytes(b"test content")
            
            result = verify_checksum(test_file, "wrong_checksum")
            
            assert result.success is False
            assert result.checksum_verified is False
            assert "mismatch" in result.error_message.lower()


class TestAtomicDownload:
    """Test atomic download with rollback."""
    
    def test_atomic_download_success(self):
        """Should move file to final location on success."""
        with tempfile.TemporaryDirectory() as tmpdir:
            final_path = Path(tmpdir) / "final.txt"
            
            with AtomicDownload(final_path) as temp_path:
                temp_path.write_text("test content")
            
            assert final_path.exists()
            assert final_path.read_text() == "test content"
    
    def test_atomic_download_rollback_on_exception(self):
        """Should rollback on exception."""
        with tempfile.TemporaryDirectory() as tmpdir:
            final_path = Path(tmpdir) / "final.txt"
            
            try:
                with AtomicDownload(final_path) as temp_path:
                    temp_path.write_text("test content")
                    raise Exception("Simulated failure")
            except Exception:
                pass
            
            # Final path should not exist
            assert not final_path.exists()
    
    def test_atomic_download_rollback_on_verification_failure(self):
        """Should rollback when verification fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            final_path = Path(tmpdir) / "final.txt"
            
            def verify_func(path):
                return False  # Always fail
            
            try:
                with AtomicDownload(final_path, verify_func) as temp_path:
                    temp_path.write_text("test content")
            except VerificationError:
                pass
            
            # Final path should not exist
            assert not final_path.exists()


class TestModelRegistrySecurity:
    """Test model registry security features."""
    
    def test_model_without_checksum_rejected_in_strict_mode(self):
        """Model without checksum should be rejected in strict mode."""
        registry = ModelRegistry(
            verification_policy=VerificationPolicy(level=VerificationLevel.STRICT)
        )
        
        # Add model without checksum
        model_info = ModelInfo(
            model_id="test",
            version="1.0.0",
            ipfs_hash="QmTest",
            checksum="",  # No checksum
            size_bytes=1000,
            architecture="bitnet",
            description="Test model",
        )
        registry._approved_models["1.0.0"] = model_info
        registry._active_model_version = "1.0.0"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model_path.mkdir()
            (model_path / "model.bin").write_bytes(b"fake model")
            
            status = registry.verify_model(str(model_path), "1.0.0", strict=True)
            
            assert status.status == ModelStatus.INVALID
            assert "no valid checksum" in status.error_message.lower()
    
    def test_download_rejected_without_checksum(self):
        """Download should be rejected if model has no checksum."""
        registry = ModelRegistry()
        
        model_info = ModelInfo(
            model_id="test",
            version="1.0.0",
            ipfs_hash="QmTest",
            checksum="",  # No checksum
            size_bytes=1000,
            architecture="bitnet",
            description="Test model",
        )
        registry._approved_models["1.0.0"] = model_info
        registry._active_model_version = "1.0.0"
        
        success, error = registry.download_model("1.0.0")
        
        assert success is False
        assert "no valid checksum" in error.lower()


class TestAdapterRegistrySecurity:
    """Test adapter registry security features."""
    
    def test_compatibility_checked_before_download(self):
        """Compatibility should be checked BEFORE download."""
        registry = AdapterRegistry(current_model_version="2.0.0")
        
        # Add incompatible adapter
        adapter_info = AdapterInfo(
            adapter_id="test_adapter",
            name="Test",
            adapter_type=AdapterType.DORA,
            version="1.0.0",
            ipfs_hash="QmTest",
            checksum="a" * 64,
            size_bytes=1000,
            compatible_model_versions=["1.0.0"],  # Only compatible with 1.0.0
            min_model_version="1.0.0",
            max_model_version="1.0.0",
            domain="general",
            description="Test",
            lora_rank=8,
            lora_alpha=16.0,
            target_modules=["q_proj"],
            approved_at=0,
            approval_tx_hash="",
            proposer="",
        )
        registry._approved_adapters["test_adapter"] = adapter_info
        
        # Try to download - should fail on compatibility check
        success, error = registry.download_adapter("test_adapter")
        
        assert success is False
        assert "not compatible" in error.lower()
    
    def test_conflict_detection(self):
        """Should detect adapter conflicts."""
        registry = AdapterRegistry()
        
        # Add two adapters targeting same module
        adapter1 = AdapterInfo(
            adapter_id="adapter1",
            name="Adapter 1",
            adapter_type=AdapterType.DORA,
            version="1.0.0",
            ipfs_hash="QmTest1",
            checksum="a" * 64,
            size_bytes=1000,
            compatible_model_versions=["1.0.0"],
            min_model_version="1.0.0",
            max_model_version=None,
            domain="general",
            description="Test",
            lora_rank=8,
            lora_alpha=16.0,
            target_modules=["q_proj", "v_proj"],
            approved_at=0,
            approval_tx_hash="",
            proposer="",
        )
        
        adapter2 = AdapterInfo(
            adapter_id="adapter2",
            name="Adapter 2",
            adapter_type=AdapterType.DORA,
            version="1.0.0",
            ipfs_hash="QmTest2",
            checksum="b" * 64,
            size_bytes=1000,
            compatible_model_versions=["1.0.0"],
            min_model_version="1.0.0",
            max_model_version=None,
            domain="general",
            description="Test",
            lora_rank=8,
            lora_alpha=16.0,
            target_modules=["q_proj"],  # Conflicts with adapter1
            approved_at=0,
            approval_tx_hash="",
            proposer="",
        )
        
        registry._approved_adapters["adapter1"] = adapter1
        registry._approved_adapters["adapter2"] = adapter2
        
        # Register first adapter
        registry._conflict_detector.register_adapter(adapter1)
        
        # Check conflicts for second adapter
        has_conflicts, conflicts = registry.check_conflicts("adapter2")
        
        assert has_conflicts is True
        assert "adapter1" in conflicts


class TestUnifiedRegistryAtomic:
    """Test unified registry atomic loading."""
    
    def test_atomic_load_rollback_on_failure(self):
        """Should rollback all components on failure."""
        # This would require mocking blockchain and IPFS clients
        # Simplified test to verify the mechanism exists
        
        registry = UnifiedRegistry()
        
        # Verify rollback method exists and works
        loaded_components = [
            ("adapter", "test_adapter"),
            ("model", "1.0.0"),
        ]
        
        # Should not raise
        registry._rollback(loaded_components)


class TestRetryMechanism:
    """Test retry with exponential backoff."""
    
    def test_retry_config_defaults(self):
        """Should have sensible defaults."""
        config = RetryConfig()
        
        assert config.max_attempts == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 30.0
        assert config.exponential_base == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

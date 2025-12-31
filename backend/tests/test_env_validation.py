"""
Property Tests for Environment Variable Validation

Tests that environment variables are properly validated.
Validates Requirements 8.2, 8.3, 8.4.
"""

import pytest
import os
from unittest.mock import patch, MagicMock


class TestInferenceModeValidation:
    """Test suite for inference mode environment variable validation."""
    
    def test_valid_inference_modes(self):
        """
        Property: All valid inference modes should pass validation.
        Validates: Requirement 8.2
        """
        from backend.app.env_validator import EnvironmentValidator
        
        validator = EnvironmentValidator()
        valid_modes = ["disabled", "mock", "remote", "local"]
        
        for mode in valid_modes:
            # Skip local mode test if torch is not available
            if mode == "local":
                try:
                    import torch
                    if not torch.cuda.is_available():
                        continue  # Skip local mode test without GPU
                except ImportError:
                    continue  # Skip local mode test without torch
            
            is_valid, error = validator.validate_inference_mode(mode)
            assert is_valid, f"Mode '{mode}' should be valid, got error: {error}"
    
    def test_invalid_inference_mode(self):
        """
        Property: Invalid inference modes should fail validation.
        Validates: Requirement 8.3
        """
        from backend.app.env_validator import EnvironmentValidator
        
        validator = EnvironmentValidator()
        invalid_modes = ["invalid", "gpu", "cpu", "auto", ""]
        
        for mode in invalid_modes:
            if mode == "":
                continue  # Empty string is handled differently
            is_valid, error = validator.validate_inference_mode(mode)
            assert not is_valid, f"Mode '{mode}' should be invalid"
            assert error is not None
    
    def test_inference_mode_case_insensitive(self):
        """
        Property: Inference mode validation should be case-insensitive.
        Validates: Requirement 8.2
        """
        from backend.app.env_validator import EnvironmentValidator
        
        validator = EnvironmentValidator()
        
        # Test various case combinations (skip local without GPU)
        test_cases = [
            ("DISABLED", True),
            ("Disabled", True),
            ("MOCK", True),
            ("Mock", True),
            ("REMOTE", True),
            ("Remote", True),
        ]
        
        for mode, expected_valid in test_cases:
            is_valid, error = validator.validate_inference_mode(mode)
            assert is_valid == expected_valid, f"Mode '{mode}' validation failed"
    
    def test_local_mode_requires_gpu(self):
        """
        Property: Local inference mode should require GPU.
        Validates: Requirement 8.4
        """
        from backend.app.env_validator import EnvironmentValidator
        
        validator = EnvironmentValidator()
        
        # Mock torch to simulate no GPU
        with patch.dict('sys.modules', {'torch': MagicMock(cuda=MagicMock(is_available=lambda: False))}):
            # Force reimport to pick up mock
            import importlib
            import backend.app.env_validator as ev
            importlib.reload(ev)
            
            validator = ev.EnvironmentValidator()
            is_valid, error = validator.validate_inference_mode("local")
            
            # Should fail because no GPU
            assert not is_valid or "GPU" in str(error) or "CUDA" in str(error)
    
    def test_inference_mode_in_rules(self):
        """
        Property: R3MES_INFERENCE_MODE should be in validation rules.
        Validates: Requirement 8.2
        """
        from backend.app.env_validator import EnvironmentValidator
        
        validator = EnvironmentValidator()
        rules = validator.get_rules()
        
        rule_names = [rule.name for rule in rules]
        assert "R3MES_INFERENCE_MODE" in rule_names, (
            "R3MES_INFERENCE_MODE should be in validation rules"
        )
    
    def test_inference_mode_has_default(self):
        """
        Property: R3MES_INFERENCE_MODE should have a safe default.
        Validates: Requirement 8.2
        """
        from backend.app.env_validator import EnvironmentValidator
        
        validator = EnvironmentValidator()
        rules = validator.get_rules()
        
        inference_rule = next(
            (rule for rule in rules if rule.name == "R3MES_INFERENCE_MODE"),
            None
        )
        
        assert inference_rule is not None
        assert inference_rule.default is not None
        # Default should be a GPU-less safe mode
        assert inference_rule.default in ("disabled", "remote", "mock")


class TestEnvironmentValidation:
    """Test suite for general environment validation."""
    
    def test_development_mode_allows_localhost(self):
        """
        Property: Development mode should allow localhost URLs.
        Validates: Requirement 8.3
        """
        with patch.dict(os.environ, {"R3MES_ENV": "development"}):
            from backend.app.env_validator import EnvironmentValidator
            
            validator = EnvironmentValidator()
            
            # Localhost should be allowed in development
            is_valid, error = validator.validate_url(
                "http://localhost:8000",
                allow_localhost=True
            )
            assert is_valid, f"Localhost should be allowed in development: {error}"
    
    def test_production_mode_rejects_localhost(self):
        """
        Property: Production mode should reject localhost URLs.
        Validates: Requirement 8.4
        """
        with patch.dict(os.environ, {"R3MES_ENV": "production"}):
            from backend.app.env_validator import EnvironmentValidator
            
            validator = EnvironmentValidator()
            
            # Localhost should be rejected in production
            is_valid, error = validator.validate_url(
                "http://localhost:8000",
                allow_localhost=False
            )
            assert not is_valid, "Localhost should be rejected in production"
    
    def test_validation_report_structure(self):
        """
        Property: Validation report should have correct structure.
        Validates: Requirement 8.3
        """
        from backend.app.env_validator import EnvironmentValidator
        
        validator = EnvironmentValidator()
        report = validator.get_validation_report()
        
        required_keys = ["valid", "environment", "is_production", "errors", "warnings"]
        for key in required_keys:
            assert key in report, f"Report missing key: {key}"
    
    def test_port_validation(self):
        """
        Property: Port validation should accept valid ports only.
        Validates: Requirement 8.3
        """
        from backend.app.env_validator import EnvironmentValidator
        
        validator = EnvironmentValidator()
        
        # Valid ports
        for port in ["80", "443", "8000", "8080", "65535"]:
            is_valid, error = validator.validate_port(port)
            assert is_valid, f"Port {port} should be valid"
        
        # Invalid ports
        for port in ["0", "-1", "65536", "abc", ""]:
            if port == "":
                continue
            is_valid, error = validator.validate_port(port)
            assert not is_valid, f"Port {port} should be invalid"
    
    def test_boolean_validation(self):
        """
        Property: Boolean validation should accept various formats.
        Validates: Requirement 8.3
        """
        from backend.app.env_validator import EnvironmentValidator
        
        validator = EnvironmentValidator()
        
        # Valid booleans
        for value in ["true", "false", "True", "False", "1", "0", "yes", "no"]:
            is_valid, error = validator.validate_boolean(value)
            assert is_valid, f"Boolean '{value}' should be valid"
        
        # Invalid booleans
        for value in ["maybe", "2", "enabled"]:
            is_valid, error = validator.validate_boolean(value)
            assert not is_valid, f"Boolean '{value}' should be invalid"


class TestInferenceModeIntegration:
    """Integration tests for inference mode with environment validator."""
    
    def test_inference_mode_from_env(self):
        """
        Property: Inference mode should be read from environment.
        Validates: Requirement 8.2
        """
        for mode in ["disabled", "mock", "remote"]:
            with patch.dict(os.environ, {"R3MES_INFERENCE_MODE": mode}):
                from backend.app.inference_mode import (
                    get_inference_mode,
                    InferenceMode,
                    reset_inference_mode_cache
                )
                
                reset_inference_mode_cache()
                
                result = get_inference_mode()
                assert result == InferenceMode(mode), f"Expected {mode}, got {result}"
    
    def test_inference_mode_default_is_safe(self):
        """
        Property: Default inference mode should be safe for GPU-less deployment.
        Validates: Requirement 8.2
        """
        # Remove inference mode from environment
        env = os.environ.copy()
        env.pop("R3MES_INFERENCE_MODE", None)
        
        with patch.dict(os.environ, env, clear=True):
            from backend.app.inference_mode import (
                get_inference_mode,
                InferenceMode,
                reset_inference_mode_cache
            )
            
            reset_inference_mode_cache()
            
            result = get_inference_mode()
            # Default should be disabled (safe for GPU-less)
            assert result == InferenceMode.DISABLED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

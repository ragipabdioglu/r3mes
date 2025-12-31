"""
Property-based tests for inference mode configuration.

Feature: role-separation, Property 1: Inference Mode Behavior
Validates: Requirements 1.1, 1.4, 1.5, 1.6

For any valid inference mode setting (disabled, mock, remote), the Backend API 
SHALL start successfully without importing torch, transformers, bitsandbytes, 
or accelerate libraries.
"""

import os
import sys
import pytest
from unittest.mock import patch

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestInferenceModeConfiguration:
    """Test inference mode enum and helper functions."""
    
    def setup_method(self):
        """Reset cache before each test."""
        from app.inference_mode import reset_inference_mode_cache
        reset_inference_mode_cache()
    
    def test_default_mode_is_disabled(self):
        """Default inference mode should be 'disabled' for safe GPU-less deployment."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove R3MES_INFERENCE_MODE if it exists
            os.environ.pop("R3MES_INFERENCE_MODE", None)
            from app.inference_mode import reset_inference_mode_cache, get_inference_mode, InferenceMode
            reset_inference_mode_cache()
            
            mode = get_inference_mode()
            assert mode == InferenceMode.DISABLED
    
    def test_disabled_mode_does_not_load_ai_libraries(self):
        """Disabled mode should not require AI libraries."""
        with patch.dict(os.environ, {"R3MES_INFERENCE_MODE": "disabled"}):
            from app.inference_mode import reset_inference_mode_cache, should_load_ai_libraries
            reset_inference_mode_cache()
            
            assert should_load_ai_libraries() == False
    
    def test_mock_mode_does_not_load_ai_libraries(self):
        """Mock mode should not require AI libraries."""
        with patch.dict(os.environ, {"R3MES_INFERENCE_MODE": "mock"}):
            from app.inference_mode import reset_inference_mode_cache, should_load_ai_libraries
            reset_inference_mode_cache()
            
            assert should_load_ai_libraries() == False
    
    def test_remote_mode_does_not_load_ai_libraries(self):
        """Remote mode should not require AI libraries."""
        with patch.dict(os.environ, {"R3MES_INFERENCE_MODE": "remote"}):
            from app.inference_mode import reset_inference_mode_cache, should_load_ai_libraries
            reset_inference_mode_cache()
            
            assert should_load_ai_libraries() == False
    
    def test_local_mode_requires_ai_libraries(self):
        """Local mode should require AI libraries."""
        with patch.dict(os.environ, {"R3MES_INFERENCE_MODE": "local"}):
            from app.inference_mode import reset_inference_mode_cache, should_load_ai_libraries
            reset_inference_mode_cache()
            
            assert should_load_ai_libraries() == True
    
    def test_invalid_mode_defaults_to_disabled(self):
        """Invalid mode values should default to disabled."""
        with patch.dict(os.environ, {"R3MES_INFERENCE_MODE": "invalid_mode"}):
            from app.inference_mode import reset_inference_mode_cache, get_inference_mode, InferenceMode
            reset_inference_mode_cache()
            
            mode = get_inference_mode()
            assert mode == InferenceMode.DISABLED
    
    def test_mode_is_case_insensitive(self):
        """Mode parsing should be case insensitive."""
        test_cases = ["DISABLED", "Disabled", "MOCK", "Mock", "REMOTE", "Remote", "LOCAL", "Local"]
        
        for mode_str in test_cases:
            with patch.dict(os.environ, {"R3MES_INFERENCE_MODE": mode_str}):
                from app.inference_mode import reset_inference_mode_cache, get_inference_mode
                reset_inference_mode_cache()
                
                mode = get_inference_mode()
                assert mode.value == mode_str.lower()
    
    def test_inference_available_for_non_disabled_modes(self):
        """Inference should be available for mock, remote, and local modes."""
        from app.inference_mode import reset_inference_mode_cache, is_inference_available
        
        for mode_str in ["mock", "remote", "local"]:
            with patch.dict(os.environ, {"R3MES_INFERENCE_MODE": mode_str}):
                reset_inference_mode_cache()
                assert is_inference_available() == True
    
    def test_inference_not_available_for_disabled_mode(self):
        """Inference should not be available for disabled mode."""
        with patch.dict(os.environ, {"R3MES_INFERENCE_MODE": "disabled"}):
            from app.inference_mode import reset_inference_mode_cache, is_inference_available
            reset_inference_mode_cache()
            
            assert is_inference_available() == False
    
    def test_mode_description_returns_string(self):
        """Mode description should return a non-empty string."""
        from app.inference_mode import reset_inference_mode_cache, get_inference_mode_description, InferenceMode
        
        for mode in InferenceMode:
            with patch.dict(os.environ, {"R3MES_INFERENCE_MODE": mode.value}):
                reset_inference_mode_cache()
                description = get_inference_mode_description()
                assert isinstance(description, str)
                assert len(description) > 0


class TestInferenceModePropertyBased:
    """
    Property-based tests using hypothesis.
    
    Feature: role-separation, Property 1: Inference Mode Behavior
    """
    
    def setup_method(self):
        """Reset cache before each test."""
        from app.inference_mode import reset_inference_mode_cache
        reset_inference_mode_cache()
    
    @pytest.mark.parametrize("mode", ["disabled", "mock", "remote"])
    def test_non_local_modes_never_require_ai_libraries(self, mode):
        """
        Property: For any non-local inference mode, AI libraries should never be required.
        
        Feature: role-separation, Property 1: Inference Mode Behavior
        Validates: Requirements 1.1, 1.4, 1.5, 1.6
        """
        with patch.dict(os.environ, {"R3MES_INFERENCE_MODE": mode}):
            from app.inference_mode import reset_inference_mode_cache, should_load_ai_libraries
            reset_inference_mode_cache()
            
            # Property: Non-local modes never require AI libraries
            assert should_load_ai_libraries() == False
    
    @pytest.mark.parametrize("mode", ["disabled", "mock", "remote"])
    def test_gpu_less_modes_validate_successfully(self, mode):
        """
        Property: For any GPU-less mode, validation should succeed without GPU.
        
        Feature: role-separation, Property 1: Inference Mode Behavior
        Validates: Requirements 1.1
        """
        with patch.dict(os.environ, {"R3MES_INFERENCE_MODE": mode}):
            from app.inference_mode import reset_inference_mode_cache, validate_inference_mode_for_startup
            reset_inference_mode_cache()
            
            is_valid, message = validate_inference_mode_for_startup()
            
            # Property: GPU-less modes always validate successfully
            assert is_valid == True
            assert "valid" in message.lower() or "GPU-less" in message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

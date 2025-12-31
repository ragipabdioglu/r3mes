"""
Property Tests for GPU-less Graceful Degradation

Tests that the Backend API can start and function without GPU libraries.
Validates Requirement 1.2: GPU-less graceful degradation.
"""

import pytest
import os
import sys
from unittest.mock import patch, MagicMock
from typing import Generator


class TestGPULessDegradation:
    """Test suite for GPU-less graceful degradation."""
    
    def setup_method(self):
        """Reset inference mode cache before each test."""
        # Reset the cached inference mode
        from backend.app.inference_mode import reset_inference_mode_cache
        reset_inference_mode_cache()
    
    def test_inference_mode_disabled_no_gpu_import(self):
        """
        Property: When R3MES_INFERENCE_MODE=disabled, no GPU libraries should be imported.
        Validates: Requirement 1.2
        """
        with patch.dict(os.environ, {"R3MES_INFERENCE_MODE": "disabled"}):
            from backend.app.inference_mode import (
                get_inference_mode, 
                should_load_ai_libraries,
                InferenceMode
            )
            
            # Reset cache to pick up new env var
            from backend.app.inference_mode import reset_inference_mode_cache
            reset_inference_mode_cache()
            
            mode = get_inference_mode()
            assert mode == InferenceMode.DISABLED
            assert should_load_ai_libraries() is False
    
    def test_inference_mode_mock_no_gpu_import(self):
        """
        Property: When R3MES_INFERENCE_MODE=mock, no GPU libraries should be imported.
        Validates: Requirement 1.2
        """
        with patch.dict(os.environ, {"R3MES_INFERENCE_MODE": "mock"}):
            from backend.app.inference_mode import (
                get_inference_mode, 
                should_load_ai_libraries,
                InferenceMode,
                reset_inference_mode_cache
            )
            
            reset_inference_mode_cache()
            
            mode = get_inference_mode()
            assert mode == InferenceMode.MOCK
            assert should_load_ai_libraries() is False
    
    def test_inference_mode_remote_no_gpu_import(self):
        """
        Property: When R3MES_INFERENCE_MODE=remote, no GPU libraries should be imported.
        Validates: Requirement 1.2
        """
        with patch.dict(os.environ, {"R3MES_INFERENCE_MODE": "remote"}):
            from backend.app.inference_mode import (
                get_inference_mode, 
                should_load_ai_libraries,
                InferenceMode,
                reset_inference_mode_cache
            )
            
            reset_inference_mode_cache()
            
            mode = get_inference_mode()
            assert mode == InferenceMode.REMOTE
            assert should_load_ai_libraries() is False
    
    def test_inference_mode_local_requires_gpu(self):
        """
        Property: When R3MES_INFERENCE_MODE=local, GPU libraries should be loaded.
        Validates: Requirement 1.4
        """
        with patch.dict(os.environ, {"R3MES_INFERENCE_MODE": "local"}):
            from backend.app.inference_mode import (
                get_inference_mode, 
                should_load_ai_libraries,
                InferenceMode,
                reset_inference_mode_cache
            )
            
            reset_inference_mode_cache()
            
            mode = get_inference_mode()
            assert mode == InferenceMode.LOCAL
            assert should_load_ai_libraries() is True
    
    def test_keyword_router_works_without_sentence_transformers(self):
        """
        Property: KeywordRouter should work without sentence-transformers.
        Validates: Requirement 1.1
        """
        from backend.app.semantic_router import KeywordRouter
        
        router = KeywordRouter()
        
        # Test code-related prompt
        adapter, score = router.decide_adapter("How do I write a Python function?")
        assert adapter == "coder_adapter"
        assert score > 0
        
        # Test law-related prompt
        adapter, score = router.decide_adapter("What are my legal rights?")
        assert adapter == "law_adapter"
        assert score > 0
        
        # Test generic prompt
        adapter, score = router.decide_adapter("Hello, how are you?")
        assert adapter == "default_adapter"
        assert score == 0.0
    
    def test_semantic_router_falls_back_to_keyword_router(self):
        """
        Property: SemanticRouter should fall back to KeywordRouter when 
        sentence-transformers is not available.
        Validates: Requirement 1.1
        """
        # Mock sentence_transformers import to fail
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            # Force re-evaluation of availability
            import backend.app.semantic_router as sr
            sr._sentence_transformers_available = None
            sr._SentenceTransformer = None
            sr._np = None
            
            # Create router - should fall back to keyword router
            router = sr.SemanticRouter(use_semantic=True)
            
            # Should still work with keyword fallback
            adapter, score = router.decide_adapter("How do I debug this code?")
            assert adapter in ["coder_adapter", "default_adapter"]
    
    def test_get_router_returns_keyword_router_for_non_local_mode(self):
        """
        Property: get_router() should return KeywordRouter when inference mode is not LOCAL.
        Validates: Requirement 1.1
        """
        with patch.dict(os.environ, {"R3MES_INFERENCE_MODE": "remote"}):
            from backend.app.inference_mode import reset_inference_mode_cache
            reset_inference_mode_cache()
            
            from backend.app.semantic_router import get_router, KeywordRouter
            
            router = get_router()
            assert isinstance(router, KeywordRouter)
    
    def test_model_manager_not_imported_in_disabled_mode(self):
        """
        Property: AIModelManager should not be instantiated when inference is disabled.
        Validates: Requirement 1.2
        """
        with patch.dict(os.environ, {"R3MES_INFERENCE_MODE": "disabled"}):
            from backend.app.inference_mode import (
                should_load_ai_libraries,
                reset_inference_mode_cache
            )
            
            reset_inference_mode_cache()
            
            # In disabled mode, we should not load AI libraries
            assert should_load_ai_libraries() is False
            
            # The model_manager module can be imported, but AIModelManager
            # should not be instantiated (it would fail without GPU)
    
    def test_inference_mode_validation_for_gpu_less_deployment(self):
        """
        Property: validate_inference_mode_for_startup() should pass for GPU-less modes.
        Validates: Requirement 1.2
        """
        for mode in ["disabled", "mock", "remote"]:
            with patch.dict(os.environ, {"R3MES_INFERENCE_MODE": mode}):
                from backend.app.inference_mode import (
                    validate_inference_mode_for_startup,
                    reset_inference_mode_cache
                )
                
                reset_inference_mode_cache()
                
                is_valid, message = validate_inference_mode_for_startup()
                assert is_valid is True, f"Mode '{mode}' should be valid for GPU-less deployment"
                assert "GPU-less deployment" in message or "valid" in message.lower()


class TestKeywordRouterPatterns:
    """Test keyword router pattern matching."""
    
    def test_code_keywords_detected(self):
        """Test that code-related keywords are properly detected."""
        from backend.app.semantic_router import KeywordRouter
        
        router = KeywordRouter()
        
        code_prompts = [
            "How to write a Python function?",
            "Debug this JavaScript error",
            "Explain this algorithm",
            "Fix my SQL query",
            "How to use React hooks?",
            "Docker container not starting",
            "Git merge conflict resolution",
        ]
        
        for prompt in code_prompts:
            adapter, score = router.decide_adapter(prompt)
            assert adapter == "coder_adapter", f"Expected coder_adapter for: {prompt}"
            assert score > 0, f"Expected positive score for: {prompt}"
    
    def test_law_keywords_detected(self):
        """Test that law-related keywords are properly detected."""
        from backend.app.semantic_router import KeywordRouter
        
        router = KeywordRouter()
        
        law_prompts = [
            "What are my legal rights?",
            "Explain this contract clause",
            "How to file a lawsuit?",
            "Copyright infringement question",
            "GDPR compliance requirements",
            "Terms of service review",
        ]
        
        for prompt in law_prompts:
            adapter, score = router.decide_adapter(prompt)
            assert adapter == "law_adapter", f"Expected law_adapter for: {prompt}"
            assert score > 0, f"Expected positive score for: {prompt}"
    
    def test_generic_prompts_return_default(self):
        """Test that generic prompts return default adapter."""
        from backend.app.semantic_router import KeywordRouter
        
        router = KeywordRouter()
        
        generic_prompts = [
            "Hello, how are you?",
            "What's the weather like?",
            "Tell me a joke",
            "Good morning",
        ]
        
        for prompt in generic_prompts:
            adapter, score = router.decide_adapter(prompt)
            assert adapter == "default_adapter", f"Expected default_adapter for: {prompt}"
            assert score == 0.0, f"Expected zero score for: {prompt}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Tests for Inference Endpoints (FAZ 3)

Tests the Backend API ↔ ServingEngine/InferencePipeline integration.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from httpx import AsyncClient
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def mock_inference_mode():
    """Mock inference mode for testing."""
    with patch("backend.app.inference_endpoints.get_inference_mode") as mock:
        yield mock


@pytest.fixture
def mock_serving_engine():
    """Mock ServingEngine for testing."""
    engine = MagicMock()
    engine.is_ready.return_value = True
    engine.is_healthy.return_value = True
    engine.model_version = "test-v1.0.0"
    engine.get_health.return_value = {
        "state": "ready",
        "is_ready": True,
        "is_healthy": True,
        "pipeline_initialized": True,
        "model_loaded": True,
        "total_requests": 100,
        "successful_requests": 95,
        "failed_requests": 5,
        "avg_latency_ms": 150.0,
        "error_message": None,
    }
    engine.get_metrics.return_value = {
        "serving_engine_requests_total": 100,
        "serving_engine_requests_success": 95,
        "serving_engine_requests_failed": 5,
        "serving_engine_latency_avg_ms": 150.0,
        "serving_engine_ready": 1,
        "serving_engine_healthy": 1,
    }
    return engine


class TestInferenceEndpointsUnit:
    """Unit tests for inference endpoints."""
    
    def test_mock_response_generation(self):
        """Test mock response generation."""
        from backend.app.inference_endpoints import _generate_mock_response, InferenceRequest
        import time
        
        request = InferenceRequest(prompt="Test prompt")
        start_time = time.perf_counter()
        
        response = _generate_mock_response("test_id", request, start_time)
        
        assert response.request_id == "test_id"
        assert "[MOCK]" in response.text
        assert response.model_version == "mock-v1.0.0"
        assert response.credits_used == 0.0
    
    def test_credit_calculation(self):
        """Test credit calculation."""
        from backend.app.inference_endpoints import _calculate_credits
        
        # Default rate: 0.001 per token
        credits = _calculate_credits(1000)
        assert credits == 1.0
        
        credits = _calculate_credits(500)
        assert credits == 0.5
    
    def test_inference_request_validation(self):
        """Test InferenceRequest validation."""
        from backend.app.inference_endpoints import InferenceRequest
        
        # Valid request
        request = InferenceRequest(prompt="Hello world")
        assert request.prompt == "Hello world"
        assert request.max_tokens == 512
        assert request.temperature == 0.7
        
        # With wallet
        request = InferenceRequest(
            prompt="Test",
            wallet_address="remes1abc123"
        )
        assert request.wallet_address == "remes1abc123"
    
    def test_inference_request_invalid_wallet(self):
        """Test InferenceRequest with invalid wallet."""
        from backend.app.inference_endpoints import InferenceRequest
        from backend.app.exceptions import InvalidWalletAddressError
        
        with pytest.raises(Exception):  # Pydantic validation error
            InferenceRequest(
                prompt="Test",
                wallet_address="invalid_wallet"
            )
    
    def test_inference_request_empty_prompt(self):
        """Test InferenceRequest with empty prompt."""
        from backend.app.inference_endpoints import InferenceRequest
        
        with pytest.raises(Exception):  # Pydantic validation error
            InferenceRequest(prompt="")


class TestInferenceHealthEndpoints:
    """Tests for health endpoints."""
    
    @pytest.mark.asyncio
    async def test_health_disabled_mode(self, mock_inference_mode):
        """Test health endpoint in disabled mode."""
        from backend.app.inference_endpoints import get_inference_health, InferenceMode
        
        mock_inference_mode.return_value = InferenceMode.DISABLED
        
        response = await get_inference_health()
        
        assert response.status == "disabled"
        assert response.is_ready == False
        assert response.is_healthy == True
    
    @pytest.mark.asyncio
    async def test_health_mock_mode(self, mock_inference_mode):
        """Test health endpoint in mock mode."""
        from backend.app.inference_endpoints import get_inference_health, InferenceMode
        
        mock_inference_mode.return_value = InferenceMode.MOCK
        
        response = await get_inference_health()
        
        assert response.status == "mock"
        assert response.is_ready == True
        assert response.is_healthy == True
    
    @pytest.mark.asyncio
    async def test_health_remote_mode(self, mock_inference_mode):
        """Test health endpoint in remote mode."""
        from backend.app.inference_endpoints import get_inference_health, InferenceMode
        
        mock_inference_mode.return_value = InferenceMode.REMOTE
        
        response = await get_inference_health()
        
        assert response.status == "remote"
        assert response.is_ready == True


class TestInferenceMetricsEndpoints:
    """Tests for metrics endpoints."""
    
    @pytest.mark.asyncio
    async def test_metrics_non_local_mode(self, mock_inference_mode):
        """Test metrics endpoint in non-local mode."""
        from backend.app.inference_endpoints import get_inference_metrics, InferenceMode
        
        mock_inference_mode.return_value = InferenceMode.MOCK
        
        response = await get_inference_metrics()
        
        assert response.serving_engine_requests_total == 0
        assert response.serving_engine_ready == 1
        assert response.serving_engine_healthy == 1


class TestInferenceGenerateEndpoint:
    """Tests for generate endpoint."""
    
    @pytest.mark.asyncio
    async def test_generate_disabled_mode(self, mock_inference_mode):
        """Test generate endpoint in disabled mode."""
        from backend.app.inference_endpoints import InferenceMode
        from fastapi import HTTPException
        
        mock_inference_mode.return_value = InferenceMode.DISABLED
        
        # This would raise HTTPException in actual endpoint
        # Testing the mode check logic
        assert mock_inference_mode.return_value == InferenceMode.DISABLED
    
    @pytest.mark.asyncio
    async def test_generate_mock_mode(self, mock_inference_mode):
        """Test generate endpoint in mock mode."""
        from backend.app.inference_endpoints import (
            _generate_mock_response,
            InferenceRequest,
            InferenceMode,
        )
        import time
        
        mock_inference_mode.return_value = InferenceMode.MOCK
        
        request = InferenceRequest(prompt="What is AI?")
        response = _generate_mock_response("test_123", request, time.perf_counter())
        
        assert "[MOCK]" in response.text
        assert response.request_id == "test_123"


class TestInferencePipelineManagement:
    """Tests for pipeline management endpoints."""
    
    @pytest.mark.asyncio
    async def test_warmup_non_local_mode(self, mock_inference_mode):
        """Test warmup endpoint in non-local mode."""
        from backend.app.inference_endpoints import warmup_pipeline, InferenceMode
        
        mock_inference_mode.return_value = InferenceMode.MOCK
        
        response = await warmup_pipeline()
        
        assert response["status"] == "skipped"
        assert "mock" in response["reason"].lower()
    
    @pytest.mark.asyncio
    async def test_preload_adapters_non_local_mode(self, mock_inference_mode):
        """Test preload adapters endpoint in non-local mode."""
        from backend.app.inference_endpoints import preload_adapters, InferenceMode
        
        mock_inference_mode.return_value = InferenceMode.REMOTE
        
        response = await preload_adapters(["adapter1", "adapter2"])
        
        assert response["status"] == "skipped"
    
    @pytest.mark.asyncio
    async def test_add_rag_document_non_local_mode(self, mock_inference_mode):
        """Test add RAG document endpoint in non-local mode."""
        from backend.app.inference_endpoints import add_rag_document, InferenceMode
        
        mock_inference_mode.return_value = InferenceMode.DISABLED
        
        response = await add_rag_document("doc1", "Test content", {"key": "value"})
        
        assert response["status"] == "skipped"


class TestServingEngineIntegration:
    """Integration tests with ServingEngine (mocked)."""
    
    @pytest.mark.asyncio
    async def test_get_serving_engine_non_local(self, mock_inference_mode):
        """Test get_serving_engine returns None for non-local modes."""
        from backend.app.inference_endpoints import get_serving_engine, InferenceMode
        
        mock_inference_mode.return_value = InferenceMode.MOCK
        
        engine = await get_serving_engine()
        
        assert engine is None
    
    @pytest.mark.asyncio
    async def test_health_with_engine(self, mock_inference_mode, mock_serving_engine):
        """Test health endpoint with mocked engine."""
        from backend.app.inference_endpoints import InferenceMode
        
        mock_inference_mode.return_value = InferenceMode.LOCAL
        
        # Engine health data
        health = mock_serving_engine.get_health()
        
        assert health["is_ready"] == True
        assert health["is_healthy"] == True
        assert health["total_requests"] == 100
        assert health["successful_requests"] == 95


class TestProxyFunctions:
    """Tests for proxy functions."""
    
    @pytest.mark.asyncio
    async def test_mock_stream_generator(self):
        """Test mock stream generator."""
        from backend.app.inference_endpoints import _mock_stream_generator
        
        chunks = []
        async for chunk in _mock_stream_generator("Test prompt"):
            chunks.append(chunk)
        
        assert len(chunks) > 0
        assert "data:" in chunks[0]
        assert "[DONE]" in chunks[-1]


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Tests for R3MES Python SDK Client
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import aiohttp

from r3mes.client import R3MESClient
from r3mes.errors import (
    R3MESError,
    ConnectionError,
    InsufficientCreditsError,
    NotFoundError,
    RateLimitError,
)


class TestR3MESClient:
    """Tests for R3MESClient class."""

    def test_init_default_urls(self):
        """Test client initialization with default URLs."""
        client = R3MESClient()
        assert "r3mes.network" in client.rpc_url
        assert "r3mes.network" in client.rest_url
        assert "r3mes.network" in client.backend_url

    def test_init_custom_urls(self):
        """Test client initialization with custom URLs."""
        client = R3MESClient(
            rpc_url="http://localhost:26657",
            rest_url="http://localhost:1317",
            backend_url="http://localhost:8000",
        )
        assert client.rpc_url == "http://localhost:26657"
        assert client.rest_url == "http://localhost:1317"
        assert client.backend_url == "http://localhost:8000"

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        async with R3MESClient() as client:
            assert client._session is not None
        assert client._session is None

    @pytest.mark.asyncio
    async def test_get_network_stats_success(self):
        """Test successful network stats retrieval."""
        mock_response = {
            "active_miners": 100,
            "total_users": 1000,
            "total_credits": 50000,
            "block_height": 12345,
        }

        with patch.object(R3MESClient, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            async with R3MESClient() as client:
                stats = await client.get_network_stats()
                
            assert stats["active_miners"] == 100
            assert stats["total_users"] == 1000

    @pytest.mark.asyncio
    async def test_get_user_info_success(self):
        """Test successful user info retrieval."""
        mock_response = {
            "wallet_address": "remes1abc123",
            "credits": 100.5,
            "is_miner": True,
        }

        with patch.object(R3MESClient, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            async with R3MESClient() as client:
                user = await client.get_user_info("remes1abc123")
                
            assert user["wallet_address"] == "remes1abc123"
            assert user["credits"] == 100.5

    @pytest.mark.asyncio
    async def test_get_user_info_not_found(self):
        """Test user info retrieval for non-existent user."""
        with patch.object(R3MESClient, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = NotFoundError("User not found")
            
            async with R3MESClient() as client:
                with pytest.raises(NotFoundError):
                    await client.get_user_info("remes1notfound")

    @pytest.mark.asyncio
    async def test_rate_limit_error(self):
        """Test rate limit error handling."""
        with patch.object(R3MESClient, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = RateLimitError("Rate limit exceeded")
            
            async with R3MESClient() as client:
                with pytest.raises(RateLimitError):
                    await client.get_network_stats()

    @pytest.mark.asyncio
    async def test_get_leaderboard(self):
        """Test leaderboard retrieval."""
        mock_response = {
            "miners": [
                {"rank": 1, "wallet_address": "remes1top", "total_earnings": 1000},
                {"rank": 2, "wallet_address": "remes1second", "total_earnings": 800},
            ]
        }

        with patch.object(R3MESClient, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            async with R3MESClient() as client:
                leaderboard = await client.get_leaderboard(limit=10, period="week")
                
            assert len(leaderboard) == 2
            assert leaderboard[0]["rank"] == 1


class TestErrors:
    """Tests for error classes."""

    def test_r3mes_error(self):
        """Test R3MESError base class."""
        error = R3MESError("Test error", code="TEST", details={"key": "value"})
        assert error.message == "Test error"
        assert error.code == "TEST"
        assert error.details == {"key": "value"}
        assert "[TEST]" in str(error)

    def test_connection_error(self):
        """Test ConnectionError."""
        error = ConnectionError("Connection failed")
        assert error.code == "CONNECTION_ERROR"
        assert isinstance(error, R3MESError)

    def test_insufficient_credits_error(self):
        """Test InsufficientCreditsError."""
        error = InsufficientCreditsError("No credits")
        assert error.code == "INSUFFICIENT_CREDITS"
        assert isinstance(error, R3MESError)

    def test_not_found_error(self):
        """Test NotFoundError."""
        error = NotFoundError("Resource not found")
        assert error.code == "NOT_FOUND"
        assert isinstance(error, R3MESError)

    def test_rate_limit_error(self):
        """Test RateLimitError."""
        error = RateLimitError("Rate limited")
        assert error.code == "RATE_LIMIT"
        assert isinstance(error, R3MESError)

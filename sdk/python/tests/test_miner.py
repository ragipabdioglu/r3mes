"""
Tests for R3MES Miner Client
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import aiohttp

from r3mes.miner import MinerClient
from r3mes.errors import ConnectionError, NotFoundError, RateLimitError


class TestMinerClient:
    """Tests for MinerClient class."""

    def test_init_default_url(self):
        """Test client initialization with default URL."""
        client = MinerClient()
        assert "r3mes.network" in client.backend_url

    def test_init_custom_url(self):
        """Test client initialization with custom URL."""
        client = MinerClient(backend_url="http://localhost:8000")
        assert client.backend_url == "http://localhost:8000"

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        async with MinerClient() as client:
            assert client._session is not None
        assert client._session is None

    @pytest.mark.asyncio
    async def test_get_stats_success(self):
        """Test successful miner stats retrieval."""
        mock_stats = {
            "wallet_address": "remes1miner",
            "hashrate": 1500.5,
            "total_earnings": 250.75,
            "blocks_found": 10,
            "uptime_percentage": 99.5,
            "is_active": True,
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_stats)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession.get", return_value=mock_response):
            async with MinerClient(backend_url="http://test") as client:
                stats = await client.get_stats("remes1miner")

            assert stats["hashrate"] == 1500.5
            assert stats["is_active"] is True

    @pytest.mark.asyncio
    async def test_get_stats_not_found(self):
        """Test miner stats for non-existent miner."""
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession.get", return_value=mock_response):
            async with MinerClient(backend_url="http://test") as client:
                stats = await client.get_stats("remes1notfound")

            assert stats["is_active"] is False
            assert stats["hashrate"] == 0

    @pytest.mark.asyncio
    async def test_get_earnings_history_success(self):
        """Test successful earnings history retrieval."""
        mock_earnings = {
            "earnings": [
                {"date": "2025-01-01", "amount": 10.5},
                {"date": "2025-01-02", "amount": 12.3},
            ]
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_earnings)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession.get", return_value=mock_response):
            async with MinerClient(backend_url="http://test") as client:
                earnings = await client.get_earnings_history("remes1miner")

            assert len(earnings) == 2
            assert earnings[0]["amount"] == 10.5

    @pytest.mark.asyncio
    async def test_get_earnings_history_not_found(self):
        """Test earnings history for non-existent miner."""
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession.get", return_value=mock_response):
            async with MinerClient(backend_url="http://test") as client:
                with pytest.raises(NotFoundError):
                    await client.get_earnings_history("remes1notfound")

    @pytest.mark.asyncio
    async def test_get_hashrate_history(self):
        """Test hashrate history retrieval."""
        mock_hashrate = {
            "hashrate": [
                {"timestamp": "2025-01-01T00:00:00Z", "value": 1500},
                {"timestamp": "2025-01-01T01:00:00Z", "value": 1550},
            ]
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_hashrate)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession.get", return_value=mock_response):
            async with MinerClient(backend_url="http://test") as client:
                hashrate = await client.get_hashrate_history("remes1miner", hours=24)

            assert len(hashrate) == 2

    @pytest.mark.asyncio
    async def test_get_leaderboard(self):
        """Test leaderboard retrieval."""
        mock_leaderboard = {
            "miners": [
                {"rank": 1, "wallet_address": "remes1top", "total_earnings": 1000},
                {"rank": 2, "wallet_address": "remes1second", "total_earnings": 800},
            ]
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_leaderboard)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession.get", return_value=mock_response):
            async with MinerClient(backend_url="http://test") as client:
                leaderboard = await client.get_leaderboard(limit=10, period="week")

            assert len(leaderboard) == 2
            assert leaderboard[0]["rank"] == 1

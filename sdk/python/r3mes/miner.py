"""
R3MES Miner Client

Client for miner-specific operations.
"""

import os
from typing import Optional, Dict, Any, List
import aiohttp

from .errors import ConnectionError, NotFoundError, RateLimitError, TimeoutError


class MinerClient:
    """
    Client for miner-specific operations.

    Example:
        async with MinerClient() as miner:
            stats = await miner.get_stats("remes1...")
            print(f"Hashrate: {stats['hashrate']}")
    """

    def __init__(
        self,
        backend_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize the miner client.

        Args:
            backend_url: R3MES backend service URL
            timeout: Request timeout in seconds
        """
        self.backend_url = backend_url or os.getenv(
            "R3MES_BACKEND_URL", "https://backend.r3mes.network"
        )
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "MinerClient":
        """Async context manager entry."""
        self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:
        """Close the client session."""
        if self._session:
            await self._session.close()
            self._session = None

    def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the HTTP session."""
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def get_stats(self, wallet_address: str) -> Dict[str, Any]:
        """
        Get miner statistics.

        Args:
            wallet_address: Miner's wallet address

        Returns:
            Dictionary containing miner stats:
            - wallet_address: Miner's wallet address
            - hashrate: Current hashrate
            - total_earnings: Total earnings
            - blocks_found: Number of blocks found
            - uptime_percentage: Uptime percentage
            - is_active: Whether miner is currently active
        """
        session = self._get_session()

        try:
            async with session.get(
                f"{self.backend_url}/miner/stats/{wallet_address}"
            ) as response:
                if response.status == 404:
                    return {
                        "wallet_address": wallet_address,
                        "hashrate": 0,
                        "total_earnings": 0,
                        "blocks_found": 0,
                        "uptime_percentage": 0,
                        "is_active": False,
                    }
                if response.status == 429:
                    raise RateLimitError("Rate limit exceeded")
                return await response.json()

        except aiohttp.ClientError as e:
            raise ConnectionError(f"Connection failed: {e}")

    async def get_earnings_history(
        self,
        wallet_address: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Get miner earnings history.

        Args:
            wallet_address: Miner's wallet address
            limit: Maximum number of entries
            offset: Pagination offset

        Returns:
            List of earnings entries
        """
        session = self._get_session()

        try:
            async with session.get(
                f"{self.backend_url}/miner/earnings/{wallet_address}",
                params={"limit": limit, "offset": offset},
            ) as response:
                if response.status == 404:
                    raise NotFoundError(f"Miner not found: {wallet_address}")
                data = await response.json()
                return data.get("earnings", [])

        except aiohttp.ClientError as e:
            raise ConnectionError(f"Connection failed: {e}")

    async def get_hashrate_history(
        self,
        wallet_address: str,
        hours: int = 24,
    ) -> List[Dict[str, Any]]:
        """
        Get miner hashrate history.

        Args:
            wallet_address: Miner's wallet address
            hours: Number of hours of history

        Returns:
            List of hashrate data points
        """
        session = self._get_session()

        try:
            async with session.get(
                f"{self.backend_url}/miner/hashrate/{wallet_address}",
                params={"hours": hours},
            ) as response:
                if response.status == 404:
                    raise NotFoundError(f"Miner not found: {wallet_address}")
                data = await response.json()
                return data.get("hashrate", [])

        except aiohttp.ClientError as e:
            raise ConnectionError(f"Connection failed: {e}")

    async def get_leaderboard(
        self,
        limit: int = 100,
        period: str = "all",
    ) -> List[Dict[str, Any]]:
        """
        Get miner leaderboard.

        Args:
            limit: Maximum number of entries
            period: Time period ('day', 'week', 'month', 'all')

        Returns:
            List of leaderboard entries
        """
        session = self._get_session()

        try:
            async with session.get(
                f"{self.backend_url}/leaderboard",
                params={"limit": limit, "period": period},
            ) as response:
                data = await response.json()
                return data.get("miners", [])

        except aiohttp.ClientError as e:
            raise ConnectionError(f"Connection failed: {e}")

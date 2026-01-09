"""
R3MES Main Client

Primary client for interacting with the R3MES network.
"""

import os
import asyncio
from typing import Optional, Dict, Any, AsyncIterator
import aiohttp

from .errors import (
    R3MESError,
    ConnectionError,
    InsufficientCreditsError,
    NotFoundError,
    RateLimitError,
    TimeoutError,
)


class R3MESClient:
    """
    Main R3MES SDK client for general operations.

    Example:
        async with R3MESClient() as client:
            stats = await client.get_network_stats()
            print(f"Active miners: {stats['active_miners']}")
    """

    def __init__(
        self,
        rpc_url: Optional[str] = None,
        rest_url: Optional[str] = None,
        backend_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize the R3MES client.

        Args:
            rpc_url: Tendermint RPC endpoint URL
            rest_url: Cosmos REST API endpoint URL
            backend_url: R3MES backend service URL
            timeout: Request timeout in seconds
        """
        self.rpc_url = rpc_url or os.getenv(
            "R3MES_RPC_URL", "https://rpc.r3mes.network"
        )
        self.rest_url = rest_url or os.getenv(
            "R3MES_REST_URL", "https://api.r3mes.network"
        )
        self.backend_url = backend_url or os.getenv(
            "R3MES_BACKEND_URL", "https://backend.r3mes.network"
        )
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "R3MESClient":
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

    async def _request(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make an HTTP request with error handling."""
        session = self._get_session()

        try:
            async with session.request(method, url, **kwargs) as response:
                if response.status == 402:
                    raise InsufficientCreditsError("Insufficient credits")
                if response.status == 404:
                    raise NotFoundError(f"Resource not found: {url}")
                if response.status == 429:
                    raise RateLimitError("Rate limit exceeded")
                if response.status >= 400:
                    text = await response.text()
                    raise R3MESError(f"Request failed: {response.status} - {text}")

                return await response.json()

        except aiohttp.ClientError as e:
            raise ConnectionError(f"Connection failed: {e}")
        except asyncio.TimeoutError:
            raise TimeoutError(f"Request timed out: {url}")

    async def get_network_stats(self) -> Dict[str, Any]:
        """
        Get network statistics.

        Returns:
            Dictionary containing network stats:
            - active_miners: Number of active miners
            - total_users: Total registered users
            - total_credits: Total credits in circulation
            - block_height: Current block height
        """
        return await self._request("GET", f"{self.backend_url}/network/stats")

    async def get_user_info(self, wallet_address: str) -> Dict[str, Any]:
        """
        Get user information by wallet address.

        Args:
            wallet_address: The user's wallet address

        Returns:
            Dictionary containing user info:
            - wallet_address: User's wallet address
            - credits: Available credits
            - is_miner: Whether user is a miner
        """
        return await self._request(
            "GET", f"{self.backend_url}/user/info/{wallet_address}"
        )

    async def chat(
        self,
        message: str,
        wallet: Optional[Any] = None,
        adapter: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Send a chat message and stream the response.

        Args:
            message: The message to send
            wallet: Optional wallet for authentication
            adapter: Optional LoRA adapter name

        Yields:
            Response tokens as they are generated
        """
        session = self._get_session()
        headers = {"Content-Type": "application/json"}

        if wallet:
            headers["X-Wallet-Address"] = wallet.address
        if adapter:
            headers["X-Adapter"] = adapter

        payload = {
            "message": message,
            "wallet_address": wallet.address if wallet else None,
        }

        try:
            async with session.post(
                f"{self.backend_url}/chat",
                json=payload,
                headers=headers,
            ) as response:
                if response.status == 402:
                    raise InsufficientCreditsError("Insufficient credits for chat")
                if response.status == 429:
                    raise RateLimitError("Rate limit exceeded")
                if response.status >= 400:
                    text = await response.text()
                    raise R3MESError(f"Chat request failed: {response.status} - {text}")

                async for line in response.content:
                    if line:
                        yield line.decode("utf-8").strip()

        except aiohttp.ClientError as e:
            raise ConnectionError(f"Connection failed: {e}")
        except asyncio.TimeoutError:
            raise TimeoutError("Chat request timed out")

    async def get_leaderboard(
        self,
        limit: int = 100,
        period: str = "all",
    ) -> list:
        """
        Get miner leaderboard.

        Args:
            limit: Maximum number of entries to return
            period: Time period ('day', 'week', 'month', 'all')

        Returns:
            List of leaderboard entries
        """
        result = await self._request(
            "GET",
            f"{self.backend_url}/leaderboard",
            params={"limit": limit, "period": period},
        )
        return result.get("miners", [])

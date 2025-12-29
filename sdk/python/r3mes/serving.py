"""
R3MES Serving Client

Client for serving node operations.
"""

import os
from typing import Optional, Dict, Any, List
import aiohttp

from .errors import ConnectionError, NotFoundError


class ServingClient:
    """
    Client for serving node operations.

    Example:
        async with ServingClient() as serving:
            nodes = await serving.list_nodes()
            print(f"Total nodes: {len(nodes)}")
    """

    def __init__(
        self,
        backend_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize the serving client.

        Args:
            backend_url: R3MES backend service URL
            timeout: Request timeout in seconds
        """
        self.backend_url = backend_url or os.getenv(
            "R3MES_BACKEND_URL", "https://backend.r3mes.network"
        )
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "ServingClient":
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

    async def list_nodes(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List all serving nodes.

        Args:
            limit: Maximum number of nodes
            offset: Pagination offset

        Returns:
            List of serving node info
        """
        session = self._get_session()

        try:
            async with session.get(
                f"{self.backend_url}/serving/nodes",
                params={"limit": limit, "offset": offset},
            ) as response:
                data = await response.json()
                return data.get("nodes", [])

        except aiohttp.ClientError as e:
            raise ConnectionError(f"Connection failed: {e}")

    async def get_node(self, wallet_address: str) -> Dict[str, Any]:
        """
        Get serving node details.

        Args:
            wallet_address: Node's wallet address

        Returns:
            Node details
        """
        session = self._get_session()

        try:
            async with session.get(
                f"{self.backend_url}/serving/node/{wallet_address}"
            ) as response:
                if response.status == 404:
                    raise NotFoundError(f"Node not found: {wallet_address}")
                return await response.json()

        except aiohttp.ClientError as e:
            raise ConnectionError(f"Connection failed: {e}")

    async def get_node_stats(self, wallet_address: str) -> Dict[str, Any]:
        """
        Get serving node statistics.

        Args:
            wallet_address: Node's wallet address

        Returns:
            Node statistics
        """
        session = self._get_session()

        try:
            async with session.get(
                f"{self.backend_url}/serving/node/{wallet_address}/stats"
            ) as response:
                if response.status == 404:
                    raise NotFoundError(f"Node not found: {wallet_address}")
                return await response.json()

        except aiohttp.ClientError as e:
            raise ConnectionError(f"Connection failed: {e}")

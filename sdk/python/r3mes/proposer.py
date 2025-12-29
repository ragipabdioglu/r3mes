"""
R3MES Proposer Client

Client for proposer node operations.
"""

import aiohttp
from typing import Optional, Dict, Any, List
from .exceptions import R3MESError, ConnectionError, NodeNotFoundError


class ProposerClient:
    """
    Client for proposer node operations on the R3MES network.
    
    Provides methods for:
    - Listing proposer nodes
    - Getting aggregation history
    - Querying gradient pool
    """
    
    def __init__(
        self,
        backend_url: str = "https://backend.r3mes.network",
    ):
        """
        Initialize proposer client.
        
        Args:
            backend_url: Backend API URL
        """
        self.backend_url = backend_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def _ensure_session(self):
        """Ensure HTTP session is created."""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def list_nodes(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List all proposer nodes.
        
        Args:
            limit: Maximum number of nodes to return
            offset: Pagination offset
            
        Returns:
            List of proposer nodes
        """
        await self._ensure_session()
        
        try:
            async with self.session.get(
                f"{self.backend_url}/proposer/nodes",
                params={"limit": limit, "offset": offset},
            ) as response:
                if response.status != 200:
                    raise R3MESError(f"Failed to list proposer nodes: HTTP {response.status}")
                data = await response.json()
                return data.get("nodes", [])
        except aiohttp.ClientError as e:
            raise ConnectionError(f"Failed to connect to backend: {e}")
    
    async def get_node(self, address: str) -> Dict[str, Any]:
        """
        Get proposer node details.
        
        Args:
            address: Node wallet address
            
        Returns:
            Node details
        """
        await self._ensure_session()
        
        try:
            async with self.session.get(
                f"{self.backend_url}/proposer/nodes/{address}"
            ) as response:
                if response.status == 404:
                    raise NodeNotFoundError(f"Proposer node not found: {address}")
                if response.status != 200:
                    raise R3MESError(f"Failed to get proposer node: HTTP {response.status}")
                return await response.json()
        except aiohttp.ClientError as e:
            raise ConnectionError(f"Failed to connect to backend: {e}")
    
    async def list_aggregations(
        self,
        limit: int = 50,
        offset: int = 0,
        proposer: Optional[str] = None,
        training_round_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        List aggregations.
        
        Args:
            limit: Maximum number of aggregations to return
            offset: Pagination offset
            proposer: Filter by proposer address
            training_round_id: Filter by training round ID
            
        Returns:
            List of aggregations
        """
        await self._ensure_session()
        
        params = {"limit": limit, "offset": offset}
        if proposer:
            params["proposer"] = proposer
        if training_round_id is not None:
            params["training_round_id"] = training_round_id
        
        try:
            async with self.session.get(
                f"{self.backend_url}/proposer/aggregations",
                params=params,
            ) as response:
                if response.status != 200:
                    raise R3MESError(f"Failed to list aggregations: HTTP {response.status}")
                data = await response.json()
                return data.get("aggregations", [])
        except aiohttp.ClientError as e:
            raise ConnectionError(f"Failed to connect to backend: {e}")
    
    async def get_aggregation(self, aggregation_id: int) -> Dict[str, Any]:
        """
        Get aggregation details.
        
        Args:
            aggregation_id: Aggregation ID
            
        Returns:
            Aggregation details
        """
        await self._ensure_session()
        
        try:
            async with self.session.get(
                f"{self.backend_url}/proposer/aggregations/{aggregation_id}"
            ) as response:
                if response.status == 404:
                    raise R3MESError(f"Aggregation not found: {aggregation_id}")
                if response.status != 200:
                    raise R3MESError(f"Failed to get aggregation: HTTP {response.status}")
                return await response.json()
        except aiohttp.ClientError as e:
            raise ConnectionError(f"Failed to connect to backend: {e}")
    
    async def get_gradient_pool(
        self,
        limit: int = 100,
        offset: int = 0,
        status: str = "pending",
    ) -> Dict[str, Any]:
        """
        Get gradient pool.
        
        Args:
            limit: Maximum number of gradients to return
            offset: Pagination offset
            status: Filter by status
            
        Returns:
            Gradient pool with pending gradients
        """
        await self._ensure_session()
        
        try:
            async with self.session.get(
                f"{self.backend_url}/proposer/pool",
                params={"limit": limit, "offset": offset, "status": status},
            ) as response:
                if response.status != 200:
                    raise R3MESError(f"Failed to get gradient pool: HTTP {response.status}")
                return await response.json()
        except aiohttp.ClientError as e:
            raise ConnectionError(f"Failed to connect to backend: {e}")
    
    async def list_commits(
        self,
        limit: int = 50,
        offset: int = 0,
        proposer: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List pending aggregation commits.
        
        Args:
            limit: Maximum number of commits to return
            offset: Pagination offset
            proposer: Filter by proposer address
            status: Filter by status
            
        Returns:
            List of commits
        """
        await self._ensure_session()
        
        params = {"limit": limit, "offset": offset}
        if proposer:
            params["proposer"] = proposer
        if status:
            params["status"] = status
        
        try:
            async with self.session.get(
                f"{self.backend_url}/proposer/commits",
                params=params,
            ) as response:
                if response.status != 200:
                    raise R3MESError(f"Failed to list commits: HTTP {response.status}")
                data = await response.json()
                return data.get("commits", [])
        except aiohttp.ClientError as e:
            raise ConnectionError(f"Failed to connect to backend: {e}")

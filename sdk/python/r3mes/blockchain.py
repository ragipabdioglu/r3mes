"""
R3MES Blockchain Client

Client for blockchain queries.
"""

import os
from typing import Optional, Dict, Any, List
import aiohttp

from .errors import ConnectionError, NotFoundError, RateLimitError


class BlockchainClient:
    """
    Client for blockchain queries.

    Example:
        async with BlockchainClient() as blockchain:
            block = await blockchain.get_latest_block()
            print(f"Block height: {block['height']}")
    """

    def __init__(
        self,
        rpc_url: Optional[str] = None,
        rest_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize the blockchain client.

        Args:
            rpc_url: Tendermint RPC endpoint URL
            rest_url: Cosmos REST API endpoint URL
            timeout: Request timeout in seconds
        """
        self.rpc_url = rpc_url or os.getenv(
            "R3MES_RPC_URL", "https://rpc.r3mes.network"
        )
        self.rest_url = rest_url or os.getenv(
            "R3MES_REST_URL", "https://api.r3mes.network"
        )
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "BlockchainClient":
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

    async def get_latest_block(self) -> Dict[str, Any]:
        """
        Get the latest block.

        Returns:
            Dictionary containing block info:
            - height: Block height
            - hash: Block hash
            - timestamp: Block timestamp
            - proposer: Block proposer address
            - tx_count: Number of transactions
        """
        session = self._get_session()

        try:
            async with session.get(f"{self.rpc_url}/block") as response:
                data = await response.json()
                result = data.get("result", {})
                block = result.get("block", {})
                header = block.get("header", {})

                return {
                    "height": int(header.get("height", 0)),
                    "hash": result.get("block_id", {}).get("hash", ""),
                    "timestamp": header.get("time", ""),
                    "proposer": header.get("proposer_address", ""),
                    "tx_count": len(block.get("data", {}).get("txs", [])),
                }

        except aiohttp.ClientError as e:
            raise ConnectionError(f"Connection failed: {e}")

    async def get_block(self, height: int) -> Dict[str, Any]:
        """
        Get a block by height.

        Args:
            height: Block height

        Returns:
            Dictionary containing block info
        """
        session = self._get_session()

        try:
            async with session.get(
                f"{self.rpc_url}/block", params={"height": height}
            ) as response:
                data = await response.json()
                result = data.get("result", {})
                block = result.get("block", {})
                header = block.get("header", {})

                return {
                    "height": int(header.get("height", 0)),
                    "hash": result.get("block_id", {}).get("hash", ""),
                    "timestamp": header.get("time", ""),
                    "proposer": header.get("proposer_address", ""),
                    "tx_count": len(block.get("data", {}).get("txs", [])),
                }

        except aiohttp.ClientError as e:
            raise ConnectionError(f"Connection failed: {e}")

    async def get_validators(
        self,
        status: str = "BOND_STATUS_BONDED",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get validators.

        Args:
            status: Validator status filter
            limit: Maximum number of validators

        Returns:
            List of validator info
        """
        session = self._get_session()

        try:
            async with session.get(
                f"{self.rest_url}/cosmos/staking/v1beta1/validators",
                params={"status": status, "pagination.limit": limit},
            ) as response:
                data = await response.json()
                validators = data.get("validators", [])

                return [
                    {
                        "operator_address": v.get("operator_address", ""),
                        "moniker": v.get("description", {}).get("moniker", ""),
                        "tokens": v.get("tokens", "0"),
                        "status": v.get("status", ""),
                        "jailed": v.get("jailed", False),
                        "commission": v.get("commission", {})
                        .get("commission_rates", {})
                        .get("rate", "0"),
                    }
                    for v in validators
                ]

        except aiohttp.ClientError as e:
            raise ConnectionError(f"Connection failed: {e}")

    async def get_balance(self, address: str) -> List[Dict[str, str]]:
        """
        Get account balance.

        Args:
            address: Account address

        Returns:
            List of balance entries with denom and amount
        """
        session = self._get_session()

        try:
            async with session.get(
                f"{self.rest_url}/cosmos/bank/v1beta1/balances/{address}"
            ) as response:
                data = await response.json()
                return data.get("balances", [])

        except aiohttp.ClientError as e:
            raise ConnectionError(f"Connection failed: {e}")

    async def get_transaction(self, tx_hash: str) -> Dict[str, Any]:
        """
        Get transaction by hash.

        Args:
            tx_hash: Transaction hash

        Returns:
            Transaction details
        """
        session = self._get_session()

        try:
            async with session.get(
                f"{self.rest_url}/cosmos/tx/v1beta1/txs/{tx_hash}"
            ) as response:
                if response.status == 404:
                    raise NotFoundError(f"Transaction not found: {tx_hash}")
                return await response.json()

        except aiohttp.ClientError as e:
            raise ConnectionError(f"Connection failed: {e}")

    async def get_status(self) -> Dict[str, Any]:
        """
        Get node status.

        Returns:
            Node status information
        """
        session = self._get_session()

        try:
            async with session.get(f"{self.rpc_url}/status") as response:
                data = await response.json()
                return data.get("result", {})

        except aiohttp.ClientError as e:
            raise ConnectionError(f"Connection failed: {e}")

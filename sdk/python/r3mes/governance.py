"""
R3MES Governance Client

Client for governance operations including proposals and voting.
"""

import os
from typing import Optional, Dict, Any, List
import aiohttp

from .errors import (
    R3MESError,
    ConnectionError,
    NotFoundError,
    RateLimitError,
    TimeoutError,
)


class GovernanceClient:
    """
    Client for R3MES governance operations.

    Example:
        async with GovernanceClient() as gov:
            proposals = await gov.get_proposals()
            for p in proposals:
                print(f"{p['id']}: {p['title']}")
    """

    def __init__(
        self,
        rest_url: Optional[str] = None,
        backend_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize the governance client.

        Args:
            rest_url: Cosmos REST API endpoint URL
            backend_url: R3MES backend service URL
            timeout: Request timeout in seconds
        """
        self.rest_url = rest_url or os.getenv(
            "R3MES_REST_URL", "https://api.r3mes.network"
        )
        self.backend_url = backend_url or os.getenv(
            "R3MES_BACKEND_URL", "https://backend.r3mes.network"
        )
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "GovernanceClient":
        self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def get_proposals(
        self,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get governance proposals.

        Args:
            status: Filter by status ('voting', 'passed', 'rejected', 'deposit')
            limit: Maximum number of proposals to return

        Returns:
            List of proposal dictionaries
        """
        session = self._get_session()
        params = {"pagination.limit": limit}
        if status:
            params["proposal_status"] = status

        url = f"{self.rest_url}/cosmos/gov/v1beta1/proposals"
        
        async with session.get(url, params=params) as response:
            if response.status != 200:
                raise R3MESError(f"Failed to get proposals: {response.status}")
            data = await response.json()
            return data.get("proposals", [])

    async def get_proposal(self, proposal_id: int) -> Dict[str, Any]:
        """
        Get a specific proposal by ID.

        Args:
            proposal_id: The proposal ID

        Returns:
            Proposal dictionary
        """
        session = self._get_session()
        url = f"{self.rest_url}/cosmos/gov/v1beta1/proposals/{proposal_id}"
        
        async with session.get(url) as response:
            if response.status == 404:
                raise NotFoundError(f"Proposal not found: {proposal_id}")
            if response.status != 200:
                raise R3MESError(f"Failed to get proposal: {response.status}")
            data = await response.json()
            return data.get("proposal", {})

    async def get_proposal_votes(
        self,
        proposal_id: int,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get votes for a proposal.

        Args:
            proposal_id: The proposal ID
            limit: Maximum number of votes to return

        Returns:
            List of vote dictionaries
        """
        session = self._get_session()
        url = f"{self.rest_url}/cosmos/gov/v1beta1/proposals/{proposal_id}/votes"
        params = {"pagination.limit": limit}
        
        async with session.get(url, params=params) as response:
            if response.status != 200:
                raise R3MESError(f"Failed to get votes: {response.status}")
            data = await response.json()
            return data.get("votes", [])

    async def get_proposal_tally(self, proposal_id: int) -> Dict[str, Any]:
        """
        Get current tally for a proposal.

        Args:
            proposal_id: The proposal ID

        Returns:
            Tally dictionary with yes, no, abstain, no_with_veto counts
        """
        session = self._get_session()
        url = f"{self.rest_url}/cosmos/gov/v1beta1/proposals/{proposal_id}/tally"
        
        async with session.get(url) as response:
            if response.status != 200:
                raise R3MESError(f"Failed to get tally: {response.status}")
            data = await response.json()
            return data.get("tally", {})

    async def get_gov_params(self) -> Dict[str, Any]:
        """
        Get governance parameters.

        Returns:
            Dictionary with voting, deposit, and tally params
        """
        session = self._get_session()
        
        params = {}
        for param_type in ["voting", "deposit", "tallying"]:
            url = f"{self.rest_url}/cosmos/gov/v1beta1/params/{param_type}"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    params[param_type] = data.get(f"{param_type}_params", {})
        
        return params


class StakingClient:
    """
    Client for R3MES staking operations.

    Example:
        async with StakingClient() as staking:
            validators = await staking.get_validators()
            for v in validators:
                print(f"{v['moniker']}: {v['tokens']} tokens")
    """

    def __init__(
        self,
        rest_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize the staking client.

        Args:
            rest_url: Cosmos REST API endpoint URL
            timeout: Request timeout in seconds
        """
        self.rest_url = rest_url or os.getenv(
            "R3MES_REST_URL", "https://api.r3mes.network"
        )
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "StakingClient":
        self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def get_validators(
        self,
        status: str = "BOND_STATUS_BONDED",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get validators.

        Args:
            status: Validator status filter
            limit: Maximum number of validators to return

        Returns:
            List of validator dictionaries
        """
        session = self._get_session()
        url = f"{self.rest_url}/cosmos/staking/v1beta1/validators"
        params = {"status": status, "pagination.limit": limit}
        
        async with session.get(url, params=params) as response:
            if response.status != 200:
                raise R3MESError(f"Failed to get validators: {response.status}")
            data = await response.json()
            return data.get("validators", [])

    async def get_validator(self, operator_address: str) -> Dict[str, Any]:
        """
        Get a specific validator.

        Args:
            operator_address: Validator operator address

        Returns:
            Validator dictionary
        """
        session = self._get_session()
        url = f"{self.rest_url}/cosmos/staking/v1beta1/validators/{operator_address}"
        
        async with session.get(url) as response:
            if response.status == 404:
                raise NotFoundError(f"Validator not found: {operator_address}")
            if response.status != 200:
                raise R3MESError(f"Failed to get validator: {response.status}")
            data = await response.json()
            return data.get("validator", {})

    async def get_delegations(
        self,
        delegator_address: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get delegations for a delegator.

        Args:
            delegator_address: Delegator wallet address
            limit: Maximum number of delegations to return

        Returns:
            List of delegation dictionaries
        """
        session = self._get_session()
        url = f"{self.rest_url}/cosmos/staking/v1beta1/delegations/{delegator_address}"
        params = {"pagination.limit": limit}
        
        async with session.get(url, params=params) as response:
            if response.status != 200:
                raise R3MESError(f"Failed to get delegations: {response.status}")
            data = await response.json()
            return data.get("delegation_responses", [])

    async def get_unbonding_delegations(
        self,
        delegator_address: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get unbonding delegations for a delegator.

        Args:
            delegator_address: Delegator wallet address
            limit: Maximum number of unbonding delegations to return

        Returns:
            List of unbonding delegation dictionaries
        """
        session = self._get_session()
        url = f"{self.rest_url}/cosmos/staking/v1beta1/delegators/{delegator_address}/unbonding_delegations"
        params = {"pagination.limit": limit}
        
        async with session.get(url, params=params) as response:
            if response.status != 200:
                raise R3MESError(f"Failed to get unbonding delegations: {response.status}")
            data = await response.json()
            return data.get("unbonding_responses", [])

    async def get_staking_pool(self) -> Dict[str, Any]:
        """
        Get staking pool information.

        Returns:
            Dictionary with bonded_tokens and not_bonded_tokens
        """
        session = self._get_session()
        url = f"{self.rest_url}/cosmos/staking/v1beta1/pool"
        
        async with session.get(url) as response:
            if response.status != 200:
                raise R3MESError(f"Failed to get staking pool: {response.status}")
            data = await response.json()
            return data.get("pool", {})

    async def get_staking_params(self) -> Dict[str, Any]:
        """
        Get staking parameters.

        Returns:
            Dictionary with unbonding_time, max_validators, etc.
        """
        session = self._get_session()
        url = f"{self.rest_url}/cosmos/staking/v1beta1/params"
        
        async with session.get(url) as response:
            if response.status != 200:
                raise R3MESError(f"Failed to get staking params: {response.status}")
            data = await response.json()
            return data.get("params", {})

    async def get_rewards(self, delegator_address: str) -> Dict[str, Any]:
        """
        Get delegation rewards for a delegator.

        Args:
            delegator_address: Delegator wallet address

        Returns:
            Dictionary with rewards and total
        """
        session = self._get_session()
        url = f"{self.rest_url}/cosmos/distribution/v1beta1/delegators/{delegator_address}/rewards"
        
        async with session.get(url) as response:
            if response.status != 200:
                raise R3MESError(f"Failed to get rewards: {response.status}")
            return await response.json()

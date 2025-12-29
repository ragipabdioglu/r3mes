"""
Blockchain Query Client for R3MES

Provides gRPC query interface to Cosmos SDK blockchain.
"""

import os
import logging
from typing import List, Dict, Optional, Any

from .network_resilience import (
    retry_with_backoff,
    BLOCKCHAIN_RETRY_CONFIG,
    get_blockchain_circuit_breaker,
)
from .exceptions import (
    MissingEnvironmentVariableError,
    ProductionConfigurationError,
    BlockchainConnectionError,
    BlockchainQueryError,
)

logger = logging.getLogger(__name__)


class BlockchainQueryClient:
    """
    Client for querying R3MES blockchain via gRPC.
    
    Supports both gRPC (preferred) and HTTP REST fallback.
    """
    
    def __init__(self, grpc_url: Optional[str] = None, rest_url: Optional[str] = None):
        """
        Initialize blockchain query client.
        
        Args:
            grpc_url: gRPC endpoint URL (default: from BLOCKCHAIN_GRPC_URL env var, not used currently)
            rest_url: REST endpoint URL (default: from BLOCKCHAIN_REST_URL env var)
        """
        # Get URLs from parameters or environment variables
        # In production, environment variables must be set (no localhost fallback)
        is_production = os.getenv("R3MES_ENV", "development").lower() == "production"
        
        if grpc_url:
            self.grpc_url = grpc_url
        else:
            grpc_url_env = os.getenv("BLOCKCHAIN_GRPC_URL")
            if not grpc_url_env:
                if is_production:
                    raise MissingEnvironmentVariableError(
                        "BLOCKCHAIN_GRPC_URL environment variable must be set in production. "
                        "Do not use localhost in production."
                    )
                # Development fallback
                self.grpc_url = "localhost:9090"
                logger.warning("BLOCKCHAIN_GRPC_URL not set, using localhost fallback (development only)")
            else:
                self.grpc_url = grpc_url_env
                # Validate that production doesn't use localhost
                if is_production and ("localhost" in self.grpc_url or "127.0.0.1" in self.grpc_url):
                    raise ProductionConfigurationError(
                        f"BLOCKCHAIN_GRPC_URL cannot use localhost in production: {self.grpc_url}"
                    )
        
        if rest_url:
            self.rest_url = rest_url
        else:
            rest_url_env = os.getenv("BLOCKCHAIN_REST_URL")
            if not rest_url_env:
                if is_production:
                    raise MissingEnvironmentVariableError(
                        "BLOCKCHAIN_REST_URL environment variable must be set in production. "
                        "Do not use localhost in production."
                    )
                # Development fallback
                self.rest_url = "http://localhost:1317"
                logger.warning("BLOCKCHAIN_REST_URL not set, using localhost fallback (development only)")
            else:
                self.rest_url = rest_url_env
                # Validate that production doesn't use localhost
                if is_production and ("localhost" in self.rest_url or "127.0.0.1" in self.rest_url):
                    raise ProductionConfigurationError(
                        f"BLOCKCHAIN_REST_URL cannot use localhost in production: {self.rest_url}"
                    )
        
        logger.info(f"Blockchain query client initialized (REST): {self.rest_url}")
    
    def _query_rest(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Query blockchain via REST API (fallback) with retry mechanism and circuit breaker."""
        import requests
        
        @retry_with_backoff(config=BLOCKCHAIN_RETRY_CONFIG, operation_name="REST query")
        def make_request():
            url = f"{self.rest_url}{endpoint}"
            circuit_breaker = get_blockchain_circuit_breaker()
            blockchain_query_timeout = int(os.getenv("BACKEND_BLOCKCHAIN_QUERY_TIMEOUT", "10"))
            response = requests.get(url, params=params, timeout=blockchain_query_timeout)
            
            if response.status_code != 200:
                raise NetworkError(f"REST query failed: {response.status_code} - {response.text}")
            
            return response.json()
        
        return make_request()
    
    def get_miner_score(self, miner_address: str) -> Optional[Dict[str, Any]]:
        """
        Get miner reputation score and statistics.
        
        Args:
            miner_address: Miner wallet address
            
        Returns:
            Miner score data or None if not found
        """
        try:
            # HTTP REST query
            endpoint = f"/remes/remes/v1/miner_score/{miner_address}"
            data = self._query_rest(endpoint)
            
            # Response structure: {"miner_score": {...}}
            if "miner_score" in data:
                score = data["miner_score"]
                return {
                    "miner": score.get("miner", miner_address),
                    "trust_score": float(score.get("trust_score", "0.0")),
                    "reputation_tier": score.get("reputation_tier", "new"),
                    "total_submissions": score.get("total_submissions", 0),
                    "successful_submissions": score.get("successful_submissions", 0),
                    "slashing_events": score.get("slashing_events", 0),
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get miner score for {miner_address}: {e}")
            return None
    
    def get_all_miners(self, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """
        Get all miners with pagination.
        
        Args:
            limit: Maximum number of miners to return
            offset: Pagination offset
            
        Returns:
            Dictionary with miners list and total count
        """
        try:
            # HTTP REST query
            endpoint = "/remes/remes/v1/dashboard/miners"
            params = {"pagination.limit": limit, "pagination.offset": offset}
            data = self._query_rest(endpoint, params)
            
            miners = []
            for miner_info in data.get("miners", []):
                trust_score = float(miner_info.get("trust_score", "0.0"))
                reputation = trust_score * 1000  # Convert to reputation score (0-1000 scale)
                
                miners.append({
                    "address": miner_info.get("address", ""),
                    "tier": miner_info.get("reputation_tier", "new"),
                    "total_submissions": miner_info.get("total_submissions", 0),
                    "reputation": reputation,
                    "trust_score": trust_score,
                    "successful_submissions": miner_info.get("successful_submissions", 0),
                    "slashing_events": miner_info.get("slashing_events", 0),
                    "last_submission_height": miner_info.get("last_submission_height", 0),
                })
            
            return {
                "miners": miners,
                "total": data.get("total", len(miners)),
            }
        except Exception as e:
            logger.error(f"Failed to get all miners: {e}")
            return {"miners": [], "total": 0}
    
    def get_validator_info(self, validator_address: str) -> Optional[Dict[str, Any]]:
        """
        Get validator information (trust score, uptime, voting power).
        
        Args:
            validator_address: Validator operator address
            
        Returns:
            Validator info or None
        """
        try:
            # Query validator from Cosmos SDK staking module
            endpoint = f"/cosmos/staking/v1beta1/validators/{validator_address}"
            staking_data = self._query_rest(endpoint)
            
            if "validator" not in staking_data:
                return None
            
            validator = staking_data["validator"]
            
            # Extract validator information from staking module
            operator_address = validator.get("operator_address", validator_address)
            status = validator.get("status", "UNBONDED")
            tokens = validator.get("tokens", "0")
            delegator_shares = validator.get("delegator_shares", "0")
            commission_rate = validator.get("commission", {}).get("commission_rates", {}).get("rate", "0")
            moniker = validator.get("description", {}).get("moniker", "")
            
            # Convert tokens and shares to float
            try:
                tokens_float = float(tokens) / 1_000_000.0  # Convert from base unit to main unit
                delegator_shares_float = float(delegator_shares) / 1_000_000.0
                commission_rate_float = float(commission_rate) if commission_rate else 0.0
            except (ValueError, TypeError):
                tokens_float = 0.0
                delegator_shares_float = 0.0
                commission_rate_float = 0.0
            
            # Get validator verification record from R3MES keeper for trust score
            trust_score = 0.0
            total_verifications = 0
            successful_verifications = 0
            false_verdicts = 0
            lazy_validation_count = 0
            
            try:
                # Query ValidatorVerificationRecord from R3MES keeper
                # Note: This endpoint may not exist yet, so we'll try and handle gracefully
                verification_endpoint = f"/remes/remes/v1/validator_verification_record/{validator_address}"
                verification_data = self._query_rest(verification_endpoint)
                
                if "verification_record" in verification_data:
                    record = verification_data["verification_record"]
                    total_verifications = record.get("total_verifications", 0)
                    successful_verifications = record.get("successful_verifications", 0)
                    false_verdicts = record.get("false_verdicts", 0)
                    lazy_validation_count = record.get("lazy_validation_count", 0)
                    
                    # Calculate trust score from verification record
                    if total_verifications > 0:
                        success_rate = successful_verifications / total_verifications
                        # Trust score = success_rate - (false_verdicts_penalty + lazy_penalty)
                        false_penalty = (false_verdicts / total_verifications) * 0.5  # 50% penalty per false verdict
                        lazy_penalty = (lazy_validation_count / total_verifications) * 0.1  # 10% penalty per lazy validation
                        trust_score = max(0.0, min(1.0, success_rate - false_penalty - lazy_penalty))
                    else:
                        trust_score = 0.5  # Default trust score for new validators
            except Exception as e:
                logger.debug(f"Could not fetch validator verification record: {e}, using default trust score")
                trust_score = 0.5  # Default trust score
            
            # Calculate uptime (simplified - would need historical data for accurate calculation)
            # For now, use status to determine if validator is active
            uptime_percentage = 100.0 if status == "BONDED" else 0.0
            
            # Voting power is proportional to tokens staked
            voting_power = tokens_float
            
            return {
                "operator_address": operator_address,
                "moniker": moniker,
                "status": status,
                "tokens": tokens_float,
                "delegator_shares": delegator_shares_float,
                "commission_rate": commission_rate_float,
                "voting_power": voting_power,
                "uptime_percentage": uptime_percentage,
                "trust_score": trust_score,
                "total_verifications": total_verifications,
                "successful_verifications": successful_verifications,
                "false_verdicts": false_verdicts,
                "lazy_validation_count": lazy_validation_count,
            }
        except Exception as e:
            logger.error(f"Failed to get validator info for {validator_address}: {e}")
            return None
    
    def get_all_validators(self, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """
        Get all validators with pagination.
        
        Args:
            limit: Maximum number of validators to return
            offset: Pagination offset
            
        Returns:
            Dictionary with validators list and total count
        """
        try:
            # Query validators from Cosmos SDK staking module
            endpoint = "/cosmos/staking/v1beta1/validators"
            params = {
                "pagination.limit": limit,
                "pagination.offset": offset,
            }
            staking_data = self._query_rest(endpoint, params)
            
            validators = []
            total = 0
            
            if "validators" in staking_data:
                validator_list = staking_data["validators"]
                total = staking_data.get("pagination", {}).get("total", len(validator_list))
                
                # Process validators directly from the list to avoid N+1 queries
                # Only fetch verification records for validators that need trust scores
                for validator in validator_list:
                    operator_address = validator.get("operator_address", "")
                    if not operator_address:
                        continue
                    
                    # Extract basic validator info directly (avoid extra HTTP call)
                    status = validator.get("status", "UNBONDED")
                    tokens = validator.get("tokens", "0")
                    delegator_shares = validator.get("delegator_shares", "0")
                    commission_rate = validator.get("commission", {}).get("commission_rates", {}).get("rate", "0")
                    moniker = validator.get("description", {}).get("moniker", "")
                    
                    # Convert tokens and shares to float
                    try:
                        tokens_float = float(tokens) / 1_000_000.0
                        delegator_shares_float = float(delegator_shares) / 1_000_000.0
                        commission_rate_float = float(commission_rate) if commission_rate else 0.0
                    except (ValueError, TypeError):
                        tokens_float = 0.0
                        delegator_shares_float = 0.0
                        commission_rate_float = 0.0
                    
                    # Default trust score (skip individual verification record queries for performance)
                    # Trust scores can be fetched on-demand via get_validator_info for detailed view
                    trust_score = 0.5  # Default trust score
                    uptime_percentage = 100.0 if status == "BONDED" else 0.0
                    voting_power = tokens_float
                    
                    validators.append({
                        "operator_address": operator_address,
                        "moniker": moniker,
                        "status": status,
                        "tokens": tokens_float,
                        "delegator_shares": delegator_shares_float,
                        "commission_rate": commission_rate_float,
                        "voting_power": voting_power,
                        "uptime_percentage": uptime_percentage,
                        "trust_score": trust_score,
                    })
            
            return {
                "validators": validators,
                "total": total,
            }
        except Exception as e:
            logger.error(f"Failed to get all validators: {e}")
            return {"validators": [], "total": 0}
    
    def get_network_statistics(self) -> Optional[Dict[str, Any]]:
        """
        Get network statistics from blockchain.
        
        Returns:
            Network statistics or None if query fails
        """
        try:
            endpoint = "/remes/remes/v1/dashboard/statistics"
            data = self._query_rest(endpoint)
            
            if "statistics" in data:
                stats = data["statistics"]
                return {
                    "total_miners": stats.get("total_miners", 0),
                    "active_miners": stats.get("active_miners", 0),
                    "total_gradients": stats.get("total_gradients", 0),
                    "total_aggregations": stats.get("total_aggregations", 0),
                    "pending_gradients": stats.get("pending_gradients", 0),
                    "pending_aggregations": stats.get("pending_aggregations", 0),
                    "average_gradient_size": stats.get("average_gradient_size", 0),
                    "last_updated": stats.get("last_updated", 0),
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get network statistics: {e}")
            return None
    
    def get_token_supply(self) -> Optional[Dict[str, Any]]:
        """
        Get token supply information from Cosmos SDK bank module.
        
        Returns:
            Token supply data or None if query fails
        """
        try:
            # Query total supply from Cosmos SDK bank module
            endpoint = "/cosmos/bank/v1beta1/supply"
            data = self._query_rest(endpoint)
            
            if "supply" in data:
                supply_list = data["supply"]
                total_supply = 0.0
                for coin in supply_list:
                    if coin.get("denom") == "uremes" or coin.get("denom") == "remes":
                        amount = float(coin.get("amount", "0"))
                        # Convert from base unit (uremes) to main unit (remes)
                        if coin.get("denom") == "uremes":
                            amount = amount / 1_000_000.0
                        total_supply = amount
                        break
                
                return {
                    "total_supply": total_supply,
                    "circulating_supply": total_supply,  # For now, assume all is circulating
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get token supply: {e}")
            return None
    
    def get_reward_params(self) -> Optional[Dict[str, Any]]:
        """
        Get reward parameters from R3MES module params.
        
        Returns:
            Reward parameters or None if query fails
        """
        try:
            # Query params from R3MES module
            endpoint = "/remes/remes/v1/params"
            data = self._query_rest(endpoint)
            
            if "params" in data:
                params = data["params"]
                # Extract reward-related params
                # Note: Actual param structure depends on keeper implementation
                return {
                    "base_reward_per_gradient": float(params.get("base_reward_per_gradient", "10.0")),
                    "miner_reward_ratio": float(params.get("miner_reward_ratio", "0.7")),
                    "validator_reward_ratio": float(params.get("validator_reward_ratio", "0.2")),
                    "treasury_ratio": float(params.get("treasury_ratio", "0.1")),
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get reward params: {e}")
            return None
    
    def get_staking_info(self) -> Optional[Dict[str, Any]]:
        """
        Get staking information from Cosmos SDK staking module.
        
        Returns:
            Staking data or None if query fails
        """
        try:
            # Query pool info from staking module
            endpoint = "/cosmos/staking/v1beta1/pool"
            data = self._query_rest(endpoint)
            
            if "pool" in data:
                pool = data["pool"]
                bonded = float(pool.get("bonded_tokens", "0"))
                not_bonded = float(pool.get("not_bonded_tokens", "0"))
                total = bonded + not_bonded
                
                # Convert from base unit to main unit
                bonded = bonded / 1_000_000.0
                not_bonded = not_bonded / 1_000_000.0
                total = total / 1_000_000.0
                
                return {
                    "total_stake": bonded,
                    "total_bonded": bonded,
                    "total_not_bonded": not_bonded,
                    "total_staking": total,
                    "staking_ratio": (bonded / total * 100) if total > 0 else 0.0,
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get staking info: {e}")
            return None
    
    def close(self):
        """Close client (no-op for REST client)."""
        pass


# Global client instance
_blockchain_client: Optional[BlockchainQueryClient] = None


def get_blockchain_client() -> BlockchainQueryClient:
    """Get global blockchain query client instance."""
    global _blockchain_client
    if _blockchain_client is None:
        _blockchain_client = BlockchainQueryClient()
    return _blockchain_client


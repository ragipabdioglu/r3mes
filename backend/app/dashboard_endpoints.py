"""
Dashboard API Endpoints for R3MES Web Dashboard

Provides REST API endpoints for network dashboard, including:
- Network status and statistics
- Miner locations for globe visualization
- Block information
- Real-time metrics

These endpoints are designed to work with the web dashboard's NetworkExplorer component.
"""

import logging
import hashlib
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from .blockchain_query_client import get_blockchain_client
from .cache_middleware import cache_response
from .config_manager import get_config_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["dashboard"])

# Rate limiter
limiter = Limiter(key_func=get_remote_address)
config_manager = get_config_manager()
config = config_manager.load()


# =============================================================================
# Response Models
# =============================================================================

class NetworkStatusResponse(BaseModel):
    """Network status for dashboard."""
    active_miners: int
    total_gradients: int
    model_updates: int
    block_height: int
    block_time: float
    network_hash_rate: float
    timestamp: int


class MinerLocationResponse(BaseModel):
    """Miner location for globe visualization."""
    address: str
    lat: float
    lng: float
    size: float
    status: str
    reputation: float


class MinerLocationsResponse(BaseModel):
    """Response containing miner locations."""
    locations: List[MinerLocationResponse]
    total: int


class NetworkStatsResponse(BaseModel):
    """Network statistics response."""
    active_miners: int
    total_users: int
    total_credits: float
    block_height: int


class BlockResponse(BaseModel):
    """Block information."""
    height: int
    hash: str
    miner: Optional[str]
    timestamp: str
    tx_count: int


# =============================================================================
# Known Data Center Locations for Globe Visualization
# =============================================================================

DATACENTER_LOCATIONS = [
    {"city": "New York", "country": "USA", "lat": 40.7128, "lng": -74.0060},
    {"city": "London", "country": "UK", "lat": 51.5074, "lng": -0.1278},
    {"city": "Tokyo", "country": "Japan", "lat": 35.6762, "lng": 139.6503},
    {"city": "Singapore", "country": "Singapore", "lat": 1.3521, "lng": 103.8198},
    {"city": "Frankfurt", "country": "Germany", "lat": 50.1109, "lng": 8.6821},
    {"city": "Sydney", "country": "Australia", "lat": -33.8688, "lng": 151.2093},
    {"city": "São Paulo", "country": "Brazil", "lat": -23.5505, "lng": -46.6333},
    {"city": "Dubai", "country": "UAE", "lat": 25.2048, "lng": 55.2708},
    {"city": "Seoul", "country": "South Korea", "lat": 37.5665, "lng": 126.9780},
    {"city": "Mumbai", "country": "India", "lat": 19.0760, "lng": 72.8777},
    {"city": "Toronto", "country": "Canada", "lat": 43.6532, "lng": -79.3832},
    {"city": "Amsterdam", "country": "Netherlands", "lat": 52.3676, "lng": 4.9041},
    {"city": "Paris", "country": "France", "lat": 48.8566, "lng": 2.3522},
    {"city": "Hong Kong", "country": "Hong Kong", "lat": 22.3193, "lng": 114.1694},
    {"city": "Los Angeles", "country": "USA", "lat": 34.0522, "lng": -118.2437},
]


def get_location_from_address(address: str) -> Dict[str, float]:
    """Generate deterministic location based on miner address hash."""
    hash_val = int(hashlib.sha256(address.encode()).hexdigest(), 16)
    location_index = hash_val % len(DATACENTER_LOCATIONS)
    
    # Add small offset for visual variety
    offset_lat = ((hash_val % 1000) - 500) / 5000
    offset_lng = ((hash_val % 2000) - 1000) / 5000
    
    loc = DATACENTER_LOCATIONS[location_index]
    return {
        "lat": loc["lat"] + offset_lat,
        "lng": loc["lng"] + offset_lng,
    }


# =============================================================================
# Dashboard Status Endpoint (for NetworkExplorer)
# =============================================================================

@router.get("/api/blockchain/dashboard/status", response_model=NetworkStatusResponse)
@limiter.limit(config.rate_limit_get)
@cache_response(ttl=5, key_prefix="dashboard_status")
async def get_dashboard_status(request: Request) -> NetworkStatusResponse:
    """
    Get network status for dashboard.
    
    This endpoint provides real-time network metrics for the NetworkExplorer component.
    Data is cached for 5 seconds to reduce blockchain query load.
    """
    try:
        blockchain_client = get_blockchain_client()
        
        # Get network statistics
        stats = blockchain_client.get_network_statistics()
        
        # Get latest block info
        block_height = 0
        block_time = 6.0  # Default block time
        
        try:
            # Query latest block from Tendermint RPC
            import httpx
            rpc_url = blockchain_client.rest_url.replace(":1317", ":26657")
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{rpc_url}/status")
                if response.status_code == 200:
                    data = response.json()
                    sync_info = data.get("result", {}).get("sync_info", {})
                    block_height = int(sync_info.get("latest_block_height", 0))
        except Exception as e:
            logger.debug(f"Could not fetch block height from RPC: {e}")
            # Fallback: try REST API
            try:
                endpoint = "/cosmos/base/tendermint/v1beta1/blocks/latest"
                block_data = blockchain_client._query_rest(endpoint)
                block_height = int(block_data.get("block", {}).get("header", {}).get("height", 0))
            except Exception:
                pass
        
        # Calculate network hash rate (gradients per hour)
        total_gradients = stats.get("total_gradients", 0) if stats else 0
        active_miners = stats.get("active_miners", 0) if stats else 0
        
        # Estimate hash rate based on recent activity
        # In production, this would be calculated from actual gradient submission rate
        network_hash_rate = active_miners * 2.5  # Approximate gradients/hour per miner
        
        return NetworkStatusResponse(
            active_miners=active_miners,
            total_gradients=total_gradients,
            model_updates=stats.get("total_aggregations", 0) if stats else 0,
            block_height=block_height,
            block_time=block_time,
            network_hash_rate=network_hash_rate,
            timestamp=int(time.time() * 1000),
        )
        
    except Exception as e:
        logger.error(f"Failed to get dashboard status: {e}", exc_info=True)
        # Return default values instead of error to keep dashboard functional
        return NetworkStatusResponse(
            active_miners=0,
            total_gradients=0,
            model_updates=0,
            block_height=0,
            block_time=6.0,
            network_hash_rate=0.0,
            timestamp=int(time.time() * 1000),
        )


# =============================================================================
# Miner Locations Endpoint (for Globe Visualization)
# =============================================================================

@router.get("/api/blockchain/dashboard/locations", response_model=MinerLocationsResponse)
@limiter.limit(config.rate_limit_get)
@cache_response(ttl=30, key_prefix="dashboard_locations")
async def get_miner_locations(
    request: Request,
    limit: int = Query(default=100, ge=1, le=500, description="Maximum miners to return"),
) -> MinerLocationsResponse:
    """
    Get miner locations for globe visualization.
    
    Returns miner coordinates for displaying on the network globe.
    Locations are deterministically generated from miner addresses.
    """
    try:
        blockchain_client = get_blockchain_client()
        
        # Get all miners
        miners_data = blockchain_client.get_all_miners(limit=limit, offset=0)
        miners = miners_data.get("miners", [])
        
        locations: List[MinerLocationResponse] = []
        
        for miner in miners:
            address = miner.get("address", "")
            if not address:
                continue
            
            # Get location from address hash
            loc = get_location_from_address(address)
            
            # Calculate size based on reputation (0.1 to 1.0)
            reputation = miner.get("reputation", 50)
            size = 0.1 + (reputation / 100) * 0.9
            
            locations.append(MinerLocationResponse(
                address=address,
                lat=loc["lat"],
                lng=loc["lng"],
                size=size,
                status=miner.get("status", "active"),
                reputation=reputation,
            ))
        
        return MinerLocationsResponse(
            locations=locations,
            total=len(locations),
        )
        
    except Exception as e:
        logger.error(f"Failed to get miner locations: {e}", exc_info=True)
        # Return empty list instead of error
        return MinerLocationsResponse(locations=[], total=0)


# =============================================================================
# Miners List Endpoint (for MinersTable)
# =============================================================================

@router.get("/api/blockchain/dashboard/miners")
@limiter.limit(config.rate_limit_get)
@cache_response(ttl=30, key_prefix="dashboard_miners")
async def get_dashboard_miners(
    request: Request,
    limit: int = Query(default=100, ge=1, le=500, description="Maximum miners to return"),
) -> Dict[str, Any]:
    """
    Get miners list for MinersTable component.
    
    Returns miners with their stats, tier, and status.
    """
    try:
        blockchain_client = get_blockchain_client()
        
        # Get all miners
        miners_data = blockchain_client.get_all_miners(limit=limit, offset=0)
        miners = miners_data.get("miners", [])
        
        # Transform miners for frontend
        result_miners = []
        for miner in miners:
            address = miner.get("address", "")
            reputation = miner.get("reputation", miner.get("trust_score", 0) * 100)
            
            result_miners.append({
                "address": address,
                "moniker": miner.get("moniker", f"Miner-{address[:8]}"),
                "reputation_score": reputation,
                "total_staked": miner.get("total_staked", "0"),
                "active_tasks": miner.get("active_tasks", 0),
                "completed_tasks": miner.get("completed_tasks", miner.get("total_submissions", 0)),
                "gpu_count": miner.get("gpu_count", 1),
                "status": "active" if miner.get("status", "active") == "active" else "inactive",
            })
        
        return {
            "miners": result_miners,
            "total": len(result_miners),
        }
        
    except Exception as e:
        logger.error(f"Failed to get dashboard miners: {e}", exc_info=True)
        return {"miners": [], "total": 0}


# =============================================================================
# Network Stats Endpoint (for Network Page)
# =============================================================================

@router.get("/network/stats", response_model=NetworkStatsResponse)
@limiter.limit(config.rate_limit_get)
@cache_response(ttl=15, key_prefix="network_stats")
async def get_network_stats(request: Request) -> NetworkStatsResponse:
    """
    Get network statistics for the network page.
    
    Returns active miners, total users, credits, and block height.
    """
    try:
        blockchain_client = get_blockchain_client()
        
        # Get network statistics
        stats = blockchain_client.get_network_statistics()
        staking_info = blockchain_client.get_staking_info()
        
        # Get block height
        block_height = 0
        try:
            endpoint = "/cosmos/base/tendermint/v1beta1/blocks/latest"
            block_data = blockchain_client._query_rest(endpoint)
            block_height = int(block_data.get("block", {}).get("header", {}).get("height", 0))
        except Exception:
            pass
        
        # Calculate total credits (staked tokens)
        total_credits = staking_info.get("total_stake", 0) if staking_info else 0
        
        return NetworkStatsResponse(
            active_miners=stats.get("active_miners", 0) if stats else 0,
            total_users=stats.get("total_miners", 0) if stats else 0,
            total_credits=total_credits,
            block_height=block_height,
        )
        
    except Exception as e:
        logger.error(f"Failed to get network stats: {e}", exc_info=True)
        return NetworkStatsResponse(
            active_miners=0,
            total_users=0,
            total_credits=0,
            block_height=0,
        )


# =============================================================================
# Blocks Endpoint
# =============================================================================

@router.get("/blocks")
@limiter.limit(config.rate_limit_get)
@cache_response(ttl=10, key_prefix="recent_blocks")
async def get_recent_blocks(
    request: Request,
    limit: int = Query(default=10, ge=1, le=100, description="Number of blocks to return"),
) -> Dict[str, Any]:
    """
    Get recent blocks for the network page.
    
    Returns the most recent blocks with their details.
    """
    return await _fetch_blocks(limit)


@router.get("/api/blockchain/dashboard/blocks")
@limiter.limit(config.rate_limit_get)
@cache_response(ttl=10, key_prefix="dashboard_blocks")
async def get_dashboard_blocks(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100, description="Number of blocks to return"),
) -> Dict[str, Any]:
    """
    Get recent blocks for RecentBlocks component.
    
    Returns blocks with time, hash, tx_count, and proposer.
    """
    return await _fetch_blocks(limit)


async def _fetch_blocks(limit: int) -> Dict[str, Any]:
    """Shared function to fetch blocks."""
    try:
        blockchain_client = get_blockchain_client()
        blocks: List[Dict[str, Any]] = []
        
        # Get latest block height
        try:
            endpoint = "/cosmos/base/tendermint/v1beta1/blocks/latest"
            latest_block = blockchain_client._query_rest(endpoint)
            latest_height = int(latest_block.get("block", {}).get("header", {}).get("height", 0))
        except Exception as e:
            logger.warning(f"Could not get latest block: {e}")
            return {"blocks": [], "limit": limit, "total": 0}
        
        # Fetch recent blocks
        for i in range(min(limit, latest_height)):
            height = latest_height - i
            if height <= 0:
                break
            
            try:
                endpoint = f"/cosmos/base/tendermint/v1beta1/blocks/{height}"
                block_data = blockchain_client._query_rest(endpoint)
                
                block = block_data.get("block", {})
                header = block.get("header", {})
                
                # Get block hash from block_id
                block_id = block_data.get("block_id", {})
                block_hash = block_id.get("hash", "")
                
                # Get proposer (miner) address
                proposer = header.get("proposer_address", "")
                
                # Get transaction count
                txs = block.get("data", {}).get("txs", [])
                tx_count = len(txs) if txs else 0
                
                blocks.append({
                    "height": height,
                    "hash": block_hash,
                    "miner": proposer,
                    "proposer": proposer,
                    "timestamp": header.get("time", ""),
                    "time": header.get("time", ""),
                    "tx_count": tx_count,
                    "validator_set_hash": header.get("validators_hash", ""),
                })
            except Exception as e:
                logger.debug(f"Could not fetch block {height}: {e}")
                continue
        
        return {
            "blocks": blocks,
            "limit": limit,
            "total": latest_height,
        }
        
    except Exception as e:
        logger.error(f"Failed to get recent blocks: {e}", exc_info=True)
        return {"blocks": [], "limit": limit, "total": 0}



# =============================================================================
# User Transaction History Endpoint
# =============================================================================

@router.get("/user/{address}/transactions")
@limiter.limit(config.rate_limit_get)
@cache_response(ttl=30, key_prefix="user_transactions")
async def get_user_transactions(
    request: Request,
    address: str,
    limit: int = Query(default=50, ge=1, le=200, description="Number of transactions to return"),
) -> Dict[str, Any]:
    """
    Get transaction history for a user address.
    
    Queries blockchain for transactions involving the specified address.
    """
    try:
        blockchain_client = get_blockchain_client()
        transactions: List[Dict[str, Any]] = []
        
        # Query transactions from Cosmos SDK tx module
        try:
            # Query sent transactions
            sent_endpoint = "/cosmos/tx/v1beta1/txs"
            sent_params = {
                "events": f"transfer.sender='{address}'",
                "pagination.limit": str(limit),
                "order_by": "ORDER_BY_DESC",
            }
            sent_data = blockchain_client._query_rest(sent_endpoint, sent_params)
            
            for tx in sent_data.get("tx_responses", []):
                tx_hash = tx.get("txhash", "")
                timestamp = tx.get("timestamp", "")
                height = int(tx.get("height", 0))
                
                # Parse transaction messages
                for msg in tx.get("tx", {}).get("body", {}).get("messages", []):
                    msg_type = msg.get("@type", "")
                    
                    if "MsgSend" in msg_type:
                        amount_list = msg.get("amount", [])
                        amount = int(amount_list[0].get("amount", 0)) if amount_list else 0
                        
                        transactions.append({
                            "id": len(transactions) + 1,
                            "hash": tx_hash,
                            "type": "send",
                            "amount": amount / 1_000_000,  # Convert to main unit
                            "from": msg.get("from_address", address),
                            "to": msg.get("to_address", ""),
                            "timestamp": int(datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp() * 1000) if timestamp else 0,
                            "status": "confirmed",
                            "block_height": height,
                        })
                    elif "MsgDelegate" in msg_type:
                        amount_obj = msg.get("amount", {})
                        amount = int(amount_obj.get("amount", 0))
                        
                        transactions.append({
                            "id": len(transactions) + 1,
                            "hash": tx_hash,
                            "type": "stake",
                            "amount": amount / 1_000_000,
                            "from": address,
                            "to": msg.get("validator_address", ""),
                            "timestamp": int(datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp() * 1000) if timestamp else 0,
                            "status": "confirmed",
                            "block_height": height,
                        })
        except Exception as e:
            logger.debug(f"Could not fetch sent transactions: {e}")
        
        # Query received transactions
        try:
            recv_endpoint = "/cosmos/tx/v1beta1/txs"
            recv_params = {
                "events": f"transfer.recipient='{address}'",
                "pagination.limit": str(limit),
                "order_by": "ORDER_BY_DESC",
            }
            recv_data = blockchain_client._query_rest(recv_endpoint, recv_params)
            
            for tx in recv_data.get("tx_responses", []):
                tx_hash = tx.get("txhash", "")
                
                # Skip if already processed
                if any(t.get("hash") == tx_hash for t in transactions):
                    continue
                
                timestamp = tx.get("timestamp", "")
                height = int(tx.get("height", 0))
                
                for msg in tx.get("tx", {}).get("body", {}).get("messages", []):
                    msg_type = msg.get("@type", "")
                    
                    if "MsgSend" in msg_type and msg.get("to_address") == address:
                        amount_list = msg.get("amount", [])
                        amount = int(amount_list[0].get("amount", 0)) if amount_list else 0
                        
                        transactions.append({
                            "id": len(transactions) + 1,
                            "hash": tx_hash,
                            "type": "receive",
                            "amount": amount / 1_000_000,
                            "from": msg.get("from_address", ""),
                            "to": address,
                            "timestamp": int(datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp() * 1000) if timestamp else 0,
                            "status": "confirmed",
                            "block_height": height,
                        })
        except Exception as e:
            logger.debug(f"Could not fetch received transactions: {e}")
        
        # Sort by timestamp descending
        transactions.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        
        return {
            "transactions": transactions[:limit],
            "total": len(transactions),
        }
        
    except Exception as e:
        logger.error(f"Failed to get transactions for {address}: {e}", exc_info=True)
        return {"transactions": [], "total": 0}


# =============================================================================
# Network Statistics Endpoint (for NetworkStats component)
# =============================================================================

@router.get("/api/blockchain/dashboard/statistics")
@limiter.limit(config.rate_limit_get)
@cache_response(ttl=30, key_prefix="dashboard_statistics")
async def get_dashboard_statistics(request: Request) -> Dict[str, Any]:
    """
    Get detailed network statistics for NetworkStats component.
    
    Returns staking info, inflation, validators, miners, and hashrate.
    """
    try:
        blockchain_client = get_blockchain_client()
        
        # Get network statistics
        stats = blockchain_client.get_network_statistics()
        staking_info = blockchain_client.get_staking_info()
        validators_data = blockchain_client.get_all_validators(limit=100, offset=0)
        
        # Count active validators
        active_validators = sum(
            1 for v in validators_data.get("validators", [])
            if v.get("status") == "BONDED"
        )
        
        # Get inflation rate (from mint module)
        inflation_rate = "0.0"
        try:
            mint_endpoint = "/cosmos/mint/v1beta1/inflation"
            mint_data = blockchain_client._query_rest(mint_endpoint)
            inflation_rate = mint_data.get("inflation", "0.0")
        except Exception:
            pass
        
        # Calculate network hashrate (gradients per hour)
        total_miners = stats.get("total_miners", 0) if stats else 0
        active_miners = stats.get("active_miners", 0) if stats else 0
        network_hashrate = active_miners * 2.5  # Approximate gradients/hour per miner
        
        return {
            "total_stake": str(int(staking_info.get("total_stake", 0) * 1_000_000)) if staking_info else "0",
            "inflation_rate": inflation_rate,
            "model_version": "BitNet b1.58",
            "active_validators": active_validators,
            "total_miners": total_miners,
            "network_hashrate": network_hashrate,
            "bonded_tokens": str(int(staking_info.get("total_bonded", 0) * 1_000_000)) if staking_info else "0",
            "unbonded_tokens": str(int(staking_info.get("total_not_bonded", 0) * 1_000_000)) if staking_info else "0",
        }
        
    except Exception as e:
        logger.error(f"Failed to get dashboard statistics: {e}", exc_info=True)
        return {
            "total_stake": "0",
            "inflation_rate": "0.0",
            "model_version": "BitNet b1.58",
            "active_validators": 0,
            "total_miners": 0,
            "network_hashrate": 0,
            "bonded_tokens": "0",
            "unbonded_tokens": "0",
        }

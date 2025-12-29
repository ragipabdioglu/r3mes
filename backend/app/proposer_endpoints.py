"""
Proposer API Endpoints

Provides API endpoints for proposer node management, aggregations, and gradient pool.
"""

from fastapi import APIRouter, HTTPException, Request, Path, Query
from slowapi import Limiter
from slowapi.util import get_remote_address
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
import logging

from .database_async import AsyncDatabase
from .config_manager import get_config_manager
from .blockchain_query_client import get_blockchain_client
from .input_validation import (
    validate_wallet_address,
    validate_pagination,
    validate_status_filter,
    InvalidInputError,
    InvalidWalletAddressError,
)
from .exceptions import BlockchainError, BlockchainQueryError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/proposer", tags=["proposer"])

# Rate limiter (will use app.state.limiter from main.py)
limiter = Limiter(key_func=get_remote_address)

config_manager = get_config_manager()
config = config_manager.load()
database = AsyncDatabase(db_path=config.database_path, chain_json_path=config.chain_json_path)


class ProposerNode(BaseModel):
    """Proposer node model."""
    node_address: str
    status: str
    total_aggregations: int
    total_rewards: str
    last_aggregation_height: Optional[int] = None


class AggregationRecord(BaseModel):
    """Aggregation record model."""
    aggregation_id: int
    proposer: str
    aggregated_gradient_ipfs_hash: str
    merkle_root: str
    participant_count: int
    training_round_id: int
    block_height: Optional[int] = None
    timestamp: Optional[str] = None


class GradientPool(BaseModel):
    """Gradient pool model."""
    pending_gradients: List[Dict[str, Any]]
    total_count: int
    total_size_bytes: Optional[int] = None


@router.get("/nodes")
@limiter.limit(config.rate_limit_get)
async def list_proposer_nodes(request: Request, limit: int = 100, offset: int = 0):
    """
    List all proposer nodes.
    
    Args:
        limit: Maximum number of nodes to return (default: 100)
        offset: Pagination offset (default: 0)
        
    Returns:
        List of proposer nodes
    """
    try:
        blockchain_client = get_blockchain_client()
        
        # Query all node registrations from blockchain
        endpoint = "/remes/remes/v1/node_registrations"
        params = {
            "pagination.limit": limit,
            "pagination.offset": offset,
        }
        
        try:
            data = blockchain_client._query_rest(endpoint, params)
        except Exception as e:
            logger.warning(f"Failed to query node registrations: {e}")
            return {
                "nodes": [],
                "total": 0,
            }
        
        proposer_nodes = []
        node_registrations = data.get("node_registrations", [])
        
        for node_reg in node_registrations:
            # Check if node has proposer role (NODE_TYPE_PROPOSER = 4)
            roles = node_reg.get("roles", [])
            has_proposer_role = 4 in roles or "NODE_TYPE_PROPOSER" in [str(r) for r in roles]
            
            if not has_proposer_role:
                continue
            
            node_address = node_reg.get("node_address", "")
            if not node_address:
                continue
            
            proposer_nodes.append({
                "node_address": node_address,
                "status": node_reg.get("status", "UNSPECIFIED"),
                "total_aggregations": 0,  # Would need to query from aggregations
                "total_rewards": "0",
                "last_aggregation_height": None,
                "resources": node_reg.get("resources", {}),
                "stake": node_reg.get("stake", ""),
            })
        
        return {
            "nodes": proposer_nodes,
            "total": len(proposer_nodes),
        }
    except Exception as e:
        logger.error(f"Error listing proposer nodes: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch proposer nodes")


@router.get("/nodes/{address}")
@limiter.limit(config.rate_limit_get)
async def get_proposer_node(
    request: Request,
    address: str = Path(..., description="Proposer node address")
):
    """
    Get proposer node details.
    
    Args:
        address: Proposer node address
        
    Returns:
        Proposer node details
    """
    try:
        blockchain_client = get_blockchain_client()
        
        # Get node registration
        try:
            reg_endpoint = f"/remes/remes/v1/node_registration/{address}"
            reg_data = blockchain_client._query_rest(reg_endpoint)
            
            if "node_registration" not in reg_data:
                raise HTTPException(status_code=404, detail="Proposer node not found")
            
            node_reg = reg_data["node_registration"]
            roles = node_reg.get("roles", [])
            has_proposer_role = 4 in roles or "NODE_TYPE_PROPOSER" in [str(r) for r in roles]
            
            if not has_proposer_role:
                raise HTTPException(status_code=404, detail="Node does not have proposer role")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get node registration: {e}")
            raise HTTPException(status_code=404, detail="Proposer node not found")
        
        # Query aggregations by this proposer (would need aggregation list endpoint)
        # For now, return basic info
        return {
            "node_address": address,
            "status": node_reg.get("status", "UNSPECIFIED"),
            "total_aggregations": 0,  # Would query from aggregations
            "total_rewards": "0",
            "last_aggregation_height": None,
            "resources": node_reg.get("resources", {}),
            "stake": node_reg.get("stake", ""),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting proposer node {address}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch proposer node details")


@router.get("/aggregations")
@limiter.limit(config.rate_limit_get)
async def list_aggregations(
    request: Request,
    limit: int = Query(default=50, ge=1, le=500, description="Maximum number of aggregations to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    proposer: Optional[str] = Query(default=None, description="Filter by proposer address"),
    training_round_id: Optional[int] = Query(default=None, ge=0, description="Filter by training round ID")
):
    """
    List recent aggregations.
    
    Queries blockchain events and state for aggregation history.
    
    Args:
        limit: Maximum number of aggregations to return (default: 50)
        offset: Pagination offset (default: 0)
        proposer: Filter by proposer address (optional)
        training_round_id: Filter by training round ID (optional)
        
    Returns:
        List of aggregations with pagination
    """
    try:
        blockchain_client = get_blockchain_client()
        aggregations_list = []
        
        # Method 1: Query from blockchain events via Tendermint RPC
        try:
            events_endpoint = "/cosmos/tx/v1beta1/txs"
            events_params = {
                "events": "aggregation_submitted.proposer EXISTS",
                "pagination.limit": str(limit * 2),
                "pagination.offset": str(offset),
                "order_by": "ORDER_BY_DESC",
            }
            
            # Add proposer filter if provided
            if proposer:
                events_params["events"] = f"aggregation_submitted.proposer='{proposer}'"
            
            events_data = blockchain_client._query_rest(events_endpoint, events_params)
            
            tx_responses = events_data.get("tx_responses", [])
            
            for tx in tx_responses:
                for event in tx.get("events", []):
                    if event.get("type") == "aggregation_submitted":
                        attrs = {attr["key"]: attr["value"] for attr in event.get("attributes", [])}
                        
                        # Apply training_round_id filter if provided
                        round_id = int(attrs.get("training_round_id", 0))
                        if training_round_id is not None and round_id != training_round_id:
                            continue
                        
                        aggregations_list.append({
                            "aggregation_id": int(attrs.get("aggregation_id", 0)),
                            "proposer": attrs.get("proposer", ""),
                            "aggregated_gradient_ipfs_hash": attrs.get("aggregated_gradient_ipfs_hash", ""),
                            "merkle_root": attrs.get("merkle_root", ""),
                            "participant_count": int(attrs.get("participant_count", 0)),
                            "training_round_id": round_id,
                            "block_height": int(tx.get("height", 0)),
                            "timestamp": tx.get("timestamp", ""),
                            "tx_hash": tx.get("txhash", ""),
                        })
                        
                        if len(aggregations_list) >= limit:
                            break
        except Exception as e:
            logger.debug(f"Could not query aggregations from blockchain events: {e}")
        
        # Method 2: Query from local database (if indexed)
        if not aggregations_list:
            try:
                db_aggregations = await database.get_aggregations(
                    limit=limit,
                    offset=offset,
                    proposer=proposer,
                    training_round_id=training_round_id
                )
                if db_aggregations:
                    aggregations_list = db_aggregations
            except Exception as e:
                logger.debug(f"Could not query aggregations from database: {e}")
        
        # Method 3: Query from blockchain state
        if not aggregations_list:
            try:
                agg_endpoint = "/remes/remes/v1/aggregations"
                agg_params = {
                    "pagination.limit": str(limit * 2),
                    "pagination.offset": str(offset),
                    "pagination.reverse": "true",
                }
                
                agg_data = blockchain_client._query_rest(agg_endpoint, agg_params)
                
                for agg in agg_data.get("aggregations", []):
                    # Apply filters
                    if proposer and agg.get("proposer") != proposer:
                        continue
                    if training_round_id is not None and agg.get("training_round_id") != training_round_id:
                        continue
                    
                    aggregations_list.append({
                        "aggregation_id": agg.get("aggregation_id", 0),
                        "proposer": agg.get("proposer", ""),
                        "aggregated_gradient_ipfs_hash": agg.get("aggregated_gradient_ipfs_hash", ""),
                        "merkle_root": agg.get("merkle_root", ""),
                        "participant_count": agg.get("participant_count", 0),
                        "training_round_id": agg.get("training_round_id", 0),
                        "block_height": agg.get("block_height"),
                        "timestamp": agg.get("timestamp"),
                    })
                    
                    if len(aggregations_list) >= limit:
                        break
            except Exception as e:
                logger.debug(f"Could not query aggregations from blockchain state: {e}")
        
        return {
            "aggregations": aggregations_list[:limit],
            "total": len(aggregations_list),
            "limit": limit,
            "offset": offset,
        }
    except Exception as e:
        logger.error(f"Error listing aggregations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch aggregations")


@router.get("/aggregations/{aggregation_id}")
@limiter.limit(config.rate_limit_get)
async def get_aggregation(
    request: Request,
    aggregation_id: int = Path(..., description="Aggregation ID")
):
    """
    Get aggregation details.
    
    Args:
        aggregation_id: Aggregation ID
        
    Returns:
        Aggregation details
    """
    try:
        blockchain_client = get_blockchain_client()
        
        # Query aggregation from blockchain
        endpoint = f"/remes/remes/v1/aggregation/{aggregation_id}"
        
        try:
            data = blockchain_client._query_rest(endpoint)
            
            if "aggregation" not in data:
                raise HTTPException(status_code=404, detail="Aggregation not found")
            
            agg = data["aggregation"]
            
            return {
                "aggregation_id": agg.get("aggregation_id", aggregation_id),
                "proposer": agg.get("proposer", ""),
                "aggregated_gradient_ipfs_hash": agg.get("aggregated_gradient_ipfs_hash", ""),
                "merkle_root": agg.get("merkle_root", ""),
                "participant_count": agg.get("participant_count", 0),
                "training_round_id": agg.get("training_round_id", 0),
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to query aggregation: {e}")
            raise HTTPException(status_code=404, detail="Aggregation not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting aggregation {aggregation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch aggregation")


@router.get("/pool")
@limiter.limit(config.rate_limit_get)
async def get_gradient_pool(
    request: Request,
    limit: int = 100,
    offset: int = 0,
    status: str = "pending"
):
    """
    Get pending gradients pool.
    
    Args:
        limit: Maximum number of gradients to return (default: 100)
        offset: Pagination offset (default: 0)
        status: Gradient status filter (default: "pending")
        
    Returns:
        Gradient pool with pending gradients
    """
    try:
        blockchain_client = get_blockchain_client()
        
        # Query stored gradients from blockchain
        endpoint = "/remes/remes/v1/stored_gradient"
        params = {
            "pagination.limit": limit,
            "pagination.offset": offset,
        }
        
        try:
            data = blockchain_client._query_rest(endpoint, params)
        except Exception as e:
            logger.warning(f"Failed to query stored gradients: {e}")
            return {
                "pending_gradients": [],
                "total_count": 0,
            }
        
        gradients = data.get("stored_gradient", [])
        
        # Filter by status
        pending_gradients = [
            grad for grad in gradients
            if grad.get("status", "").lower() == status.lower()
        ]
        
        return {
            "pending_gradients": pending_gradients,
            "total_count": len(pending_gradients),
        }
    except Exception as e:
        logger.error(f"Error getting gradient pool: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch gradient pool")


@router.get("/commits")
@limiter.limit(config.rate_limit_get)
async def list_pending_commits(
    request: Request,
    limit: int = 50,
    offset: int = 0,
    proposer: Optional[str] = None,
    status: Optional[str] = None
):
    """
    Get pending aggregation commitments.
    
    Queries blockchain for pending aggregation commitments that proposers
    have submitted but not yet finalized.
    
    Args:
        limit: Maximum number of commits to return (default: 50)
        offset: Pagination offset (default: 0)
        proposer: Filter by proposer address (optional)
        status: Filter by status: "pending", "committed", "revealed" (optional)
        
    Returns:
        List of pending commitments with pagination
    """
    try:
        blockchain_client = get_blockchain_client()
        commits_list = []
        
        # Method 1: Query from blockchain events
        try:
            events_endpoint = "/cosmos/tx/v1beta1/txs"
            events_params = {
                "events": "aggregation_commitment.proposer EXISTS",
                "pagination.limit": str(limit * 2),
                "pagination.offset": str(offset),
                "order_by": "ORDER_BY_DESC",
            }
            
            if proposer:
                events_params["events"] = f"aggregation_commitment.proposer='{proposer}'"
            
            events_data = blockchain_client._query_rest(events_endpoint, events_params)
            
            tx_responses = events_data.get("tx_responses", [])
            
            for tx in tx_responses:
                for event in tx.get("events", []):
                    if event.get("type") in ["aggregation_commitment", "commit_aggregation"]:
                        attrs = {attr["key"]: attr["value"] for attr in event.get("attributes", [])}
                        
                        commit_status = attrs.get("status", "pending")
                        
                        # Apply status filter if provided
                        if status and commit_status.lower() != status.lower():
                            continue
                        
                        commits_list.append({
                            "commitment_id": attrs.get("commitment_id", ""),
                            "proposer": attrs.get("proposer", ""),
                            "commitment_hash": attrs.get("commitment_hash", ""),
                            "training_round_id": int(attrs.get("training_round_id", 0)),
                            "gradient_count": int(attrs.get("gradient_count", 0)),
                            "status": commit_status,
                            "commit_height": int(tx.get("height", 0)),
                            "commit_time": tx.get("timestamp", ""),
                            "reveal_deadline": attrs.get("reveal_deadline", ""),
                            "tx_hash": tx.get("txhash", ""),
                        })
                        
                        if len(commits_list) >= limit:
                            break
        except Exception as e:
            logger.debug(f"Could not query commits from blockchain events: {e}")
        
        # Method 2: Query from local database
        if not commits_list:
            try:
                db_commits = await database.get_aggregation_commits(
                    limit=limit,
                    offset=offset,
                    proposer=proposer,
                    status=status
                )
                if db_commits:
                    commits_list = db_commits
            except Exception as e:
                logger.debug(f"Could not query commits from database: {e}")
        
        # Method 3: Query from blockchain state
        if not commits_list:
            try:
                commits_endpoint = "/remes/remes/v1/aggregation_commitments"
                commits_params = {
                    "pagination.limit": str(limit * 2),
                    "pagination.offset": str(offset),
                }
                
                commits_data = blockchain_client._query_rest(commits_endpoint, commits_params)
                
                for commit in commits_data.get("aggregation_commitments", []):
                    # Apply filters
                    if proposer and commit.get("proposer") != proposer:
                        continue
                    
                    commit_status = commit.get("status", "pending")
                    if status and commit_status.lower() != status.lower():
                        continue
                    
                    commits_list.append({
                        "commitment_id": commit.get("commitment_id", ""),
                        "proposer": commit.get("proposer", ""),
                        "commitment_hash": commit.get("commitment_hash", ""),
                        "training_round_id": commit.get("training_round_id", 0),
                        "gradient_count": commit.get("gradient_count", 0),
                        "status": commit_status,
                        "commit_height": commit.get("commit_height"),
                        "commit_time": commit.get("commit_time"),
                        "reveal_deadline": commit.get("reveal_deadline"),
                    })
                    
                    if len(commits_list) >= limit:
                        break
            except Exception as e:
                logger.debug(f"Could not query commits from blockchain state: {e}")
        
        return {
            "commits": commits_list[:limit],
            "total": len(commits_list),
            "limit": limit,
            "offset": offset,
        }
    except Exception as e:
        logger.error(f"Error listing commits: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch commits")


@router.post("/aggregate")
@limiter.limit(config.rate_limit_get)
async def submit_aggregation(
    request: Request,
    proposer: str,
    gradient_ids: List[int],
    aggregated_hash: str,
    merkle_root: str,
    training_round_id: int
):
    """
    Submit aggregation (via blockchain).
    
    Note: This endpoint is informational. Actual aggregation submission must be done
    via blockchain transaction (MsgSubmitAggregation).
    
    Args:
        proposer: Proposer address
        gradient_ids: List of gradient IDs to aggregate
        aggregated_hash: IPFS hash of aggregated gradient
        merkle_root: Merkle root of included gradients
        training_round_id: Training round ID
        
    Returns:
        Success message
    """
    return {
        "message": "Aggregation submission initiated. Actual submission requires blockchain transaction.",
        "note": "Use blockchain client to send MsgSubmitAggregation transaction.",
        "proposer": proposer,
        "gradient_count": len(gradient_ids),
    }


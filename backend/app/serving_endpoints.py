"""
Serving Node API Endpoints

Provides API endpoints for serving node management, inference requests, and statistics.
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

router = APIRouter(prefix="/serving", tags=["serving"])

# Rate limiter (will use app.state.limiter from main.py)
limiter = Limiter(key_func=get_remote_address)

config_manager = get_config_manager()
config = config_manager.load()
database = AsyncDatabase(db_path=config.database_path, chain_json_path=config.chain_json_path)


class ServingNodeStatus(BaseModel):
    """Serving node status model."""
    node_address: str
    model_version: str
    model_ipfs_hash: Optional[str] = None
    is_available: bool
    total_requests: int
    successful_requests: int
    average_latency_ms: float
    last_heartbeat: Optional[str] = None


class InferenceRequest(BaseModel):
    """Inference request model."""
    request_id: str
    requester: str
    serving_node: str
    model_version: str
    input_data_ipfs_hash: str
    fee: str
    status: str
    request_time: Optional[str] = None
    result_ipfs_hash: Optional[str] = None
    latency_ms: Optional[int] = None


class ServingNodeStats(BaseModel):
    """Serving node statistics model."""
    node_address: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    average_latency_ms: float
    total_fees_earned: str


@router.get("/nodes")
@limiter.limit(config.rate_limit_get)
async def list_serving_nodes(request: Request, limit: int = 100, offset: int = 0):
    """
    List all serving nodes.
    
    Args:
        limit: Maximum number of nodes to return (default: 100)
        offset: Pagination offset (default: 0)
        
    Returns:
        List of serving nodes with their status
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
        
        serving_nodes = []
        node_registrations = data.get("node_registrations", [])
        
        # Filter serving nodes first
        serving_node_addresses = []
        serving_node_regs = {}
        for node_reg in node_registrations:
            # Check if node has serving role (NODE_TYPE_SERVING = 2)
            roles = node_reg.get("roles", [])
            has_serving_role = 2 in roles or "NODE_TYPE_SERVING" in [str(r) for r in roles]
            
            if not has_serving_role:
                continue
            
            node_address = node_reg.get("node_address", "")
            if not node_address:
                continue
            
            serving_node_addresses.append(node_address)
            serving_node_regs[node_address] = node_reg
        
        # FIX: Batch fetch serving node statuses using ThreadPoolExecutor
        # This solves the N+1 query problem by parallelizing requests
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def fetch_serving_status(node_address: str):
            """Fetch serving node status - runs in thread pool."""
            try:
                status_endpoint = f"/remes/remes/v1/serving_node_status/{node_address}"
                status_data = blockchain_client._query_rest(status_endpoint)
                return node_address, status_data.get("serving_node_status", {})
            except Exception as e:
                logger.debug(f"Could not fetch serving status for {node_address}: {e}")
                return node_address, {}
        
        # Parallel fetch with max 10 concurrent requests to avoid overwhelming the RPC
        status_map = {}
        max_workers = min(10, len(serving_node_addresses))
        
        if serving_node_addresses:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(fetch_serving_status, addr): addr 
                    for addr in serving_node_addresses
                }
                for future in as_completed(futures):
                    try:
                        node_address, status = future.result(timeout=5)
                        status_map[node_address] = status
                    except Exception as e:
                        addr = futures[future]
                        logger.debug(f"Failed to fetch status for {addr}: {e}")
                        status_map[addr] = {}
        
        # Build response using cached statuses
        for node_address in serving_node_addresses:
            node_reg = serving_node_regs[node_address]
            status = status_map.get(node_address, {})
            
            if status:
                serving_nodes.append({
                    "node_address": node_address,
                    "model_version": status.get("model_version", ""),
                    "model_ipfs_hash": status.get("model_ipfs_hash", ""),
                    "is_available": status.get("is_available", False),
                    "total_requests": status.get("total_requests", 0),
                    "successful_requests": status.get("successful_requests", 0),
                    "average_latency_ms": status.get("average_latency_ms", 0),
                    "last_heartbeat": status.get("last_heartbeat", ""),
                    "status": node_reg.get("status", "UNSPECIFIED"),
                })
            else:
                # Include node even without status
                serving_nodes.append({
                    "node_address": node_address,
                    "model_version": "",
                    "model_ipfs_hash": "",
                    "is_available": False,
                    "total_requests": 0,
                    "successful_requests": 0,
                    "average_latency_ms": 0,
                    "last_heartbeat": "",
                    "status": node_reg.get("status", "UNSPECIFIED"),
                })
        
        return {
            "nodes": serving_nodes,
            "total": len(serving_nodes),
        }
    except Exception as e:
        logger.error(f"Error listing serving nodes: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch serving nodes")


@router.get("/nodes/{address}")
@limiter.limit(config.rate_limit_get)
async def get_serving_node(
    request: Request,
    address: str = Path(..., description="Serving node address")
):
    """
    Get serving node details.
    
    Args:
        address: Serving node address
        
    Returns:
        Serving node details with status and statistics
    """
    try:
        blockchain_client = get_blockchain_client()
        
        # Get node registration
        try:
            reg_endpoint = f"/remes/remes/v1/node_registration/{address}"
            reg_data = blockchain_client._query_rest(reg_endpoint)
            
            if "node_registration" not in reg_data:
                raise HTTPException(status_code=404, detail="Serving node not found")
            
            node_reg = reg_data["node_registration"]
            roles = node_reg.get("roles", [])
            has_serving_role = 2 in roles or "NODE_TYPE_SERVING" in [str(r) for r in roles]
            
            if not has_serving_role:
                raise HTTPException(status_code=404, detail="Node does not have serving role")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get node registration: {e}")
            raise HTTPException(status_code=404, detail="Serving node not found")
        
        # Get serving node status
        try:
            status_endpoint = f"/remes/remes/v1/serving_node_status/{address}"
            status_data = blockchain_client._query_rest(status_endpoint)
            
            serving_status = status_data.get("serving_node_status", {})
        except Exception as e:
            logger.debug(f"Could not fetch serving status: {e}")
            serving_status = {}
        
        # Calculate statistics
        total_requests = serving_status.get("total_requests", 0)
        successful_requests = serving_status.get("successful_requests", 0)
        failed_requests = total_requests - successful_requests
        success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0.0
        
        return {
            "node_address": address,
            "model_version": serving_status.get("model_version", ""),
            "model_ipfs_hash": serving_status.get("model_ipfs_hash", ""),
            "is_available": serving_status.get("is_available", False),
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": round(success_rate, 2),
            "average_latency_ms": serving_status.get("average_latency_ms", 0),
            "last_heartbeat": serving_status.get("last_heartbeat", ""),
            "status": node_reg.get("status", "UNSPECIFIED"),
            "resources": node_reg.get("resources", {}),
            "stake": node_reg.get("stake", ""),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting serving node {address}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch serving node details")


@router.get("/nodes/{address}/requests")
@limiter.limit(config.rate_limit_get)
async def get_serving_node_requests(
    request: Request,
    address: str = Path(..., description="Serving node address"),
    limit: int = Query(default=50, ge=1, le=500, description="Maximum number of requests to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    status: Optional[str] = Query(default=None, description="Filter by status: pending, completed, failed")
):
    """
    Get inference requests for a serving node.
    
    Queries blockchain events and local database for inference request history.
    
    Args:
        address: Serving node address
        limit: Maximum number of requests to return (default: 50, max: 500)
        offset: Pagination offset (default: 0)
        status: Filter by status (optional): "pending", "completed", "failed"
        
    Returns:
        List of inference requests with pagination
    """
    # Input validation
    try:
        address = validate_wallet_address(address)
    except InvalidWalletAddressError as e:
        logger.warning(f"Invalid wallet address in request: {address}")
        raise HTTPException(status_code=400, detail=str(e))
    
    try:
        limit, offset = validate_pagination(limit, offset, max_limit=500)
    except InvalidInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    if status:
        try:
            status = validate_status_filter(status, ["pending", "completed", "failed"])
        except InvalidInputError as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    try:
        blockchain_client = get_blockchain_client()
        requests_list = []
        
        # Method 1: Query from blockchain events via Tendermint RPC
        try:
            # Query inference request events where serving_node matches
            events_endpoint = "/cosmos/tx/v1beta1/txs"
            events_params = {
                "events": f"inference_request.serving_node='{address}'",
                "pagination.limit": str(limit),
                "pagination.offset": str(offset),
                "order_by": "ORDER_BY_DESC",
            }
            
            events_data = blockchain_client._query_rest(events_endpoint, events_params)
            
            tx_responses = events_data.get("tx_responses", [])
            
            for tx in tx_responses:
                # Parse inference request from transaction events
                for event in tx.get("events", []):
                    if event.get("type") == "inference_request":
                        attrs = {attr["key"]: attr["value"] for attr in event.get("attributes", [])}
                        
                        request_status = attrs.get("status", "pending")
                        
                        # Apply status filter if provided
                        if status and request_status.lower() != status.lower():
                            continue
                        
                        requests_list.append({
                            "request_id": attrs.get("request_id", ""),
                            "requester": attrs.get("requester", ""),
                            "serving_node": attrs.get("serving_node", address),
                            "model_version": attrs.get("model_version", ""),
                            "input_data_ipfs_hash": attrs.get("input_data_ipfs_hash", ""),
                            "fee": attrs.get("fee", "0"),
                            "status": request_status,
                            "request_time": tx.get("timestamp", ""),
                            "result_ipfs_hash": attrs.get("result_ipfs_hash", ""),
                            "latency_ms": int(attrs.get("latency_ms", 0)) if attrs.get("latency_ms") else None,
                            "tx_hash": tx.get("txhash", ""),
                            "block_height": int(tx.get("height", 0)),
                        })
        except Exception as e:
            logger.debug(f"Could not query inference requests from blockchain events: {e}")
        
        # Method 2: Query from local database (if indexed)
        if not requests_list:
            try:
                # Try to get from local database index
                db_requests = await database.get_inference_requests_by_serving_node(
                    serving_node=address,
                    limit=limit,
                    offset=offset,
                    status=status
                )
                if db_requests:
                    requests_list = db_requests
            except Exception as e:
                logger.debug(f"Could not query inference requests from database: {e}")
        
        # Method 3: Query stored inference requests from blockchain state
        if not requests_list:
            try:
                # Query all inference requests and filter by serving node
                requests_endpoint = "/remes/remes/v1/inference_requests"
                requests_params = {
                    "pagination.limit": str(limit * 2),  # Fetch more to account for filtering
                    "pagination.offset": str(offset),
                }
                
                requests_data = blockchain_client._query_rest(requests_endpoint, requests_params)
                
                for req in requests_data.get("inference_requests", []):
                    if req.get("serving_node") == address:
                        request_status = req.get("status", "pending")
                        
                        # Apply status filter if provided
                        if status and request_status.lower() != status.lower():
                            continue
                        
                        requests_list.append({
                            "request_id": req.get("request_id", ""),
                            "requester": req.get("requester", ""),
                            "serving_node": req.get("serving_node", ""),
                            "model_version": req.get("model_version", ""),
                            "input_data_ipfs_hash": req.get("input_data_ipfs_hash", ""),
                            "fee": req.get("fee", "0"),
                            "status": request_status,
                            "request_time": req.get("request_time", ""),
                            "result_ipfs_hash": req.get("result_ipfs_hash", ""),
                            "latency_ms": req.get("latency_ms"),
                        })
                        
                        if len(requests_list) >= limit:
                            break
            except Exception as e:
                logger.debug(f"Could not query inference requests from blockchain state: {e}")
        
        return {
            "requests": requests_list[:limit],
            "total": len(requests_list),
            "limit": limit,
            "offset": offset,
            "serving_node": address,
        }
    except Exception as e:
        logger.error(f"Error getting requests for serving node {address}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch inference requests")


@router.get("/nodes/{address}/stats")
@limiter.limit(config.rate_limit_get)
async def get_serving_node_stats(
    request: Request,
    address: str = Path(..., description="Serving node address")
):
    """
    Get serving node statistics.
    
    Args:
        address: Serving node address
        
    Returns:
        Serving node statistics
    """
    try:
        blockchain_client = get_blockchain_client()
        
        # Get serving node status
        try:
            status_endpoint = f"/remes/remes/v1/serving_node_status/{address}"
            status_data = blockchain_client._query_rest(status_endpoint)
            
            serving_status = status_data.get("serving_node_status", {})
        except Exception as e:
            logger.debug(f"Could not fetch serving status: {e}")
            serving_status = {}
        
        total_requests = serving_status.get("total_requests", 0)
        successful_requests = serving_status.get("successful_requests", 0)
        failed_requests = total_requests - successful_requests
        success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0.0
        
        return {
            "node_address": address,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": round(success_rate, 2),
            "average_latency_ms": serving_status.get("average_latency_ms", 0),
            "model_version": serving_status.get("model_version", ""),
            "is_available": serving_status.get("is_available", False),
        }
    except Exception as e:
        logger.error(f"Error getting stats for serving node {address}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch serving node statistics")


@router.post("/nodes/{address}/status")
@limiter.limit(config.rate_limit_get)
async def update_serving_node_status(
    request: Request,
    address: str = Path(..., description="Serving node address"),
    is_available: Optional[bool] = None,
    model_version: Optional[str] = None,
    model_ipfs_hash: Optional[str] = None
):
    """
    Update serving node status.
    
    Note: This endpoint is informational. Actual status updates must be done
    via blockchain transaction (MsgUpdateServingNodeStatus).
    
    Args:
        address: Serving node address
        is_available: Whether node is available
        model_version: Model version
        model_ipfs_hash: Model IPFS hash
        
    Returns:
        Success message
    """
    return {
        "message": "Status update submitted. Actual update requires blockchain transaction.",
        "note": "Use blockchain client to send MsgUpdateServingNodeStatus transaction."
    }


@router.get("/requests/{request_id}")
@limiter.limit(config.rate_limit_get)
async def get_inference_request(
    request: Request,
    request_id: str = Path(..., description="Inference request ID")
):
    """
    Get inference request details.
    
    Args:
        request_id: Inference request ID
        
    Returns:
        Inference request details
    """
    try:
        blockchain_client = get_blockchain_client()
        
        # Query inference request from blockchain
        endpoint = f"/remes/remes/v1/inference_request/{request_id}"
        
        try:
            data = blockchain_client._query_rest(endpoint)
            
            if "inference_request" not in data:
                raise HTTPException(status_code=404, detail="Inference request not found")
            
            req = data["inference_request"]
            
            return {
                "request_id": req.get("request_id", request_id),
                "requester": req.get("requester", ""),
                "serving_node": req.get("serving_node", ""),
                "model_version": req.get("model_version", ""),
                "input_data_ipfs_hash": req.get("input_data_ipfs_hash", ""),
                "fee": req.get("fee", ""),
                "status": req.get("status", "pending"),
                "request_time": req.get("request_time", ""),
                "result_ipfs_hash": req.get("result_ipfs_hash", ""),
                "latency_ms": req.get("latency_ms", 0),
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to query inference request: {e}")
            raise HTTPException(status_code=404, detail="Inference request not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting inference request {request_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch inference request")


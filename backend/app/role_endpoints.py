"""
Role Management API Endpoints

Provides API endpoints for node role registration and management.
"""

from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
import logging

from .database_async import AsyncDatabase
from .config_manager import get_config_manager
from .blockchain_query_client import get_blockchain_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/roles", tags=["roles"])

# Rate limiter (will use app.state.limiter from main.py)
limiter = Limiter(key_func=get_remote_address)

config_manager = get_config_manager()
config = config_manager.load()
database = AsyncDatabase(db_path=config.database_path, chain_json_path=config.chain_json_path)


class NodeRole(BaseModel):
    """Node role model."""
    role_id: int
    role_name: str
    description: str


class NodeRoles(BaseModel):
    """Node roles for an address."""
    node_address: str
    roles: List[int]
    role_names: List[str]
    status: str
    resources: Optional[Dict[str, Any]] = None


class RoleStats(BaseModel):
    """Role statistics."""
    role_id: int
    role_name: str
    total_nodes: int
    active_nodes: int


# Role definitions
ROLES = {
    1: {"name": "Miner", "description": "AI model training node"},
    2: {"name": "Serving", "description": "Inference serving node"},
    3: {"name": "Validator", "description": "Blockchain validator node"},
    4: {"name": "Proposer", "description": "Gradient aggregation proposer node"},
}

# Role access control configuration
ROLE_ACCESS_CONTROL = {
    1: {  # Miner
        "public": True,
        "min_stake": "1000remes",
        "requires_approval": False,
    },
    2: {  # Serving
        "public": True,
        "min_stake": "1000remes",
        "requires_approval": False,
    },
    3: {  # Validator
        "public": False,  # Requires authorization
        "min_stake": "100000remes",  # Higher minimum stake
        "requires_approval": True,  # Whitelist/governance approval required
        "whitelist_only": True,  # Only whitelisted addresses
    },
    4: {  # Proposer
        "public": False,  # Requires authorization
        "min_stake": "50000remes",
        "requires_approval": True,
        "requires_validator_role": True,  # Must be a validator first
    },
}


@router.get("")
@limiter.limit(config.rate_limit_get)
async def list_roles(request: Request):
    """
    List all available roles with access control information.
    
    Returns:
        List of all roles with descriptions and access control info
    """
    try:
        roles_list = [
            {
                "role_id": role_id,
                "role_name": role_info["name"],
                "description": role_info["description"],
                "access_control": ROLE_ACCESS_CONTROL.get(role_id, {}),
            }
            for role_id, role_info in ROLES.items()
        ]
        
        return {
            "roles": roles_list,
            "total": len(roles_list),
        }
    except Exception as e:
        logger.error(f"Error listing roles: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch roles")


@router.get("/{address}")
@limiter.limit(config.rate_limit_get)
async def get_node_roles(
    request: Request,
    address: str
):
    """
    Get node roles for an address.
    
    Args:
        address: Node address
        
    Returns:
        Node roles and status
    """
    try:
        blockchain_client = get_blockchain_client()
        
        # Query node registration from blockchain
        endpoint = f"/remes/remes/v1/node_registration/{address}"
        
        try:
            data = blockchain_client._query_rest(endpoint)
            
            if "node_registration" not in data:
                raise HTTPException(status_code=404, detail="Node not found")
            
            node_reg = data["node_registration"]
            roles = node_reg.get("roles", [])
            
            # Convert role IDs to role names
            role_names = [
                ROLES.get(role_id, {}).get("name", f"Unknown({role_id})")
                for role_id in roles
            ]
            
            return {
                "node_address": address,
                "roles": roles,
                "role_names": role_names,
                "status": node_reg.get("status", "UNSPECIFIED"),
                "resources": node_reg.get("resources", {}),
                "stake": node_reg.get("stake", ""),
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to query node registration: {e}")
            raise HTTPException(status_code=404, detail="Node not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting node roles for {address}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch node roles")


@router.post("/register")
@limiter.limit(config.rate_limit_get)
async def register_node_roles(
    request: Request,
    node_address: str,
    roles: List[int],
    stake: str,
    resources: Optional[Dict[str, Any]] = None
):
    """
    Register node with roles.
    
    Note: This endpoint is informational. Actual registration must be done
    via blockchain transaction (MsgRegisterNode).
    
    Args:
        node_address: Node address
        roles: List of role IDs (1=Miner, 2=Serving, 3=Validator, 4=Proposer)
        stake: Staked amount
        resources: Resource specification (optional)
        
    Returns:
        Success message or error if role requires authorization
    """
    # Validate roles
    invalid_roles = [r for r in roles if r not in ROLES]
    if invalid_roles:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid role IDs: {invalid_roles}. Valid roles: {list(ROLES.keys())}"
        )
    
    # Check access control for each role
    unauthorized_roles = []
    for role_id in roles:
        access_control = ROLE_ACCESS_CONTROL.get(role_id)
        if not access_control:
            continue
        
        # Check if role is public or requires authorization
        if not access_control["public"]:
            # For non-public roles, inform user that authorization is required
            # Actual authorization check happens on blockchain
            unauthorized_roles.append({
                "role_id": role_id,
                "role_name": ROLES[role_id]["name"],
                "requires_authorization": True,
                "min_stake": access_control.get("min_stake", "N/A"),
                "message": f"Role '{ROLES[role_id]['name']}' requires authorization. Please request access or ensure you meet the requirements.",
            })
    
    role_names = [ROLES[r]["name"] for r in roles]
    
    response = {
        "message": "Node registration initiated. Actual registration requires blockchain transaction.",
        "note": "Use blockchain client to send MsgRegisterNode transaction.",
        "node_address": node_address,
        "roles": roles,
        "role_names": role_names,
        "stake": stake,
    }
    
    # Add warnings for unauthorized roles
    if unauthorized_roles:
        response["warnings"] = unauthorized_roles
        response["note"] = "Some selected roles require authorization. Registration may fail if authorization is not granted."
    
    return response


@router.post("/update")
@limiter.limit(config.rate_limit_get)
async def update_node_roles(
    request: Request,
    node_address: str,
    roles: Optional[List[int]] = None,
    resources: Optional[Dict[str, Any]] = None
):
    """
    Update node roles.
    
    Note: This endpoint is informational. Actual update must be done
    via blockchain transaction (MsgUpdateNodeRegistration).
    
    Args:
        node_address: Node address
        roles: Updated list of role IDs (optional)
        resources: Updated resource specification (optional)
        
    Returns:
        Success message
    """
    if roles is not None:
        invalid_roles = [r for r in roles if r not in ROLES]
        if invalid_roles:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid role IDs: {invalid_roles}. Valid roles: {list(ROLES.keys())}"
            )
    
    return {
        "message": "Node role update initiated. Actual update requires blockchain transaction.",
        "note": "Use blockchain client to send MsgUpdateNodeRegistration transaction.",
        "node_address": node_address,
    }


@router.get("/stats/summary")
@limiter.limit(config.rate_limit_get)
async def get_role_statistics(request: Request):
    """
    Get role statistics (count per role).
    
    Returns:
        Statistics for each role
    """
    try:
        blockchain_client = get_blockchain_client()
        
        # Query all node registrations
        endpoint = "/remes/remes/v1/node_registrations"
        params = {
            "pagination.limit": 10000,  # Get all nodes
            "pagination.offset": 0,
        }
        
        try:
            data = blockchain_client._query_rest(endpoint, params)
        except Exception as e:
            logger.warning(f"Failed to query node registrations: {e}")
            return {
                "stats": [
                    {
                        "role_id": role_id,
                        "role_name": role_info["name"],
                        "total_nodes": 0,
                        "active_nodes": 0,
                    }
                    for role_id, role_info in ROLES.items()
                ]
            }
        
        node_registrations = data.get("node_registrations", [])
        
        # Count nodes per role
        role_counts = {role_id: {"total": 0, "active": 0} for role_id in ROLES.keys()}
        
        for node_reg in node_registrations:
            roles = node_reg.get("roles", [])
            status = node_reg.get("status", "UNSPECIFIED")
            is_active = status == "NODE_STATUS_ACTIVE" or status == "ACTIVE"
            
            for role_id in roles:
                if role_id in role_counts:
                    role_counts[role_id]["total"] += 1
                    if is_active:
                        role_counts[role_id]["active"] += 1
        
        stats = [
            {
                "role_id": role_id,
                "role_name": ROLES[role_id]["name"],
                "total_nodes": role_counts[role_id]["total"],
                "active_nodes": role_counts[role_id]["active"],
            }
            for role_id in sorted(ROLES.keys())
        ]
        
        return {
            "stats": stats,
        }
    except Exception as e:
        logger.error(f"Error getting role statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch role statistics")


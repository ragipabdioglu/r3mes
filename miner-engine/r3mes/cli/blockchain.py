"""
Blockchain CLI utilities for R3MES

Provides functions for interacting with the R3MES blockchain from CLI and Desktop Launcher.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def register_node(
    private_key: str,
    node_address: str,
    roles: List[int],
    stake: str,
    grpc_url: Optional[str] = None,
    chain_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Register a node with specified roles on the blockchain.
    
    Args:
        private_key: Node's private key for signing transaction
        node_address: Node's blockchain address
        roles: List of role IDs (1=MINER, 2=SERVING, 3=VALIDATOR, 4=PROPOSER)
        stake: Amount to stake (e.g., "1000uremes")
        grpc_url: gRPC endpoint URL (default: from env or localhost:9090)
        chain_id: Chain ID (default: from env or "remes-test")
        
    Returns:
        Dictionary with success status, tx_hash, and error message if failed
    """
    try:
        # Import blockchain client
        import sys
        from pathlib import Path
        
        # Add miner-engine to path
        miner_engine_dir = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(miner_engine_dir))
        
        from bridge.blockchain_client import BlockchainClient
        
        # Get configuration from environment or parameters
        if grpc_url is None:
            grpc_url = os.getenv("R3MES_NODE_GRPC_URL", "localhost:9090")
        
        if chain_id is None:
            chain_id = os.getenv("R3MES_CHAIN_ID", "remes-test")
        
        # Create blockchain client
        client = BlockchainClient(
            node_url=grpc_url,
            private_key=private_key,
            chain_id=chain_id,
        )
        
        # Build and send MsgRegisterNode transaction
        # Note: This requires the blockchain client to support node registration
        result = client.register_node(
            node_address=node_address,
            roles=roles,
            stake=stake,
        )
        
        if result.get("success"):
            return {
                "success": True,
                "tx_hash": result.get("tx_hash"),
                "message": f"Node registered successfully with roles: {roles}",
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "Unknown error"),
            }
            
    except ImportError as e:
        logger.error(f"Failed to import blockchain client: {e}")
        return {
            "success": False,
            "error": f"Blockchain client not available: {e}",
        }
    except Exception as e:
        logger.error(f"Failed to register node: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
        }


def get_node_roles(
    node_address: str,
    grpc_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get registered roles for a node.
    
    Args:
        node_address: Node's blockchain address
        grpc_url: gRPC endpoint URL (default: from env or localhost:9090)
        
    Returns:
        Dictionary with roles and status
    """
    try:
        import sys
        from pathlib import Path
        
        miner_engine_dir = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(miner_engine_dir))
        
        from bridge.blockchain_client import BlockchainClient
        
        if grpc_url is None:
            grpc_url = os.getenv("R3MES_NODE_GRPC_URL", "localhost:9090")
        
        # Create blockchain client (no private key needed for queries)
        client = BlockchainClient(
            node_url=grpc_url,
            private_key="",  # Empty for read-only
            chain_id="",
        )
        
        # Query node registration
        result = client.get_node_registration(node_address)
        
        if result:
            return {
                "success": True,
                "node_address": node_address,
                "roles": result.get("roles", []),
                "status": result.get("status", "UNSPECIFIED"),
                "stake": result.get("stake", "0"),
            }
        else:
            return {
                "success": False,
                "error": "Node not registered",
            }
            
    except Exception as e:
        logger.error(f"Failed to get node roles: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
        }


def update_node_roles(
    private_key: str,
    node_address: str,
    roles: List[int],
    grpc_url: Optional[str] = None,
    chain_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Update roles for an existing node registration.
    
    Args:
        private_key: Node's private key for signing transaction
        node_address: Node's blockchain address
        roles: New list of role IDs
        grpc_url: gRPC endpoint URL
        chain_id: Chain ID
        
    Returns:
        Dictionary with success status and tx_hash
    """
    try:
        import sys
        from pathlib import Path
        
        miner_engine_dir = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(miner_engine_dir))
        
        from bridge.blockchain_client import BlockchainClient
        
        if grpc_url is None:
            grpc_url = os.getenv("R3MES_NODE_GRPC_URL", "localhost:9090")
        
        if chain_id is None:
            chain_id = os.getenv("R3MES_CHAIN_ID", "remes-test")
        
        client = BlockchainClient(
            node_url=grpc_url,
            private_key=private_key,
            chain_id=chain_id,
        )
        
        result = client.update_node_roles(
            node_address=node_address,
            roles=roles,
        )
        
        if result.get("success"):
            return {
                "success": True,
                "tx_hash": result.get("tx_hash"),
                "message": f"Node roles updated to: {roles}",
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "Unknown error"),
            }
            
    except Exception as e:
        logger.error(f"Failed to update node roles: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
        }


def unregister_node(
    private_key: str,
    node_address: str,
    grpc_url: Optional[str] = None,
    chain_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Unregister a node from the network.
    
    Args:
        private_key: Node's private key for signing transaction
        node_address: Node's blockchain address
        grpc_url: gRPC endpoint URL
        chain_id: Chain ID
        
    Returns:
        Dictionary with success status and tx_hash
    """
    try:
        import sys
        from pathlib import Path
        
        miner_engine_dir = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(miner_engine_dir))
        
        from bridge.blockchain_client import BlockchainClient
        
        if grpc_url is None:
            grpc_url = os.getenv("R3MES_NODE_GRPC_URL", "localhost:9090")
        
        if chain_id is None:
            chain_id = os.getenv("R3MES_CHAIN_ID", "remes-test")
        
        client = BlockchainClient(
            node_url=grpc_url,
            private_key=private_key,
            chain_id=chain_id,
        )
        
        result = client.unregister_node(node_address=node_address)
        
        if result.get("success"):
            return {
                "success": True,
                "tx_hash": result.get("tx_hash"),
                "message": "Node unregistered successfully",
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "Unknown error"),
            }
            
    except Exception as e:
        logger.error(f"Failed to unregister node: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
        }


if __name__ == "__main__":
    # CLI interface
    import argparse
    
    parser = argparse.ArgumentParser(description="R3MES Blockchain CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Register node command
    register_parser = subparsers.add_parser("register", help="Register a node")
    register_parser.add_argument("--private-key", required=True, help="Node private key")
    register_parser.add_argument("--address", required=True, help="Node address")
    register_parser.add_argument("--roles", required=True, help="Comma-separated role IDs (1=MINER, 2=SERVING, 3=VALIDATOR, 4=PROPOSER)")
    register_parser.add_argument("--stake", required=True, help="Stake amount (e.g., 1000uremes)")
    register_parser.add_argument("--grpc-url", help="gRPC endpoint URL")
    register_parser.add_argument("--chain-id", help="Chain ID")
    
    # Get roles command
    get_parser = subparsers.add_parser("get-roles", help="Get node roles")
    get_parser.add_argument("--address", required=True, help="Node address")
    get_parser.add_argument("--grpc-url", help="gRPC endpoint URL")
    
    args = parser.parse_args()
    
    if args.command == "register":
        roles = [int(r.strip()) for r in args.roles.split(",")]
        result = register_node(
            private_key=args.private_key,
            node_address=args.address,
            roles=roles,
            stake=args.stake,
            grpc_url=args.grpc_url,
            chain_id=args.chain_id,
        )
        print(json.dumps(result, indent=2))
    
    elif args.command == "get-roles":
        result = get_node_roles(
            node_address=args.address,
            grpc_url=args.grpc_url,
        )
        print(json.dumps(result, indent=2))
    
    else:
        parser.print_help()

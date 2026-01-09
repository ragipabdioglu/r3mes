"""
Tendermint RPC Client for Transaction Broadcasting

Provides functionality to broadcast transactions to Tendermint RPC
and retrieve transaction hashes.
"""

import json
import logging
import time
from typing import Dict, Any, Optional
import requests
import base64

logger = logging.getLogger(__name__)


class TendermintClient:
    """
    Client for interacting with Tendermint RPC.
    
    Used for broadcasting transactions and retrieving transaction hashes.
    """
    
    def __init__(self, rpc_url: str = "http://localhost:26657", timeout: int = 30):
        """
        Initialize Tendermint RPC client.
        
        Args:
            rpc_url: Tendermint RPC URL (default: http://localhost:26657)
            timeout: Request timeout in seconds
        """
        self.rpc_url = rpc_url.rstrip('/')
        self.timeout = timeout
    
    def _make_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make a JSON-RPC request to Tendermint RPC.
        
        Args:
            method: RPC method name
            params: Method parameters
            
        Returns:
            Response dictionary
        """
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or {},
        }
        
        try:
            response = requests.post(
                f"{self.rpc_url}",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()
            
            if "error" in result:
                raise Exception(f"RPC error: {result['error']}")
            
            return result.get("result", {})
        except requests.exceptions.RequestException as e:
            logger.error(f"Tendermint RPC request failed: {e}")
            raise
    
    def broadcast_tx_sync(self, tx_bytes: bytes) -> Dict[str, Any]:
        """
        Broadcast transaction synchronously.
        
        Returns immediately after transaction is accepted by mempool.
        
        Args:
            tx_bytes: Transaction bytes (base64 encoded)
            
        Returns:
            Response dictionary with transaction hash
        """
        tx_base64 = base64.b64encode(tx_bytes).decode("utf-8")
        
        result = self._make_request("broadcast_tx_sync", {"tx": tx_base64})
        
        return {
            "code": result.get("code", 0),
            "hash": result.get("hash", ""),
            "log": result.get("log", ""),
        }
    
    def broadcast_tx_async(self, tx_bytes: bytes) -> Dict[str, Any]:
        """
        Broadcast transaction asynchronously.
        
        Returns immediately without waiting for transaction to be included in a block.
        
        Args:
            tx_bytes: Transaction bytes (base64 encoded)
            
        Returns:
            Response dictionary with transaction hash
        """
        tx_base64 = base64.b64encode(tx_bytes).decode("utf-8")
        
        result = self._make_request("broadcast_tx_async", {"tx": tx_base64})
        
        return {
            "hash": result.get("hash", ""),
        }
    
    def broadcast_tx_commit(self, tx_bytes: bytes, max_wait: int = 10) -> Dict[str, Any]:
        """
        Broadcast transaction and wait for commit.
        
        Waits for transaction to be included in a block (up to max_wait seconds).
        
        Args:
            tx_bytes: Transaction bytes (base64 encoded)
            max_wait: Maximum time to wait for commit (seconds)
            
        Returns:
            Response dictionary with transaction hash and check_tx/deliver_tx results
        """
        tx_base64 = base64.b64encode(tx_bytes).decode("utf-8")
        
        result = self._make_request("broadcast_tx_commit", {"tx": tx_base64})
        
        return {
            "check_tx": result.get("check_tx", {}),
            "deliver_tx": result.get("deliver_tx", {}),
            "hash": result.get("hash", ""),
            "height": result.get("height", 0),
        }
    
    def get_tx(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        """
        Query transaction by hash.
        
        Args:
            tx_hash: Transaction hash (hex encoded)
            
        Returns:
            Transaction information or None if not found
        """
        try:
            result = self._make_request("tx", {"hash": tx_hash, "prove": False})
            return result
        except Exception as e:
            logger.debug(f"Transaction not found: {e}")
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get Tendermint node status.
        
        Returns:
            Node status information
        """
        return self._make_request("status")


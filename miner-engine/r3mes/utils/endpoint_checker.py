"""
Endpoint Checker Utility
Checks DNS resolution and connectivity for blockchain endpoints.
"""

import socket
import requests
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

# Default endpoints
DEFAULT_ENDPOINTS = {
    "mainnet": {
        "grpc": "node.r3mes.network:9090",
        "rest": "https://api.r3mes.network",
        "rpc": "https://rpc.r3mes.network",
    },
    "testnet": {
        "grpc": "testnet-node.r3mes.network:9090",
        "rest": "https://testnet-api.r3mes.network",
        "rpc": "https://testnet-rpc.r3mes.network",
    },
    "local": {
        "grpc": "localhost:9090",
        "rest": "http://localhost:1317",
        "rpc": "http://localhost:26657",
    },
}

# Fallback IPs (if DNS fails)
# Note: These will be set during production deployment
# For now, DNS resolution is preferred over hardcoded IPs
FALLBACK_IPS = {
    # "node.r3mes.network": "1.2.3.4",  # Set during production
    # "api.r3mes.network": "1.2.3.4",  # Set during production
    # "rpc.r3mes.network": "1.2.3.4",  # Set during production
}


def resolve_dns(hostname: str) -> Optional[str]:
    """
    Resolve DNS hostname to IP address.
    
    Args:
        hostname: Hostname to resolve
    
    Returns:
        IP address if successful, None otherwise
    """
    try:
        # Remove port if present
        host = hostname.split(":")[0]
        ip = socket.gethostbyname(host)
        return ip
    except socket.gaierror:
        return None
    except Exception as e:
        logger.error(f"DNS resolution error for {hostname}: {e}")
        return None


def check_endpoint_connectivity(url: str, timeout: int = 5) -> Dict[str, Any]:
    """
    Check if endpoint is reachable.
    
    Args:
        url: Endpoint URL
        timeout: Connection timeout in seconds
    
    Returns:
        Dict with 'reachable', 'latency_ms', 'error', etc.
    """
    import time
    
    try:
        start_time = time.time()
        response = requests.get(url, timeout=timeout, allow_redirects=False)
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "reachable": True,
            "status_code": response.status_code,
            "latency_ms": round(latency_ms, 2),
            "error": None,
        }
    except requests.exceptions.Timeout:
        return {
            "reachable": False,
            "error": "Connection timeout",
            "latency_ms": None,
        }
    except requests.exceptions.ConnectionError:
        return {
            "reachable": False,
            "error": "Connection refused",
            "latency_ms": None,
        }
    except Exception as e:
        return {
            "reachable": False,
            "error": str(e),
            "latency_ms": None,
        }


def check_blockchain_endpoints(network: str = "mainnet") -> Dict[str, Any]:
    """
    Check all blockchain endpoints for a network.
    
    Args:
        network: Network name (mainnet, testnet, local)
    
    Returns:
        Dict with endpoint statuses
    """
    if network not in DEFAULT_ENDPOINTS:
        return {
            "error": f"Unknown network: {network}",
            "endpoints": {},
        }
    
    endpoints = DEFAULT_ENDPOINTS[network]
    results = {}
    
    # Check REST API
    if "rest" in endpoints:
        rest_url = endpoints["rest"]
        # Add health check endpoint
        health_url = f"{rest_url}/health" if not rest_url.endswith("/health") else rest_url
        results["rest"] = {
            "url": rest_url,
            "dns_resolved": resolve_dns(rest_url.split("://")[-1].split("/")[0]) is not None,
            "connectivity": check_endpoint_connectivity(health_url),
        }
    
    # Check RPC
    if "rpc" in endpoints:
        rpc_url = endpoints["rpc"]
        # Add status endpoint
        status_url = f"{rpc_url}/status" if not rpc_url.endswith("/status") else rpc_url
        results["rpc"] = {
            "url": rpc_url,
            "dns_resolved": resolve_dns(rpc_url.split("://")[-1].split("/")[0]) is not None,
            "connectivity": check_endpoint_connectivity(status_url),
        }
    
    # Check gRPC (simplified - just DNS check)
    if "grpc" in endpoints:
        grpc_host = endpoints["grpc"].split(":")[0]
        results["grpc"] = {
            "url": endpoints["grpc"],
            "dns_resolved": resolve_dns(grpc_host) is not None,
            "connectivity": None,  # gRPC requires special client
        }
    
    return {
        "network": network,
        "endpoints": results,
    }


def get_best_endpoint(network: str = "mainnet") -> Optional[Dict[str, str]]:
    """
    Get the best available endpoint for a network.
    
    Args:
        network: Network name
    
    Returns:
        Dict with endpoint URLs, or None if none available
    """
    # Try mainnet first
    result = check_blockchain_endpoints(network)
    
    # Check if any endpoint is reachable
    for endpoint_name, endpoint_data in result.get("endpoints", {}).items():
        if endpoint_data.get("dns_resolved") and endpoint_data.get("connectivity", {}).get("reachable"):
            # Return first available endpoint set
            return DEFAULT_ENDPOINTS[network]
    
    # Try fallback network
    if network == "mainnet":
        logger.warning("Mainnet endpoints not available, trying testnet...")
        testnet_result = check_blockchain_endpoints("testnet")
        for endpoint_name, endpoint_data in testnet_result.get("endpoints", {}).items():
            if endpoint_data.get("dns_resolved") and endpoint_data.get("connectivity", {}).get("reachable"):
                return DEFAULT_ENDPOINTS["testnet"]
    
    # Try local
    logger.warning("Remote endpoints not available, trying local...")
    local_result = check_blockchain_endpoints("local")
    for endpoint_name, endpoint_data in local_result.get("endpoints", {}).items():
        if endpoint_data.get("dns_resolved"):
            return DEFAULT_ENDPOINTS["local"]
    
    return None


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    print("Checking mainnet endpoints...")
    result = check_blockchain_endpoints("mainnet")
    print(f"Results: {result}")
    
    print("\nGetting best endpoint...")
    best = get_best_endpoint("mainnet")
    print(f"Best endpoint: {best}")


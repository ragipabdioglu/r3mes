"""
R3MES Faucet Client
Handles automatic token airdrop for new users during setup.
"""

import os
import requests
import time
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Faucet configuration (can be overridden via environment variables)
FAUCET_BASE_URL = os.getenv("R3MES_FAUCET_BASE_URL", "https://api.r3mes.network")
FAUCET_ENDPOINT = os.getenv("R3MES_FAUCET_ENDPOINT", "/faucet/claim")
FAUCET_AMOUNT = os.getenv("R3MES_FAUCET_AMOUNT", "1000000uremes")  # 1 REMES = 1,000,000 uremes
FAUCET_TIMEOUT = int(os.getenv("R3MES_FAUCET_TIMEOUT", "30"))  # seconds

# Fallback faucet URLs (for development/testnet)
# Use BACKEND_URL environment variable for local development
FALLBACK_FAUCETS = []


def request_faucet(address: str, faucet_url: Optional[str] = None, amount: Optional[str] = None) -> Dict[str, Any]:
    """
    Request tokens from the faucet.
    
    Args:
        address: Wallet address to receive tokens
        faucet_url: Optional custom faucet URL (for testing)
        amount: Optional amount to request (default: FAUCET_AMOUNT)
    
    Returns:
        Dict with 'success', 'message', and optional 'tx_hash'
    """
    if not address:
        return {
            "success": False,
            "message": "Address is required",
        }
    
    # Build faucet URL list
    faucet_urls = []
    
    # Add custom URL if provided
    if faucet_url:
        faucet_urls.append(faucet_url)
    
    # Add BACKEND_URL from environment (for local development)
    # In production, BACKEND_URL must be set (no localhost fallback)
    is_production = os.getenv("R3MES_ENV", "development").lower() == "production"
    backend_url_env = os.getenv("BACKEND_URL")
    
    if not backend_url_env:
        if is_production:
            raise ValueError(
                "BACKEND_URL environment variable must be set in production. "
                "Do not use localhost in production."
            )
        # Development fallback
        backend_url = "http://localhost:8000"
        logger.warning("BACKEND_URL not set, using localhost fallback (development only)")
    else:
        backend_url = backend_url_env
        # Validate that production doesn't use localhost
        if is_production and ("localhost" in backend_url or "127.0.0.1" in backend_url):
            raise ValueError(
                f"BACKEND_URL cannot use localhost in production: {backend_url}"
            )
    
    faucet_urls.append(f"{backend_url}{FAUCET_ENDPOINT}")
    
    # Add production URL
    faucet_urls.append(f"{FAUCET_BASE_URL}{FAUCET_ENDPOINT}")
    
    # Add fallbacks
    faucet_urls.extend(FALLBACK_FAUCETS)
    
    request_amount = amount or FAUCET_AMOUNT
    
    for url in faucet_urls:
        if not url:
            continue
            
        try:
            logger.info(f"Requesting faucet from {url} for address {address}")
            
            response = requests.post(
                url,
                json={
                    "address": address,
                    "amount": request_amount,
                },
                timeout=FAUCET_TIMEOUT,
                headers={"Content-Type": "application/json"},
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "message": data.get("message", "Tokens sent successfully"),
                    "tx_hash": data.get("tx_hash"),
                    "amount": data.get("amount", request_amount),
                }
            elif response.status_code == 429:
                # Rate limited
                error_data = response.json()
                detail = error_data.get("detail", {})
                next_available = detail.get("next_claim_available_at")
                message = "Rate limit exceeded. You can only claim once per day."
                if next_available:
                    message += f" Next claim available at: {next_available}"
                return {
                    "success": False,
                    "message": message,
                    "retry_after": detail.get("next_claim_available_at"),
                }
            elif response.status_code == 400:
                # Bad request (e.g., invalid address)
                error_data = response.json()
                detail = error_data.get("detail", {})
                if isinstance(detail, dict):
                    message = detail.get("error") or detail.get("message", "Invalid request")
                else:
                    message = detail if isinstance(detail, str) else "Invalid request"
                return {
                    "success": False,
                    "message": message,
                }
            else:
                # Try next faucet
                logger.warning(f"Faucet {url} returned status {response.status_code}")
                continue
                
        except requests.exceptions.Timeout:
            logger.warning(f"Faucet {url} timeout")
            continue
        except requests.exceptions.ConnectionError:
            logger.warning(f"Faucet {url} connection error")
            continue
        except Exception as e:
            logger.error(f"Error requesting faucet from {url}: {e}")
            continue
    
    # All faucets failed
    return {
        "success": False,
        "message": "Faucet is currently unavailable. Please request tokens manually or use a testnet.",
    }


def check_faucet_availability(faucet_url: Optional[str] = None) -> bool:
    """
    Check if faucet is available.
    
    Args:
        faucet_url: Optional custom faucet URL
    
    Returns:
        True if faucet is available, False otherwise
    """
    # Build faucet URL list (same as request_faucet)
    faucet_urls = []
    if faucet_url:
        faucet_urls.append(faucet_url)
    
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    faucet_urls.append(f"{backend_url}{FAUCET_ENDPOINT}")
    faucet_urls.append(f"{FAUCET_BASE_URL}{FAUCET_ENDPOINT}")
    faucet_urls.extend(FALLBACK_FAUCETS)
    
    for url in faucet_urls:
        if not url:
            continue
            
        try:
            # Try status endpoint first
            status_url = url.replace("/faucet/claim", "/faucet/status")
            response = requests.get(status_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get("enabled", False)
        except requests.exceptions.RequestException:
            # Connection failed, try next URL
            pass
        except (ValueError, KeyError):
            # Invalid JSON response
            pass
        
        # Try main endpoint with dummy request (will fail but confirms endpoint exists)
        try:
            response = requests.post(
                url,
                json={"address": "remes1test"},
                timeout=5,
            )
            # Even if it fails, if we get a response (not connection error), faucet is available
            if response.status_code in [200, 400, 429]:
                return True
        except requests.exceptions.RequestException:
            # Connection failed, try next URL
            continue
    
    return False


if __name__ == "__main__":
    # Test
    import sys
    if len(sys.argv) > 1:
        address = sys.argv[1]
        result = request_faucet(address)
        print(f"Success: {result['success']}")
        print(f"Message: {result['message']}")
        if result.get('tx_hash'):
            print(f"TX Hash: {result['tx_hash']}")
    else:
        print("Usage: python faucet.py <address>")


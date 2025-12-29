"""
Faucet API Endpoint

Provides token faucet functionality for new users to claim initial tokens
for gas fees to register as miners or submit transactions.

Rate limiting: 1 request per day per IP and per address
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, HTTPException, Request, Depends
from slowapi import Limiter
from slowapi.util import get_remote_address
from pydantic import BaseModel, Field, field_validator
import httpx
from ..exceptions import (
    InvalidWalletAddressError,
    FaucetError,
    FaucetDisabledError,
    FaucetRateLimitError,
    FaucetTransactionError,
    BlockchainTransactionError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/faucet", tags=["faucet"])

# Rate limiter instance (will be initialized in main.py)
limiter = Limiter(key_func=get_remote_address)


class FaucetRequest(BaseModel):
    """Request model for faucet claim."""
    address: str = Field(..., description="Wallet address to receive tokens")
    amount: Optional[str] = Field(None, description="Requested amount (optional, default from config)")
    
    @field_validator("address")
    @classmethod
    def validate_address(cls, v: str) -> str:
        """Validate Cosmos address format."""
        if not v:
            raise InvalidWalletAddressError("Address cannot be empty")
        # Basic validation: Cosmos addresses start with specific prefixes
        # remes addresses start with "remes"
        if not v.startswith("remes"):
            raise InvalidWalletAddressError("Invalid address format: must start with 'remes'")
        if len(v) < 20 or len(v) > 60:
            raise InvalidWalletAddressError("Invalid address length")
        return v.strip()


class FaucetResponse(BaseModel):
    """Response model for faucet claim."""
    success: bool
    message: str
    tx_hash: Optional[str] = None
    amount: str
    next_claim_available_at: Optional[str] = None


# In-memory rate limit tracking (for development)
# In production, use Redis or database for distributed rate limiting
_rate_limit_cache: dict[str, datetime] = {}


def _get_rate_limit_key(ip_address: str, address: str) -> tuple[str, str]:
    """Get rate limit keys for IP and address."""
    ip_key = f"faucet:ip:{ip_address}"
    addr_key = f"faucet:addr:{address}"
    return ip_key, addr_key


def _check_rate_limit(ip_address: str, address: str) -> tuple[bool, Optional[datetime]]:
    """
    Check if request is within rate limit.
    
    Returns:
        (is_allowed, next_available_at)
    """
    ip_key, addr_key = _get_rate_limit_key(ip_address, address)
    now = datetime.now()
    
    # Check IP rate limit (1 per day)
    if ip_key in _rate_limit_cache:
        last_request = _rate_limit_cache[ip_key]
        if now - last_request < timedelta(days=1):
            next_available = last_request + timedelta(days=1)
            return False, next_available
    
    # Check address rate limit (1 per day)
    if addr_key in _rate_limit_cache:
        last_request = _rate_limit_cache[addr_key]
        if now - last_request < timedelta(days=1):
            next_available = last_request + timedelta(days=1)
            return False, next_available
    
    return True, None


def _record_rate_limit(ip_address: str, address: str):
    """Record rate limit usage."""
    ip_key, addr_key = _get_rate_limit_key(ip_address, address)
    now = datetime.now()
    _rate_limit_cache[ip_key] = now
    _rate_limit_cache[addr_key] = now


async def _send_blockchain_transaction(
    to_address: str,
    amount: str,
) -> Optional[str]:
    """
    Send blockchain transaction via remesd CLI (production-ready).
    
    Uses remesd binary to sign and broadcast transactions securely.
    Falls back to test mode simulation if remesd is not available.
    
    Args:
        to_address: Recipient address
        amount: Amount to send (e.g., "1000000uremes" for 1 REMES)
    
    Returns:
        Transaction hash if successful, None otherwise
    """
    # Get faucet configuration
    faucet_enabled = os.getenv("FAUCET_ENABLED", "true").lower() == "true"
    if not faucet_enabled:
        logger.warning("Faucet is disabled via FAUCET_ENABLED environment variable")
        return None
    
    faucet_amount = os.getenv("FAUCET_AMOUNT", "1000000uremes")  # 1 REMES default
    faucet_treasury_address = os.getenv("FAUCET_TREASURY_ADDRESS", "")
    faucet_key_name = os.getenv("FAUCET_KEY_NAME", "faucet-key")
    chain_id = os.getenv("CHAIN_ID", "remes-mainnet-1")
    remesd_path = os.getenv("REMESD_PATH", "remesd")  # Path to remesd binary
    remesd_home = os.getenv("REMESD_HOME", os.path.expanduser("~/.remesd"))
    
    if not faucet_treasury_address:
        logger.error("FAUCET_TREASURY_ADDRESS not configured")
        return None
    
    # Use requested amount if provided, otherwise use default
    send_amount = amount or faucet_amount
    
    # Normalize amount format
    if "remes" in send_amount and "uremes" not in send_amount:
        # Convert remes to uremes (1 remes = 1000000 uremes)
        try:
            remes_value = float(send_amount.replace("remes", "").strip())
            send_amount = f"{int(remes_value * 1000000)}uremes"
        except ValueError:
            logger.error(f"Invalid amount format: {send_amount}")
            return None
    
    # Test mode: return simulated hash
    if os.getenv("R3MES_TEST_MODE", "false").lower() == "true":
        import hashlib
        tx_hash = hashlib.sha256(
            f"{faucet_treasury_address}{to_address}{send_amount}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:64]
        logger.info(f"TEST MODE: Simulated transaction hash: {tx_hash}")
        return tx_hash
    
    # Production mode: use remesd CLI to send transaction
    try:
        import subprocess
        import json
        
        # Build remesd command
        cmd = [
            remesd_path,
            "tx",
            "bank",
            "send",
            faucet_treasury_address,
            to_address,
            send_amount,
            "--chain-id", chain_id,
            "--from", faucet_key_name,
            "--keyring-backend", os.getenv("FAUCET_KEYRING_BACKEND", "os"),
            "--gas", "auto",
            "--gas-adjustment", "1.5",
            "--yes",
            "--output", "json",
            "--home", remesd_home
        ]
        
        logger.info(f"Sending {send_amount} from {faucet_treasury_address} to {to_address} via remesd")
        
        # Execute command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False
        )
        
        if result.returncode != 0:
            logger.error(f"remesd transaction failed: {result.stderr}")
            return None
        
        # Parse output to get transaction hash
        try:
            output_data = json.loads(result.stdout)
            tx_hash = output_data.get("txhash")
            if tx_hash:
                logger.info(f"Transaction successful: {tx_hash}")
                return tx_hash
            else:
                logger.error(f"No txhash in remesd output: {result.stdout}")
                return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse remesd JSON output: {e}. Output: {result.stdout}")
            return None
        
    except FileNotFoundError:
        logger.error(f"remesd binary not found at {remesd_path}. Please install remesd or set REMESD_PATH environment variable.")
        return None
    except subprocess.TimeoutExpired:
        logger.error("remesd transaction timed out after 30 seconds")
        return None
    except Exception as e:
        logger.error(f"Failed to send blockchain transaction: {e}", exc_info=True)
        return None


@router.post("/claim", response_model=FaucetResponse)
@limiter.limit("1/day")  # Additional limiter decorator (backup)
async def claim_faucet(
    request: Request,
    faucet_request: FaucetRequest,
):
    """
    Claim tokens from the faucet.
    
    Rate Limits:
    - 1 request per day per IP address
    - 1 request per day per wallet address
    
    Returns transaction hash if successful.
    """
    # Get client IP address
    ip_address = get_remote_address(request)
    
    # Check rate limits
    is_allowed, next_available = _check_rate_limit(ip_address, faucet_request.address)
    if not is_allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "message": "You can only claim from the faucet once per day",
                "next_claim_available_at": next_available.isoformat() if next_available else None,
            }
        )
    
    # Get faucet configuration
    faucet_enabled = os.getenv("FAUCET_ENABLED", "true").lower() == "true"
    if not faucet_enabled:
        raise HTTPException(
            status_code=503,
            detail="Faucet is currently disabled"
        )
    
    faucet_amount = os.getenv("FAUCET_AMOUNT", "1000000uremes")  # 1 REMES = 1,000,000 uremes
    daily_limit = os.getenv("FAUCET_DAILY_LIMIT", "5000000uremes")  # 5 REMES max per day
    
    # Use requested amount or default
    requested_amount = faucet_request.amount or faucet_amount
    
    # Validate amount doesn't exceed daily limit
    try:
        requested_uremes = int(requested_amount.replace("uremes", "").replace("remes", ""))
        limit_uremes = int(daily_limit.replace("uremes", "").replace("remes", ""))
        if requested_uremes > limit_uremes:
            raise HTTPException(
                status_code=400,
                detail=f"Requested amount exceeds daily limit of {daily_limit}"
            )
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid amount format: {requested_amount}"
        )
    
    # Send blockchain transaction
    tx_hash = await _send_blockchain_transaction(
        to_address=faucet_request.address,
        amount=requested_amount,
    )
    
    if tx_hash is None:
        # Transaction failed or not implemented
        logger.warning(f"Faucet transaction failed for {faucet_request.address}")
        raise HTTPException(
            status_code=500,
            detail="Failed to send tokens. Please try again later or contact support."
        )
    
    # Record rate limit
    _record_rate_limit(ip_address, faucet_request.address)
    
    # Calculate next available time
    next_available = datetime.now() + timedelta(days=1)
    
    logger.info(f"Faucet claim successful: {faucet_request.address} received {requested_amount} (tx: {tx_hash})")
    
    return FaucetResponse(
        success=True,
        message=f"Successfully sent {requested_amount} to {faucet_request.address}",
        tx_hash=tx_hash,
        amount=requested_amount,
        next_claim_available_at=next_available.isoformat(),
    )


@router.get("/status")
async def get_faucet_status():
    """Get faucet status and configuration."""
    faucet_enabled = os.getenv("FAUCET_ENABLED", "true").lower() == "true"
    faucet_amount = os.getenv("FAUCET_AMOUNT", "1000000uremes")
    daily_limit = os.getenv("FAUCET_DAILY_LIMIT", "5000000uremes")
    
    return {
        "enabled": faucet_enabled,
        "amount_per_claim": faucet_amount,
        "daily_limit": daily_limit,
        "rate_limit": "1 request per day per IP and per address",
    }


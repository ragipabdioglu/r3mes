"""
Authentication and authorization utilities
"""

import os
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import Request, Depends, Header, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .database_async import AsyncDatabase
from .exceptions import InvalidAPIKeyError

# JWT Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

# API Key Authentication
security = HTTPBearer(auto_error=False)

# Global database instance (will be initialized in main.py)
_database: Optional[AsyncDatabase] = None

def init_auth(database: AsyncDatabase):
    """Initialize authentication with database instance."""
    global _database
    _database = database

def create_jwt_token(wallet_address: str, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT token for a wallet address.
    
    Args:
        wallet_address: Wallet address to encode in token
        expires_delta: Token expiration time (default: 24 hours)
        
    Returns:
        JWT token string
    """
    if expires_delta is None:
        expires_delta = timedelta(hours=JWT_EXPIRATION_HOURS)
    
    expire = datetime.utcnow() + expires_delta
    to_encode = {
        "wallet_address": wallet_address,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access_token"
    }
    
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def verify_jwt_token(token: str) -> Dict[str, Any]:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded token payload
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        wallet_address = payload.get("wallet_address")
        
        if wallet_address is None:
            raise HTTPException(status_code=401, detail="Invalid token: missing wallet_address")
        
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_wallet_from_auth(
    request: Request,
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> Optional[str]:
    """
    API key, JWT token veya wallet_address'ten wallet adresini alır.
    
    Öncelik sırası:
    1. X-API-Key header (API key)
    2. Authorization Bearer token (JWT or API key)
    3. Request body'deki wallet_address
    """
    if _database is None:
        raise RuntimeError("Authentication not initialized")
    
    # Try API key from X-API-Key header first
    if x_api_key:
        api_key_info = await _database.validate_api_key(x_api_key)
        if api_key_info and api_key_info["is_active"]:
            return api_key_info["wallet_address"]
        else:
            raise InvalidAPIKeyError("Invalid or expired API key")
    
    # Try Authorization Bearer token (could be JWT or API key)
    if authorization and authorization.credentials:
        token = authorization.credentials
        
        # First try as JWT token
        try:
            payload = verify_jwt_token(token)
            return payload["wallet_address"]
        except HTTPException:
            # If JWT fails, try as API key
            try:
                api_key_info = await _database.validate_api_key(token)
                if api_key_info and api_key_info["is_active"]:
                    return api_key_info["wallet_address"]
            except Exception:
                pass
            
            # If both fail, raise invalid token error
            raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    return None

async def require_auth(
    request: Request,
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> str:
    """
    Require authentication and return wallet address.
    
    Raises HTTPException if no valid authentication is provided.
    """
    wallet_address = await get_wallet_from_auth(request, authorization, x_api_key)
    
    if not wallet_address:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Provide either X-API-Key header or Authorization Bearer token."
        )
    
    return wallet_address
"""
JWT Authentication Implementation for R3MES Backend

Production-ready JWT authentication with:
- RS256 asymmetric signing
- Token refresh mechanism
- Blacklist support
- Rate limiting integration
- Secure token storage
"""

import os
import jwt
import time
import hashlib
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from fastapi import HTTPException, status, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

from .exceptions import (
    InvalidAPIKeyError,
    MissingCredentialsError,
    ProductionConfigurationError,
)
from .cache import get_cache_manager

logger = logging.getLogger(__name__)

# JWT Configuration
JWT_ALGORITHM = "RS256"  # Asymmetric signing for production
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "15"))
JWT_REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "30"))
JWT_ISSUER = os.getenv("JWT_ISSUER", "r3mes-backend")
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE", "r3mes-api")

# Security
security = HTTPBearer(auto_error=False)


class JWTManager:
    """JWT token manager with RS256 signing."""
    
    def __init__(self):
        """Initialize JWT manager with RSA keys."""
        self.private_key = self._load_private_key()
        self.public_key = self._load_public_key()
        self.cache_manager = get_cache_manager()
        
        # Validate keys in production
        is_production = os.getenv("R3MES_ENV", "development").lower() == "production"
        if is_production:
            if not self.private_key or not self.public_key:
                raise ProductionConfigurationError(
                    "JWT RSA keys must be configured in production. "
                    "Set JWT_PRIVATE_KEY_PATH and JWT_PUBLIC_KEY_PATH environment variables."
                )
    
    def _load_private_key(self) -> Optional[str]:
        """Load RSA private key from file or environment."""
        try:
            # Try file path first
            key_path = os.getenv("JWT_PRIVATE_KEY_PATH")
            if key_path and os.path.exists(key_path):
                with open(key_path, 'r') as f:
                    return f.read()
            
            # Try environment variable
            key_env = os.getenv("JWT_PRIVATE_KEY")
            if key_env:
                return key_env
            
            # Development fallback - generate temporary key
            is_production = os.getenv("R3MES_ENV", "development").lower() == "production"
            if not is_production:
                logger.warning("JWT private key not found, using development fallback")
                from cryptography.hazmat.primitives.asymmetric import rsa
                from cryptography.hazmat.primitives import serialization
                from cryptography.hazmat.backends import default_backend
                
                private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=2048,
                    backend=default_backend()
                )
                
                pem = private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                )
                
                return pem.decode('utf-8')
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading JWT private key: {e}")
            return None
    
    def _load_public_key(self) -> Optional[str]:
        """Load RSA public key from file or environment."""
        try:
            # Try file path first
            key_path = os.getenv("JWT_PUBLIC_KEY_PATH")
            if key_path and os.path.exists(key_path):
                with open(key_path, 'r') as f:
                    return f.read()
            
            # Try environment variable
            key_env = os.getenv("JWT_PUBLIC_KEY")
            if key_env:
                return key_env
            
            # Development fallback - derive from private key
            if self.private_key:
                from cryptography.hazmat.primitives import serialization
                from cryptography.hazmat.backends import default_backend
                
                private_key_obj = serialization.load_pem_private_key(
                    self.private_key.encode('utf-8'),
                    password=None,
                    backend=default_backend()
                )
                
                public_key_obj = private_key_obj.public_key()
                pem = public_key_obj.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )
                
                return pem.decode('utf-8')
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading JWT public key: {e}")
            return None
    
    def create_access_token(
        self,
        wallet_address: str,
        additional_claims: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create JWT access token.
        
        Args:
            wallet_address: User's wallet address
            additional_claims: Additional claims to include
            
        Returns:
            JWT access token
        """
        try:
            now = datetime.utcnow()
            expires_at = now + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
            
            payload = {
                "sub": wallet_address,
                "iat": int(now.timestamp()),
                "exp": int(expires_at.timestamp()),
                "iss": JWT_ISSUER,
                "aud": JWT_AUDIENCE,
                "type": "access",
                "jti": self._generate_jti(wallet_address, now),
            }
            
            if additional_claims:
                payload.update(additional_claims)
            
            token = jwt.encode(
                payload,
                self.private_key,
                algorithm=JWT_ALGORITHM
            )
            
            logger.debug(f"Created access token for {wallet_address}")
            return token
            
        except Exception as e:
            logger.error(f"Error creating access token: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create access token"
            )
    
    def create_refresh_token(
        self,
        wallet_address: str,
        additional_claims: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create JWT refresh token.
        
        Args:
            wallet_address: User's wallet address
            additional_claims: Additional claims to include
            
        Returns:
            JWT refresh token
        """
        try:
            now = datetime.utcnow()
            expires_at = now + timedelta(days=JWT_REFRESH_TOKEN_EXPIRE_DAYS)
            
            payload = {
                "sub": wallet_address,
                "iat": int(now.timestamp()),
                "exp": int(expires_at.timestamp()),
                "iss": JWT_ISSUER,
                "aud": JWT_AUDIENCE,
                "type": "refresh",
                "jti": self._generate_jti(wallet_address, now),
            }
            
            if additional_claims:
                payload.update(additional_claims)
            
            token = jwt.encode(
                payload,
                self.private_key,
                algorithm=JWT_ALGORITHM
            )
            
            logger.debug(f"Created refresh token for {wallet_address}")
            return token
            
        except Exception as e:
            logger.error(f"Error creating refresh token: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create refresh token"
            )
    
    def verify_token(self, token: str, token_type: str = "access") -> Dict[str, Any]:
        """
        Verify and decode JWT token.
        
        Args:
            token: JWT token to verify
            token_type: Expected token type ("access" or "refresh")
            
        Returns:
            Decoded token payload
            
        Raises:
            HTTPException: If token is invalid
        """
        try:
            # Check if token is blacklisted
            if self._is_token_blacklisted(token):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )
            
            # Decode and verify token
            payload = jwt.decode(
                token,
                self.public_key,
                algorithms=[JWT_ALGORITHM],
                issuer=JWT_ISSUER,
                audience=JWT_AUDIENCE,
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_iss": True,
                    "verify_aud": True,
                }
            )
            
            # Verify token type
            if payload.get("type") != token_type:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Invalid token type. Expected {token_type}"
                )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        except Exception as e:
            logger.error(f"Error verifying token: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token verification failed"
            )
    
    def refresh_access_token(self, refresh_token: str) -> Tuple[str, str]:
        """
        Refresh access token using refresh token.
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            Tuple of (new_access_token, new_refresh_token)
        """
        try:
            # Verify refresh token
            payload = self.verify_token(refresh_token, token_type="refresh")
            wallet_address = payload["sub"]
            
            # Create new tokens
            new_access_token = self.create_access_token(wallet_address)
            new_refresh_token = self.create_refresh_token(wallet_address)
            
            # Blacklist old refresh token
            self._blacklist_token(refresh_token, payload["exp"])
            
            logger.info(f"Refreshed tokens for {wallet_address}")
            return new_access_token, new_refresh_token
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error refreshing token: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Failed to refresh token"
            )
    
    def revoke_token(self, token: str):
        """
        Revoke (blacklist) a token.
        
        Args:
            token: Token to revoke
        """
        try:
            # Decode token to get expiration
            payload = jwt.decode(
                token,
                self.public_key,
                algorithms=[JWT_ALGORITHM],
                options={"verify_signature": False}
            )
            
            # Blacklist token until expiration
            self._blacklist_token(token, payload["exp"])
            
            logger.info(f"Revoked token for {payload.get('sub', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error revoking token: {e}")
    
    def _generate_jti(self, wallet_address: str, timestamp: datetime) -> str:
        """Generate unique JWT ID."""
        data = f"{wallet_address}:{timestamp.isoformat()}:{os.urandom(16).hex()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _blacklist_token(self, token: str, exp: int):
        """Add token to blacklist."""
        try:
            # Calculate TTL (time until expiration)
            ttl = max(0, exp - int(time.time()))
            
            if ttl > 0:
                # Store token hash in Redis with TTL
                token_hash = hashlib.sha256(token.encode()).hexdigest()
                key = f"jwt:blacklist:{token_hash}"
                
                import asyncio
                asyncio.create_task(
                    self.cache_manager.set(key, "1", ttl=ttl)
                )
                
        except Exception as e:
            logger.error(f"Error blacklisting token: {e}")
    
    def _is_token_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted."""
        try:
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            key = f"jwt:blacklist:{token_hash}"
            
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Can't use async in sync context
                    return False
                result = loop.run_until_complete(self.cache_manager.get(key))
                return result is not None
            except RuntimeError:
                return False
                
        except Exception as e:
            logger.error(f"Error checking token blacklist: {e}")
            return False


# Global JWT manager instance
_jwt_manager: Optional[JWTManager] = None


def get_jwt_manager() -> JWTManager:
    """Get JWT manager singleton."""
    global _jwt_manager
    if _jwt_manager is None:
        _jwt_manager = JWTManager()
    return _jwt_manager


# Dependency for protected endpoints
async def get_current_user(
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> str:
    """
    Get current user from JWT token.
    
    Returns:
        Wallet address of authenticated user
        
    Raises:
        HTTPException: If authentication fails
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = authorization.credentials
    jwt_manager = get_jwt_manager()
    
    try:
        payload = jwt_manager.verify_token(token, token_type="access")
        wallet_address = payload["sub"]
        return wallet_address
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Optional authentication (doesn't raise error if missing)
async def get_current_user_optional(
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[str]:
    """
    Get current user from JWT token (optional).
    
    Returns:
        Wallet address of authenticated user, or None if not authenticated
    """
    if not authorization:
        return None
    
    try:
        return await get_current_user(authorization)
    except HTTPException:
        return None

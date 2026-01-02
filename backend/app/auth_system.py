"""
Authentication System - Secure JWT-based authentication with Vault integration

Provides secure authentication with:
- JWT tokens with Vault-managed signing keys
- Rate limiting for brute force protection
- Secure credential validation
- Session management
- Authentication logging
"""

import os
import time
import logging
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from functools import lru_cache

import jwt
from passlib.context import CryptContext
from passlib.hash import bcrypt

logger = logging.getLogger(__name__)


@dataclass
class AuthConfig:
    """Authentication configuration."""
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    password_min_length: int = 8
    require_strong_passwords: bool = True


class AuthenticationError(Exception):
    """Base authentication error."""
    pass


class InvalidCredentialsError(AuthenticationError):
    """Invalid username or password."""
    pass


class AccountLockedError(AuthenticationError):
    """Account is locked due to too many failed attempts."""
    pass


class TokenExpiredError(AuthenticationError):
    """JWT token has expired."""
    pass


class InvalidTokenError(AuthenticationError):
    """JWT token is invalid."""
    pass


class RateLimitExceededError(AuthenticationError):
    """Rate limit exceeded."""
    pass


class RateLimiter:
    """
    Simple in-memory rate limiter for authentication attempts.
    
    In production, this should be replaced with Redis-based rate limiting.
    """
    
    def __init__(self):
        self._attempts: Dict[str, list] = {}
        self._lockouts: Dict[str, datetime] = {}
    
    def check_rate_limit(self, identifier: str, max_attempts: int, window_minutes: int) -> bool:
        """
        Check if identifier is within rate limits.
        
        Args:
            identifier: IP address or username
            max_attempts: Maximum attempts allowed
            window_minutes: Time window in minutes
            
        Returns:
            True if within limits, False if rate limited
        """
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=window_minutes)
        
        # Check if currently locked out
        if identifier in self._lockouts:
            lockout_time = self._lockouts[identifier]
            if now < lockout_time:
                return False
            else:
                # Lockout expired, remove it
                del self._lockouts[identifier]
        
        # Clean old attempts
        if identifier in self._attempts:
            self._attempts[identifier] = [
                attempt_time for attempt_time in self._attempts[identifier]
                if attempt_time > window_start
            ]
        
        # Check current attempt count
        current_attempts = len(self._attempts.get(identifier, []))
        return current_attempts < max_attempts
    
    def record_attempt(self, identifier: str, success: bool, lockout_duration_minutes: int = 15):
        """
        Record an authentication attempt.
        
        Args:
            identifier: IP address or username
            success: Whether the attempt was successful
            lockout_duration_minutes: How long to lock out after max attempts
        """
        now = datetime.utcnow()
        
        if identifier not in self._attempts:
            self._attempts[identifier] = []
        
        if not success:
            self._attempts[identifier].append(now)
            
            # Check if we should lock out
            if len(self._attempts[identifier]) >= 5:  # Max attempts reached
                self._lockouts[identifier] = now + timedelta(minutes=lockout_duration_minutes)
                logger.warning(f"Account locked due to too many failed attempts: {identifier}")
        else:
            # Successful login, clear attempts
            if identifier in self._attempts:
                del self._attempts[identifier]
            if identifier in self._lockouts:
                del self._lockouts[identifier]


class AuthenticationSystem:
    """
    Secure authentication system with JWT tokens and Vault integration.
    
    Features:
    - JWT tokens with Vault-managed signing keys
    - Rate limiting for brute force protection
    - Secure password hashing with bcrypt
    - Account lockout after failed attempts
    - Authentication audit logging
    """
    
    def __init__(self, config: Optional[AuthConfig] = None):
        """
        Initialize authentication system.
        
        Args:
            config: Authentication configuration
        """
        self.config = config or AuthConfig()
        self._jwt_secret: Optional[str] = None
        self._password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self._rate_limiter = RateLimiter()
        self._vault_client = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize authentication system with Vault secrets."""
        if self._initialized:
            return
        
        try:
            # Get Vault client if available
            vault_addr = os.getenv("VAULT_ADDR")
            if vault_addr:
                from .vault_client import get_vault_client
                self._vault_client = get_vault_client()
                await self._vault_client.initialize()
                
                # Load JWT secret from Vault
                try:
                    jwt_config = await self._vault_client.get_secret("jwt")
                    if isinstance(jwt_config, dict):
                        self._jwt_secret = jwt_config.get("secret_key")
                        logger.info("✅ JWT secret loaded from Vault")
                except Exception as e:
                    logger.warning(f"Failed to load JWT secret from Vault: {e}")
            
            # Fallback to environment variable
            if not self._jwt_secret:
                self._jwt_secret = os.getenv("JWT_SECRET_KEY")
                if self._jwt_secret:
                    logger.info("JWT secret loaded from environment variable")
            
            # Generate random secret if none found (development only)
            if not self._jwt_secret:
                is_production = os.getenv("R3MES_ENV", "development").lower() == "production"
                if is_production:
                    raise AuthenticationError("JWT secret must be configured in production")
                
                self._jwt_secret = secrets.token_urlsafe(32)
                logger.warning("⚠️  Generated random JWT secret (development only)")
            
            self._initialized = True
            logger.info("✅ Authentication system initialized")
            
        except Exception as e:
            logger.error(f"❌ Authentication system initialization failed: {e}")
            raise AuthenticationError(f"Authentication initialization failed: {e}")
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        return self._password_context.hash(password)
    
    def _verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return self._password_context.verify(password, hashed)
    
    def _validate_password_strength(self, password: str) -> None:
        """
        Validate password strength.
        
        Args:
            password: Password to validate
            
        Raises:
            AuthenticationError: If password is too weak
        """
        if len(password) < self.config.password_min_length:
            raise AuthenticationError(f"Password must be at least {self.config.password_min_length} characters")
        
        if not self.config.require_strong_passwords:
            return
        
        # Check for required character types
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        missing = []
        if not has_upper:
            missing.append("uppercase letter")
        if not has_lower:
            missing.append("lowercase letter")
        if not has_digit:
            missing.append("digit")
        if not has_special:
            missing.append("special character")
        
        if missing:
            raise AuthenticationError(f"Password must contain: {', '.join(missing)}")
    
    async def create_user(self, username: str, password: str, wallet_address: str, 
                         additional_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a new user account.
        
        Args:
            username: Username
            password: Plain text password
            wallet_address: User's wallet address
            additional_data: Additional user data
            
        Returns:
            User data dictionary
            
        Raises:
            AuthenticationError: If user creation fails
        """
        if not self._initialized:
            await self.initialize()
        
        # Validate inputs
        from .input_validator import validate_string, validate_wallet_address
        
        username = validate_string(username, "username", min_length=3, max_length=50)
        wallet_address = validate_wallet_address(wallet_address)
        
        # Validate password strength
        self._validate_password_strength(password)
        
        # Hash password
        password_hash = self._hash_password(password)
        
        # Create user data
        user_data = {
            "username": username,
            "password_hash": password_hash,
            "wallet_address": wallet_address,
            "created_at": datetime.utcnow().isoformat(),
            "is_active": True,
            "failed_login_attempts": 0,
            "last_login": None,
            **(additional_data or {})
        }
        
        # Store user in Vault if available
        if self._vault_client:
            try:
                await self._vault_client.put_secret(f"users/{username}", user_data)
                logger.info(f"User created and stored in Vault: {username}")
            except Exception as e:
                logger.error(f"Failed to store user in Vault: {e}")
                raise AuthenticationError(f"Failed to create user: {e}")
        
        # Remove sensitive data from return value
        safe_user_data = user_data.copy()
        del safe_user_data["password_hash"]
        
        return safe_user_data
    
    async def authenticate_user(self, username: str, password: str, 
                              client_ip: str) -> Dict[str, Any]:
        """
        Authenticate user with rate limiting.
        
        Args:
            username: Username
            password: Password
            client_ip: Client IP address for rate limiting
            
        Returns:
            Authentication result with JWT token
            
        Raises:
            AuthenticationError: If authentication fails
        """
        if not self._initialized:
            await self.initialize()
        
        # Check rate limiting
        if not self._rate_limiter.check_rate_limit(client_ip, 5, 15):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise RateLimitExceededError("Too many authentication attempts. Please try again later.")
        
        if not self._rate_limiter.check_rate_limit(username, 3, 15):
            logger.warning(f"Rate limit exceeded for user: {username}")
            raise AccountLockedError("Account temporarily locked due to too many failed attempts.")
        
        try:
            # Get user from Vault
            user_data = None
            if self._vault_client:
                try:
                    user_data = await self._vault_client.get_secret(f"users/{username}")
                except Exception as e:
                    logger.debug(f"User not found in Vault: {username}")
            
            if not user_data:
                # Record failed attempt
                self._rate_limiter.record_attempt(client_ip, False)
                self._rate_limiter.record_attempt(username, False)
                raise InvalidCredentialsError("Invalid username or password")
            
            # Check if account is active
            if not user_data.get("is_active", True):
                logger.warning(f"Inactive account login attempt: {username}")
                raise InvalidCredentialsError("Account is disabled")
            
            # Verify password
            password_hash = user_data.get("password_hash")
            if not password_hash or not self._verify_password(password, password_hash):
                # Record failed attempt
                self._rate_limiter.record_attempt(client_ip, False)
                self._rate_limiter.record_attempt(username, False)
                logger.warning(f"Invalid password for user: {username}")
                raise InvalidCredentialsError("Invalid username or password")
            
            # Successful authentication
            self._rate_limiter.record_attempt(client_ip, True)
            self._rate_limiter.record_attempt(username, True)
            
            # Update last login time
            user_data["last_login"] = datetime.utcnow().isoformat()
            if self._vault_client:
                try:
                    await self._vault_client.put_secret(f"users/{username}", user_data)
                except Exception as e:
                    logger.warning(f"Failed to update last login time: {e}")
            
            # Generate JWT token
            token = self._generate_jwt_token(user_data)
            
            logger.info(f"Successful authentication: {username} from {client_ip}")
            
            return {
                "token": token,
                "user": {
                    "username": user_data["username"],
                    "wallet_address": user_data["wallet_address"],
                    "last_login": user_data["last_login"]
                }
            }
            
        except AuthenticationError:
            raise
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise AuthenticationError(f"Authentication failed: {e}")
    
    def _generate_jwt_token(self, user_data: Dict[str, Any]) -> str:
        """
        Generate JWT token for authenticated user.
        
        Args:
            user_data: User data dictionary
            
        Returns:
            JWT token string
        """
        now = datetime.utcnow()
        payload = {
            "username": user_data["username"],
            "wallet_address": user_data["wallet_address"],
            "iat": now,
            "exp": now + timedelta(hours=self.config.jwt_expiration_hours),
            "iss": "r3mes-backend",
            "sub": user_data["username"]
        }
        
        return jwt.encode(payload, self._jwt_secret, algorithm=self.config.jwt_algorithm)
    
    def verify_jwt_token(self, token: str) -> Dict[str, Any]:
        """
        Verify and decode JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded token payload
            
        Raises:
            TokenExpiredError: If token has expired
            InvalidTokenError: If token is invalid
        """
        if not self._initialized:
            raise InvalidTokenError("Authentication system not initialized")
        
        try:
            payload = jwt.decode(
                token, 
                self._jwt_secret, 
                algorithms=[self.config.jwt_algorithm],
                options={"verify_exp": True}
            )
            return payload
            
        except jwt.ExpiredSignatureError:
            raise TokenExpiredError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise InvalidTokenError(f"Invalid token: {e}")
    
    async def refresh_token(self, token: str) -> str:
        """
        Refresh JWT token if it's close to expiration.
        
        Args:
            token: Current JWT token
            
        Returns:
            New JWT token
            
        Raises:
            InvalidTokenError: If token is invalid
        """
        payload = self.verify_jwt_token(token)
        
        # Check if token is close to expiration (within 1 hour)
        exp_time = datetime.fromtimestamp(payload["exp"])
        if exp_time - datetime.utcnow() > timedelta(hours=1):
            return token  # Token is still fresh
        
        # Get user data and generate new token
        username = payload["username"]
        if self._vault_client:
            try:
                user_data = await self._vault_client.get_secret(f"users/{username}")
                return self._generate_jwt_token(user_data)
            except Exception as e:
                logger.error(f"Failed to refresh token for user {username}: {e}")
                raise InvalidTokenError("Failed to refresh token")
        
        raise InvalidTokenError("Cannot refresh token: Vault not available")
    
    async def change_password(self, username: str, old_password: str, 
                            new_password: str) -> None:
        """
        Change user password.
        
        Args:
            username: Username
            old_password: Current password
            new_password: New password
            
        Raises:
            AuthenticationError: If password change fails
        """
        if not self._initialized:
            await self.initialize()
        
        # Validate new password strength
        self._validate_password_strength(new_password)
        
        # Get user data
        if not self._vault_client:
            raise AuthenticationError("Password change not available: Vault not configured")
        
        try:
            user_data = await self._vault_client.get_secret(f"users/{username}")
        except Exception:
            raise InvalidCredentialsError("User not found")
        
        # Verify old password
        password_hash = user_data.get("password_hash")
        if not password_hash or not self._verify_password(old_password, password_hash):
            raise InvalidCredentialsError("Current password is incorrect")
        
        # Update password
        user_data["password_hash"] = self._hash_password(new_password)
        user_data["password_changed_at"] = datetime.utcnow().isoformat()
        
        await self._vault_client.put_secret(f"users/{username}", user_data)
        
        logger.info(f"Password changed for user: {username}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform authentication system health check.
        
        Returns:
            Health status information
        """
        try:
            if not self._initialized:
                return {"status": "not_initialized", "healthy": False}
            
            # Check JWT secret
            if not self._jwt_secret:
                return {"status": "no_jwt_secret", "healthy": False}
            
            # Check Vault connection if configured
            vault_status = "not_configured"
            if self._vault_client:
                vault_health = await self._vault_client.health_check()
                vault_status = "healthy" if vault_health.get("healthy") else "unhealthy"
            
            return {
                "status": "healthy",
                "healthy": True,
                "vault_status": vault_status,
                "jwt_algorithm": self.config.jwt_algorithm,
                "jwt_expiration_hours": self.config.jwt_expiration_hours
            }
            
        except Exception as e:
            logger.error(f"Authentication health check failed: {e}")
            return {
                "status": "unhealthy",
                "healthy": False,
                "error": str(e)
            }


# Global authentication system instance
_auth_system: Optional[AuthenticationSystem] = None


def get_auth_system() -> AuthenticationSystem:
    """
    Get or create global authentication system instance.
    
    Returns:
        AuthenticationSystem instance
    """
    global _auth_system
    
    if _auth_system is None:
        _auth_system = AuthenticationSystem()
    
    return _auth_system


async def initialize_auth_system() -> AuthenticationSystem:
    """
    Initialize authentication system.
    
    Returns:
        Initialized AuthenticationSystem instance
    """
    auth_system = get_auth_system()
    await auth_system.initialize()
    return auth_system


# Convenience functions
async def authenticate_user(username: str, password: str, client_ip: str) -> Dict[str, Any]:
    """Convenience function to authenticate user."""
    auth_system = get_auth_system()
    return await auth_system.authenticate_user(username, password, client_ip)


def verify_token(token: str) -> Dict[str, Any]:
    """Convenience function to verify JWT token."""
    auth_system = get_auth_system()
    return auth_system.verify_jwt_token(token)


async def auth_health_check() -> Dict[str, Any]:
    """Convenience function for authentication health check."""
    auth_system = get_auth_system()
    return await auth_system.health_check()
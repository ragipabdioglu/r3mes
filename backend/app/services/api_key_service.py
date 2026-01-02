"""
API Key Service

Business logic for API key management operations.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from .base_service import BaseService
from ..repositories.api_key_repository import APIKeyRepository
from ..input_validation import validate_wallet_address, sanitize_string
from ..exceptions import ResourceNotFoundError, InvalidInputError, AuthenticationError


class APIKeyService(BaseService):
    """Service for API key management operations."""
    
    def __init__(self, database, cache_manager=None):
        """
        Initialize API key service.
        
        Args:
            database: Database instance
            cache_manager: Cache manager instance (optional)
        """
        super().__init__(database, cache_manager)
        self.api_key_repo = APIKeyRepository(database)
    
    async def create_api_key(
        self,
        wallet_address: str,
        name: Optional[str] = None,
        expires_in_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create a new API key with business logic validation.
        
        Args:
            wallet_address: Owner wallet address
            name: API key name (optional)
            expires_in_days: Expiration in days (optional)
            
        Returns:
            Created API key information (including plaintext key)
        """
        # Validate inputs
        wallet_address = validate_wallet_address(wallet_address)
        
        if name:
            name = sanitize_string(name, max_length=100)
            if not name.strip():
                name = None
        
        if expires_in_days is not None:
            if expires_in_days < 1 or expires_in_days > 365:
                raise InvalidInputError(
                    message="Expiration days must be between 1 and 365",
                    field="expires_in_days",
                    value=expires_in_days
                )
        
        try:
            # Check if user exists (create if not)
            from .user_service import UserService
            user_service = UserService(self.db, self.cache)
            
            try:
                await user_service.get_user_info(wallet_address)
            except ResourceNotFoundError:
                # Create user if doesn't exist
                await user_service.create_user(wallet_address)
            
            # Create API key
            api_key_data = await self.api_key_repo.create_api_key(
                wallet_address=wallet_address,
                name=name or "Default",
                expires_in_days=expires_in_days
            )
            
            # Invalidate cache
            cache_key = f"api_keys:{wallet_address}"
            await self._cache_delete(cache_key)
            
            await self._log_operation("create_api_key", {
                "wallet_address": wallet_address,
                "name": name,
                "expires_in_days": expires_in_days
            })
            
            return api_key_data
            
        except Exception as e:
            await self._handle_error(e, "create_api_key", {
                "wallet_address": wallet_address,
                "name": name,
                "expires_in_days": expires_in_days
            })
    
    async def list_api_keys(self, wallet_address: str) -> List[Dict[str, Any]]:
        """
        List API keys for a wallet with caching.
        
        Args:
            wallet_address: Owner wallet address
            
        Returns:
            List of API key information (without plaintext keys)
        """
        # Validate input
        wallet_address = validate_wallet_address(wallet_address)
        
        # Check cache first
        cache_key = f"api_keys:{wallet_address}"
        cached_keys = await self._cache_get(cache_key)
        if cached_keys:
            await self._log_operation("list_api_keys_cached", {"wallet_address": wallet_address})
            return cached_keys
        
        try:
            # Get from database
            api_keys = await self.api_key_repo.list_api_keys(wallet_address)
            
            # Cache the result
            await self._cache_set(cache_key, api_keys, ttl=300)  # 5 minutes
            
            await self._log_operation("list_api_keys", {
                "wallet_address": wallet_address,
                "count": len(api_keys)
            })
            
            return api_keys
            
        except Exception as e:
            await self._handle_error(e, "list_api_keys", {"wallet_address": wallet_address})
    
    async def revoke_api_key(self, api_key_id: int, wallet_address: str) -> bool:
        """
        Revoke an API key with ownership validation.
        
        Args:
            api_key_id: API key ID to revoke
            wallet_address: Owner wallet address
            
        Returns:
            True if revoked successfully
        """
        # Validate inputs
        wallet_address = validate_wallet_address(wallet_address)
        
        if api_key_id <= 0:
            raise InvalidInputError(
                message="API key ID must be positive",
                field="api_key_id",
                value=api_key_id
            )
        
        try:
            # Revoke API key
            success = await self.api_key_repo.revoke_api_key(api_key_id, wallet_address)
            
            if success:
                # Invalidate cache
                cache_key = f"api_keys:{wallet_address}"
                await self._cache_delete(cache_key)
                
                await self._log_operation("revoke_api_key", {
                    "api_key_id": api_key_id,
                    "wallet_address": wallet_address
                })
            
            return success
            
        except Exception as e:
            await self._handle_error(e, "revoke_api_key", {
                "api_key_id": api_key_id,
                "wallet_address": wallet_address
            })
    
    async def delete_api_key(self, api_key_id: int, wallet_address: str) -> bool:
        """
        Delete an API key permanently with ownership validation.
        
        Args:
            api_key_id: API key ID to delete
            wallet_address: Owner wallet address
            
        Returns:
            True if deleted successfully
        """
        # Validate inputs
        wallet_address = validate_wallet_address(wallet_address)
        
        if api_key_id <= 0:
            raise InvalidInputError(
                message="API key ID must be positive",
                field="api_key_id",
                value=api_key_id
            )
        
        try:
            # Delete API key
            success = await self.api_key_repo.delete_api_key(api_key_id, wallet_address)
            
            if success:
                # Invalidate cache
                cache_key = f"api_keys:{wallet_address}"
                await self._cache_delete(cache_key)
                
                await self._log_operation("delete_api_key", {
                    "api_key_id": api_key_id,
                    "wallet_address": wallet_address
                })
            
            return success
            
        except Exception as e:
            await self._handle_error(e, "delete_api_key", {
                "api_key_id": api_key_id,
                "wallet_address": wallet_address
            })
    
    async def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Validate an API key with caching.
        
        Args:
            api_key: API key to validate
            
        Returns:
            API key information if valid, None otherwise
        """
        if not api_key or not api_key.strip():
            return None
        
        api_key = api_key.strip()
        
        # Check cache first
        cache_key = f"api_key_validation:{api_key[:16]}"  # Use prefix for security
        cached_result = await self._cache_get(cache_key)
        if cached_result is not None:
            await self._log_operation("validate_api_key_cached", {"api_key_prefix": api_key[:16]})
            return cached_result if cached_result != "INVALID" else None
        
        try:
            # Validate with database
            api_key_info = await self.db.validate_api_key(api_key)
            
            # Cache the result (cache invalid keys too to prevent repeated DB hits)
            cache_value = api_key_info if api_key_info else "INVALID"
            await self._cache_set(cache_key, cache_value, ttl=60)  # 1 minute
            
            await self._log_operation("validate_api_key", {
                "api_key_prefix": api_key[:16],
                "valid": api_key_info is not None,
                "wallet_address": api_key_info.get("wallet_address") if api_key_info else None
            })
            
            return api_key_info
            
        except Exception as e:
            await self._handle_error(e, "validate_api_key", {"api_key_prefix": api_key[:16]})
    
    async def get_api_key_stats(self, wallet_address: str) -> Dict[str, Any]:
        """
        Get API key statistics for a wallet.
        
        Args:
            wallet_address: Owner wallet address
            
        Returns:
            API key statistics
        """
        # Validate input
        wallet_address = validate_wallet_address(wallet_address)
        
        # Check cache first
        cache_key = f"api_key_stats:{wallet_address}"
        cached_stats = await self._cache_get(cache_key)
        if cached_stats:
            await self._log_operation("get_api_key_stats_cached", {"wallet_address": wallet_address})
            return cached_stats
        
        try:
            # Get stats from repository
            stats = await self.api_key_repo.get_api_key_stats(wallet_address)
            
            # Cache the result
            await self._cache_set(cache_key, stats, ttl=300)  # 5 minutes
            
            await self._log_operation("get_api_key_stats", {
                "wallet_address": wallet_address,
                "total_keys": stats.get("total_keys", 0),
                "active_keys": stats.get("active_keys", 0)
            })
            
            return stats
            
        except Exception as e:
            await self._handle_error(e, "get_api_key_stats", {"wallet_address": wallet_address})
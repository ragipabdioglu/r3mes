"""
API Key Repository

Handles all API key-related database operations with proper validation and error handling.
"""

import os
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from .base_repository import BaseRepository
from ..input_validation import validate_wallet_address, sanitize_string
from ..exceptions import InvalidInputError, ResourceNotFoundError, AuthenticationError

# Configuration constants
MAX_API_KEYS_PER_WALLET = int(os.getenv("MAX_API_KEYS_PER_WALLET", "10"))
API_KEY_NAME_MAX_LENGTH = int(os.getenv("API_KEY_NAME_MAX_LENGTH", "100"))


class APIKeyRepository(BaseRepository):
    """Repository for API key management."""
    
    async def create_api_key(
        self,
        wallet_address: str,
        name: Optional[str] = None,
        expires_in_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create new API key.
        
        Args:
            wallet_address: Wallet address
            name: API key name (optional)
            expires_in_days: Expiration in days (optional)
            
        Returns:
            Dictionary with API key information (includes plain key - shown only once)
            
        Raises:
            InvalidInputError: If inputs are invalid
            DatabaseError: If database operation fails
        """
        # Validate inputs
        wallet_address = await self._validate_input(
            wallet_address,
            "wallet_address",
            validate_wallet_address
        )
        
        if name is not None:
            name = await self._validate_input(
                name,
                "name",
                lambda x: sanitize_string(x, max_length=API_KEY_NAME_MAX_LENGTH),
                required=False
            )
        
        # Validate expiration
        if expires_in_days is not None:
            if not isinstance(expires_in_days, int) or expires_in_days < 1 or expires_in_days > 365:
                raise InvalidInputError(
                    message="Expiration must be between 1 and 365 days",
                    field="expires_in_days",
                    value=expires_in_days
                )
        
        # Check API key limit per wallet (configurable)
        existing_keys = await self.list_api_keys(wallet_address)
        active_keys = [key for key in existing_keys if key.get('is_active', False)]
        
        if len(active_keys) >= MAX_API_KEYS_PER_WALLET:
            raise InvalidInputError(
                message=f"Maximum number of API keys reached ({MAX_API_KEYS_PER_WALLET})",
                field="wallet_address",
                value=wallet_address,
                details={"active_keys": len(active_keys), "max_keys": MAX_API_KEYS_PER_WALLET}
            )
        
        # Create API key
        api_key_data = await self._execute_query(
            "create_api_key",
            self.database.create_api_key,
            wallet_address,
            name,
            expires_in_days
        )
        
        self._log_operation("create_api_key", {
            "wallet_address": wallet_address,
            "name": name,
            "expires_in_days": expires_in_days,
            "key_id": api_key_data.get("api_key", "")[:12] + "..."  # Log only prefix
        })
        
        return api_key_data
    
    async def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Validate API key and return information.
        
        Args:
            api_key: API key to validate
            
        Returns:
            API key info if valid, None otherwise
            
        Raises:
            DatabaseError: If database operation fails
        """
        if not api_key or not isinstance(api_key, str):
            return None
        
        # Basic format validation
        if not api_key.startswith("r3mes_") or len(api_key) < 20:
            return None
        
        api_key_info = await self._execute_query(
            "validate_api_key",
            self.database.validate_api_key,
            api_key
        )
        
        if api_key_info:
            # Check if expired
            expires_at = api_key_info.get('expires_at')
            if expires_at:
                if isinstance(expires_at, str):
                    expires_at = datetime.fromisoformat(expires_at)
                
                if datetime.now() > expires_at:
                    self._log_operation("validate_api_key_expired", {
                        "wallet_address": api_key_info.get('wallet_address'),
                        "expires_at": expires_at.isoformat()
                    })
                    return None
            
            # Check if active
            if not api_key_info.get('is_active', False):
                self._log_operation("validate_api_key_inactive", {
                    "wallet_address": api_key_info.get('wallet_address')
                })
                return None
            
            self._log_operation("validate_api_key_success", {
                "wallet_address": api_key_info.get('wallet_address')
            })
        else:
            self._log_operation("validate_api_key_not_found", {
                "key_prefix": api_key[:12] + "..."
            })
        
        return api_key_info
    
    async def list_api_keys(self, wallet_address: str) -> List[Dict[str, Any]]:
        """
        List all API keys for wallet.
        
        Args:
            wallet_address: Wallet address
            
        Returns:
            List of API keys (without plain keys for security)
            
        Raises:
            InvalidInputError: If wallet address is invalid
            DatabaseError: If database operation fails
        """
        # Validate input
        wallet_address = await self._validate_input(
            wallet_address,
            "wallet_address",
            validate_wallet_address
        )
        
        # Execute query
        keys = await self._execute_query(
            "list_api_keys",
            self.database.list_api_keys,
            wallet_address
        )
        
        self._log_operation("list_api_keys", {
            "wallet_address": wallet_address,
            "key_count": len(keys)
        })
        
        return keys
    
    async def revoke_api_key(
        self,
        api_key: str,
        wallet_address: str
    ) -> bool:
        """
        Revoke API key (set as inactive).
        
        Args:
            api_key: API key to revoke
            wallet_address: Wallet address (for authorization)
            
        Returns:
            True if successful
            
        Raises:
            InvalidInputError: If inputs are invalid
            AuthenticationError: If wallet doesn't own the key
            ResourceNotFoundError: If key not found
            DatabaseError: If database operation fails
        """
        # Validate inputs
        wallet_address = await self._validate_input(
            wallet_address,
            "wallet_address",
            validate_wallet_address
        )
        
        if not api_key or not isinstance(api_key, str):
            raise InvalidInputError(
                message="Invalid API key format",
                field="api_key",
                value="[REDACTED]"
            )
        
        # Verify ownership
        api_key_info = await self.validate_api_key(api_key)
        if not api_key_info:
            raise ResourceNotFoundError(
                resource_type="api_key",
                resource_id="[REDACTED]",
                message="API key not found or invalid"
            )
        
        if api_key_info.get('wallet_address') != wallet_address:
            raise AuthenticationError(
                message="API key does not belong to this wallet",
                details={
                    "provided_wallet": wallet_address,
                    "key_owner": api_key_info.get('wallet_address')
                }
            )
        
        # Revoke key
        success = await self._execute_query(
            "revoke_api_key",
            self.database.revoke_api_key,
            api_key,
            wallet_address
        )
        
        if not success:
            raise DatabaseError(
                message="Failed to revoke API key",
                details={"wallet_address": wallet_address}
            )
        
        self._log_operation("revoke_api_key", {
            "wallet_address": wallet_address,
            "key_name": api_key_info.get('name', 'Unnamed')
        })
        
        return success
    
    async def delete_api_key(
        self,
        api_key: str,
        wallet_address: str
    ) -> bool:
        """
        Permanently delete API key.
        
        Args:
            api_key: API key to delete
            wallet_address: Wallet address (for authorization)
            
        Returns:
            True if successful
            
        Raises:
            InvalidInputError: If inputs are invalid
            AuthenticationError: If wallet doesn't own the key
            ResourceNotFoundError: If key not found
            DatabaseError: If database operation fails
        """
        # Validate inputs
        wallet_address = await self._validate_input(
            wallet_address,
            "wallet_address",
            validate_wallet_address
        )
        
        if not api_key or not isinstance(api_key, str):
            raise InvalidInputError(
                message="Invalid API key format",
                field="api_key",
                value="[REDACTED]"
            )
        
        # Verify ownership (similar to revoke)
        api_key_info = await self.validate_api_key(api_key)
        if not api_key_info:
            # Try to get info even if inactive/expired for ownership check
            api_key_info = await self._execute_query(
                "get_api_key_info",
                self.database.get_api_key_info,
                api_key
            )
        
        if not api_key_info:
            raise ResourceNotFoundError(
                resource_type="api_key",
                resource_id="[REDACTED]",
                message="API key not found"
            )
        
        if api_key_info.get('wallet_address') != wallet_address:
            raise AuthenticationError(
                message="API key does not belong to this wallet",
                details={
                    "provided_wallet": wallet_address,
                    "key_owner": api_key_info.get('wallet_address')
                }
            )
        
        # Delete key
        success = await self._execute_query(
            "delete_api_key",
            self.database.delete_api_key,
            api_key,
            wallet_address
        )
        
        if not success:
            raise DatabaseError(
                message="Failed to delete API key",
                details={"wallet_address": wallet_address}
            )
        
        self._log_operation("delete_api_key", {
            "wallet_address": wallet_address,
            "key_name": api_key_info.get('name', 'Unnamed')
        })
        
        return success
    
    async def cleanup_expired_keys(self) -> int:
        """
        Clean up expired API keys.
        
        Returns:
            Number of keys cleaned up
            
        Raises:
            DatabaseError: If database operation fails
        """
        count = await self._execute_query(
            "cleanup_expired_keys",
            self.database.cleanup_expired_api_keys
        )
        
        self._log_operation("cleanup_expired_keys", {
            "cleaned_count": count
        })
        
        return count
    
    async def get_api_key_stats(self, wallet_address: str) -> Dict[str, Any]:
        """
        Get API key statistics for wallet.
        
        Args:
            wallet_address: Wallet address
            
        Returns:
            Dictionary with API key statistics
            
        Raises:
            InvalidInputError: If wallet address is invalid
            DatabaseError: If database operation fails
        """
        # Validate input
        wallet_address = await self._validate_input(
            wallet_address,
            "wallet_address",
            validate_wallet_address
        )
        
        # Get all keys for wallet
        keys = await self.list_api_keys(wallet_address)
        
        # Calculate statistics
        total_keys = len(keys)
        active_keys = len([k for k in keys if k.get('is_active', False)])
        expired_keys = 0
        
        for key in keys:
            expires_at = key.get('expires_at')
            if expires_at:
                if isinstance(expires_at, str):
                    expires_at = datetime.fromisoformat(expires_at)
                if datetime.now() > expires_at:
                    expired_keys += 1
        
        stats = {
            "wallet_address": wallet_address,
            "total_keys": total_keys,
            "active_keys": active_keys,
            "inactive_keys": total_keys - active_keys,
            "expired_keys": expired_keys,
            "max_keys": MAX_API_KEYS_PER_WALLET  # Configurable limit
        }
        
        self._log_operation("get_api_key_stats", stats)
        
        return stats
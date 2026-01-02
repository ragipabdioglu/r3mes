"""
User Service

Business logic for user management operations.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime

from .base_service import BaseService
from ..repositories.user_repository import UserRepository
from ..input_validation import validate_wallet_address
from ..exceptions import ResourceNotFoundError, InvalidInputError


class UserService(BaseService):
    """Service for user management operations."""
    
    def __init__(self, database, cache_manager=None):
        """
        Initialize user service.
        
        Args:
            database: Database instance
            cache_manager: Cache manager instance (optional)
        """
        super().__init__(database, cache_manager)
        self.user_repo = UserRepository(database)
    
    async def get_user_info(self, wallet_address: str) -> Dict[str, Any]:
        """
        Get user information with caching.
        
        Args:
            wallet_address: User wallet address
            
        Returns:
            User information dictionary
            
        Raises:
            ResourceNotFoundError: If user not found
        """
        # Validate input
        wallet_address = validate_wallet_address(wallet_address)
        
        # Check cache first
        cache_key = f"user_info:{wallet_address}"
        cached_info = await self._cache_get(cache_key)
        if cached_info:
            await self._log_operation("get_user_info_cached", {"wallet_address": wallet_address})
            return cached_info
        
        try:
            # Get from database
            user_info = await self.user_repo.get_by_wallet_address(wallet_address)
            
            if not user_info:
                raise ResourceNotFoundError(
                    message="User not found",
                    resource_type="user",
                    resource_id=wallet_address
                )
            
            # Cache the result
            await self._cache_set(cache_key, user_info, ttl=300)  # 5 minutes
            
            await self._log_operation("get_user_info", {"wallet_address": wallet_address})
            return user_info
            
        except Exception as e:
            await self._handle_error(e, "get_user_info", {"wallet_address": wallet_address})
    
    async def create_user(self, wallet_address: str, initial_credits: float = 0.0) -> Dict[str, Any]:
        """
        Create a new user.
        
        Args:
            wallet_address: User wallet address
            initial_credits: Initial credit amount
            
        Returns:
            Created user information
        """
        # Validate input
        wallet_address = validate_wallet_address(wallet_address)
        
        if initial_credits < 0:
            raise InvalidInputError(
                message="Initial credits cannot be negative",
                field="initial_credits",
                value=initial_credits
            )
        
        try:
            # Create user
            user_data = {
                "wallet_address": wallet_address,
                "credits": initial_credits,
                "is_miner": False,
                "created_at": datetime.utcnow().isoformat()
            }
            
            created_user = await self.user_repo.create(user_data)
            
            # Invalidate cache
            cache_key = f"user_info:{wallet_address}"
            await self._cache_delete(cache_key)
            
            await self._log_operation("create_user", {
                "wallet_address": wallet_address,
                "initial_credits": initial_credits
            })
            
            return created_user
            
        except Exception as e:
            await self._handle_error(e, "create_user", {
                "wallet_address": wallet_address,
                "initial_credits": initial_credits
            })
    
    async def update_credits(self, wallet_address: str, credit_change: float, operation: str = "manual") -> Dict[str, Any]:
        """
        Update user credits.
        
        Args:
            wallet_address: User wallet address
            credit_change: Credit change amount (positive or negative)
            operation: Operation description
            
        Returns:
            Updated user information
        """
        # Validate input
        wallet_address = validate_wallet_address(wallet_address)
        
        try:
            # Update credits
            updated_user = await self.user_repo.update_credits(wallet_address, credit_change)
            
            # Invalidate cache
            cache_key = f"user_info:{wallet_address}"
            await self._cache_delete(cache_key)
            
            await self._log_operation("update_credits", {
                "wallet_address": wallet_address,
                "credit_change": credit_change,
                "operation": operation,
                "new_balance": updated_user.get("credits", 0)
            })
            
            return updated_user
            
        except Exception as e:
            await self._handle_error(e, "update_credits", {
                "wallet_address": wallet_address,
                "credit_change": credit_change,
                "operation": operation
            })
    
    async def set_miner_status(self, wallet_address: str, is_miner: bool) -> Dict[str, Any]:
        """
        Update user miner status.
        
        Args:
            wallet_address: User wallet address
            is_miner: Miner status
            
        Returns:
            Updated user information
        """
        # Validate input
        wallet_address = validate_wallet_address(wallet_address)
        
        try:
            # Update miner status
            updated_user = await self.user_repo.update_miner_status(wallet_address, is_miner)
            
            # Invalidate cache
            cache_key = f"user_info:{wallet_address}"
            await self._cache_delete(cache_key)
            
            await self._log_operation("set_miner_status", {
                "wallet_address": wallet_address,
                "is_miner": is_miner
            })
            
            return updated_user
            
        except Exception as e:
            await self._handle_error(e, "set_miner_status", {
                "wallet_address": wallet_address,
                "is_miner": is_miner
            })
    
    async def get_user_stats(self, wallet_address: str) -> Dict[str, Any]:
        """
        Get comprehensive user statistics.
        
        Args:
            wallet_address: User wallet address
            
        Returns:
            User statistics dictionary
        """
        # Validate input
        wallet_address = validate_wallet_address(wallet_address)
        
        # Check cache first
        cache_key = f"user_stats:{wallet_address}"
        cached_stats = await self._cache_get(cache_key)
        if cached_stats:
            await self._log_operation("get_user_stats_cached", {"wallet_address": wallet_address})
            return cached_stats
        
        try:
            # Get basic user info
            user_info = await self.get_user_info(wallet_address)
            
            # Get additional stats (API keys, transactions, etc.)
            stats = {
                "wallet_address": wallet_address,
                "credits": user_info["credits"],
                "is_miner": user_info["is_miner"],
                "created_at": user_info.get("created_at"),
                "last_activity": user_info.get("last_activity"),
                # Add more stats as needed
                "total_api_requests": 0,  # TODO: Implement
                "total_credits_earned": 0,  # TODO: Implement
                "total_credits_spent": 0,  # TODO: Implement
            }
            
            # Cache the result
            await self._cache_set(cache_key, stats, ttl=600)  # 10 minutes
            
            await self._log_operation("get_user_stats", {"wallet_address": wallet_address})
            return stats
            
        except Exception as e:
            await self._handle_error(e, "get_user_stats", {"wallet_address": wallet_address})
    
    async def list_users(self, limit: int = 50, offset: int = 0, is_miner: Optional[bool] = None) -> Dict[str, Any]:
        """
        List users with pagination and filtering.
        
        Args:
            limit: Number of users to return
            offset: Pagination offset
            is_miner: Filter by miner status (optional)
            
        Returns:
            Paginated user list
        """
        try:
            users = await self.user_repo.list_users(
                limit=limit,
                offset=offset,
                is_miner=is_miner
            )
            
            await self._log_operation("list_users", {
                "limit": limit,
                "offset": offset,
                "is_miner": is_miner,
                "result_count": len(users)
            })
            
            return {
                "users": users,
                "limit": limit,
                "offset": offset,
                "total": len(users)  # TODO: Get actual total count
            }
            
        except Exception as e:
            await self._handle_error(e, "list_users", {
                "limit": limit,
                "offset": offset,
                "is_miner": is_miner
            })
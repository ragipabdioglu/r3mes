"""
User Repository

Handles all user-related database operations with proper validation and error handling.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime

from .base_repository import BaseRepository
from ..input_validation import validate_wallet_address
from ..exceptions import InvalidInputError, ResourceNotFoundError


class UserRepository(BaseRepository):
    """Repository for user-related database operations."""
    
    async def get_user_info(self, wallet_address: str) -> Dict[str, Any]:
        """
        Get user information by wallet address.
        
        Args:
            wallet_address: Wallet address to retrieve
            
        Returns:
            User information dictionary
            
        Raises:
            InvalidInputError: If wallet address is invalid
            ResourceNotFoundError: If user not found
            DatabaseError: If database operation fails
        """
        # Validate input
        wallet_address = await self._validate_input(
            wallet_address,
            "wallet_address",
            validate_wallet_address
        )
        
        # Execute query
        user_info = await self._execute_query(
            "get_user_info",
            self.database.get_user_info,
            wallet_address
        )
        
        if not user_info:
            raise ResourceNotFoundError(
                resource_type="user",
                resource_id=wallet_address,
                message=f"User not found: {wallet_address}"
            )
        
        self._log_operation("get_user_info", {"wallet_address": wallet_address})
        return user_info
    
    async def get_network_stats(self) -> Dict[str, Any]:
        """
        Get network-wide statistics.
        
        Returns:
            Network statistics dictionary
            
        Raises:
            DatabaseError: If database operation fails
        """
        stats = await self._execute_query(
            "get_network_stats",
            self.database.get_network_stats
        )
        
        self._log_operation("get_network_stats")
        return stats
    
    async def create_user(self, wallet_address: str) -> Dict[str, Any]:
        """
        Create new user.
        
        Args:
            wallet_address: Wallet address for new user
            
        Returns:
            Created user information
            
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
        
        # Check if user already exists
        existing_user = None
        try:
            existing_user = await self.database.get_user_info(wallet_address)
        except Exception:
            # User doesn't exist, which is what we want
            pass
        
        if existing_user:
            raise InvalidInputError(
                message=f"User already exists: {wallet_address}",
                field="wallet_address",
                value=wallet_address
            )
        
        # Create user
        user = await self._execute_query(
            "create_user",
            self.database.create_user,
            wallet_address
        )
        
        self._log_operation("create_user", {"wallet_address": wallet_address})
        return user
    
    async def update_credits(
        self, 
        wallet_address: str, 
        amount: float,
        operation_type: str = "update"
    ) -> bool:
        """
        Update user credits.
        
        Args:
            wallet_address: Wallet address
            amount: Amount to add/subtract (positive to add, negative to subtract)
            operation_type: Type of operation for logging
            
        Returns:
            True if successful
            
        Raises:
            InvalidInputError: If inputs are invalid
            ResourceNotFoundError: If user not found
            DatabaseError: If database operation fails
        """
        # Validate inputs
        wallet_address = await self._validate_input(
            wallet_address,
            "wallet_address",
            validate_wallet_address
        )
        
        if not isinstance(amount, (int, float)):
            raise InvalidInputError(
                message="Amount must be a number",
                field="amount",
                value=amount
            )
        
        if amount == 0:
            raise InvalidInputError(
                message="Amount cannot be zero",
                field="amount",
                value=amount
            )
        
        # Check if user exists
        user_info = await self.get_user_info(wallet_address)
        current_credits = user_info.get('credits', 0.0)
        
        # Check for insufficient credits if subtracting
        if amount < 0 and current_credits < abs(amount):
            raise InvalidInputError(
                message=f"Insufficient credits: {current_credits} < {abs(amount)}",
                field="amount",
                value=amount,
                details={
                    "current_credits": current_credits,
                    "required_credits": abs(amount)
                }
            )
        
        # Execute update
        success = await self._execute_query(
            "update_credits",
            self.database.update_credits,
            wallet_address,
            amount
        )
        
        if not success:
            raise DatabaseError(
                message="Failed to update credits",
                details={
                    "wallet_address": wallet_address,
                    "amount": amount,
                    "operation_type": operation_type
                }
            )
        
        self._log_operation("update_credits", {
            "wallet_address": wallet_address,
            "amount": amount,
            "operation_type": operation_type,
            "new_balance": current_credits + amount
        })
        
        return success
    
    async def add_credits(self, wallet_address: str, amount: float) -> bool:
        """
        Add credits to user account.
        
        Args:
            wallet_address: Wallet address
            amount: Amount to add (must be positive)
            
        Returns:
            True if successful
            
        Raises:
            InvalidInputError: If amount is not positive
            DatabaseError: If database operation fails
        """
        if amount <= 0:
            raise InvalidInputError(
                message="Amount must be positive",
                field="amount",
                value=amount
            )
        
        return await self.update_credits(wallet_address, amount, "add_credits")
    
    async def deduct_credits(self, wallet_address: str, amount: float) -> bool:
        """
        Deduct credits from user account.
        
        Args:
            wallet_address: Wallet address
            amount: Amount to deduct (must be positive)
            
        Returns:
            True if successful
            
        Raises:
            InvalidInputError: If amount is not positive or insufficient credits
            DatabaseError: If database operation fails
        """
        if amount <= 0:
            raise InvalidInputError(
                message="Amount must be positive",
                field="amount",
                value=amount
            )
        
        return await self.update_credits(wallet_address, -amount, "deduct_credits")
    
    async def check_credits(self, wallet_address: str) -> float:
        """
        Check user's current credit balance.
        
        Args:
            wallet_address: Wallet address
            
        Returns:
            Current credit balance
            
        Raises:
            InvalidInputError: If wallet address is invalid
            ResourceNotFoundError: If user not found
            DatabaseError: If database operation fails
        """
        user_info = await self.get_user_info(wallet_address)
        credits = user_info.get('credits', 0.0)
        
        self._log_operation("check_credits", {
            "wallet_address": wallet_address,
            "credits": credits
        })
        
        return credits
    
    async def set_miner_status(
        self, 
        wallet_address: str, 
        is_miner: bool
    ) -> bool:
        """
        Set user's miner status.
        
        Args:
            wallet_address: Wallet address
            is_miner: Whether user is a miner
            
        Returns:
            True if successful
            
        Raises:
            InvalidInputError: If wallet address is invalid
            ResourceNotFoundError: If user not found
            DatabaseError: If database operation fails
        """
        # Validate input
        wallet_address = await self._validate_input(
            wallet_address,
            "wallet_address",
            validate_wallet_address
        )
        
        if not isinstance(is_miner, bool):
            raise InvalidInputError(
                message="is_miner must be a boolean",
                field="is_miner",
                value=is_miner
            )
        
        # Check if user exists
        await self.get_user_info(wallet_address)
        
        # Update miner status
        success = await self._execute_query(
            "set_miner_status",
            self.database.set_miner_status,
            wallet_address,
            is_miner
        )
        
        self._log_operation("set_miner_status", {
            "wallet_address": wallet_address,
            "is_miner": is_miner
        })
        
        return success
    
    async def list_users(
        self, 
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        is_miner: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        List users with pagination and filtering.
        
        Args:
            limit: Number of users to return
            offset: Number of users to skip
            is_miner: Filter by miner status (optional)
            
        Returns:
            Dictionary with users list and pagination info
            
        Raises:
            InvalidInputError: If pagination parameters are invalid
            DatabaseError: If database operation fails
        """
        # Validate pagination
        limit, offset = await self._validate_pagination(limit, offset)
        
        # Execute query
        users = await self._execute_query(
            "list_users",
            self.database.list_users,
            limit=limit,
            offset=offset,
            is_miner=is_miner
        )
        
        # Get total count
        total_count = await self._execute_query(
            "count_users",
            self.database.count_users,
            is_miner=is_miner
        )
        
        result = {
            "users": users,
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(users) < total_count
        }
        
        self._log_operation("list_users", {
            "limit": limit,
            "offset": offset,
            "is_miner": is_miner,
            "returned_count": len(users),
            "total_count": total_count
        })
        
        return result
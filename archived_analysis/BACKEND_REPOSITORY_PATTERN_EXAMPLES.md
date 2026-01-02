# Repository Pattern Implementation Examples

## 1. Base Repository Class

```python
# backend/app/repositories/base_repository.py

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import logging
from ..exceptions import DatabaseError, InvalidInputError

logger = logging.getLogger(__name__)

class BaseRepository(ABC):
    """Base repository with common CRUD operations and error handling."""
    
    def __init__(self, database):
        self.database = database
        self.entity_name = self.__class__.__name__
    
    async def _execute_query(self, query_func, *args, **kwargs):
        """Execute query with error handling."""
        try:
            return await query_func(*args, **kwargs)
        except Exception as e:
            logger.error(f"{self.entity_name}: Query failed: {e}", exc_info=True)
            raise DatabaseError(
                message=f"Failed to execute {self.entity_name} query",
                operation=query_func.__name__,
                cause=e
            )
    
    async def _validate_input(self, value: Any, field_name: str, validator_func):
        """Validate input with error handling."""
        try:
            return validator_func(value)
        except Exception as e:
            logger.warning(f"{self.entity_name}: Validation failed for {field_name}: {e}")
            raise InvalidInputError(
                message=f"Invalid {field_name}",
                field=field_name,
                value=value
            )
```

## 2. User Repository Implementation

```python
# backend/app/repositories/user_repository.py

from typing import Optional, Dict, Any, List
from .base_repository import BaseRepository
from ..input_validation import validate_wallet_address
from ..exceptions import InvalidInputError

class UserRepository(BaseRepository):
    """Repository for user-related database operations."""
    
    async def get_user_info(self, wallet_address: str) -> Dict[str, Any]:
        """
        Get user information.
        
        Args:
            wallet_address: Wallet address to retrieve
            
        Returns:
            User information dictionary
            
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
        user_info = await self._execute_query(
            self.database.get_user_info,
            wallet_address
        )
        
        if not user_info:
            raise InvalidInputError(
                message=f"User not found: {wallet_address}",
                field="wallet_address",
                value=wallet_address
            )
        
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
            self.database.get_network_stats
        )
        
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
        
        # Execute query
        user = await self._execute_query(
            self.database.create_user,
            wallet_address
        )
        
        return user
    
    async def update_credits(self, wallet_address: str, amount: float) -> bool:
        """
        Update user credits.
        
        Args:
            wallet_address: Wallet address
            amount: Amount to add/subtract
            
        Returns:
            True if successful
            
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
        
        if not isinstance(amount, (int, float)):
            raise InvalidInputError(
                message="Amount must be a number",
                field="amount",
                value=amount
            )
        
        # Execute query
        success = await self._execute_query(
            self.database.update_credits,
            wallet_address,
            amount
        )
        
        return success
```

## 3. API Key Repository Implementation

```python
# backend/app/repositories/api_key_repository.py

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from .base_repository import BaseRepository
from ..input_validation import validate_wallet_address, sanitize_string
from ..exceptions import InvalidInputError

class APIKeyRepository(BaseRepository):
    """Repository for API key management."""
    
    async def create_api_key(
        self,
        wallet_address: str,
        name: str,
        expires_in_days: Optional[int] = None
    ) -> str:
        """
        Create new API key.
        
        Args:
            wallet_address: Wallet address
            name: API key name
            expires_in_days: Expiration in days (optional)
            
        Returns:
            Generated API key
            
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
        
        name = await self._validate_input(
            name,
            "name",
            lambda x: sanitize_string(x, max_length=100)
        )
        
        # Validate expiration
        if expires_in_days is not None:
            if not isinstance(expires_in_days, int) or expires_in_days < 1 or expires_in_days > 365:
                raise InvalidInputError(
                    message="Expiration must be between 1 and 365 days",
                    field="expires_in_days",
                    value=expires_in_days
                )
        
        # Execute query
        api_key = await self._execute_query(
            self.database.create_api_key,
            wallet_address,
            name,
            expires_in_days
        )
        
        return api_key
    
    async def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Validate API key.
        
        Args:
            api_key: API key to validate
            
        Returns:
            API key info if valid, None otherwise
            
        Raises:
            DatabaseError: If database operation fails
        """
        if not api_key or not isinstance(api_key, str):
            return None
        
        api_key_info = await self._execute_query(
            self.database.validate_api_key,
            api_key
        )
        
        return api_key_info
    
    async def list_api_keys(self, wallet_address: str) -> List[Dict[str, Any]]:
        """
        List all API keys for wallet.
        
        Args:
            wallet_address: Wallet address
            
        Returns:
            List of API keys
            
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
            self.database.list_api_keys,
            wallet_address
        )
        
        return keys
    
    async def revoke_api_key(
        self,
        api_key_id: int,
        wallet_address: str
    ) -> bool:
        """
        Revoke API key.
        
        Args:
            api_key_id: API key ID
            wallet_address: Wallet address (for authorization)
            
        Returns:
            True if successful
            
        Raises:
            InvalidInputError: If inputs are invalid
            DatabaseError: If database operation fails
        """
        # Validate inputs
        if not isinstance(api_key_id, int) or api_key_id < 1:
            raise InvalidInputError(
                message="API key ID must be a positive integer",
                field="api_key_id",
                value=api_key_id
            )
        
        wallet_address = await self._validate_input(
            wallet_address,
            "wallet_address",
            validate_wallet_address
        )
        
        # Execute query
        success = await self._execute_query(
            self.database.revoke_api_key,
            api_key_id,
            wallet_address
        )
        
        return success
```

## 4. Credit Repository Implementation

```python
# backend/app/repositories/credit_repository.py

from typing import Dict, Any
from .base_repository import BaseRepository
from ..input_validation import validate_wallet_address, validate_positive_number
from ..exceptions import InvalidInputError, InsufficientCreditsError

class CreditRepository(BaseRepository):
    """Repository for credit management."""
    
    async def reserve_credit_atomic(
        self,
        wallet_address: str,
        amount: float
    ) -> Dict[str, Any]:
        """
        Atomically reserve credits.
        
        Args:
            wallet_address: Wallet address
            amount: Amount to reserve
            
        Returns:
            Reservation info with reservation_id
            
        Raises:
            InvalidInputError: If inputs are invalid
            InsufficientCreditsError: If insufficient credits
            DatabaseError: If database operation fails
        """
        # Validate inputs
        wallet_address = await self._validate_input(
            wallet_address,
            "wallet_address",
            validate_wallet_address
        )
        
        amount = await self._validate_input(
            amount,
            "amount",
            lambda x: validate_positive_number(x, "amount")
        )
        
        # Execute query
        reservation = await self._execute_query(
            self.database.reserve_credit_atomic,
            wallet_address,
            amount
        )
        
        if not reservation.get("success"):
            raise InsufficientCreditsError(
                required=amount,
                available=reservation.get("available", 0),
                wallet=wallet_address
            )
        
        return reservation
    
    async def confirm_credit_reservation(self, reservation_id: str) -> bool:
        """
        Confirm credit reservation.
        
        Args:
            reservation_id: Reservation ID
            
        Returns:
            True if successful
            
        Raises:
            InvalidInputError: If reservation_id is invalid
            DatabaseError: If database operation fails
        """
        if not reservation_id or not isinstance(reservation_id, str):
            raise InvalidInputError(
                message="Invalid reservation ID",
                field="reservation_id",
                value=reservation_id
            )
        
        # Execute query
        success = await self._execute_query(
            self.database.confirm_credit_reservation,
            reservation_id
        )
        
        return success
    
    async def rollback_credit_reservation(self, reservation_id: str) -> bool:
        """
        Rollback credit reservation.
        
        Args:
            reservation_id: Reservation ID
            
        Returns:
            True if successful
            
        Raises:
            InvalidInputError: If reservation_id is invalid
            DatabaseError: If database operation fails
        """
        if not reservation_id or not isinstance(reservation_id, str):
            raise InvalidInputError(
                message="Invalid reservation ID",
                field="reservation_id",
                value=reservation_id
            )
        
        # Execute query
        success = await self._execute_query(
            self.database.rollback_credit_reservation,
            reservation_id
        )
        
        return success
```

## 5. Updated Endpoint Using Repository

```python
# backend/app/api/user.py

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from ..models.responses import UserInfoResponse
from ..repositories.user_repository import UserRepository
from ..exceptions import (
    InvalidInputError,
    DatabaseError,
    R3MESException
)
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/users", tags=["users"])

# Dependency injection
async def get_user_repository(database = Depends(...)) -> UserRepository:
    return UserRepository(database)

@router.get("/info/{wallet_address}", response_model=UserInfoResponse)
async def get_user_info(
    request: Request,
    wallet_address: str,
    user_repo: UserRepository = Depends(get_user_repository)
) -> UserInfoResponse:
    """
    Get user information.
    
    Args:
        wallet_address: Wallet address
        
    Returns:
        User information
        
    Raises:
        400: Invalid wallet address
        404: User not found
        500: Database error
    """
    try:
        # Use repository instead of direct database call
        user_info = await user_repo.get_user_info(wallet_address)
        
        return UserInfoResponse(
            wallet_address=user_info['wallet_address'],
            credits=user_info['credits'],
            is_miner=user_info['is_miner']
        )
        
    except InvalidInputError as e:
        logger.warning(f"Invalid input: {e}")
        raise HTTPException(status_code=400, detail=e.user_message)
        
    except DatabaseError as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database operation failed")
        
    except R3MESException as e:
        logger.error(f"R3MES error: {e}")
        raise HTTPException(status_code=500, detail=e.user_message)
        
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
```

## 6. Error Handler Middleware

```python
# backend/app/middleware/error_handler.py

from fastapi import Request
from fastapi.responses import JSONResponse
from ..exceptions import R3MESException, InvalidInputError, DatabaseError
import logging

logger = logging.getLogger(__name__)

async def error_handler_middleware(request: Request, call_next):
    """Middleware to handle all exceptions consistently."""
    try:
        response = await call_next(request)
        return response
    except InvalidInputError as e:
        logger.warning(f"Invalid input: {e}")
        return JSONResponse(
            status_code=400,
            content={
                "error": True,
                "error_code": e.error_code.value,
                "message": e.user_message,
                "details": e.details
            }
        )
    except DatabaseError as e:
        logger.error(f"Database error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "error_code": e.error_code.value,
                "message": "Database operation failed",
                "details": e.details
            }
        )
    except R3MESException as e:
        logger.error(f"R3MES error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "error_code": e.error_code.value,
                "message": e.user_message,
                "details": e.details
            }
        )
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "message": "Internal server error"
            }
        )
```


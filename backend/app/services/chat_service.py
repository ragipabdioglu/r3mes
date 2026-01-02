"""
Chat Service

Business logic for chat/inference operations.
"""

import os
from typing import Optional, Dict, Any, AsyncGenerator, List
from datetime import datetime

from .base_service import BaseService
from .user_service import UserService
from ..input_validation import validate_wallet_address, sanitize_string
from ..exceptions import InvalidInputError, InsufficientCreditsError, InferenceError
from ..inference_mode import get_inference_mode, InferenceMode


class ChatService(BaseService):
    """Service for chat/inference operations."""
    
    def __init__(self, database, cache_manager=None):
        """
        Initialize chat service.
        
        Args:
            database: Database instance
            cache_manager: Cache manager instance (optional)
        """
        super().__init__(database, cache_manager)
        self.user_service = UserService(database, cache_manager)
        self.credit_cost_per_request = float(os.getenv("CHAT_CREDIT_COST", "1.0"))
    
    async def process_chat_request(
        self,
        message: str,
        wallet_address: str,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a chat request with business logic validation.
        
        Args:
            message: User message
            wallet_address: User wallet address
            model_name: Requested model name (optional)
            
        Returns:
            Chat processing information
        """
        # Validate inputs
        wallet_address = validate_wallet_address(wallet_address)
        message = sanitize_string(message, max_length=10000)
        
        if not message.strip():
            raise InvalidInputError(
                message="Message cannot be empty",
                field="message",
                value=message
            )
        
        if model_name:
            model_name = sanitize_string(model_name, max_length=100)
        
        try:
            # Check inference mode
            inference_mode = get_inference_mode()
            
            if inference_mode == InferenceMode.DISABLED:
                raise InferenceError(
                    message="Inference service is disabled",
                    error_code="INFERENCE_DISABLED"
                )
            
            # Get user info and validate credits
            user_info = await self.user_service.get_user_info(wallet_address)
            
            if user_info["credits"] < self.credit_cost_per_request:
                raise InsufficientCreditsError(
                    message=f"Insufficient credits. Required: {self.credit_cost_per_request}, Available: {user_info['credits']}",
                    required_credits=self.credit_cost_per_request,
                    available_credits=user_info["credits"]
                )
            
            # Reserve credits atomically
            reservation = await self.db.reserve_credit_atomic(wallet_address, self.credit_cost_per_request)
            
            if not reservation["success"]:
                raise InsufficientCreditsError(
                    message=f"Failed to reserve credits: {reservation.get('error')}",
                    required_credits=self.credit_cost_per_request,
                    available_credits=user_info["credits"]
                )
            
            # Determine adapter/model
            adapter_name = await self._select_adapter(message, model_name)
            
            # Log the request
            await self._log_operation("process_chat_request", {
                "wallet_address": wallet_address,
                "message_length": len(message),
                "model_name": model_name,
                "adapter_name": adapter_name,
                "inference_mode": inference_mode.value,
                "reservation_id": reservation["reservation_id"]
            })
            
            return {
                "reservation_id": reservation["reservation_id"],
                "adapter_name": adapter_name,
                "inference_mode": inference_mode,
                "credit_cost": self.credit_cost_per_request,
                "message_processed": True
            }
            
        except Exception as e:
            await self._handle_error(e, "process_chat_request", {
                "wallet_address": wallet_address,
                "message_length": len(message) if message else 0,
                "model_name": model_name
            })
    
    async def _select_adapter(self, message: str, requested_model: Optional[str] = None) -> str:
        """
        Select appropriate adapter for the message.
        
        Args:
            message: User message
            requested_model: Requested model name (optional)
            
        Returns:
            Selected adapter name
        """
        # If specific model requested, use it
        if requested_model:
            return requested_model
        
        # Try semantic router first
        try:
            from ..semantic_router import get_semantic_router
            semantic_router = get_semantic_router()
            
            if semantic_router is not None:
                adapter_result = semantic_router.decide_adapter(message)
                if isinstance(adapter_result, tuple):
                    adapter_name, similarity_score = adapter_result
                    if similarity_score > 0:
                        return adapter_name
                else:
                    return adapter_result
        except Exception as e:
            self.logger.warning(f"Semantic router failed: {e}")
        
        # Fallback to simple keyword router
        return self._simple_keyword_router(message)
    
    def _simple_keyword_router(self, message: str) -> str:
        """
        Simple keyword-based adapter routing.
        
        Args:
            message: User message
            
        Returns:
            Adapter name based on keywords
        """
        message_lower = message.lower()
        
        # Simple keyword matching
        if any(word in message_lower for word in ["code", "program", "function", "debug", "error", "python", "javascript"]):
            return "coding"
        elif any(word in message_lower for word in ["legal", "law", "contract", "court", "lawyer"]):
            return "legal"
        elif any(word in message_lower for word in ["medical", "health", "doctor", "symptom", "medicine"]):
            return "medical"
        elif any(word in message_lower for word in ["finance", "money", "invest", "stock", "trading"]):
            return "finance"
        elif any(word in message_lower for word in ["math", "calculate", "equation", "formula"]):
            return "math"
        else:
            return "general"
    
    async def confirm_chat_completion(self, reservation_id: str, success: bool = True) -> bool:
        """
        Confirm chat completion and finalize credit transaction.
        
        Args:
            reservation_id: Credit reservation ID
            success: Whether the chat was successful
            
        Returns:
            True if confirmed successfully
        """
        try:
            if success:
                # Confirm credit deduction
                confirmed = await self.db.confirm_credit_reservation(reservation_id)
                
                await self._log_operation("confirm_chat_completion", {
                    "reservation_id": reservation_id,
                    "success": success,
                    "confirmed": confirmed
                })
                
                return confirmed
            else:
                # Rollback credit reservation
                rolled_back = await self.db.rollback_credit_reservation(reservation_id)
                
                await self._log_operation("rollback_chat_completion", {
                    "reservation_id": reservation_id,
                    "success": success,
                    "rolled_back": rolled_back
                })
                
                return rolled_back
                
        except Exception as e:
            await self._handle_error(e, "confirm_chat_completion", {
                "reservation_id": reservation_id,
                "success": success
            })
    
    async def get_chat_stats(self, wallet_address: str) -> Dict[str, Any]:
        """
        Get chat statistics for a user.
        
        Args:
            wallet_address: User wallet address
            
        Returns:
            Chat statistics
        """
        # Validate input
        wallet_address = validate_wallet_address(wallet_address)
        
        # Check cache first
        cache_key = f"chat_stats:{wallet_address}"
        cached_stats = await self._cache_get(cache_key)
        if cached_stats:
            await self._log_operation("get_chat_stats_cached", {"wallet_address": wallet_address})
            return cached_stats
        
        try:
            # Get user info
            user_info = await self.user_service.get_user_info(wallet_address)
            
            # TODO: Implement actual chat statistics from database
            stats = {
                "wallet_address": wallet_address,
                "total_requests": 0,  # TODO: Implement
                "successful_requests": 0,  # TODO: Implement
                "failed_requests": 0,  # TODO: Implement
                "total_credits_spent": 0,  # TODO: Implement
                "average_response_time": 0,  # TODO: Implement
                "favorite_adapter": "general",  # TODO: Implement
                "current_credits": user_info["credits"],
                "credit_cost_per_request": self.credit_cost_per_request
            }
            
            # Cache the result
            await self._cache_set(cache_key, stats, ttl=300)  # 5 minutes
            
            await self._log_operation("get_chat_stats", {"wallet_address": wallet_address})
            return stats
            
        except Exception as e:
            await self._handle_error(e, "get_chat_stats", {"wallet_address": wallet_address})
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models/adapters.
        
        Returns:
            List of available models
        """
        # Check cache first
        cache_key = "available_models"
        cached_models = await self._cache_get(cache_key)
        if cached_models:
            await self._log_operation("get_available_models_cached")
            return cached_models
        
        try:
            # Get inference mode
            inference_mode = get_inference_mode()
            
            # Default models/adapters
            models = [
                {
                    "name": "general",
                    "description": "General purpose conversation",
                    "category": "general",
                    "available": True
                },
                {
                    "name": "coding",
                    "description": "Programming and code assistance",
                    "category": "technical",
                    "available": True
                },
                {
                    "name": "legal",
                    "description": "Legal document analysis",
                    "category": "professional",
                    "available": inference_mode != InferenceMode.DISABLED
                },
                {
                    "name": "medical",
                    "description": "Medical information (not advice)",
                    "category": "professional",
                    "available": inference_mode != InferenceMode.DISABLED
                },
                {
                    "name": "finance",
                    "description": "Financial analysis and advice",
                    "category": "professional",
                    "available": inference_mode != InferenceMode.DISABLED
                },
                {
                    "name": "math",
                    "description": "Mathematical problem solving",
                    "category": "technical",
                    "available": True
                }
            ]
            
            # TODO: Get actual available models from model manager
            
            # Cache the result
            await self._cache_set(cache_key, models, ttl=600)  # 10 minutes
            
            await self._log_operation("get_available_models", {"count": len(models)})
            return models
            
        except Exception as e:
            await self._handle_error(e, "get_available_models")
"""
Chat API endpoints - AI inference with credit system
"""

import random
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from ..inference_mode import get_inference_mode, InferenceMode
from ..exceptions import (
    InvalidInputError, InvalidWalletAddressError, MissingCredentialsError,
    InsufficientCreditsError, InvalidAPIKeyError
)
from ..database_async import AsyncDatabase
from ..serving_node_registry import ServingNodeRegistry
from ..notifications import get_notification_service, NotificationPriority
from ..url_validator import validate_serving_endpoint

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

# Request models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000, description="User message for AI inference")
    wallet_address: Optional[str] = Field(None, description="Wallet address (optional if API key is provided)")
    
    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Validate and sanitize message input."""
        if not v or not v.strip():
            raise InvalidInputError("Message cannot be empty")
        # Remove null bytes and control characters (except newlines and tabs)
        v = v.replace('\x00', '')
        # Limit message length to prevent DoS
        if len(v) > 10000:
            raise InvalidInputError("Message too long (max 10000 characters)")
        return v.strip()
    
    @field_validator("wallet_address")
    @classmethod
    def validate_wallet_address(cls, v: Optional[str]) -> Optional[str]:
        """Validate Cosmos wallet address format."""
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        # Basic validation: Cosmos addresses start with specific prefixes
        if not v.startswith("remes"):
            raise InvalidWalletAddressError("Invalid address format: must start with 'remes'")
        if len(v) < 20 or len(v) > 60:
            raise InvalidWalletAddressError("Invalid address length (must be 20-60 characters)")
        # Check for invalid characters
        if not all(c.isalnum() or c in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'] for c in v):
            raise InvalidWalletAddressError("Invalid address format: contains invalid characters")
        return v


# Simple keyword-based adapter routing for non-local modes
def simple_keyword_router(message: str) -> str:
    """Simple keyword-based adapter routing when SemanticRouter is not available."""
    message_lower = message.lower()
    
    # Simple keyword matching
    if any(word in message_lower for word in ["code", "program", "function", "debug", "error"]):
        return "coding"
    elif any(word in message_lower for word in ["legal", "law", "contract", "court"]):
        return "legal"
    elif any(word in message_lower for word in ["medical", "health", "doctor", "symptom"]):
        return "medical"
    elif any(word in message_lower for word in ["finance", "money", "invest", "stock"]):
        return "finance"
    else:
        return "general"


class ChatService:
    """Service class for chat operations."""
    
    def __init__(self, database: AsyncDatabase, serving_node_registry: ServingNodeRegistry):
        self.database = database
        self.serving_node_registry = serving_node_registry
    
    async def get_wallet_from_auth(self, request: Request, api_key: Optional[str] = None) -> Optional[str]:
        """Extract wallet address from API key or request."""
        if api_key:
            api_key_info = await self.database.validate_api_key(api_key)
            if api_key_info and api_key_info["is_active"]:
                return api_key_info["wallet_address"]
            else:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid or expired API key"
                ) from InvalidAPIKeyError("Invalid or expired API key")
        return None
    
    async def process_chat_request(self, chat_request: ChatRequest, wallet_address: str) -> StreamingResponse:
        """Process chat request with credit management."""
        # Check inference mode first
        inference_mode = get_inference_mode()
        
        # Handle disabled mode
        if inference_mode == InferenceMode.DISABLED:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Inference service disabled",
                    "code": "INFERENCE_DISABLED",
                    "message": "AI inference is not available on this server. Please try again later or contact support."
                }
            )
        
        # Handle mock mode
        if inference_mode == InferenceMode.MOCK:
            async def generate_mock_response():
                mock_response = f"[MOCK MODE] This is a simulated response to: {chat_request.message[:50]}..."
                for word in mock_response.split():
                    yield word + " "
            return StreamingResponse(generate_mock_response(), media_type="text/plain")
        
        # ATOMIC CREDIT RESERVATION: Reserve credit before starting inference
        reservation = await self.database.reserve_credit_atomic(wallet_address, 1.0)
        
        if not reservation["success"]:
            notification_service = get_notification_service()
            await notification_service.send_notification(
                title="Low Credits Alert",
                message=f"Wallet {wallet_address} attempted to use chat with insufficient credits.",
                priority=NotificationPriority.LOW,
                metadata={"wallet_address": wallet_address, "error": reservation.get("error")}
            )
            raise HTTPException(
                status_code=402,
                detail="Insufficient credits. Please mine blocks to earn credits."
            ) from InsufficientCreditsError(f"Wallet {wallet_address}: {reservation.get('error')}")
        
        reservation_id = reservation["reservation_id"]
        logger.debug(f"Credit reserved for {wallet_address}, reservation_id: {reservation_id}")
        
        # Decide adapter using semantic router or simple keyword router
        adapter_name = self._decide_adapter(chat_request.message)
        
        # Try to route to serving node first
        serving_nodes = await self.serving_node_registry.get_serving_nodes_for_lora(
            lora_name=adapter_name,
            max_age_seconds=60
        )
        
        if serving_nodes:
            return await self._route_to_serving_node(
                serving_nodes, chat_request, wallet_address, reservation_id
            )
        
        # Handle case when no serving nodes available
        if inference_mode == InferenceMode.REMOTE:
            # In remote mode, we MUST have serving nodes
            logger.warning(f"No serving nodes available for {adapter_name} in remote mode")
            await self.database.rollback_credit_reservation(reservation_id)
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "No serving nodes available",
                    "code": "NO_SERVING_NODES",
                    "message": f"No serving nodes are currently available for adapter '{adapter_name}'. Please try again later."
                }
            )
        
        # Fallback to local inference (only for local mode)
        logger.debug(f"No serving nodes available for {adapter_name}, using local inference")
        return await self._local_inference(chat_request, adapter_name, wallet_address, reservation_id)
    
    def _decide_adapter(self, message: str) -> str:
        """Decide which adapter to use for the message."""
        try:
            from ..semantic_router import SemanticRouter
            semantic_router = SemanticRouter()
            adapter_result = semantic_router.decide_adapter(message)
            if isinstance(adapter_result, tuple):
                adapter_name, similarity_score = adapter_result
                if similarity_score > 0:
                    logger.debug(f"Semantic router: {adapter_name} (similarity: {similarity_score:.3f})")
                    return adapter_name
            else:
                return adapter_result
        except ImportError:
            pass
        
        # Use simple keyword router as fallback
        adapter_name = simple_keyword_router(message)
        logger.debug(f"Keyword router: {adapter_name}")
        return adapter_name
    
    async def _route_to_serving_node(self, serving_nodes, chat_request, wallet_address, reservation_id):
        """Route request to a serving node."""
        import httpx
        import os
        
        # Load balancing: simple random selection
        selected_node = random.choice(serving_nodes)
        endpoint_url = selected_node["endpoint_url"]
        
        # SSRF Protection: Validate serving node endpoint before proxying
        is_valid_url, validation_error = validate_serving_endpoint(endpoint_url)
        if not is_valid_url:
            logger.warning(
                f"SSRF Protection: Blocked request to serving node {selected_node['wallet_address']} "
                f"with invalid endpoint: {endpoint_url}. Reason: {validation_error}"
            )
            # Fall back to local inference instead of proxying to potentially malicious URL
            logger.info(f"Falling back to local inference due to invalid serving node endpoint")
            return await self._local_inference(chat_request, "general", wallet_address, reservation_id)
        
        logger.info(f"Routing chat request to serving node: {selected_node['wallet_address']} at {endpoint_url}")
        
        # Proxy request to serving node with atomic credit handling
        async def generate_from_node_atomic():
            stream_started = False
            reservation_confirmed = False
            try:
                http_client_timeout = float(os.getenv("BACKEND_HTTP_CLIENT_TIMEOUT", "30.0"))
                async with httpx.AsyncClient(timeout=http_client_timeout) as client:
                    # Make streaming request to serving node
                    async with client.stream(
                        "POST",
                        f"{endpoint_url}/chat",
                        json={
                            "message": chat_request.message,
                            "wallet_address": wallet_address
                        },
                        headers={"Content-Type": "application/json"}
                    ) as response:
                        response.raise_for_status()
                        
                        # Stream response from serving node
                        async for chunk in response.aiter_text():
                            if chunk:
                                if not stream_started:
                                    stream_started = True
                                    # Confirm credit reservation after stream successfully starts
                                    try:
                                        confirmed = await self.database.confirm_credit_reservation(reservation_id)
                                        if confirmed:
                                            reservation_confirmed = True
                                            logger.info(f"Credit reservation confirmed for {wallet_address} (reservation: {reservation_id})")
                                        else:
                                            logger.error(f"Failed to confirm credit reservation {reservation_id}")
                                    except Exception as e:
                                        logger.error(f"Error confirming credit reservation for {wallet_address}: {e}")
                                
                                yield chunk
            except Exception as e:
                # If serving node fails, fall back to local inference
                logger.warning(f"Serving node request failed: {e}. Falling back to local inference.")
                if not stream_started:
                    # Retry with local inference (reservation still valid)
                    async for token in self._generate_local_inference_atomic(chat_request, "general", wallet_address, reservation_id):
                        yield token
                else:
                    raise
            finally:
                # If stream started but reservation wasn't confirmed, try to confirm now
                if stream_started and not reservation_confirmed:
                    try:
                        await self.database.confirm_credit_reservation(reservation_id)
                        logger.info(f"Credit reservation confirmed in finally block for {wallet_address}")
                    except Exception as e:
                        logger.error(f"Failed to confirm reservation in finally block: {e}")
        
        return StreamingResponse(generate_from_node_atomic(), media_type="text/plain")
    
    async def _local_inference(self, chat_request, adapter_name, wallet_address, reservation_id):
        """Perform local inference."""
        return StreamingResponse(
            self._generate_local_inference_atomic(chat_request, adapter_name, wallet_address, reservation_id),
            media_type="text/plain"
        )
    
    async def _generate_local_inference_atomic(self, chat_request, adapter_name, wallet_address, reservation_id):
        """Generate local inference with atomic credit handling."""
        import os
        stream_started = False
        reservation_confirmed = False
        try:
            # Get inference executor
            from ..inference_executor import get_inference_executor
            inference_executor = await get_inference_executor(max_workers=int(os.getenv("MAX_WORKERS", "1")))
            
            # Get model manager
            from ..model_manager import AIModelManager
            from ..config_manager import get_config_manager
            config = get_config_manager().load()
            model_manager = AIModelManager(base_model_path=config.base_model_path)
            
            if model_manager is None:
                raise RuntimeError("Model manager not available. Set R3MES_INFERENCE_MODE=local and ensure GPU is available.")
            
            # Use inference executor to run in separate thread/process
            async for token in inference_executor.run_inference_streaming(
                chat_request.message,
                adapter_name,
                model_manager
            ):
                # First token received - stream has successfully started
                if not stream_started:
                    stream_started = True
                    # Confirm credit reservation after stream successfully starts
                    try:
                        confirmed = await self.database.confirm_credit_reservation(reservation_id)
                        if confirmed:
                            reservation_confirmed = True
                            logger.info(f"Credit reservation confirmed for {wallet_address} (reservation: {reservation_id})")
                        else:
                            logger.error(f"Failed to confirm credit reservation {reservation_id}")
                    except Exception as e:
                        logger.error(f"Error confirming credit reservation for {wallet_address}: {e}")
                
                yield token
                
        except Exception as e:
            # If stream fails before starting, rollback reservation
            if not stream_started and not reservation_confirmed:
                logger.warning(f"Stream failed before starting for {wallet_address}: {e}. Rolling back reservation.")
                await self.database.rollback_credit_reservation(reservation_id)
            # If stream fails after starting but reservation was confirmed, log it
            elif reservation_confirmed:
                logger.warning(f"Stream failed after starting for {wallet_address}: {e}. Credit already deducted.")
            raise
        finally:
            # If stream started but reservation wasn't confirmed, try to confirm now
            if stream_started and not reservation_confirmed:
                try:
                    await self.database.confirm_credit_reservation(reservation_id)
                    logger.info(f"Credit reservation confirmed in finally block for {wallet_address}")
                except Exception as e:
                    logger.error(f"Failed to confirm reservation in finally block: {e}")


# Global service instance (will be initialized in main.py)
_chat_service: Optional[ChatService] = None

def get_chat_service() -> ChatService:
    """Get the global chat service instance."""
    global _chat_service
    if _chat_service is None:
        raise RuntimeError("Chat service not initialized")
    return _chat_service

def init_chat_service(database: AsyncDatabase, serving_node_registry: ServingNodeRegistry):
    """Initialize the global chat service instance."""
    global _chat_service
    _chat_service = ChatService(database, serving_node_registry)


@router.post("/")
async def chat(request: Request, chat_request: ChatRequest):
    """
    Chat endpoint - AI inference with credit system.
    
    Behavior depends on R3MES_INFERENCE_MODE:
    - disabled: Returns 503 error
    - mock: Returns mock responses
    - remote: Proxies to Serving Nodes
    - local: Runs inference locally (requires GPU)
    """
    from ..auth import get_wallet_from_auth
    
    # Get wallet address from API key or request body
    wallet_from_auth = await get_wallet_from_auth(request)
    wallet_address = wallet_from_auth or chat_request.wallet_address
    
    if not wallet_address:
        raise HTTPException(
            status_code=401,
            detail="Either provide wallet_address in request body or valid API key in X-API-Key header"
        ) from MissingCredentialsError("Either provide wallet_address in request body or valid API key in X-API-Key header")
    
    chat_service = get_chat_service()
    return await chat_service.process_chat_request(chat_request, wallet_address)
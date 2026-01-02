"""
User management API endpoints
"""

from typing import List, Dict, Optional
from fastapi import APIRouter, HTTPException, Request, Path as FastAPIPath, Field
from pydantic import BaseModel, field_validator
from datetime import datetime, timedelta
import secrets

from slowapi import Limiter
from slowapi.util import get_remote_address

from ..database_async import AsyncDatabase
from ..exceptions import InvalidWalletAddressError

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

router = APIRouter(prefix="/user", tags=["users"])

# Response models
class UserInfoResponse(BaseModel):
    wallet_address: str
    credits: float
    is_miner: bool

class NetworkStatsResponse(BaseModel):
    active_miners: int
    total_users: int
    total_credits: float
    block_height: Optional[int] = None

class BlockResponse(BaseModel):
    height: int
    miner: Optional[str] = None
    timestamp: Optional[str] = None
    hash: Optional[str] = None

class BlocksResponse(BaseModel):
    blocks: List[BlockResponse]
    limit: int
    total: int

class MinerStatsResponse(BaseModel):
    wallet_address: str
    total_earnings: float
    hashrate: float
    gpu_temperature: float
    blocks_found: int
    uptime_percentage: float
    network_difficulty: float

class EarningsHistoryResponse(BaseModel):
    earnings: List[Dict]

class HashrateHistoryResponse(BaseModel):
    hashrate: List[Dict]

# API Key Management Models
class CreateAPIKeyRequest(BaseModel):
    wallet_address: str = Field(..., description="Wallet address for API key")
    name: Optional[str] = Field(None, max_length=100, description="API key name (optional)")
    expires_days: Optional[int] = Field(None, ge=1, le=365, description="Expiration in days (1-365, optional)")
    
    @field_validator("wallet_address")
    @classmethod
    def validate_wallet_address(cls, v: str) -> str:
        """Validate Cosmos wallet address format."""
        v = v.strip()
        if not v:
            raise ValueError("Wallet address cannot be empty")
        if not v.startswith("remes"):
            raise ValueError("Invalid address format: must start with 'remes'")
        if len(v) < 20 or len(v) > 60:
            raise ValueError("Invalid address length (must be 20-60 characters)")
        return v
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: Optional[str]) -> Optional[str]:
        """Validate API key name."""
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        if len(v) > 100:
            raise ValueError("API key name too long (max 100 characters)")
        # Prevent XSS in name
        if any(char in v for char in ['<', '>', '"', "'", '&']):
            raise ValueError("API key name contains invalid characters")
        return v

class RevokeAPIKeyRequest(BaseModel):
    api_key_id: int = Field(..., ge=1, description="API key ID to revoke")
    wallet_address: str = Field(..., description="Wallet address that owns the API key")
    
    @field_validator("wallet_address")
    @classmethod
    def validate_wallet_address(cls, v: str) -> str:
        """Validate Cosmos wallet address format."""
        v = v.strip()
        if not v:
            raise ValueError("Wallet address cannot be empty")
        if not v.startswith("remes"):
            raise ValueError("Invalid address format: must start with 'remes'")
        if len(v) < 20 or len(v) > 60:
            raise ValueError("Invalid address length (must be 20-60 characters)")
        return v


class UserService:
    """Service class for user operations."""
    
    def __init__(self, database: AsyncDatabase):
        self.database = database


# Global service instance
_user_service: Optional['UserService'] = None

def get_user_service() -> 'UserService':
    """Get the global user service instance."""
    global _user_service
    if _user_service is None:
        raise RuntimeError("User service not initialized")
    return _user_service

def init_user_service(database: AsyncDatabase):
    """Initialize the global user service instance."""
    global _user_service
    _user_service = UserService(database)


@router.get("/info/{wallet_address}")
async def get_user_info(
    request: Request,
    wallet_address: str = FastAPIPath(..., description="Wallet address", min_length=20, max_length=60, pattern="^remes[a-z0-9]+$")
) -> UserInfoResponse:
    """
    Kullanıcı bilgilerini döndürür.
    
    Cüzdanın kalan kredisini ve madenci olup olmadığını (VIP durumu) JSON döndür.
    """
    user_service = get_user_service()
    user_info = await user_service.database.get_user_info(wallet_address)
    
    if not user_info:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserInfoResponse(
        wallet_address=user_info['wallet_address'],
        credits=user_info['credits'],
        is_miner=user_info['is_miner']
    )

@router.get("/network/stats")
async def get_network_stats(request: Request) -> NetworkStatsResponse:
    """
    Ağ istatistiklerini döndürür.
    
    Aktif madenci sayısı, toplam blok sayısı gibi genel istatistikleri döndür.
    """
    user_service = get_user_service()
    stats = await user_service.database.get_network_stats()
    
    return NetworkStatsResponse(
        active_miners=stats['active_miners'],
        total_users=stats['total_users'],
        total_credits=stats['total_credits'],
        block_height=stats.get('block_height')
    )

@router.get("/blocks")
async def get_blocks(
    request: Request,
    limit: int = Field(10, ge=1, le=100, description="Number of blocks to return (1-100)")
) -> BlocksResponse:
    """
    Son blokları döndürür.
    
    Args:
        limit: Döndürülecek blok sayısı (default: 10, max: 100)
    
    Returns:
        Blok listesi
    """
    user_service = get_user_service()
    blocks_data = await user_service.database.get_recent_blocks(limit=limit)
    
    blocks = [
        BlockResponse(
            height=block['height'],
            miner=block.get('miner'),
            timestamp=block.get('timestamp'),
            hash=block.get('hash')
        )
        for block in blocks_data
    ]
    
    return BlocksResponse(
        blocks=blocks,
        limit=limit,
        total=len(blocks)
    )

@router.get("/miner/stats/{wallet_address}")
async def get_miner_stats(
    request: Request,
    wallet_address: str = FastAPIPath(..., description="Miner wallet address", min_length=20, max_length=60, pattern="^remes[a-z0-9]+$")
) -> MinerStatsResponse:
    """
    Miner istatistiklerini döndürür.
    
    Args:
        wallet_address: Miner cüzdan adresi
    
    Returns:
        Miner istatistikleri
    """
    user_service = get_user_service()
    stats = await user_service.database.get_miner_stats(wallet_address)
    
    return MinerStatsResponse(
        wallet_address=stats['wallet_address'],
        total_earnings=stats['total_earnings'],
        hashrate=stats['hashrate'],
        gpu_temperature=stats['gpu_temperature'],
        blocks_found=stats['blocks_found'],
        uptime_percentage=stats['uptime_percentage'],
        network_difficulty=stats['network_difficulty']
    )

@router.get("/miner/earnings/{wallet_address}")
async def get_miner_earnings(
    request: Request,
    wallet_address: str = FastAPIPath(..., description="Miner wallet address", min_length=20, max_length=60, pattern="^remes[a-z0-9]+$"),
    days: int = Field(7, ge=1, le=365, description="Number of days to retrieve (1-365)")
) -> EarningsHistoryResponse:
    """
    Miner earnings geçmişini döndürür.
    
    Args:
        wallet_address: Miner cüzdan adresi
        days: Kaç günlük geçmiş (default: 7)
    
    Returns:
        Earnings geçmişi
    """
    user_service = get_user_service()
    earnings = await user_service.database.get_earnings_history(wallet_address, days=days)
    
    return EarningsHistoryResponse(earnings=earnings)

@router.get("/miner/hashrate/{wallet_address}")
async def get_miner_hashrate(
    request: Request,
    wallet_address: str = FastAPIPath(..., description="Miner wallet address", min_length=20, max_length=60, pattern="^remes[a-z0-9]+$"),
    days: int = Field(7, ge=1, le=365, description="Number of days to retrieve (1-365)")
) -> HashrateHistoryResponse:
    """
    Miner hashrate geçmişini döndürür.
    
    Args:
        wallet_address: Miner cüzdan adresi
        days: Kaç günlük geçmiş (default: 7)
    
    Returns:
        Hashrate geçmişi
    """
    user_service = get_user_service()
    hashrate = await user_service.database.get_hashrate_history(wallet_address, days=days)
    
    return HashrateHistoryResponse(hashrate=hashrate)

# ========== API Key Management Endpoints ==========

@router.post("/api-keys/create")
@limiter.limit("5/minute")  # 5 API key creation requests per minute
async def create_api_key(request: Request, key_request: CreateAPIKeyRequest):
    """
    Yeni bir API key oluşturur.
    
    Args:
        key_request: API key oluşturma isteği
        
    Returns:
        API key bilgileri
    """
    user_service = get_user_service()
    api_key = await user_service.database.create_api_key(
        wallet_address=key_request.wallet_address,
        name=key_request.name or "Default",
        expires_in_days=key_request.expires_days
    )
    
    # Format response (api_key is plaintext, but we don't store it)
    api_key_data = {
        "api_key": api_key,
        "name": key_request.name or "Default",
        "created_at": datetime.now().isoformat(),
        "expires_at": None
    }
    
    return {
        "api_key": api_key_data["api_key"],
        "name": api_key_data["name"],
        "created_at": api_key_data["created_at"],
        "expires_at": api_key_data["expires_at"],
        "message": "⚠️  Save this API key securely. It will not be shown again."
    }

@router.get("/api-keys/list/{wallet_address}")
async def list_api_keys(request: Request, wallet_address: str):
    """
    Bir cüzdan için tüm API key'leri listeler.
    
    Args:
        wallet_address: Cüzdan adresi
        
    Returns:
        API key listesi
    """
    user_service = get_user_service()
    keys = await user_service.database.list_api_keys(wallet_address)
    return {"wallet_address": wallet_address, "api_keys": keys}

@router.post("/api-keys/revoke")
@limiter.limit("10/minute")  # 10 revoke requests per minute
async def revoke_api_key(request: Request, revoke_request: RevokeAPIKeyRequest):
    """
    Bir API key'i iptal eder.
    
    Args:
        revoke_request: API key iptal isteği
        
    Returns:
        İşlem sonucu
    """
    user_service = get_user_service()
    success = await user_service.database.revoke_api_key(
        api_key_id=revoke_request.api_key_id,
        wallet_address=revoke_request.wallet_address
    )
    
    if success:
        return {"message": "API key revoked successfully"}
    else:
        raise HTTPException(
            status_code=404,
            detail="API key not found or you don't have permission to revoke it"
        )

@router.delete("/api-keys/delete")
async def delete_api_key(request: Request, revoke_request: RevokeAPIKeyRequest):
    """
    Bir API key'i tamamen siler.
    
    Args:
        revoke_request: API key silme isteği
        
    Returns:
        İşlem sonucu
    """
    user_service = get_user_service()
    success = await user_service.database.delete_api_key(
        api_key_id=revoke_request.api_key_id,
        wallet_address=revoke_request.wallet_address
    )
    
    if success:
        return {"message": "API key deleted successfully"}
    else:
        raise HTTPException(
            status_code=404,
            detail="API key not found or you don't have permission to delete it"
        )
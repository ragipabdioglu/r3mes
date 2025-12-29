"""
Miner Endpoints for R3MES Dashboard

Provides REST API endpoints for miner information including locations and tiers.
"""

import logging
import hashlib
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from .blockchain_query_client import get_blockchain_client
from .cache import cache_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/miners", tags=["miners"])


class MinerLocation(BaseModel):
    """Miner location data for globe visualization."""
    address: str
    moniker: str
    latitude: float
    longitude: float
    status: str
    reputation_score: float
    country: str
    city: str


class MinerLocationsResponse(BaseModel):
    """Response containing miner locations."""
    locations: List[MinerLocation]
    total: int


class MinerTier(BaseModel):
    """Miner tier information."""
    tier: str
    name: str
    min_score: int
    reward_multiplier: float
    color: str


class MinerWithTier(BaseModel):
    """Miner data with tier information."""
    address: str
    moniker: str
    reputation_score: float
    tier: MinerTier
    total_staked: str
    completed_tasks: int
    gpu_count: int
    status: str


# Tier configuration
TIER_CONFIG = {
    "diamond": {"name": "Diamond", "min_score": 95, "reward_multiplier": 2.0, "color": "#b9f2ff"},
    "platinum": {"name": "Platinum", "min_score": 85, "reward_multiplier": 1.5, "color": "#e5e4e2"},
    "gold": {"name": "Gold", "min_score": 70, "reward_multiplier": 1.25, "color": "#ffd700"},
    "silver": {"name": "Silver", "min_score": 50, "reward_multiplier": 1.1, "color": "#c0c0c0"},
    "bronze": {"name": "Bronze", "min_score": 0, "reward_multiplier": 1.0, "color": "#cd7f32"},
}

# Known data center locations for IP-based geolocation fallback
DATACENTER_LOCATIONS = [
    {"city": "New York", "country": "USA", "lat": 40.7128, "lon": -74.0060},
    {"city": "London", "country": "UK", "lat": 51.5074, "lon": -0.1278},
    {"city": "Tokyo", "country": "Japan", "lat": 35.6762, "lon": 139.6503},
    {"city": "Singapore", "country": "Singapore", "lat": 1.3521, "lon": 103.8198},
    {"city": "Frankfurt", "country": "Germany", "lat": 50.1109, "lon": 8.6821},
    {"city": "Sydney", "country": "Australia", "lat": -33.8688, "lon": 151.2093},
    {"city": "São Paulo", "country": "Brazil", "lat": -23.5505, "lon": -46.6333},
    {"city": "Dubai", "country": "UAE", "lat": 25.2048, "lon": 55.2708},
    {"city": "Seoul", "country": "South Korea", "lat": 37.5665, "lon": 126.9780},
    {"city": "Mumbai", "country": "India", "lat": 19.0760, "lon": 72.8777},
    {"city": "Toronto", "country": "Canada", "lat": 43.6532, "lon": -79.3832},
    {"city": "Amsterdam", "country": "Netherlands", "lat": 52.3676, "lon": 4.9041},
    {"city": "Paris", "country": "France", "lat": 48.8566, "lon": 2.3522},
    {"city": "Hong Kong", "country": "Hong Kong", "lat": 22.3193, "lon": 114.1694},
    {"city": "Los Angeles", "country": "USA", "lat": 34.0522, "lon": -118.2437},
]


def get_tier_from_score(score: float) -> MinerTier:
    """Determine miner tier based on reputation score."""
    if score >= 95:
        tier_key = "diamond"
    elif score >= 85:
        tier_key = "platinum"
    elif score >= 70:
        tier_key = "gold"
    elif score >= 50:
        tier_key = "silver"
    else:
        tier_key = "bronze"
    
    config = TIER_CONFIG[tier_key]
    return MinerTier(
        tier=tier_key,
        name=config["name"],
        min_score=config["min_score"],
        reward_multiplier=config["reward_multiplier"],
        color=config["color"],
    )


def get_location_from_address(address: str) -> Dict[str, Any]:
    """
    Generate a deterministic location based on miner address.
    In production, this would use IP geolocation or miner-reported location.
    """
    # Use address hash to deterministically select a location
    hash_val = int(hashlib.sha256(address.encode()).hexdigest(), 16)
    location_index = hash_val % len(DATACENTER_LOCATIONS)
    
    # Add small random offset for visual variety
    offset_lat = ((hash_val % 1000) - 500) / 5000  # ±0.1 degrees
    offset_lon = ((hash_val % 2000) - 1000) / 5000  # ±0.2 degrees
    
    loc = DATACENTER_LOCATIONS[location_index]
    return {
        "city": loc["city"],
        "country": loc["country"],
        "latitude": loc["lat"] + offset_lat,
        "longitude": loc["lon"] + offset_lon,
    }


@router.get("/locations", response_model=MinerLocationsResponse)
@cache_response(ttl=60, key_prefix="miner_locations")
async def get_miner_locations(
    limit: int = Query(default=100, ge=1, le=500, description="Maximum miners to return"),
    active_only: bool = Query(default=False, description="Only return active miners"),
) -> MinerLocationsResponse:
    """
    Get miner locations for globe visualization.
    
    Returns miner coordinates for displaying on the network globe.
    Location is determined by:
    1. Miner-reported location (if available)
    2. IP geolocation (if available)
    3. Deterministic fallback based on address hash
    """
    try:
        blockchain_client = get_blockchain_client()
        
        # Get all miners
        miners_data = blockchain_client.get_all_miners(limit=limit, offset=0)
        miners = miners_data.get("miners", [])
        
        locations: List[MinerLocation] = []
        
        for miner in miners:
            address = miner.get("address", "")
            status = miner.get("status", "inactive")
            
            # Skip inactive miners if filter is enabled
            if active_only and status != "active":
                continue
            
            # Get location (from miner data or generate from address)
            location_data = miner.get("location")
            if not location_data or not location_data.get("latitude"):
                location_data = get_location_from_address(address)
            
            locations.append(MinerLocation(
                address=address,
                moniker=miner.get("moniker", f"Miner-{address[:8]}"),
                latitude=location_data.get("latitude", 0),
                longitude=location_data.get("longitude", 0),
                status=status,
                reputation_score=miner.get("reputation_score", 50),
                country=location_data.get("country", "Unknown"),
                city=location_data.get("city", "Unknown"),
            ))
        
        return MinerLocationsResponse(
            locations=locations,
            total=len(locations),
        )
    
    except Exception as e:
        logger.error(f"Failed to get miner locations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch miner locations: {str(e)}"
        )


@router.get("/leaderboard")
@cache_response(ttl=30, key_prefix="miner_leaderboard")
async def get_miner_leaderboard(
    limit: int = Query(default=50, ge=1, le=200, description="Maximum miners to return"),
    tier: Optional[str] = Query(default=None, description="Filter by tier"),
) -> Dict[str, Any]:
    """
    Get miner leaderboard with tier information.
    
    Returns miners sorted by reputation score with tier badges.
    """
    try:
        blockchain_client = get_blockchain_client()
        
        # Get all miners
        miners_data = blockchain_client.get_all_miners(limit=limit * 2, offset=0)  # Get extra for filtering
        miners = miners_data.get("miners", [])
        
        # Sort by reputation score
        miners.sort(key=lambda m: m.get("reputation_score", 0), reverse=True)
        
        leaderboard = []
        for rank, miner in enumerate(miners, 1):
            score = miner.get("reputation_score", 0)
            miner_tier = get_tier_from_score(score)
            
            # Filter by tier if specified
            if tier and miner_tier.tier != tier.lower():
                continue
            
            leaderboard.append({
                "rank": rank,
                "address": miner.get("address", ""),
                "moniker": miner.get("moniker", "Unknown"),
                "reputation_score": score,
                "tier": miner_tier.dict(),
                "total_staked": miner.get("total_staked", "0"),
                "completed_tasks": miner.get("completed_tasks", 0),
                "active_tasks": miner.get("active_tasks", 0),
                "gpu_count": miner.get("gpu_count", 0),
                "status": miner.get("status", "inactive"),
            })
            
            if len(leaderboard) >= limit:
                break
        
        # Calculate tier distribution
        tier_distribution = {
            "diamond": 0,
            "platinum": 0,
            "gold": 0,
            "silver": 0,
            "bronze": 0,
        }
        for miner in miners:
            score = miner.get("reputation_score", 0)
            miner_tier = get_tier_from_score(score)
            tier_distribution[miner_tier.tier] += 1
        
        return {
            "leaderboard": leaderboard,
            "total": len(leaderboard),
            "tier_distribution": tier_distribution,
            "tier_config": TIER_CONFIG,
        }
    
    except Exception as e:
        logger.error(f"Failed to get miner leaderboard: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch miner leaderboard: {str(e)}"
        )


@router.get("/tiers")
async def get_tier_config() -> Dict[str, Any]:
    """
    Get tier configuration.
    
    Returns tier thresholds, reward multipliers, and colors.
    """
    return {
        "tiers": TIER_CONFIG,
        "description": {
            "diamond": "Elite miners with exceptional performance (95+ score)",
            "platinum": "Top-tier miners with excellent track record (85-94 score)",
            "gold": "High-performing miners with consistent results (70-84 score)",
            "silver": "Reliable miners with good performance (50-69 score)",
            "bronze": "Entry-level miners building reputation (0-49 score)",
        },
    }


@router.get("/{miner_address}/tier")
@cache_response(ttl=30, key_prefix="miner_tier")
async def get_miner_tier(miner_address: str) -> Dict[str, Any]:
    """
    Get tier information for a specific miner.
    """
    try:
        blockchain_client = get_blockchain_client()
        
        miner_info = blockchain_client.get_miner_info(miner_address)
        
        if not miner_info:
            raise HTTPException(
                status_code=404,
                detail=f"Miner not found: {miner_address}"
            )
        
        score = miner_info.get("reputation_score", 0)
        tier = get_tier_from_score(score)
        
        # Calculate progress to next tier
        next_tier = None
        progress_to_next = 100
        
        if tier.tier == "bronze":
            next_tier = "silver"
            progress_to_next = (score / 50) * 100
        elif tier.tier == "silver":
            next_tier = "gold"
            progress_to_next = ((score - 50) / 20) * 100
        elif tier.tier == "gold":
            next_tier = "platinum"
            progress_to_next = ((score - 70) / 15) * 100
        elif tier.tier == "platinum":
            next_tier = "diamond"
            progress_to_next = ((score - 85) / 10) * 100
        
        return {
            "miner_address": miner_address,
            "reputation_score": score,
            "current_tier": tier.dict(),
            "next_tier": next_tier,
            "progress_to_next_tier": min(100, max(0, progress_to_next)),
            "reward_multiplier": tier.reward_multiplier,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get tier for miner {miner_address}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch miner tier: {str(e)}"
        )

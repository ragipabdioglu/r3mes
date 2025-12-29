"""
Leaderboard API Endpoints

Provides leaderboard data for miners and validators.
"""

from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from typing import List, Dict, Optional
import logging

from .database_async import AsyncDatabase
from .config_manager import get_config_manager
from .blockchain_query_client import get_blockchain_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/leaderboard", tags=["leaderboard"])

# Rate limiter (will use app.state.limiter from main.py)
limiter = Limiter(key_func=get_remote_address)

config_manager = get_config_manager()
config = config_manager.load()
database = AsyncDatabase(db_path=config.database_path, chain_json_path=config.chain_json_path)


def calculate_tier(reputation: float) -> str:
    """Calculate reputation tier based on reputation score."""
    if reputation >= 1000:
        return "diamond"
    elif reputation >= 500:
        return "platinum"
    elif reputation >= 200:
        return "gold"
    elif reputation >= 50:
        return "silver"
    else:
        return "bronze"


@router.get("/miners")
@limiter.limit(config.rate_limit_get)
async def get_top_miners(request: Request, limit: int = 100):
    """
    Get top miners by reputation.
    
    Args:
        limit: Maximum number of miners to return (default: 100)
        
    Returns:
        List of top miners with their stats
    """
    try:
        blockchain_client = get_blockchain_client()
        
        # Query all miners from blockchain
        result = blockchain_client.get_all_miners(limit=limit * 2, offset=0)  # Get more to sort
        
        if not result or not result.get("miners"):
            # Fallback to empty list if query fails
            logger.warning("No miners found from blockchain query")
            return {
                "miners": [],
                "total": 0,
            }
        
        miners = result["miners"]
        
        # Sort by reputation (descending)
        miners.sort(key=lambda x: x.get("reputation", 0.0), reverse=True)
        
        # Calculate trend (simplified: based on recent submissions)
        # In production, this would compare with historical data
        for miner in miners:
            # Trend calculation: positive if recent submissions, negative if slashing events
            trend = 0
            if miner.get("successful_submissions", 0) > 0:
                trend = min(20, miner.get("successful_submissions", 0) // 100)
            if miner.get("slashing_events", 0) > 0:
                trend = -min(20, miner.get("slashing_events", 0) * 5)
            miner["trend"] = trend
            
            # Ensure tier is calculated from reputation
            reputation = miner.get("reputation", 0.0)
            miner["tier"] = calculate_tier(reputation)
        
        # Limit results
        miners = miners[:limit]
        
        return {
            "miners": miners,
            "total": result.get("total", len(miners)),
        }
    except Exception as e:
        logger.error(f"Error getting top miners: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch miner leaderboard")


@router.get("/validators")
@limiter.limit(config.rate_limit_get)
async def get_top_validators(request: Request, limit: int = 100):
    """
    Get top validators by trust score.
    
    Args:
        limit: Maximum number of validators to return (default: 100)
        
    Returns:
        List of top validators with their stats
    """
    try:
        blockchain_client = get_blockchain_client()
        
        # Query validators from blockchain
        result = blockchain_client.get_all_validators(limit=limit, offset=0)
        
        if not result or not result.get("validators"):
            return {
                "validators": [],
                "total": 0,
            }
        
        validators = result["validators"]
        
        # Sort by trust score (descending), then by voting power
        validators.sort(key=lambda x: (x.get("trust_score", 0.0), x.get("voting_power", 0.0)), reverse=True)
        
        # Calculate tier from trust score
        for validator in validators:
            trust_score = validator.get("trust_score", 0.0)
            reputation = trust_score * 1000  # Convert to reputation scale
            validator["tier"] = calculate_tier(reputation)
        
        # Limit results
        validators = validators[:limit]
        
        return {
            "validators": validators,
            "total": result.get("total", len(validators)),
        }
    except Exception as e:
        logger.error(f"Error getting top validators: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch validator leaderboard")


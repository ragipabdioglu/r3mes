"""
Analytics API Endpoints

Provides analytics data for the dashboard.
"""

from fastapi import APIRouter, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
import logging

from .database_async import AsyncDatabase
from .analytics import get_analytics_engine
from .config_manager import get_config_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["analytics"])

# Rate limiter (will use app.state.limiter from main.py)
limiter = Limiter(key_func=get_remote_address)

config_manager = get_config_manager()
config = config_manager.load()
database = AsyncDatabase(db_path=config.database_path, chain_json_path=config.chain_json_path)
analytics_engine = get_analytics_engine(database)


@router.get("")
@limiter.limit(config.rate_limit_get)
async def get_analytics(request: Request, days: int = 7):
    """
    Get analytics data.
    
    Args:
        days: Number of days to analyze (default: 7)
        
    Returns:
        Analytics data including API usage, user engagement, model performance, and network health
    """
    try:
        api_usage = await analytics_engine.get_api_usage_stats(days)
        user_engagement = await analytics_engine.get_user_engagement_stats(days)
        model_performance = await analytics_engine.get_model_performance_stats(days)
        network_health = await analytics_engine.get_network_health_trends(days)
        
        # Format endpoints data for charts
        endpoints_data = [
            {"endpoint": endpoint, "count": count}
            for endpoint, count in api_usage.get("endpoints", {}).items()
        ]
        api_usage["endpoints_data"] = endpoints_data
        
        return {
            "api_usage": api_usage,
            "user_engagement": user_engagement,
            "model_performance": model_performance,
            "network_health": network_health,
        }
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise


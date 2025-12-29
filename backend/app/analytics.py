"""
Analytics Engine

Tracks user engagement, API usage patterns, model performance, and network health trends.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
from collections import defaultdict

from .database_async import AsyncDatabase
from .cache import get_cache_manager

logger = logging.getLogger(__name__)


class AnalyticsEngine:
    """Analytics engine for tracking various metrics."""
    
    def __init__(self, database: AsyncDatabase):
        self.database = database
        self.cache = get_cache_manager()
        
        # In-memory counters (would be persisted in production)
        self.api_usage: Dict[str, int] = defaultdict(int)
        self.user_engagement: Dict[str, Dict] = defaultdict(dict)
        self.model_performance: List[Dict] = []
    
    async def track_api_usage(self, endpoint: str, method: str, user_id: Optional[str] = None):
        """Track API endpoint usage."""
        key = f"{method}:{endpoint}"
        self.api_usage[key] += 1
        
        if user_id:
            if "api_calls" not in self.user_engagement[user_id]:
                self.user_engagement[user_id]["api_calls"] = 0
            self.user_engagement[user_id]["api_calls"] += 1
    
    async def track_user_engagement(self, user_id: str, action: str, metadata: Optional[Dict] = None):
        """Track user engagement actions."""
        if user_id not in self.user_engagement:
            self.user_engagement[user_id] = {
                "first_seen": datetime.now().isoformat(),
                "actions": [],
            }
        
        self.user_engagement[user_id]["actions"].append({
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        })
    
    async def track_model_performance(
        self,
        adapter: str,
        latency: float,
        tokens_per_second: Optional[float] = None,
        success: bool = True
    ):
        """Track model inference performance."""
        self.model_performance.append({
            "adapter": adapter,
            "latency": latency,
            "tokens_per_second": tokens_per_second,
            "success": success,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Keep only last 1000 entries
        if len(self.model_performance) > 1000:
            self.model_performance = self.model_performance[-1000:]
    
    async def get_api_usage_stats(self, days: int = 7) -> Dict:
        """Get API usage statistics."""
        # In production, query from database
        return {
            "total_requests": sum(self.api_usage.values()),
            "endpoints": dict(self.api_usage),
            "period_days": days,
        }
    
    async def get_user_engagement_stats(self, days: int = 7) -> Dict:
        """Get user engagement statistics."""
        active_users = len([
            uid for uid, data in self.user_engagement.items()
            if "actions" in data and len(data["actions"]) > 0
        ])
        
        total_actions = sum(
            len(data.get("actions", []))
            for data in self.user_engagement.values()
        )
        
        return {
            "active_users": active_users,
            "total_actions": total_actions,
            "average_actions_per_user": total_actions / active_users if active_users > 0 else 0,
            "period_days": days,
        }
    
    async def get_model_performance_stats(self, days: int = 7) -> Dict:
        """Get model performance statistics."""
        if not self.model_performance:
            return {
                "average_latency": 0.0,
                "average_tokens_per_second": 0.0,
                "success_rate": 0.0,
            }
        
        recent_performance = [
            p for p in self.model_performance
            if datetime.fromisoformat(p["timestamp"]) > datetime.now() - timedelta(days=days)
        ]
        
        if not recent_performance:
            return {
                "average_latency": 0.0,
                "average_tokens_per_second": 0.0,
                "success_rate": 0.0,
            }
        
        avg_latency = sum(p["latency"] for p in recent_performance) / len(recent_performance)
        avg_tps = sum(
            p.get("tokens_per_second", 0) or 0
            for p in recent_performance
        ) / len(recent_performance)
        success_rate = sum(1 for p in recent_performance if p["success"]) / len(recent_performance)
        
        return {
            "average_latency": avg_latency,
            "average_tokens_per_second": avg_tps,
            "success_rate": success_rate,
            "total_inferences": len(recent_performance),
        }
    
    async def get_network_health_trends(self, days: int = 7) -> Dict:
        """Get network health trends."""
        from .blockchain_query_client import get_blockchain_client
        from .blockchain_rpc_client import get_blockchain_rpc_client
        
        try:
            blockchain_client = get_blockchain_client()
            rpc_client = get_blockchain_rpc_client()
            
            # Get current network stats
            network_stats = await self.database.get_network_stats()
            blockchain_stats = blockchain_client.get_network_statistics()
            
            # Get current block height
            current_height = rpc_client.get_latest_block_height() or 0
            
            # Calculate trends (simplified: single data point for now)
            # In production, this would aggregate historical data
            active_miners = network_stats.get("active_miners", 0)
            total_gradients = blockchain_stats.get("total_gradients", 0) if blockchain_stats else 0
            
            # Estimate FLOPs from gradients (simplified)
            # Each gradient submission represents computational work
            estimated_flops = total_gradients * 1e12  # Estimate: 1 TFLOP per gradient
            
            # Get actual block time from blockchain RPC
            try:
                # Get recent blocks to calculate average block time
                recent_blocks = rpc_client.get_recent_blocks(limit=100)
                if len(recent_blocks) >= 2:
                    # Calculate average block time from recent blocks
                    block_times = []
                    for i in range(1, len(recent_blocks)):
                        prev_time = recent_blocks[i-1].get("time")
                        curr_time = recent_blocks[i].get("time")
                        if prev_time and curr_time:
                            from datetime import datetime
                            try:
                                if isinstance(prev_time, str):
                                    prev_dt = datetime.fromisoformat(prev_time.replace('Z', '+00:00'))
                                else:
                                    prev_dt = prev_time
                                if isinstance(curr_time, str):
                                    curr_dt = datetime.fromisoformat(curr_time.replace('Z', '+00:00'))
                                else:
                                    curr_dt = curr_time
                                time_diff = (curr_dt - prev_dt).total_seconds()
                                if time_diff > 0:
                                    block_times.append(time_diff)
                            except Exception:
                                pass
                    
                    if block_times:
                        avg_block_time = sum(block_times) / len(block_times)
                    else:
                        avg_block_time = 5.0  # Default fallback
                else:
                    avg_block_time = 5.0  # Default fallback
            except Exception as e:
                logger.warning(f"Failed to calculate block time from blockchain: {e}, using default")
                avg_block_time = 5.0  # Default fallback
            
            # Build trend data (single point for now, would be time-series in production)
            active_miners_trend = [
                {
                    "date": datetime.now().isoformat(),
                    "count": active_miners
                }
            ]
            
            total_flops_trend = [
                {
                    "date": datetime.now().isoformat(),
                    "flops": estimated_flops
                }
            ]
            
            block_time_trend = [
                {
                    "date": datetime.now().isoformat(),
                    "block_time": avg_block_time
                }
            ]
            
            return {
                "active_miners_trend": active_miners_trend,
                "total_flops_trend": total_flops_trend,
                "block_time_trend": block_time_trend,
                "period_days": days,
            }
        except Exception as e:
            logger.error(f"Error getting network health trends: {e}", exc_info=True)
            # Return empty trends on error
            return {
                "active_miners_trend": [],
                "total_flops_trend": [],
                "block_time_trend": [],
                "period_days": days,
            }


# Global analytics engine instance
_analytics_engine: Optional[AnalyticsEngine] = None


def get_analytics_engine(database: AsyncDatabase) -> AnalyticsEngine:
    """Get global analytics engine instance."""
    global _analytics_engine
    if _analytics_engine is None:
        _analytics_engine = AnalyticsEngine(database)
    return _analytics_engine


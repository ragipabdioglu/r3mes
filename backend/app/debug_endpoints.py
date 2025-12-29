"""
Debug Endpoints for R3MES Backend

Provides debug API endpoints for state inspection, cache statistics, and performance metrics.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional
import logging

from .debug_config import get_debug_config, validate_debug_config
from .performance_profiler import get_profiler
from .cache import get_cache_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/debug", tags=["debug"])


def require_debug_mode():
    """Dependency to require debug mode to be enabled"""
    debug_config = get_debug_config()
    if not debug_config.enabled or not debug_config.is_backend_enabled():
        raise HTTPException(status_code=403, detail="Debug mode is not enabled")
    if not debug_config.state_inspection:
        raise HTTPException(status_code=403, detail="State inspection is not enabled in debug mode")
    
    # Check for production mode
    error = validate_debug_config(is_production=False)  # For now, allow in non-production
    if error and "SECURITY ERROR" in error:
        raise HTTPException(status_code=403, detail=error)
    
    return debug_config


@router.get("/state")
async def get_application_state(debug_config: dict = Depends(require_debug_mode)) -> Dict[str, Any]:
    """
    Get application state information.
    
    Returns:
        Dictionary containing application state information
    """
    try:
        state = {
            "debug_mode_enabled": True,
            "debug_level": debug_config.level.value if hasattr(debug_config.level, 'value') else str(debug_config.level),
            "components_enabled": list(debug_config.components) if hasattr(debug_config, 'components') else [],
            "features": {
                "logging": debug_config.logging,
                "profiling": debug_config.profiling,
                "state_inspection": debug_config.state_inspection,
                "trace": debug_config.trace,
            },
        }
        return state
    except Exception as e:
        logger.error(f"Failed to get application state: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get application state: {str(e)}")


@router.get("/cache")
async def get_cache_stats(debug_config: dict = Depends(require_debug_mode)) -> Dict[str, Any]:
    """
    Get cache statistics.
    
    Returns:
        Dictionary containing cache statistics
    """
    try:
        cache_manager = get_cache_manager()
        
        stats = {
            "cache_enabled": cache_manager is not None,
        }
        
        if cache_manager:
            # Get cache stats if available
            if hasattr(cache_manager, 'get_stats'):
                stats.update(cache_manager.get_stats())
            else:
                stats["info"] = "Cache manager does not provide statistics"
        
        return stats
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")


@router.get("/performance")
async def get_performance_metrics(debug_config: dict = Depends(require_debug_mode)) -> Dict[str, Any]:
    """
    Get performance metrics.
    
    Returns:
        Dictionary containing performance metrics
    """
    try:
        profiler = get_profiler()
        
        if not profiler.enabled:
            return {
                "profiling_enabled": False,
                "message": "Profiling is not enabled"
            }
        
        stats = profiler.get_stats()
        
        # Convert to dict for JSON serialization
        result = {
            "profiling_enabled": True,
            "start_time": stats.start_time.isoformat() if hasattr(stats.start_time, 'isoformat') else str(stats.start_time),
            "end_time": stats.end_time.isoformat() if hasattr(stats.end_time, 'isoformat') else str(stats.end_time),
            "duration": stats.duration,
            "profiles": {k: {
                "function": v.function,
                "call_count": v.call_count,
                "total_duration": v.total_duration,
                "min_duration": v.min_duration,
                "max_duration": v.max_duration,
                "avg_duration": v.avg_duration,
                "last_call_time": v.last_call_time.isoformat() if hasattr(v.last_call_time, 'isoformat') else str(v.last_call_time) if v.last_call_time else None,
            } for k, v in stats.profiles.items()},
            "memory_stats": stats.memory_stats,
            "cpu_percent": stats.cpu_percent,
            "thread_count": stats.thread_count,
        }
        
        return result
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")


@router.post("/performance/reset")
async def reset_performance_stats(debug_config: dict = Depends(require_debug_mode)) -> Dict[str, str]:
    """
    Reset performance statistics.
    
    Returns:
        Confirmation message
    """
    try:
        profiler = get_profiler()
        profiler.reset()
        return {"message": "Performance statistics reset successfully"}
    except Exception as e:
        logger.error(f"Failed to reset performance stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reset performance stats: {str(e)}")


@router.post("/performance/export")
async def export_performance_stats(
    filename: Optional[str] = None,
    debug_config: dict = Depends(require_debug_mode)
) -> Dict[str, str]:
    """
    Export performance statistics to a file.
    
    Args:
        filename: Optional filename for export (default: auto-generated)
    
    Returns:
        Path to exported file
    """
    try:
        profiler = get_profiler()
        exported_path = profiler.export_stats(filename)
        return {
            "message": "Performance statistics exported successfully",
            "file_path": exported_path
        }
    except Exception as e:
        logger.error(f"Failed to export performance stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to export performance stats: {str(e)}")


@router.get("/trace/{trace_id}")
async def get_trace_info(trace_id: str, debug_config: dict = Depends(require_debug_mode)) -> Dict[str, Any]:
    """
    Get trace information for a specific trace ID.
    
    Args:
        trace_id: Trace ID to look up
    
    Returns:
        Dictionary containing trace information
    """
    try:
        # Trace system is planned for future implementation
        # This endpoint provides a placeholder for distributed tracing integration
        return {
            "trace_id": trace_id,
            "message": "Trace system not yet implemented",
            "status": "not_implemented"
        }
    except Exception as e:
        logger.error(f"Failed to get trace info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get trace info: {str(e)}")

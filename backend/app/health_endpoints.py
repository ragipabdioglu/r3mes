"""
Detailed Health Check Endpoints

Provides granular health checks for database, Redis, blockchain, secret management,
and connection pool statistics.
"""

import os
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any
import httpx

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/database")
async def health_database():
    """Check database connectivity and pool status."""
    try:
        from .main import database
        
        # Test connection
        if database.config.is_postgresql():
            # PostgreSQL: test connection and get pool stats
            if database._db and database._db.pool:
                async with database._db.pool.acquire() as conn:
                    await conn.execute("SELECT 1")
                
                # Get pool statistics
                pool = database._db.pool
                pool_stats = {
                    "size": pool.get_size(),
                    "idle_size": pool.get_idle_size(),
                    "min_size": database._db.min_size,
                    "max_size": database._db.max_size,
                    "used": pool.get_size() - pool.get_idle_size(),
                    "available": pool.get_idle_size()
                }
                
                return {
                    "status": "healthy",
                    "database": "postgresql",
                    "pool": pool_stats
                }
            else:
                return JSONResponse(
                    status_code=503,
                    content={"status": "unhealthy", "error": "Database pool not initialized"}
                )
        else:
            # SQLite: test connection
            if not database._connection:
                await database.connect()
            await database._connection.execute("SELECT 1")
            
            return {
                "status": "healthy",
                "database": "sqlite",
                "path": database.db_path
            }
    except Exception as e:
        logger.error(f"Database health check failed: {e}", exc_info=True)
        # Send notification for critical database health failure
        try:
            from .notifications import get_notification_service, NotificationPriority
            notification_service = get_notification_service()
            await notification_service.send_system_alert(
                component="database",
                alert_type="health_check_failure",
                message=f"Database health check failed: {e}",
                priority=NotificationPriority.CRITICAL
            )
        except Exception as notif_error:
            logger.warning(f"Failed to send database health check failure notification: {notif_error}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@router.get("/redis")
async def health_redis():
    """Check Redis connectivity."""
    try:
        from .cache import get_cache_manager
        
        cache = get_cache_manager()
        
        # Test connection
        if cache.redis:
            await cache.redis.ping()
            
            # Get Redis info
            info = await cache.redis.info("server")
            redis_version = info.get("redis_version", "unknown")
            
            return {
                "status": "healthy",
                "redis": "connected",
                "version": redis_version,
                "url": cache.redis_url
            }
        else:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "error": "Redis not connected"}
            )
    except Exception as e:
        logger.error(f"Redis health check failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@router.get("/blockchain")
async def health_blockchain():
    """Check blockchain connectivity."""
    try:
        rpc_url = os.getenv("BLOCKCHAIN_RPC_URL")
        if not rpc_url:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "error": "BLOCKCHAIN_RPC_URL not set"}
            )
        
        # Test RPC connection
        health_check_timeout = float(os.getenv("BACKEND_HEALTH_CHECK_TIMEOUT", "10.0"))
        async with httpx.AsyncClient(timeout=health_check_timeout) as client:
            response = await client.get(f"{rpc_url}/status")
            
            if response.status_code == 200:
                data = response.json()
                result = data.get("result", {})
                
                return {
                    "status": "healthy",
                    "blockchain": "connected",
                    "rpc_url": rpc_url,
                    "node_info": {
                        "network": result.get("node_info", {}).get("network", "unknown"),
                        "version": result.get("node_info", {}).get("version", "unknown")
                    }
                }
            else:
                return JSONResponse(
                    status_code=503,
                    content={"status": "unhealthy", "error": f"RPC returned status {response.status_code}"}
                )
    except Exception as e:
        logger.error(f"Blockchain health check failed: {e}", exc_info=True)
        # Send notification for critical blockchain health failure
        try:
            from .notifications import get_notification_service, NotificationPriority
            notification_service = get_notification_service()
            await notification_service.send_system_alert(
                component="blockchain",
                alert_type="health_check_failure",
                message=f"Blockchain health check failed: {e}",
                priority=NotificationPriority.CRITICAL
            )
        except Exception as notif_error:
            logger.warning(f"Failed to send blockchain health check failure notification: {notif_error}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@router.get("/secrets")
async def health_secrets():
    """Check secret management connectivity."""
    try:
        from .secrets import get_secret_manager
        
        sm = get_secret_manager()
        
        # Test connection
        if hasattr(sm, 'test_connection'):
            if sm.test_connection():
                return {
                    "status": "healthy",
                    "secret_manager": sm.__class__.__name__,
                    "type": type(sm).__name__
                }
            else:
                return JSONResponse(
                    status_code=503,
                    content={"status": "unhealthy", "error": "Secret manager connection test failed"}
                )
        else:
            # If no test_connection method, assume it's working (environment variables)
            return {
                "status": "healthy",
                "secret_manager": sm.__class__.__name__,
                "type": type(sm).__name__,
                "note": "Connection test not available for this provider"
            }
    except Exception as e:
        logger.error(f"Secret management health check failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@router.get("/indexer")
async def health_indexer():
    """Check indexer status and health."""
    try:
        from .indexer import get_indexer
        from .main import database
        
        indexer = get_indexer(database)
        status = indexer.get_status()
        
        # Get current blockchain height for comparison
        from .blockchain_rpc_client import get_blockchain_rpc_client
        rpc_client = get_blockchain_rpc_client()
        current_height = rpc_client.get_latest_block_height()
        
        if current_height is not None:
            status["current_blockchain_height"] = current_height
            status["lag_blocks"] = current_height - status.get("last_indexed_height", 0)
            status["lag_percentage"] = (
                (status["lag_blocks"] / current_height * 100) if current_height > 0 else 0
            )
        
        # Determine health status
        if not status["running"]:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "error": "Indexer is not running", **status}
            )
        
        # Check if lag is too high (more than 1000 blocks behind)
        if status.get("lag_blocks", 0) > 1000:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "degraded",
                    "error": f"Indexer lag is high: {status.get('lag_blocks', 0)} blocks behind",
                    **status
                }
            )
        
        return {
            "status": "healthy",
            **status
        }
    except Exception as e:
        logger.error(f"Indexer health check failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@router.get("/pool")
async def health_pool():
    """Get database connection pool statistics."""
    try:
        from .main import database
        
        if not database.config.is_postgresql():
            return JSONResponse(
                status_code=400,
                content={"error": "Connection pool statistics only available for PostgreSQL"}
            )
        
        if database._db and database._db.pool:
            pool = database._db.pool
            return {
                "size": pool.get_size(),
                "idle_size": pool.get_idle_size(),
                "min_size": database._db.min_size,
                "max_size": database._db.max_size,
                "used": pool.get_size() - pool.get_idle_size(),
                "available": pool.get_idle_size(),
                "utilization_percent": round((pool.get_size() - pool.get_idle_size()) / database._db.max_size * 100, 2) if database._db.max_size > 0 else 0
            }
        else:
            return JSONResponse(
                status_code=503,
                content={"error": "Database pool not initialized"}
            )
    except Exception as e:
        logger.error(f"Pool statistics check failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=503,
            content={"error": str(e)}
        )


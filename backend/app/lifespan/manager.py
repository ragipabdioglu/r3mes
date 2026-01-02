"""
Lifespan Manager for R3MES Backend

Manages the complete lifecycle of the FastAPI application.
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI

from .startup import initialize_startup_components
from .shutdown import shutdown_components

logger = logging.getLogger(__name__)


@asynccontextmanager
async def create_lifespan(app: FastAPI):
    """
    Create lifespan context manager for FastAPI application.
    
    This replaces the large lifespan function in main.py with a clean,
    structured approach to startup and shutdown.
    
    Args:
        app: FastAPI application instance
        
    Yields:
        Dictionary of initialized components
    """
    components: Dict[str, Any] = {}
    
    try:
        # Startup phase
        logger.info("🚀 R3MES Backend starting up...")
        components = await initialize_startup_components()
        
        # Store components in app state for access in endpoints
        app.state.components = components
        
        # Store individual components for backward compatibility
        app.state.database = components.get('database')
        app.state.cache_manager = components.get('cache_manager')
        app.state.model_manager = components.get('model_manager')
        app.state.semantic_router = components.get('semantic_router')
        app.state.task_queue = components.get('task_queue')
        app.state.serving_node_registry = components.get('serving_node_registry')
        app.state.notification_service = components.get('notification_service')
        app.state.config = components.get('config')
        
        logger.info("🎉 R3MES Backend startup completed successfully")
        
        # Yield control to the application
        yield components
        
    except Exception as e:
        logger.critical(f"❌ R3MES Backend startup failed: {e}")
        # Attempt cleanup of any partially initialized components
        if components:
            try:
                await shutdown_components(components)
            except Exception as cleanup_error:
                logger.error(f"Error during startup cleanup: {cleanup_error}")
        raise
    
    finally:
        # Shutdown phase
        logger.info("🛑 R3MES Backend shutting down...")
        try:
            await shutdown_components(components)
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        logger.info("👋 R3MES Backend shutdown completed")


# Dependency functions for accessing components in endpoints
def get_database(app: FastAPI):
    """Get database instance from app state."""
    return app.state.database


def get_cache_manager(app: FastAPI):
    """Get cache manager instance from app state."""
    return app.state.cache_manager


def get_model_manager(app: FastAPI):
    """Get model manager instance from app state."""
    return app.state.model_manager


def get_semantic_router(app: FastAPI):
    """Get semantic router instance from app state."""
    return app.state.semantic_router


def get_task_queue(app: FastAPI):
    """Get task queue instance from app state."""
    return app.state.task_queue


def get_serving_node_registry(app: FastAPI):
    """Get serving node registry instance from app state."""
    return app.state.serving_node_registry


def get_notification_service(app: FastAPI):
    """Get notification service instance from app state."""
    return app.state.notification_service


def get_config(app: FastAPI):
    """Get configuration instance from app state."""
    return app.state.config
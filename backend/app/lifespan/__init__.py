"""
Lifespan Management Package

Provides structured startup and shutdown management for R3MES Backend.
"""

from .manager import create_lifespan
from .startup import initialize_startup_components
from .shutdown import shutdown_components

__all__ = [
    'create_lifespan',
    'initialize_startup_components', 
    'shutdown_components'
]
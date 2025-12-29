"""
WebSocket API Endpoints

Provides WebSocket endpoints for real-time updates.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Optional
import logging

from .websocket_manager import connection_manager, websocket_endpoint

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/miner_stats")
async def miner_stats_websocket(websocket: WebSocket):
    """WebSocket endpoint for miner statistics."""
    await websocket_endpoint(websocket, "miner_stats")


@router.websocket("/ws/training_metrics")
async def training_metrics_websocket(websocket: WebSocket):
    """WebSocket endpoint for training metrics."""
    await websocket_endpoint(websocket, "training_metrics")


@router.websocket("/ws/network_status")
async def network_status_websocket(websocket: WebSocket):
    """WebSocket endpoint for network status."""
    await websocket_endpoint(websocket, "network_status")


@router.websocket("/ws/blocks")
async def blocks_websocket(websocket: WebSocket):
    """WebSocket endpoint for block updates."""
    await websocket_endpoint(websocket, "block_updates")


"""
Notification API Endpoints for R3MES

Provides REST API for notification management:
- List notifications
- Mark as read
- Create notifications
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel
import uuid
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/notifications", tags=["notifications"])


# In-memory storage (production should use database)
notifications_store: List[Dict[str, Any]] = []


class NotificationCreate(BaseModel):
    title: str
    message: str
    priority: str = "medium"  # low, medium, high, critical
    type: str = "system"  # mining, system, economic, governance
    wallet_address: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class NotificationResponse(BaseModel):
    id: str
    title: str
    message: str
    priority: str
    type: str
    read: bool
    created_at: str
    metadata: Optional[Dict[str, Any]] = None


@router.get("", response_model=List[NotificationResponse])
async def get_notifications(
    wallet_address: Optional[str] = Query(None, description="Filter by wallet address"),
    unread_only: bool = Query(False, description="Only return unread notifications"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of notifications to return"),
    notification_type: Optional[str] = Query(None, description="Filter by notification type"),
):
    """
    Get notifications for user.
    
    Args:
        wallet_address: Optional wallet address to filter notifications
        unread_only: If true, only return unread notifications
        limit: Maximum number of notifications to return
        notification_type: Filter by notification type (mining, system, economic, governance)
    
    Returns:
        List of notifications
    """
    filtered = notifications_store.copy()
    
    # Filter by wallet address (include global notifications)
    if wallet_address:
        filtered = [
            n for n in filtered 
            if n.get("wallet_address") == wallet_address or n.get("wallet_address") is None
        ]
    
    # Filter by read status
    if unread_only:
        filtered = [n for n in filtered if not n.get("read", False)]
    
    # Filter by type
    if notification_type:
        filtered = [n for n in filtered if n.get("type") == notification_type]
    
    # Sort by created_at descending (newest first)
    filtered.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    
    return filtered[:limit]


@router.post("/{notification_id}/read")
async def mark_as_read(notification_id: str):
    """
    Mark a notification as read.
    
    Args:
        notification_id: ID of the notification to mark as read
    
    Returns:
        Success status
    """
    for notification in notifications_store:
        if notification.get("id") == notification_id:
            notification["read"] = True
            return {"success": True, "message": "Notification marked as read"}
    
    raise HTTPException(status_code=404, detail="Notification not found")


@router.post("/read-all")
async def mark_all_as_read(
    wallet_address: Optional[str] = Query(None, description="Wallet address to filter")
):
    """
    Mark all notifications as read.
    
    Args:
        wallet_address: Optional wallet address to filter which notifications to mark
    
    Returns:
        Number of notifications marked as read
    """
    count = 0
    for notification in notifications_store:
        if wallet_address is None or notification.get("wallet_address") == wallet_address or notification.get("wallet_address") is None:
            if not notification.get("read", False):
                notification["read"] = True
                count += 1
    
    return {"success": True, "marked_count": count}


@router.post("", response_model=NotificationResponse)
async def create_notification(notification: NotificationCreate):
    """
    Create a new notification.
    
    Args:
        notification: Notification data
    
    Returns:
        Created notification
    """
    new_notification = {
        "id": str(uuid.uuid4()),
        "title": notification.title,
        "message": notification.message,
        "priority": notification.priority,
        "type": notification.type,
        "wallet_address": notification.wallet_address,
        "read": False,
        "created_at": datetime.utcnow().isoformat(),
        "metadata": notification.metadata or {},
    }
    
    notifications_store.append(new_notification)
    
    # Keep only last 1000 notifications in memory
    if len(notifications_store) > 1000:
        notifications_store.pop(0)
    
    logger.info(f"Created notification: {new_notification['title']}")
    
    return new_notification


@router.delete("/{notification_id}")
async def delete_notification(notification_id: str):
    """
    Delete a notification.
    
    Args:
        notification_id: ID of the notification to delete
    
    Returns:
        Success status
    """
    for i, notification in enumerate(notifications_store):
        if notification.get("id") == notification_id:
            notifications_store.pop(i)
            return {"success": True, "message": "Notification deleted"}
    
    raise HTTPException(status_code=404, detail="Notification not found")


@router.get("/stats")
async def get_notification_stats(
    wallet_address: Optional[str] = Query(None, description="Wallet address to filter")
):
    """
    Get notification statistics.
    
    Args:
        wallet_address: Optional wallet address to filter
    
    Returns:
        Notification statistics
    """
    filtered = notifications_store
    
    if wallet_address:
        filtered = [
            n for n in filtered 
            if n.get("wallet_address") == wallet_address or n.get("wallet_address") is None
        ]
    
    total = len(filtered)
    unread = len([n for n in filtered if not n.get("read", False)])
    
    by_type = {}
    by_priority = {}
    
    for n in filtered:
        n_type = n.get("type", "system")
        n_priority = n.get("priority", "medium")
        
        by_type[n_type] = by_type.get(n_type, 0) + 1
        by_priority[n_priority] = by_priority.get(n_priority, 0) + 1
    
    return {
        "total": total,
        "unread": unread,
        "by_type": by_type,
        "by_priority": by_priority,
    }


# Helper function to create system notifications
def create_system_notification(
    title: str,
    message: str,
    priority: str = "medium",
    notification_type: str = "system",
    wallet_address: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Helper function to create notifications from other parts of the application.
    
    Args:
        title: Notification title
        message: Notification message
        priority: Priority level (low, medium, high, critical)
        notification_type: Type (mining, system, economic, governance)
        wallet_address: Optional target wallet address
        metadata: Optional additional metadata
    """
    new_notification = {
        "id": str(uuid.uuid4()),
        "title": title,
        "message": message,
        "priority": priority,
        "type": notification_type,
        "wallet_address": wallet_address,
        "read": False,
        "created_at": datetime.utcnow().isoformat(),
        "metadata": metadata or {},
    }
    
    notifications_store.append(new_notification)
    
    # Keep only last 1000 notifications
    if len(notifications_store) > 1000:
        notifications_store.pop(0)
    
    return new_notification


# WebSocket integration for real-time notifications
async def broadcast_notification_realtime(notification: Dict[str, Any]):
    """
    Broadcast notification via WebSocket for real-time delivery.
    
    Args:
        notification: Notification data to broadcast
    """
    try:
        from .websocket_manager import broadcast_notification
        await broadcast_notification(notification, notification.get("wallet_address"))
    except Exception as e:
        logger.warning(f"Failed to broadcast notification via WebSocket: {e}")


async def create_and_broadcast_notification(
    title: str,
    message: str,
    priority: str = "medium",
    notification_type: str = "system",
    wallet_address: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a notification and broadcast it via WebSocket.
    
    Args:
        title: Notification title
        message: Notification message
        priority: Priority level
        notification_type: Type of notification
        wallet_address: Target wallet address
        metadata: Additional metadata
    
    Returns:
        Created notification
    """
    notification = create_system_notification(
        title=title,
        message=message,
        priority=priority,
        notification_type=notification_type,
        wallet_address=wallet_address,
        metadata=metadata,
    )
    
    # Broadcast via WebSocket
    await broadcast_notification_realtime(notification)
    
    return notification

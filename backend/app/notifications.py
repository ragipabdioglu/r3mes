"""
Notification System for R3MES

Supports multiple notification channels:
- Email (SMTP)
- Slack webhooks
- In-app notifications
- Push notifications (future)
"""

import smtplib
import aiohttp
import asyncio
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Optional
from enum import Enum
import logging
import os
import json

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """Notification channel types."""
    EMAIL = "email"
    SLACK = "slack"
    IN_APP = "in_app"
    PUSH = "push"


class NotificationPriority(Enum):
    """Notification priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationService:
    """Notification service for sending alerts and updates."""
    
    def __init__(self):
        self.smtp_host = os.getenv("SMTP_HOST")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        self.smtp_from = os.getenv("SMTP_FROM", "noreply@r3mes.network")
        
        self.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        self.slack_channel = os.getenv("SLACK_CHANNEL", "#r3mes-alerts")
        
        self.enabled_channels = os.getenv("NOTIFICATION_CHANNELS", "email,slack").split(",")
    
    async def send_notification(
        self,
        title: str,
        message: str,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        channels: Optional[List[NotificationChannel]] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, bool]:
        """
        Send notification to specified channels.
        
        Args:
            title: Notification title
            message: Notification message
            priority: Priority level
            channels: List of channels to send to (None = all enabled)
            metadata: Additional metadata
            
        Returns:
            Dictionary mapping channel to success status
        """
        if channels is None:
            channels = [NotificationChannel(c) for c in self.enabled_channels if c in [ch.value for ch in NotificationChannel]]
        
        results = {}
        
        for channel in channels:
            try:
                if channel == NotificationChannel.EMAIL:
                    results["email"] = await self._send_email(title, message, priority)
                elif channel == NotificationChannel.SLACK:
                    results["slack"] = await self._send_slack(title, message, priority, metadata)
                elif channel == NotificationChannel.IN_APP:
                    results["in_app"] = await self._send_in_app(title, message, priority, metadata)
                else:
                    results[channel.value] = False
            except Exception as e:
                logger.error(f"Failed to send notification via {channel.value}: {e}")
                results[channel.value] = False
        
        return results
    
    async def _send_email(
        self,
        title: str,
        message: str,
        priority: NotificationPriority
    ) -> bool:
        """Send email notification."""
        if not all([self.smtp_host, self.smtp_user, self.smtp_password]):
            logger.warning("Email configuration incomplete, skipping email notification")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_from
            msg['To'] = os.getenv("ALERT_EMAIL", "alerts@r3mes.network")
            msg['Subject'] = f"[{priority.value.upper()}] {title}"
            
            body = f"""
            <html>
              <body>
                <h2>{title}</h2>
                <p>{message}</p>
                <hr>
                <p style="color: #888; font-size: 12px;">
                  This is an automated notification from R3MES.
                </p>
              </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # Send email asynchronously
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._send_email_sync,
                msg
            )
            
            return True
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    def _send_email_sync(self, msg: MIMEMultipart):
        """Synchronous email sending."""
        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.starttls()
            server.login(self.smtp_user, self.smtp_password)
            server.send_message(msg)
    
    async def _send_slack(
        self,
        title: str,
        message: str,
        priority: NotificationPriority,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Send Slack notification."""
        if not self.slack_webhook_url:
            logger.warning("Slack webhook URL not configured, skipping Slack notification")
            return False
        
        try:
            # Determine color based on priority
            color_map = {
                NotificationPriority.LOW: "#36a64f",  # Green
                NotificationPriority.MEDIUM: "#ffa500",  # Orange
                NotificationPriority.HIGH: "#ff0000",  # Red
                NotificationPriority.CRITICAL: "#8b0000",  # Dark red
            }
            
            payload = {
                "channel": self.slack_channel,
                "username": "R3MES Alerts",
                "icon_emoji": ":warning:",
                "attachments": [
                    {
                        "color": color_map.get(priority, "#ffa500"),
                        "title": title,
                        "text": message,
                        "fields": [
                            {
                                "title": "Priority",
                                "value": priority.value.upper(),
                                "short": True
                            }
                        ],
                        "footer": "R3MES Notification System",
                        "ts": int(asyncio.get_event_loop().time())
                    }
                ]
            }
            
            # Add metadata fields
            if metadata:
                for key, value in metadata.items():
                    payload["attachments"][0]["fields"].append({
                        "title": key.replace("_", " ").title(),
                        "value": str(value),
                        "short": True
                    })
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.slack_webhook_url,
                    json=payload
                ) as response:
                    if response.status == 200:
                        return True
                    else:
                        logger.error(f"Slack API returned status {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False
    
    async def _send_in_app(
        self,
        title: str,
        message: str,
        priority: NotificationPriority,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Send in-app notification (store in database for user retrieval)."""
        # In-app notifications would be stored in database
        # and retrieved by frontend via API
        # For now, just log it
        logger.info(f"In-app notification: {title} - {message}")
        return True
    
    async def send_mining_alert(
        self,
        wallet_address: str,
        alert_type: str,
        message: str
    ):
        """Send mining-specific alert."""
        return await self.send_notification(
            title=f"Mining Alert: {alert_type}",
            message=f"Wallet {wallet_address}: {message}",
            priority=NotificationPriority.HIGH,
            metadata={
                "wallet_address": wallet_address,
                "alert_type": alert_type,
            }
        )
    
    async def send_system_alert(
        self,
        component: str,
        alert_type: str,
        message: str,
        priority: NotificationPriority = NotificationPriority.CRITICAL
    ):
        """Send system-level alert."""
        return await self.send_notification(
            title=f"System Alert: {component} - {alert_type}",
            message=message,
            priority=priority,
            metadata={
                "component": component,
                "alert_type": alert_type,
            }
        )


# Global notification service instance
_notification_service: Optional[NotificationService] = None


def get_notification_service() -> NotificationService:
    """Get global notification service instance."""
    global _notification_service
    if _notification_service is None:
        _notification_service = NotificationService()
    return _notification_service


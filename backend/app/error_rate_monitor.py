"""
Error Rate Monitor

Monitors API error rates and sends notifications when error rate exceeds threshold.
"""

import asyncio
import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Optional
import os

from .metrics import api_requests_total
from .notifications import get_notification_service, NotificationPriority
from prometheus_client import REGISTRY

logger = logging.getLogger(__name__)


class ErrorRateMonitor:
    """
    Monitors API error rates and sends alerts when threshold is exceeded.
    """
    
    def __init__(
        self,
        error_rate_threshold: float = 0.1,  # 10% error rate threshold
        check_interval: int = 60,  # Check every 60 seconds
        min_requests: int = 100,  # Minimum requests before alerting
    ):
        """
        Initialize error rate monitor.
        
        Args:
            error_rate_threshold: Error rate threshold (0.0-1.0)
            check_interval: Check interval in seconds
            min_requests: Minimum number of requests before alerting
        """
        self.error_rate_threshold = error_rate_threshold
        self.check_interval = check_interval
        self.min_requests = min_requests
        self.running = False
        self._task: Optional[asyncio.Task] = None
        self._last_error_counts: Dict[str, int] = {}
        self._last_total_counts: Dict[str, int] = {}
        self._last_check_time = datetime.now()
    
    async def start(self):
        """Start error rate monitoring."""
        if self.running:
            logger.warning("Error rate monitor is already running")
            return
        
        self.running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("Error rate monitor started")
    
    async def stop(self):
        """Stop error rate monitoring."""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Error rate monitor stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                await self._check_error_rate()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in error rate monitor loop: {e}", exc_info=True)
                await asyncio.sleep(self.check_interval)
    
    async def _check_error_rate(self):
        """Check current error rate and send notification if threshold exceeded."""
        try:
            # Get current metrics from Prometheus
            current_time = datetime.now()
            time_diff = (current_time - self._last_check_time).total_seconds()
            
            # Collect metrics from Prometheus registry
            metrics_data = REGISTRY.collect()
            
            total_requests = 0
            error_requests = 0
            
            for metric_family in metrics_data:
                if metric_family.name == 'api_requests_total':
                    for sample in metric_family.samples:
                        labels = sample.labels
                        status_code = labels.get('status_code', '')
                        count = int(sample.value)
                        
                        # Get previous count for this label combination
                        label_key = f"{labels.get('method', '')}:{labels.get('endpoint', '')}:{status_code}"
                        prev_count = self._last_total_counts.get(label_key, 0)
                        
                        # Calculate delta (requests since last check)
                        delta = count - prev_count
                        total_requests += delta
                        
                        # Count 4xx and 5xx as errors
                        if status_code and status_code.startswith(('4', '5')):
                            error_requests += delta
                        
                        # Store current count for next check
                        self._last_total_counts[label_key] = count
            
            # Calculate error rate
            if total_requests >= self.min_requests:
                error_rate = error_requests / total_requests if total_requests > 0 else 0.0
                
                if error_rate >= self.error_rate_threshold:
                    # Send notification
                    notification_service = get_notification_service()
                    await notification_service.send_system_alert(
                        component="api",
                        alert_type="high_error_rate",
                        message=(
                            f"High API error rate detected: {error_rate:.2%} "
                            f"({error_requests}/{total_requests} errors in last {time_diff:.0f}s). "
                            f"Threshold: {self.error_rate_threshold:.2%}"
                        ),
                        priority=NotificationPriority.HIGH,
                        metadata={
                            "error_rate": error_rate,
                            "error_count": error_requests,
                            "total_requests": total_requests,
                            "time_window_seconds": time_diff,
                        }
                    )
                    logger.warning(
                        f"High error rate detected: {error_rate:.2%} "
                        f"({error_requests}/{total_requests} errors)"
                    )
            
            self._last_check_time = current_time
            
        except Exception as e:
            logger.error(f"Failed to check error rate: {e}", exc_info=True)


# Global error rate monitor instance
_error_rate_monitor: Optional[ErrorRateMonitor] = None


def get_error_rate_monitor() -> ErrorRateMonitor:
    """Get global error rate monitor instance."""
    global _error_rate_monitor
    if _error_rate_monitor is None:
        error_rate_threshold = float(os.getenv("ERROR_RATE_THRESHOLD", "0.1"))
        check_interval = int(os.getenv("ERROR_RATE_CHECK_INTERVAL", "60"))
        min_requests = int(os.getenv("ERROR_RATE_MIN_REQUESTS", "100"))
        _error_rate_monitor = ErrorRateMonitor(
            error_rate_threshold=error_rate_threshold,
            check_interval=check_interval,
            min_requests=min_requests,
        )
    return _error_rate_monitor


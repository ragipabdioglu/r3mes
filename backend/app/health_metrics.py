"""
Health Check Metrics for Prometheus

Exports health check metrics for monitoring and alerting.
"""

from prometheus_client import Gauge
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# Health check metrics
health_check_status = Gauge(
    'health_check_status',
    'Health check status (1=healthy, 0=unhealthy)',
    ['service', 'component']
)

health_check_duration_seconds = Gauge(
    'health_check_duration_seconds',
    'Health check duration in seconds',
    ['service', 'component']
)

health_check_last_success = Gauge(
    'health_check_last_success',
    'Timestamp of last successful health check',
    ['service', 'component']
)

health_check_last_failure = Gauge(
    'health_check_last_failure',
    'Timestamp of last failed health check',
    ['service', 'component']
)


class HealthMetrics:
    """Health check metrics exporter."""
    
    @staticmethod
    def record_health_check(
        service: str,
        component: str,
        is_healthy: bool,
        duration: float
    ):
        """
        Record health check result.
        
        Args:
            service: Service name (e.g., 'backend', 'database')
            component: Component name (e.g., 'api', 'postgresql')
            is_healthy: Whether the check passed
            duration: Check duration in seconds
        """
        health_check_status.labels(service=service, component=component).set(
            1.0 if is_healthy else 0.0
        )
        health_check_duration_seconds.labels(service=service, component=component).set(duration)
        
        import time
        timestamp = time.time()
        
        if is_healthy:
            health_check_last_success.labels(service=service, component=component).set(timestamp)
        else:
            health_check_last_failure.labels(service=service, component=component).set(timestamp)
    
    @staticmethod
    def get_health_status() -> Dict[str, Any]:
        """Get current health status from metrics."""
        # This would query the metrics to get current status
        # For now, return a placeholder
        return {
            'status': 'healthy',
            'checks': {}
        }


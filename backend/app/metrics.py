"""
Prometheus Metrics - Metrics exporter for monitoring

Provides metrics for API requests, latency, cache performance, and system health.
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
from typing import Optional
import time
import logging

logger = logging.getLogger(__name__)

# API Metrics
api_requests_total = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status_code']
)

api_request_duration_seconds = Histogram(
    'api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# Cache Metrics
cache_hits_total = Counter(
    'cache_hits_total',
    'Total number of cache hits'
)

cache_misses_total = Counter(
    'cache_misses_total',
    'Total number of cache misses'
)

# Database Metrics
database_connections_active = Gauge(
    'database_connections_active',
    'Number of active database connections'
)

database_query_duration_seconds = Histogram(
    'database_query_duration_seconds',
    'Database query duration in seconds',
    ['operation'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
)

# Model Inference Metrics
model_inference_duration_seconds = Histogram(
    'model_inference_duration_seconds',
    'Model inference duration in seconds',
    ['adapter'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

model_inference_requests_total = Counter(
    'model_inference_requests_total',
    'Total number of model inference requests',
    ['adapter', 'status']
)

# System Metrics
system_memory_usage_bytes = Gauge(
    'system_memory_usage_bytes',
    'System memory usage in bytes'
)

system_cpu_usage_percent = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage'
)

# GPU Metrics (if available)
gpu_utilization_percent = Gauge(
    'gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id']
)

gpu_memory_usage_bytes = Gauge(
    'gpu_memory_usage_bytes',
    'GPU memory usage in bytes',
    ['gpu_id']
)

gpu_temperature_celsius = Gauge(
    'gpu_temperature_celsius',
    'GPU temperature in Celsius',
    ['gpu_id']
)


class MetricsMiddleware:
    """Middleware to track API metrics."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        method = scope["method"]
        path = scope["path"]
        
        # Start timer
        start_time = time.time()
        
        # Track request
        status_code = 200
        
        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)
        
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            # Record metrics
            duration = time.time() - start_time
            
            api_requests_total.labels(
                method=method,
                endpoint=path,
                status_code=status_code
            ).inc()
            
            api_request_duration_seconds.labels(
                method=method,
                endpoint=path
            ).observe(duration)


def get_metrics_response() -> Response:
    """Get Prometheus metrics endpoint response."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


def record_cache_hit():
    """Record a cache hit."""
    cache_hits_total.inc()


def record_cache_miss():
    """Record a cache miss."""
    cache_misses_total.inc()


def record_database_query(operation: str, duration: float):
    """Record a database query."""
    database_query_duration_seconds.labels(operation=operation).observe(duration)


def record_model_inference(adapter: str, duration: float, success: bool = True):
    """Record a model inference."""
    model_inference_duration_seconds.labels(adapter=adapter).observe(duration)
    model_inference_requests_total.labels(
        adapter=adapter,
        status="success" if success else "error"
    ).inc()


def update_system_metrics(memory_bytes: int, cpu_percent: float):
    """Update system metrics."""
    system_memory_usage_bytes.set(memory_bytes)
    system_cpu_usage_percent.set(cpu_percent)


def update_gpu_metrics(gpu_id: int, utilization: float, memory_bytes: int, temperature: float):
    """Update GPU metrics."""
    gpu_utilization_percent.labels(gpu_id=str(gpu_id)).set(utilization)
    gpu_memory_usage_bytes.labels(gpu_id=str(gpu_id)).set(memory_bytes)
    gpu_temperature_celsius.labels(gpu_id=str(gpu_id)).set(temperature)


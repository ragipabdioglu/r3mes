"""
OpenTelemetry Setup for Distributed Tracing

Configures OpenTelemetry for distributed tracing across R3MES services.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# OpenTelemetry imports (optional, fail gracefully if not installed)
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    from opentelemetry.instrumentation.sqlite3 import SQLite3Instrumentor
    from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    logger.warning("OpenTelemetry packages not installed. Distributed tracing will be disabled.")


def setup_opentelemetry(
    service_name: str = "r3mes-backend",
    jaeger_endpoint: Optional[str] = None,
    otlp_endpoint: Optional[str] = None,
    enable_console_exporter: bool = False
) -> bool:
    """
    Set up OpenTelemetry for distributed tracing.
    
    Args:
        service_name: Name of the service
        jaeger_endpoint: Jaeger endpoint (e.g., "http://jaeger:14268/api/traces")
        otlp_endpoint: OTLP endpoint (e.g., "http://otel-collector:4317")
        enable_console_exporter: Enable console exporter for debugging
    
    Returns:
        True if setup successful, False otherwise
    """
    if not OPENTELEMETRY_AVAILABLE:
        logger.warning("OpenTelemetry not available, skipping setup")
        return False
    
    try:
        # Create resource
        resource = Resource.create({
            "service.name": service_name,
            "service.version": os.getenv("R3MES_VERSION", "1.0.0"),
            "deployment.environment": os.getenv("R3MES_ENV", "development"),
        })
        
        # Create tracer provider
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)
        
        # Add span processors
        processors = []
        
        # OTLP exporter (preferred)
        if otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(
                endpoint=otlp_endpoint,
                insecure=True  # Set to False for TLS
            )
            processors.append(BatchSpanProcessor(otlp_exporter))
            logger.info(f"OpenTelemetry OTLP exporter configured: {otlp_endpoint}")
        
        # Jaeger exporter (fallback)
        elif jaeger_endpoint:
            jaeger_exporter = JaegerExporter(
                agent_host_name=jaeger_endpoint.split("://")[1].split(":")[0] if "://" in jaeger_endpoint else jaeger_endpoint.split(":")[0],
                agent_port=int(jaeger_endpoint.split(":")[-1]) if ":" in jaeger_endpoint else 14268,
            )
            processors.append(BatchSpanProcessor(jaeger_exporter))
            logger.info(f"OpenTelemetry Jaeger exporter configured: {jaeger_endpoint}")
        
        # Console exporter (for debugging)
        if enable_console_exporter:
            console_exporter = ConsoleSpanExporter()
            processors.append(BatchSpanProcessor(console_exporter))
            logger.info("OpenTelemetry console exporter enabled")
        
        # Add processors to tracer provider
        for processor in processors:
            tracer_provider.add_span_processor(processor)
        
        logger.info("OpenTelemetry tracing initialized successfully")
        return True
    
    except Exception as e:
        logger.error(f"Failed to setup OpenTelemetry: {e}")
        return False


def instrument_fastapi(app) -> bool:
    """
    Instrument FastAPI application with OpenTelemetry.
    
    Args:
        app: FastAPI application instance
    
    Returns:
        True if instrumentation successful, False otherwise
    """
    if not OPENTELEMETRY_AVAILABLE:
        return False
    
    try:
        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI instrumented with OpenTelemetry")
        return True
    except Exception as e:
        logger.error(f"Failed to instrument FastAPI: {e}")
        return False


def instrument_http_clients() -> bool:
    """
    Instrument HTTP clients (requests, httpx) with OpenTelemetry.
    
    Returns:
        True if instrumentation successful, False otherwise
    """
    if not OPENTELEMETRY_AVAILABLE:
        return False
    
    try:
        RequestsInstrumentor().instrument()
        HTTPXClientInstrumentor().instrument()
        logger.info("HTTP clients instrumented with OpenTelemetry")
        return True
    except Exception as e:
        logger.error(f"Failed to instrument HTTP clients: {e}")
        return False


def instrument_databases() -> bool:
    """
    Instrument databases (Redis, SQLite, PostgreSQL) with OpenTelemetry.
    
    Returns:
        True if instrumentation successful, False otherwise
    """
    if not OPENTELEMETRY_AVAILABLE:
        return False
    
    try:
        RedisInstrumentor().instrument()
        SQLite3Instrumentor().instrument()
        AsyncPGInstrumentor().instrument()
        logger.info("Databases instrumented with OpenTelemetry")
        return True
    except Exception as e:
        logger.error(f"Failed to instrument databases: {e}")
        return False


def get_tracer(name: str = None):
    """
    Get OpenTelemetry tracer.
    
    Args:
        name: Tracer name (default: service name)
    
    Returns:
        Tracer instance or None if not available
    """
    if not OPENTELEMETRY_AVAILABLE:
        return None
    
    try:
        return trace.get_tracer(name or "r3mes-backend")
    except Exception as e:
        logger.error(f"Failed to get tracer: {e}")
        return None


def get_current_trace_id() -> Optional[str]:
    """
    Get current trace ID from active span.
    
    Returns:
        Trace ID as hex string or None
    """
    if not OPENTELEMETRY_AVAILABLE:
        return None
    
    try:
        span = trace.get_current_span()
        if span:
            context = span.get_span_context()
            if context.is_valid:
                return format(context.trace_id, '032x')
    except Exception:
        pass
    
    return None


def get_current_span_id() -> Optional[str]:
    """
    Get current span ID from active span.
    
    Returns:
        Span ID as hex string or None
    """
    if not OPENTELEMETRY_AVAILABLE:
        return None
    
    try:
        span = trace.get_current_span()
        if span:
            context = span.get_span_context()
            if context.is_valid:
                return format(context.span_id, '016x')
    except Exception:
        pass
    
    return None


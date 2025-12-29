"""
Trace Middleware for Log Correlation

Adds trace ID and span ID to request state and logs for correlation.
"""

import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .opentelemetry_setup import get_current_trace_id, get_current_span_id

logger = logging.getLogger(__name__)


class TraceMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add trace ID and span ID to request state and logs.
    
    This enables log correlation across services by including trace IDs
    in all log messages.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and add trace context."""
        # Get trace ID from OpenTelemetry context
        trace_id = get_current_trace_id()
        span_id = get_current_span_id()
        
        # If no trace ID from OpenTelemetry, generate one
        if not trace_id:
            import uuid
            trace_id = uuid.uuid4().hex[:32]
            span_id = uuid.uuid4().hex[:16]
        
        # Add to request state
        request.state.trace_id = trace_id
        request.state.span_id = span_id
        
        # Add trace context to logger
        # This will be included in all log messages for this request
        old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.trace_id = trace_id
            record.span_id = span_id
            return record
        
        logging.setLogRecordFactory(record_factory)
        
        try:
            response = await call_next(request)
            
            # Add trace ID to response headers (for debugging)
            response.headers["X-Trace-ID"] = trace_id
            response.headers["X-Span-ID"] = span_id
            
            return response
        finally:
            # Restore original factory
            logging.setLogRecordFactory(old_factory)


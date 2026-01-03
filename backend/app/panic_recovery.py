"""
Panic Recovery Middleware for FastAPI

Handles unhandled exceptions and panics gracefully.
"""

import logging
import traceback
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .exceptions import R3MESException
from .sentry import capture_exception

logger = logging.getLogger(__name__)


class PanicRecoveryMiddleware(BaseHTTPMiddleware):
    """
    Middleware to catch and handle unhandled exceptions.
    
    Prevents the application from crashing on unexpected errors.
    """
    
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except R3MESException as e:
            # Known R3MES exceptions - log and return appropriate error
            logger.error(f"R3MES exception: {e}", exc_info=True)
            capture_exception(e)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "R3MES_ERROR",
                    "message": str(e),
                    "type": type(e).__name__
                }
            )
        except Exception as e:
            # Unknown exceptions - log with full traceback and return generic error
            logger.critical(
                f"Unhandled exception in {request.url.path}: {e}",
                exc_info=True,
                extra={
                    "path": str(request.url.path),
                    "method": request.method,
                    "client": request.client.host if request.client else None,
                }
            )
            capture_exception(e)
            
            # In production, don't expose internal error details
            is_production = getattr(request.app.state, "is_production", False)
            if is_production:
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": "INTERNAL_SERVER_ERROR",
                        "message": "An internal error occurred. Please try again later.",
                        "request_id": getattr(request.state, "request_id", "unknown")
                    }
                )
            else:
                # In development, show full error details
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": "INTERNAL_SERVER_ERROR",
                        "message": str(e),
                        "type": type(e).__name__,
                        "traceback": traceback.format_exc(),
                        "path": str(request.url.path),
                        "method": request.method,
                    }
                )


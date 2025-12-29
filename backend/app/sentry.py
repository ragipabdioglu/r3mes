"""
Sentry Integration for R3MES Backend

Error tracking and performance monitoring with Sentry.
"""

import os
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
# Optional SQLAlchemy integration (only if sqlalchemy is installed)
try:
    from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
    SQLALCHEMY_AVAILABLE = True
except Exception:
    SqlalchemyIntegration = None
    SQLALCHEMY_AVAILABLE = False

def init_sentry():
    """
    Initialize Sentry SDK for error tracking.
    """
    sentry_dsn = os.getenv("SENTRY_DSN")
    
    if not sentry_dsn:
        # Sentry is optional - don't fail if DSN is not set
        return
    
    environment = os.getenv("R3MES_ENV", "development")
    traces_sample_rate = 0.1 if environment == "production" else 1.0
    
    # Build integrations list
    integrations = [
        FastApiIntegration(),
        LoggingIntegration(
            level=None,  # Capture all log levels
            event_level=None,  # Send all log levels as events
        ),
    ]
    
    # Add SQLAlchemy integration only if available
    if SQLALCHEMY_AVAILABLE and SqlalchemyIntegration is not None:
        integrations.append(SqlalchemyIntegration())
    
    sentry_sdk.init(
        dsn=sentry_dsn,
        environment=environment,
        traces_sample_rate=traces_sample_rate,
        profiles_sample_rate=traces_sample_rate,
        
        # Integrations
        integrations=integrations,
        
        # Filter out sensitive data
        before_send=lambda event, hint: filter_sensitive_data(event, hint),
        
        # Release tracking
        release=os.getenv("R3MES_VERSION", "unknown"),
        
        # Server name
        server_name=os.getenv("HOSTNAME", "r3mes-backend"),
    )


def filter_sensitive_data(event, hint):
    """
    Filter out sensitive information from Sentry events.
    """
    # Don't send events in development unless explicitly enabled
    if os.getenv("R3MES_ENV", "development") == "development":
        if os.getenv("SENTRY_DEBUG", "false").lower() != "true":
            return None
    
    # Remove sensitive headers
    if event.get("request") and event["request"].get("headers"):
        sensitive_headers = [
            "authorization",
            "cookie",
            "x-api-key",
            "api-key",
        ]
        for header in sensitive_headers:
            event["request"]["headers"].pop(header, None)
    
    # Remove sensitive data from request body
    if event.get("request") and event["request"].get("data"):
        if isinstance(event["request"]["data"], dict):
            sensitive_fields = [
                "password",
                "private_key",
                "mnemonic",
                "api_key",
            ]
            for field in sensitive_fields:
                event["request"]["data"].pop(field, None)
    
    return event


def capture_exception(error: Exception, **kwargs):
    """
    Capture an exception to Sentry.
    
    Args:
        error: The exception to capture
        **kwargs: Additional context (tags, extra, etc.)
    """
    sentry_sdk.capture_exception(error, **kwargs)


def capture_message(message: str, level: str = "info", **kwargs):
    """
    Capture a message to Sentry.
    
    Args:
        message: The message to capture
        level: Log level (debug, info, warning, error, fatal)
        **kwargs: Additional context
    """
    sentry_sdk.capture_message(message, level=level, **kwargs)


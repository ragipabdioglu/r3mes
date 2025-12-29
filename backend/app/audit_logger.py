"""
Audit Logger Module for R3MES Backend

Provides comprehensive audit logging for sensitive operations.
All audit logs are structured and can be exported to external systems.
"""

import logging
import json
import os
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass, asdict
from functools import wraps


class AuditOperation(str, Enum):
    """Audit operation types."""
    # Authentication
    LOGIN = "login"
    LOGOUT = "logout"
    API_KEY_CREATE = "api_key_create"
    API_KEY_REVOKE = "api_key_revoke"
    API_KEY_USE = "api_key_use"
    
    # Wallet operations
    WALLET_CREATE = "wallet_create"
    WALLET_IMPORT = "wallet_import"
    WALLET_EXPORT = "wallet_export"
    WALLET_BALANCE_CHECK = "wallet_balance_check"
    
    # Credit operations
    CREDIT_DEDUCT = "credit_deduct"
    CREDIT_ADD = "credit_add"
    CREDIT_TRANSFER = "credit_transfer"
    
    # Inference operations
    INFERENCE_REQUEST = "inference_request"
    INFERENCE_COMPLETE = "inference_complete"
    INFERENCE_FAILED = "inference_failed"
    
    # Node operations
    NODE_REGISTER = "node_register"
    NODE_UNREGISTER = "node_unregister"
    NODE_STATUS_UPDATE = "node_status_update"
    
    # Governance
    PROPOSAL_CREATE = "proposal_create"
    PROPOSAL_VOTE = "proposal_vote"
    
    # Admin operations
    ADMIN_CONFIG_CHANGE = "admin_config_change"
    ADMIN_USER_MODIFY = "admin_user_modify"
    
    # Security events
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INVALID_AUTH_ATTEMPT = "invalid_auth_attempt"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"


class AuditSeverity(str, Enum):
    """Audit log severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditLogEntry:
    """Structured audit log entry."""
    timestamp: str
    operation: str
    user_id: str
    resource: str
    success: bool
    severity: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class AuditLogger:
    """
    Audit logger for sensitive operations.
    
    Features:
    - Structured logging with JSON format
    - Configurable output (file, stdout, external service)
    - Request correlation via trace_id
    - Automatic timestamp in UTC
    """
    
    def __init__(self, logger_name: str = "audit"):
        """Initialize audit logger."""
        self.logger = logging.getLogger(logger_name)
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup audit logger with appropriate handlers."""
        # Don't add handlers if already configured
        if self.logger.handlers:
            return
        
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        
        # Console handler for audit logs
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s | AUDIT | %(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S%z'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for audit logs (if enabled)
        audit_log_path = os.getenv("R3MES_AUDIT_LOG_PATH")
        if audit_log_path:
            try:
                file_handler = logging.FileHandler(audit_log_path, encoding='utf-8')
                file_handler.setLevel(logging.INFO)
                file_handler.setFormatter(console_formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                self.logger.warning(f"Failed to setup audit file handler: {e}")
    
    def log(
        self,
        operation: AuditOperation,
        user_id: str,
        resource: str,
        success: bool = True,
        severity: AuditSeverity = AuditSeverity.INFO,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ):
        """
        Log an audit event.
        
        Args:
            operation: Type of operation being audited
            user_id: User identifier (wallet address, API key, etc.)
            resource: Resource being accessed/modified
            success: Whether the operation succeeded
            severity: Log severity level
            ip_address: Client IP address
            user_agent: Client user agent
            details: Additional details about the operation
            error_message: Error message if operation failed
            request_id: Request correlation ID
            trace_id: Distributed trace ID
        """
        entry = AuditLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            operation=operation.value,
            user_id=user_id,
            resource=resource,
            success=success,
            severity=severity.value,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details,
            error_message=error_message,
            request_id=request_id,
            trace_id=trace_id,
        )
        
        log_message = entry.to_json()
        
        if severity == AuditSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif severity == AuditSeverity.ERROR:
            self.logger.error(log_message)
        elif severity == AuditSeverity.WARNING:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def log_auth_success(
        self,
        user_id: str,
        auth_method: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ):
        """Log successful authentication."""
        self.log(
            operation=AuditOperation.LOGIN,
            user_id=user_id,
            resource="authentication",
            success=True,
            details={"auth_method": auth_method},
            ip_address=ip_address,
            user_agent=user_agent,
        )
    
    def log_auth_failure(
        self,
        user_id: str,
        reason: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ):
        """Log failed authentication attempt."""
        self.log(
            operation=AuditOperation.INVALID_AUTH_ATTEMPT,
            user_id=user_id,
            resource="authentication",
            success=False,
            severity=AuditSeverity.WARNING,
            error_message=reason,
            ip_address=ip_address,
            user_agent=user_agent,
        )
    
    def log_credit_operation(
        self,
        user_id: str,
        operation_type: str,
        amount: float,
        success: bool = True,
        error_message: Optional[str] = None,
    ):
        """Log credit operation (deduct, add, transfer)."""
        op = AuditOperation.CREDIT_DEDUCT if operation_type == "deduct" else AuditOperation.CREDIT_ADD
        self.log(
            operation=op,
            user_id=user_id,
            resource="credits",
            success=success,
            details={"amount": amount, "operation_type": operation_type},
            error_message=error_message,
            severity=AuditSeverity.ERROR if not success else AuditSeverity.INFO,
        )
    
    def log_inference_request(
        self,
        user_id: str,
        adapter_name: str,
        message_length: int,
        request_id: Optional[str] = None,
    ):
        """Log inference request."""
        self.log(
            operation=AuditOperation.INFERENCE_REQUEST,
            user_id=user_id,
            resource=f"inference/{adapter_name}",
            success=True,
            details={"adapter": adapter_name, "message_length": message_length},
            request_id=request_id,
        )
    
    def log_inference_complete(
        self,
        user_id: str,
        adapter_name: str,
        latency_ms: float,
        tokens_generated: int,
        request_id: Optional[str] = None,
    ):
        """Log successful inference completion."""
        self.log(
            operation=AuditOperation.INFERENCE_COMPLETE,
            user_id=user_id,
            resource=f"inference/{adapter_name}",
            success=True,
            details={
                "adapter": adapter_name,
                "latency_ms": latency_ms,
                "tokens_generated": tokens_generated,
            },
            request_id=request_id,
        )
    
    def log_rate_limit_exceeded(
        self,
        user_id: str,
        endpoint: str,
        ip_address: Optional[str] = None,
    ):
        """Log rate limit exceeded event."""
        self.log(
            operation=AuditOperation.RATE_LIMIT_EXCEEDED,
            user_id=user_id,
            resource=endpoint,
            success=False,
            severity=AuditSeverity.WARNING,
            ip_address=ip_address,
            error_message="Rate limit exceeded",
        )
    
    def log_suspicious_activity(
        self,
        user_id: str,
        activity_type: str,
        details: Dict[str, Any],
        ip_address: Optional[str] = None,
    ):
        """Log suspicious activity for security monitoring."""
        self.log(
            operation=AuditOperation.SUSPICIOUS_ACTIVITY,
            user_id=user_id,
            resource="security",
            success=False,
            severity=AuditSeverity.CRITICAL,
            details={"activity_type": activity_type, **details},
            ip_address=ip_address,
        )


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get or create the global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def audit_log(
    operation: AuditOperation,
    resource: str,
    get_user_id: Optional[callable] = None,
):
    """
    Decorator for automatic audit logging of function calls.
    
    Usage:
        @audit_log(AuditOperation.WALLET_EXPORT, "wallet")
        async def export_wallet(wallet_address: str):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            audit_logger = get_audit_logger()
            user_id = get_user_id(*args, **kwargs) if get_user_id else "unknown"
            
            try:
                result = await func(*args, **kwargs)
                audit_logger.log(
                    operation=operation,
                    user_id=user_id,
                    resource=resource,
                    success=True,
                )
                return result
            except Exception as e:
                audit_logger.log(
                    operation=operation,
                    user_id=user_id,
                    resource=resource,
                    success=False,
                    severity=AuditSeverity.ERROR,
                    error_message=str(e),
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            audit_logger = get_audit_logger()
            user_id = get_user_id(*args, **kwargs) if get_user_id else "unknown"
            
            try:
                result = func(*args, **kwargs)
                audit_logger.log(
                    operation=operation,
                    user_id=user_id,
                    resource=resource,
                    success=True,
                )
                return result
            except Exception as e:
                audit_logger.log(
                    operation=operation,
                    user_id=user_id,
                    resource=resource,
                    success=False,
                    severity=AuditSeverity.ERROR,
                    error_message=str(e),
                )
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator

"""
Logging Setup - Centralized logging configuration

Sets up structured logging with file rotation and proper log levels.
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

# Add TRACE level (lower than DEBUG)
TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")


def trace(self, message, *args, **kws):
    """Log a message with severity 'TRACE'."""
    if self.isEnabledFor(TRACE_LEVEL):
        self._log(TRACE_LEVEL, message, args, **kws)


# Add trace method to Logger class
logging.Logger.trace = trace


class SensitiveDataFilter(logging.Filter):
    """
    Filter to prevent logging of sensitive information.
    
    Removes or masks sensitive data from log messages including:
    - Passwords
    - Private keys
    - API keys
    - Mnemonics
    - Authorization tokens
    """
    
    # Patterns to detect sensitive data
    SENSITIVE_PATTERNS = [
        r'password["\']?\s*[:=]\s*["\']?([^"\']+)',
        r'private_key["\']?\s*[:=]\s*["\']?([^"\']+)',
        r'api_key["\']?\s*[:=]\s*["\']?([^"\']+)',
        r'mnemonic["\']?\s*[:=]\s*["\']?([^"\']+)',
        r'authorization["\']?\s*[:=]\s*["\']?([^"\']+)',
        r'secret["\']?\s*[:=]\s*["\']?([^"\']+)',
        r'token["\']?\s*[:=]\s*["\']?([^"\']+)',
    ]
    
    def filter(self, record):
        """Filter log records to mask sensitive data."""
        import re
        
        if hasattr(record, 'msg') and record.msg:
            message = str(record.msg)
            
            # Mask sensitive patterns
            for pattern in self.SENSITIVE_PATTERNS:
                message = re.sub(pattern, lambda m: m.group(0).split('=')[0] + '=***MASKED***', message, flags=re.IGNORECASE)
            
            # Also check args for sensitive data
            if hasattr(record, 'args') and record.args:
                new_args = []
                for arg in record.args:
                    if isinstance(arg, str):
                        for pattern in self.SENSITIVE_PATTERNS:
                            arg = re.sub(pattern, lambda m: m.group(0).split('=')[0] + '=***MASKED***', arg, flags=re.IGNORECASE)
                    new_args.append(arg)
                record.args = tuple(new_args)
            
            record.msg = message
        
        return True


def setup_logging(
    log_dir: Optional[str] = None,
    log_level: str = "INFO",
    enable_file_logging: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """
    Set up application-wide logging.
    
    Args:
        log_dir: Directory for log files (default: ~/.r3mes/logs)
        log_level: Logging level (TRACE, DEBUG, INFO, WARNING, ERROR)
        enable_file_logging: Whether to enable file logging
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
    """
    # Check debug config for log level override
    try:
        from .debug_config import get_debug_config
        debug_config = get_debug_config()
        if debug_config.enabled and debug_config.is_backend_enabled() and debug_config.logging:
            log_level = debug_config.log_level
    except ImportError:
        # Debug config not available, use provided log_level
        pass
    
    # Determine log directory
    if log_dir is None:
        log_dir = os.getenv("R3MES_LOG_DIR", str(Path.home() / ".r3mes" / "logs"))
    
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    # Map log level strings to logging constants
    level_map = {
        "TRACE": TRACE_LEVEL,
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    numeric_level = level_map.get(log_level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler with colored output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # Include trace_id and span_id if available
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [trace_id=%(trace_id)s] [span_id=%(span_id)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        defaults={'trace_id': '-', 'span_id': '-'}
    )
    console_handler.setFormatter(console_format)
    # Add sensitive data filter to console handler
    console_handler.addFilter(SensitiveDataFilter())
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if enable_file_logging:
        log_file = log_path / "r3mes_backend.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # File logs everything
        # Include trace_id and span_id if available
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [trace_id=%(trace_id)s] [span_id=%(span_id)s] - %(pathname)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            defaults={'trace_id': '-', 'span_id': '-'}
        )
        file_handler.setFormatter(file_format)
        # Add sensitive data filter to file handler
        file_handler.addFilter(SensitiveDataFilter())
        root_logger.addHandler(file_handler)
        
        # Error log file (only errors)
        error_log_file = log_path / "r3mes_backend_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_format)
        # Add sensitive data filter to error handler
        error_handler.addFilter(SensitiveDataFilter())
        root_logger.addHandler(error_handler)
    
    logging.info(f"Logging initialized (level: {log_level}, file: {enable_file_logging})")
    logging.info(f"Log rotation: max_bytes={max_bytes}, backup_count={backup_count}")
    logging.info("Sensitive data filtering enabled (passwords, keys, tokens will be masked)")


"""
Structured logging utility for R3MES Miner Engine.

Provides JSON-formatted logs suitable for production monitoring systems.
"""

import logging
import logging.handlers
import json
import sys
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

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


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs JSON logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data)


def setup_logger(
    name: str,
    level: int = logging.INFO,
    use_json: bool = False,
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB default
    backup_count: int = 5,  # Keep 5 backup files
) -> logging.Logger:
    """
    Set up a logger with structured output.
    
    Args:
        name: Logger name
        level: Logging level (default: INFO). Can be TRACE_LEVEL, logging.DEBUG, etc.
        use_json: Whether to use JSON formatting (default: False)
        log_file: Optional file path for file logging
        max_bytes: Maximum log file size before rotation (default: 10MB)
        backup_count: Number of backup log files to keep (default: 5)
    
    Returns:
        Configured logger instance
    """
    # Check debug config for log level override
    try:
        from r3mes.debug_config import get_debug_config
        debug_config = get_debug_config()
        if debug_config.enabled and debug_config.is_miner_enabled() and debug_config.logging:
            # Map debug config log level to logging constant
            level_map = {
                "TRACE": TRACE_LEVEL,
                "DEBUG": logging.DEBUG,
                "INFO": logging.INFO,
                "WARNING": logging.WARNING,
                "WARN": logging.WARNING,
                "ERROR": logging.ERROR,
            }
            level = level_map.get(debug_config.log_level, level)
            
            # Use JSON format if configured
            if debug_config.log_format == "json":
                use_json = True
            
            # Use debug config log file if specified
            if debug_config.log_file:
                log_file = debug_config.log_file
    except ImportError:
        # Debug config not available, use provided parameters
        pass
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    if use_json:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    console_handler.setFormatter(formatter)
    # Add sensitive data filter to console handler
    console_handler.addFilter(SensitiveDataFilter())
    logger.addHandler(console_handler)
    
    # File handler with rotation (if specified)
    if log_file:
        # Create directory if it doesn't exist
        log_file_path = Path(log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use RotatingFileHandler for log rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        # Add sensitive data filter to file handler
        file_handler.addFilter(SensitiveDataFilter())
        logger.addHandler(file_handler)
    
    return logger


# Default logger instance
default_logger = setup_logger("r3mes.miner", use_json=False)


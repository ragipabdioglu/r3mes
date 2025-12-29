"""
Logging Configuration for Production

Configures structured logging with file rotation and log aggregation support.
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_file_logging: bool = True
) -> None:
    """
    Setup logging configuration for production.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        enable_file_logging: Enable file logging (default: True)
    """
    # Get log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler (always enabled)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if enabled)
    if enable_file_logging and log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Production-optimized rotation settings
        is_production = os.getenv("R3MES_ENV", "development").lower() == "production"
        
        if is_production:
            # Production: configurable via environment variables
            max_bytes = int(os.getenv("BACKEND_LOG_MAX_BYTES_PRODUCTION", str(100 * 1024 * 1024)))  # Default: 100 MB
            backup_count = int(os.getenv("BACKEND_LOG_BACKUP_COUNT_PRODUCTION", "10"))
        else:
            # Development: configurable via environment variables
            max_bytes = int(os.getenv("BACKEND_LOG_MAX_BYTES_DEVELOPMENT", str(10 * 1024 * 1024)))  # Default: 10 MB
            backup_count = int(os.getenv("BACKEND_LOG_BACKUP_COUNT_DEVELOPMENT", "5"))
        
        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        
        # Structured format for file logging (JSON-like for easier parsing)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Configure third-party loggers
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    # Production: Reduce noise from asyncpg
    logging.getLogger("asyncpg").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


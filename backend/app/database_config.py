"""
Database Configuration - Connection pool management for PostgreSQL
"""

import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def read_secret_file(file_path: str) -> Optional[str]:
    """Read secret from file (Docker secrets support)."""
    if not file_path or not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r') as f:
            return f.read().strip()
    except Exception as e:
        logger.warning(f"Failed to read secret file {file_path}: {e}")
        return None


class DatabaseConfig:
    """Database configuration manager."""
    
    def __init__(self):
        # Check for Docker secrets (POSTGRES_PASSWORD_FILE)
        postgres_password_file = os.getenv("POSTGRES_PASSWORD_FILE")
        postgres_password = None
        
        if postgres_password_file:
            postgres_password = read_secret_file(postgres_password_file)
            if postgres_password:
                # Construct DATABASE_URL from components
                postgres_user = os.getenv("POSTGRES_USER", "r3mes")
                postgres_db = os.getenv("POSTGRES_DB", "r3mes")
                postgres_host = os.getenv("POSTGRES_HOST", "postgres")
                postgres_port = os.getenv("POSTGRES_PORT", "5432")
                self.database_url = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
                logger.info("Using Docker secrets for PostgreSQL password")
            else:
                logger.warning(f"POSTGRES_PASSWORD_FILE specified but file not readable: {postgres_password_file}")
                self.database_url = os.getenv("DATABASE_URL")
        else:
            self.database_url = os.getenv("DATABASE_URL")
        
        self.database_type: str = os.getenv("DATABASE_TYPE", "sqlite").lower()
        
        # Production pool settings (higher for production)
        is_production = os.getenv("R3MES_ENV", "development").lower() == "production"
        if is_production:
            default_min_size = 10
            default_max_size = 50
        else:
            default_min_size = 5
            default_max_size = 20
        
        self.pool_min_size: int = int(os.getenv("DATABASE_POOL_MIN_SIZE", str(default_min_size)))
        self.pool_max_size: int = int(os.getenv("DATABASE_POOL_MAX_SIZE", str(default_max_size)))
        
        # SQLite fallback
        if not self.database_url and self.database_type == "sqlite":
            self.database_path = os.getenv("DATABASE_PATH", "backend/database.db")
        elif not self.database_url:
            raise ValueError("DATABASE_URL environment variable is required for PostgreSQL")
    
    def get_connection_string(self) -> str:
        """Get database connection string."""
        if self.database_type == "postgresql":
            return self.database_url
        else:
            return self.database_path
    
    def is_postgresql(self) -> bool:
        """Check if using PostgreSQL."""
        return self.database_type == "postgresql"
    
    def is_sqlite(self) -> bool:
        """Check if using SQLite."""
        return self.database_type == "sqlite"


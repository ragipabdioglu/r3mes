"""
Database Query Optimization

Provides query analysis, index optimization, and performance tuning utilities.
"""

import logging
from typing import List, Dict, Any
import asyncpg

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """Analyze and optimize database queries."""
    
    @staticmethod
    async def analyze_query(conn: asyncpg.Connection, query: str) -> Dict[str, Any]:
        """
        Analyze a query using EXPLAIN ANALYZE.
        
        Args:
            conn: Database connection
            query: SQL query to analyze
            
        Returns:
            Analysis results
        """
        explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"
        result = await conn.fetchval(explain_query)
        return result[0] if result else {}
    
    @staticmethod
    async def get_slow_queries(conn: asyncpg.Connection, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get slow queries from pg_stat_statements.
        
        Requires pg_stat_statements extension.
        """
        try:
            # Enable extension if not already enabled
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements")
            
            query = """
                SELECT 
                    query,
                    calls,
                    total_exec_time,
                    mean_exec_time,
                    max_exec_time,
                    stddev_exec_time
                FROM pg_stat_statements
                ORDER BY mean_exec_time DESC
                LIMIT $1
            """
            
            return await conn.fetch(query, limit)
        except Exception as e:
            logger.warning(f"Failed to get slow queries: {e}")
            return []


class IndexAuditor:
    """Audit and optimize database indexes."""
    
    @staticmethod
    async def get_unused_indexes(conn: asyncpg.Connection) -> List[Dict[str, Any]]:
        """
        Find unused indexes that can be dropped.
        
        Requires pg_stat_statements extension.
        """
        try:
            query = """
                SELECT
                    schemaname,
                    tablename,
                    indexname,
                    idx_scan as index_scans,
                    pg_size_pretty(pg_relation_size(indexrelid)) as index_size
                FROM pg_stat_user_indexes
                WHERE idx_scan = 0
                AND indexrelid NOT IN (
                    SELECT conindid FROM pg_constraint
                )
                ORDER BY pg_relation_size(indexrelid) DESC
            """
            
            return await conn.fetch(query)
        except Exception as e:
            logger.warning(f"Failed to get unused indexes: {e}")
            return []
    
    @staticmethod
    async def get_missing_indexes(conn: asyncpg.Connection) -> List[Dict[str, Any]]:
        """
        Find tables that might benefit from indexes.
        """
        try:
            query = """
                SELECT
                    schemaname,
                    tablename,
                    seq_scan,
                    seq_tup_read,
                    idx_scan,
                    seq_tup_read / seq_scan as avg_seq_read
                FROM pg_stat_user_tables
                WHERE seq_scan > 0
                AND seq_tup_read / seq_scan > 1000
                ORDER BY seq_tup_read DESC
            """
            
            return await conn.fetch(query)
        except Exception as e:
            logger.warning(f"Failed to get missing indexes: {e}")
            return []


async def apply_indexes_postgresql(conn: asyncpg.Connection):
    """
    Apply recommended indexes for PostgreSQL.
    
    This function is called during database initialization.
    """
    indexes = [
        # Composite indexes for time-based queries
        "CREATE INDEX IF NOT EXISTS idx_mining_stats_wallet_recorded ON mining_stats(wallet_address, recorded_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_earnings_history_wallet_recorded ON earnings_history(wallet_address, recorded_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_hashrate_history_wallet_recorded ON hashrate_history(wallet_address, recorded_at DESC)",
        
        # Partial index for active API keys
        "CREATE INDEX IF NOT EXISTS idx_api_keys_wallet_active ON api_keys(wallet_address, is_active) WHERE is_active = TRUE",
        
        # Blockchain events indexes
        "CREATE INDEX IF NOT EXISTS idx_blockchain_events_type_height ON blockchain_events(event_type, block_height DESC)",
    ]
    
    for index_sql in indexes:
        try:
            await conn.execute(index_sql)
        except Exception as e:
            logger.warning(f"Failed to create index: {e}")


def get_sqlite_index_queries() -> List[str]:
    """
    Get recommended index queries for SQLite.
    
    Returns:
        List of CREATE INDEX statements
    """
    return [
        "CREATE INDEX IF NOT EXISTS idx_users_is_miner ON users(is_miner)",
        "CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at)",
        "CREATE INDEX IF NOT EXISTS idx_mining_stats_wallet ON mining_stats(wallet_address)",
        "CREATE INDEX IF NOT EXISTS idx_mining_stats_recorded_at ON mining_stats(recorded_at)",
        "CREATE INDEX IF NOT EXISTS idx_mining_stats_wallet_recorded ON mining_stats(wallet_address, recorded_at)",
        "CREATE INDEX IF NOT EXISTS idx_earnings_history_wallet ON earnings_history(wallet_address)",
        "CREATE INDEX IF NOT EXISTS idx_earnings_history_recorded_at ON earnings_history(recorded_at)",
        "CREATE INDEX IF NOT EXISTS idx_earnings_history_wallet_recorded ON earnings_history(wallet_address, recorded_at)",
        "CREATE INDEX IF NOT EXISTS idx_hashrate_history_wallet ON hashrate_history(wallet_address)",
        "CREATE INDEX IF NOT EXISTS idx_hashrate_history_recorded_at ON hashrate_history(recorded_at)",
        "CREATE INDEX IF NOT EXISTS idx_hashrate_history_wallet_recorded ON hashrate_history(wallet_address, recorded_at)",
        "CREATE INDEX IF NOT EXISTS idx_api_keys_wallet ON api_keys(wallet_address)",
        "CREATE INDEX IF NOT EXISTS idx_api_keys_is_active ON api_keys(is_active)",
        "CREATE INDEX IF NOT EXISTS idx_api_keys_expires_at ON api_keys(expires_at)",
    ]

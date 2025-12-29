"""
Migration script to migrate data from SQLite to PostgreSQL.

Usage:
    python -m backend.app.migrations.migrate_sqlite_to_postgresql \
        --sqlite-path backend/database.db \
        --postgresql-url postgresql://user:password@localhost:5432/r3mes
"""

import argparse
import sqlite3
import asyncpg
import asyncio
import logging
from typing import Dict, List, Any
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def migrate_table(
    sqlite_conn: sqlite3.Connection,
    pg_pool: asyncpg.Pool,
    table_name: str,
    columns: List[str],
    primary_key: str = "id"
) -> int:
    """
    Migrate a single table from SQLite to PostgreSQL.
    
    Args:
        sqlite_conn: SQLite connection
        pg_pool: PostgreSQL connection pool
        columns: List of column names to migrate
        primary_key: Primary key column name
        
    Returns:
        Number of rows migrated
    """
    logger.info(f"Migrating table: {table_name}")
    
    # Read data from SQLite
    cursor = sqlite_conn.cursor()
    cursor.execute(f"SELECT {', '.join(columns)} FROM {table_name}")
    rows = cursor.fetchall()
    
    if not rows:
        logger.info(f"  No data to migrate for {table_name}")
        return 0
    
    # Prepare insert statement
    placeholders = ', '.join([f'${i+1}' for i in range(len(columns))])
    insert_sql = f"""
        INSERT INTO {table_name} ({', '.join(columns)})
        VALUES ({placeholders})
        ON CONFLICT ({primary_key}) DO NOTHING
    """
    
    # Insert data into PostgreSQL
    async with pg_pool.acquire() as conn:
        count = 0
        for row in rows:
            try:
                await conn.execute(insert_sql, *row)
                count += 1
            except Exception as e:
                logger.warning(f"  Failed to insert row into {table_name}: {e}")
                continue
    
    logger.info(f"  Migrated {count} rows from {table_name}")
    return count


async def migrate_database(sqlite_path: str, postgresql_url: str, dry_run: bool = False):
    """
    Migrate all data from SQLite to PostgreSQL.
    
    Args:
        sqlite_path: Path to SQLite database file
        postgresql_url: PostgreSQL connection URL
        dry_run: If True, only validate without migrating
    """
    logger.info(f"Starting migration from SQLite ({sqlite_path}) to PostgreSQL")
    
    if dry_run:
        logger.info("DRY RUN MODE - No data will be migrated")
    
    # Connect to SQLite
    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_conn.row_factory = sqlite3.Row
    
    # Connect to PostgreSQL
    pg_pool = await asyncpg.create_pool(postgresql_url, min_size=1, max_size=5)
    
    try:
        # Verify PostgreSQL tables exist
        async with pg_pool.acquire() as conn:
            tables = await conn.fetch("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            table_names = [t['table_name'] for t in tables]
            logger.info(f"PostgreSQL tables found: {', '.join(table_names)}")
        
        if dry_run:
            logger.info("Dry run complete - validation successful")
            return
        
        # Migrate each table
        total_rows = 0
        
        # Users table
        if 'users' in table_names:
            total_rows += await migrate_table(
                sqlite_conn, pg_pool, 'users',
                ['wallet_address', 'credits', 'is_miner', 'last_mining_time', 'created_at'],
                'wallet_address'
            )
        
        # Mining stats table
        if 'mining_stats' in table_names:
            total_rows += await migrate_table(
                sqlite_conn, pg_pool, 'mining_stats',
                ['wallet_address', 'hashrate', 'gpu_temperature', 'blocks_found', 
                 'uptime_percentage', 'network_difficulty', 'recorded_at'],
                'id'
            )
        
        # Earnings history table
        if 'earnings_history' in table_names:
            total_rows += await migrate_table(
                sqlite_conn, pg_pool, 'earnings_history',
                ['wallet_address', 'earnings', 'recorded_at'],
                'id'
            )
        
        # Hashrate history table
        if 'hashrate_history' in table_names:
            total_rows += await migrate_table(
                sqlite_conn, pg_pool, 'hashrate_history',
                ['wallet_address', 'hashrate', 'recorded_at'],
                'id'
            )
        
        # API keys table
        if 'api_keys' in table_names:
            total_rows += await migrate_table(
                sqlite_conn, pg_pool, 'api_keys',
                ['api_key_hash', 'wallet_address', 'name', 'is_active', 
                 'last_used', 'created_at', 'expires_at'],
                'id'
            )
        
        logger.info(f"Migration complete! Migrated {total_rows} total rows")
        
    finally:
        sqlite_conn.close()
        await pg_pool.close()


def main():
    parser = argparse.ArgumentParser(description='Migrate SQLite database to PostgreSQL')
    parser.add_argument('--sqlite-path', required=True, help='Path to SQLite database file')
    parser.add_argument('--postgresql-url', required=True, help='PostgreSQL connection URL')
    parser.add_argument('--dry-run', action='store_true', help='Validate without migrating')
    
    args = parser.parse_args()
    
    asyncio.run(migrate_database(args.sqlite_path, args.postgresql_url, args.dry_run))


if __name__ == '__main__':
    main()


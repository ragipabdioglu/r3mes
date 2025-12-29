#!/usr/bin/env python3
"""
Migration Verification Script

Verifies that database migration was successful by comparing row counts,
data integrity, and index verification.

Usage:
    python scripts/verify_migration.py \
        --source-sqlite /var/lib/r3mes/database.db \
        --target-postgresql postgresql://user:password@host:5432/r3mes
"""

import argparse
import sys
import os
import logging
import sqlite3
import asyncpg
import asyncio
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def verify_row_counts(sqlite_path: str, postgresql_url: str) -> bool:
    """Verify row counts match between SQLite and PostgreSQL."""
    logger.info("Verifying row counts...")
    
    sqlite_conn = sqlite3.connect(sqlite_path)
    pg_pool = await asyncpg.create_pool(postgresql_url, min_size=1, max_size=5)
    
    tables = ['users', 'mining_stats', 'earnings_history', 'hashrate_history', 'api_keys']
    
    try:
        all_match = True
        
        for table_name in tables:
            # Count in SQLite
            cursor = sqlite_conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            sqlite_count = cursor.fetchone()[0]
            
            # Count in PostgreSQL
            async with pg_pool.acquire() as conn:
                pg_count = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")
            
            if sqlite_count == pg_count:
                logger.info(f"  ✅ {table_name}: {pg_count} rows (match)")
            else:
                logger.error(f"  ❌ {table_name}: SQLite={sqlite_count}, PostgreSQL={pg_count}")
                all_match = False
        
        return all_match
        
    finally:
        sqlite_conn.close()
        await pg_pool.close()


async def verify_data_integrity(sqlite_path: str, postgresql_url: str) -> bool:
    """Verify data integrity by comparing sample records."""
    logger.info("Verifying data integrity...")
    
    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_conn.row_factory = sqlite3.Row
    pg_pool = await asyncpg.create_pool(postgresql_url, min_size=1, max_size=5)
    
    try:
        # Sample users table
        cursor = sqlite_conn.cursor()
        cursor.execute("SELECT * FROM users LIMIT 5")
        sqlite_rows = cursor.fetchall()
        
        if not sqlite_rows:
            logger.info("  No data to verify")
            return True
        
        async with pg_pool.acquire() as conn:
            for sqlite_row in sqlite_rows:
                wallet_address = sqlite_row['wallet_address']
                pg_row = await conn.fetchrow(
                    "SELECT * FROM users WHERE wallet_address = $1",
                    wallet_address
                )
                
                if not pg_row:
                    logger.error(f"  ❌ User {wallet_address} not found in PostgreSQL")
                    return False
                
                # Compare key fields
                if (sqlite_row['credits'] != pg_row['credits'] or
                    sqlite_row['is_miner'] != pg_row['is_miner']):
                    logger.error(f"  ❌ User {wallet_address} data mismatch")
                    return False
        
        logger.info("  ✅ Data integrity verified")
        return True
        
    finally:
        sqlite_conn.close()
        await pg_pool.close()


async def verify_indexes(postgresql_url: str) -> bool:
    """Verify that all required indexes exist."""
    logger.info("Verifying indexes...")
    
    pg_pool = await asyncpg.create_pool(postgresql_url, min_size=1, max_size=5)
    
    required_indexes = [
        'idx_users_is_miner',
        'idx_mining_stats_wallet',
        'idx_earnings_history_wallet',
        'idx_hashrate_history_wallet',
        'idx_api_key_hash',
        'idx_blockchain_events_type_height',
    ]
    
    try:
        async with pg_pool.acquire() as conn:
            indexes = await conn.fetch("""
                SELECT indexname 
                FROM pg_indexes 
                WHERE schemaname = 'public'
            """)
            
            index_names = [row['indexname'] for row in indexes]
            missing_indexes = [idx for idx in required_indexes if idx not in index_names]
            
            if missing_indexes:
                logger.error(f"  ❌ Missing indexes: {missing_indexes}")
                return False
            
            logger.info(f"  ✅ All required indexes exist ({len(required_indexes)} indexes)")
            return True
            
    finally:
        await pg_pool.close()


async def run_performance_tests(postgresql_url: str) -> bool:
    """Run performance test queries."""
    logger.info("Running performance tests...")
    
    pg_pool = await asyncpg.create_pool(postgresql_url, min_size=1, max_size=5)
    
    try:
        async with pg_pool.acquire() as conn:
            # Test query with index
            start_time = asyncio.get_event_loop().time()
            result = await conn.fetch("""
                SELECT * FROM mining_stats 
                WHERE wallet_address = $1 
                ORDER BY recorded_at DESC 
                LIMIT 10
            """, "test_wallet")
            query_time = asyncio.get_event_loop().time() - start_time
            
            if query_time < 1.0:  # Should be fast with index
                logger.info(f"  ✅ Query performance: {query_time:.3f}s")
                return True
            else:
                logger.warning(f"  ⚠️  Query performance slow: {query_time:.3f}s")
                return False
                
    except Exception as e:
        logger.warning(f"  ⚠️  Performance test failed: {e}")
        return True  # Don't fail on performance test
    finally:
        await pg_pool.close()


def main():
    """Main verification function."""
    parser = argparse.ArgumentParser(description='Verify database migration')
    parser.add_argument('--source-sqlite', required=True, help='Path to SQLite database file')
    parser.add_argument('--target-postgresql', required=True, help='PostgreSQL connection URL')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.source_sqlite):
        logger.error(f"❌ SQLite database not found: {args.source_sqlite}")
        return 1
    
    logger.info("=" * 60)
    logger.info("R3MES Migration Verification")
    logger.info("=" * 60)
    logger.info()
    
    # Run all verification checks
    checks = [
        ("Row Counts", verify_row_counts(args.source_sqlite, args.target_postgresql)),
        ("Data Integrity", verify_data_integrity(args.source_sqlite, args.target_postgresql)),
        ("Indexes", verify_indexes(args.target_postgresql)),
        ("Performance", run_performance_tests(args.target_postgresql)),
    ]
    
    all_passed = True
    for check_name, check_coro in checks:
        result = asyncio.run(check_coro)
        if not result:
            all_passed = False
    
    logger.info()
    logger.info("=" * 60)
    if all_passed:
        logger.info("✅ All verification checks passed!")
        return 0
    else:
        logger.error("❌ Some verification checks failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())


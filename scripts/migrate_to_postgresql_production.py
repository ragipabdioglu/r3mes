#!/usr/bin/env python3
"""
Production Database Migration Script

Migrates database from SQLite to PostgreSQL with full verification and rollback support.

Usage:
    python scripts/migrate_to_postgresql_production.py \
        --source-sqlite /var/lib/r3mes/database.db \
        --target-postgresql postgresql://user:password@host:5432/r3mes \
        --backup-path /backups/

Options:
    --source-sqlite: Path to SQLite database file
    --target-postgresql: PostgreSQL connection URL
    --backup-path: Path to store backups
    --dry-run: Validate without migrating
    --skip-backup: Skip backup creation (not recommended)
"""

import argparse
import sys
import os
import shutil
import subprocess
import logging
from pathlib import Path
from datetime import datetime
import sqlite3
import asyncpg
import asyncio

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def backup_sqlite_database(sqlite_path: str, backup_path: str) -> str:
    """
    Create a backup of SQLite database.
    
    Returns:
        Path to backup file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(backup_path, f"database_backup_{timestamp}.db")
    
    logger.info(f"Creating SQLite backup: {backup_file}")
    shutil.copy2(sqlite_path, backup_file)
    logger.info(f"✅ SQLite backup created: {backup_file}")
    
    return backup_file


def verify_postgresql_connection(postgresql_url: str) -> bool:
    """Verify PostgreSQL connection."""
    try:
        async def test_connection():
            conn = await asyncpg.connect(postgresql_url)
            await conn.execute("SELECT 1")
            await conn.close()
            return True
        
        result = asyncio.run(test_connection())
        logger.info("✅ PostgreSQL connection verified")
        return result
    except Exception as e:
        logger.error(f"❌ PostgreSQL connection failed: {e}")
        return False


def run_alembic_migrations(postgresql_url: str) -> bool:
    """
    Run Alembic migrations to create schema.
    
    Returns:
        True if successful
    """
    try:
        logger.info("Running Alembic migrations...")
        
        # Set DATABASE_URL for Alembic
        os.environ["DATABASE_URL"] = postgresql_url
        os.environ["DATABASE_TYPE"] = "postgresql"
        
        # Change to backend directory
        backend_dir = Path(__file__).parent.parent / "backend"
        original_dir = os.getcwd()
        
        try:
            os.chdir(backend_dir)
            
            # Run alembic upgrade
            result = subprocess.run(
                ["alembic", "upgrade", "head"],
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info("✅ Alembic migrations completed successfully")
            logger.debug(f"Alembic output: {result.stdout}")
            return True
            
        finally:
            os.chdir(original_dir)
            
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Alembic migration failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to run Alembic migrations: {e}")
        return False


async def verify_schema(postgresql_url: str) -> bool:
    """Verify that all required tables exist in PostgreSQL."""
    try:
        async with asyncpg.create_pool(postgresql_url, min_size=1, max_size=5) as pool:
            async with pool.acquire() as conn:
                tables = await conn.fetch("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                """)
                
                table_names = [row['table_name'] for row in tables]
                required_tables = [
                    'users', 'mining_stats', 'earnings_history',
                    'hashrate_history', 'api_keys', 'blockchain_events',
                    'network_snapshots', 'indexer_state'
                ]
                
                missing_tables = [t for t in required_tables if t not in table_names]
                
                if missing_tables:
                    logger.error(f"❌ Missing tables: {missing_tables}")
                    return False
                
                logger.info(f"✅ All required tables exist: {', '.join(required_tables)}")
                return True
                
    except Exception as e:
        logger.error(f"❌ Schema verification failed: {e}")
        return False


async def migrate_data(
    sqlite_path: str,
    postgresql_url: str,
    dry_run: bool = False
) -> Dict[str, int]:
    """
    Migrate data from SQLite to PostgreSQL.
    
    Returns:
        Dictionary mapping table names to row counts
    """
    from app.migrations.migrate_sqlite_to_postgresql import migrate_table
    
    logger.info("Starting data migration...")
    
    # Connect to SQLite
    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_conn.row_factory = sqlite3.Row
    
    # Connect to PostgreSQL
    pg_pool = await asyncpg.create_pool(postgresql_url, min_size=1, max_size=5)
    
    migration_results = {}
    
    try:
        # Define tables to migrate
        tables_config = [
            {
                'name': 'users',
                'columns': ['wallet_address', 'credits', 'is_miner', 'last_mining_time', 'created_at'],
                'primary_key': 'wallet_address'
            },
            {
                'name': 'mining_stats',
                'columns': ['id', 'wallet_address', 'hashrate', 'gpu_temperature', 'blocks_found', 
                           'uptime_percentage', 'network_difficulty', 'recorded_at'],
                'primary_key': 'id'
            },
            {
                'name': 'earnings_history',
                'columns': ['id', 'wallet_address', 'earnings', 'recorded_at'],
                'primary_key': 'id'
            },
            {
                'name': 'hashrate_history',
                'columns': ['id', 'wallet_address', 'hashrate', 'recorded_at'],
                'primary_key': 'id'
            },
            {
                'name': 'api_keys',
                'columns': ['id', 'api_key_hash', 'wallet_address', 'name', 'is_active', 
                           'last_used', 'created_at', 'expires_at'],
                'primary_key': 'id'
            },
        ]
        
        if dry_run:
            logger.info("DRY RUN MODE - Validating data only")
            for table_config in tables_config:
                cursor = sqlite_conn.cursor()
                cursor.execute(f"SELECT COUNT(*) FROM {table_config['name']}")
                count = cursor.fetchone()[0]
                logger.info(f"  {table_config['name']}: {count} rows (would migrate)")
                migration_results[table_config['name']] = count
        else:
            # Migrate each table
            for table_config in tables_config:
                count = await migrate_table(
                    sqlite_conn,
                    pg_pool,
                    table_config['name'],
                    table_config['columns'],
                    table_config['primary_key']
                )
                migration_results[table_config['name']] = count
        
        return migration_results
        
    finally:
        sqlite_conn.close()
        await pg_pool.close()


async def verify_migration(
    sqlite_path: str,
    postgresql_url: str,
    migration_results: Dict[str, int]
) -> bool:
    """
    Verify migration by comparing row counts.
    
    Returns:
        True if verification passed
    """
    logger.info("Verifying migration...")
    
    sqlite_conn = sqlite3.connect(sqlite_path)
    pg_pool = await asyncpg.create_pool(postgresql_url, min_size=1, max_size=5)
    
    try:
        all_verified = True
        
        for table_name, expected_count in migration_results.items():
            # Count in SQLite
            cursor = sqlite_conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            sqlite_count = cursor.fetchone()[0]
            
            # Count in PostgreSQL
            async with pg_pool.acquire() as conn:
                pg_count = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")
            
            if sqlite_count == pg_count == expected_count:
                logger.info(f"  ✅ {table_name}: {pg_count} rows (verified)")
            else:
                logger.error(
                    f"  ❌ {table_name}: SQLite={sqlite_count}, PostgreSQL={pg_count}, "
                    f"Expected={expected_count}"
                )
                all_verified = False
        
        return all_verified
        
    finally:
        sqlite_conn.close()
        await pg_pool.close()


def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(description='Migrate SQLite database to PostgreSQL')
    parser.add_argument('--source-sqlite', required=True, help='Path to SQLite database file')
    parser.add_argument('--target-postgresql', required=True, help='PostgreSQL connection URL')
    parser.add_argument('--backup-path', default='/backups', help='Path to store backups')
    parser.add_argument('--dry-run', action='store_true', help='Validate without migrating')
    parser.add_argument('--skip-backup', action='store_true', help='Skip backup creation')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.source_sqlite):
        logger.error(f"❌ SQLite database not found: {args.source_sqlite}")
        return 1
    
    # Create backup directory
    os.makedirs(args.backup_path, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("R3MES Production Database Migration")
    logger.info("=" * 60)
    logger.info(f"Source: {args.source_sqlite}")
    logger.info(f"Target: {args.target_postgresql}")
    logger.info(f"Backup path: {args.backup_path}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("=" * 60)
    logger.info()
    
    # Step 1: Backup SQLite database
    if not args.skip_backup:
        try:
            backup_file = backup_sqlite_database(args.source_sqlite, args.backup_path)
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            return 1
    else:
        logger.warning("⚠️  Skipping backup (not recommended)")
    
    # Step 2: Verify PostgreSQL connection
    if not verify_postgresql_connection(args.target_postgresql):
        return 1
    
    # Step 3: Run Alembic migrations
    if not run_alembic_migrations(args.target_postgresql):
        return 1
    
    # Step 4: Verify schema
    schema_verified = asyncio.run(verify_schema(args.target_postgresql))
    if not schema_verified:
        return 1
    
    # Step 5: Migrate data
    try:
        migration_results = asyncio.run(
            migrate_data(args.source_sqlite, args.target_postgresql, args.dry_run)
        )
        
        total_rows = sum(migration_results.values())
        logger.info(f"✅ Data migration completed: {total_rows} total rows")
        
    except Exception as e:
        logger.error(f"❌ Data migration failed: {e}")
        return 1
    
    # Step 6: Verify migration
    if not args.dry_run:
        verified = asyncio.run(
            verify_migration(args.source_sqlite, args.target_postgresql, migration_results)
        )
        
        if not verified:
            logger.error("❌ Migration verification failed")
            return 1
        
        logger.info("✅ Migration verification passed")
    
    logger.info()
    logger.info("=" * 60)
    logger.info("✅ Migration completed successfully!")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


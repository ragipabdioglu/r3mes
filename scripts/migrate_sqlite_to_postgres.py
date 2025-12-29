#!/usr/bin/env python3
"""
SQLite to PostgreSQL Migration Script

Migrates data from SQLite database to PostgreSQL database.
"""

import sqlite3
import asyncpg
import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import json
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.database_config import DatabaseConfig


async def migrate_sqlite_to_postgres(
    sqlite_path: str,
    postgres_url: str
):
    """
    Migrate data from SQLite to PostgreSQL.
    
    Args:
        sqlite_path: Path to SQLite database file
        postgres_url: PostgreSQL connection URL
    """
    print(f"📦 Starting migration from {sqlite_path} to PostgreSQL...")
    
    # Connect to SQLite
    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_conn.row_factory = sqlite3.Row
    sqlite_cursor = sqlite_conn.cursor()
    
    # Connect to PostgreSQL
    postgres_pool = await asyncpg.create_pool(postgres_url)
    
    try:
        async with postgres_pool.acquire() as postgres_conn:
            # Migrate users table
            print("📊 Migrating users table...")
            sqlite_cursor.execute("SELECT * FROM users")
            users = sqlite_cursor.fetchall()
            
            for user in users:
                await postgres_conn.execute("""
                    INSERT INTO users (wallet_address, credits, is_miner, last_mining_time, created_at)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (wallet_address) DO UPDATE
                    SET credits = EXCLUDED.credits,
                        is_miner = EXCLUDED.is_miner,
                        last_mining_time = EXCLUDED.last_mining_time
                """, 
                    user['wallet_address'],
                    user['credits'],
                    bool(user['is_miner']),
                    user['last_mining_time'],
                    user['created_at']
                )
            
            print(f"✅ Migrated {len(users)} users")
            
            # Migrate api_keys table
            print("🔑 Migrating API keys table...")
            sqlite_cursor.execute("SELECT * FROM api_keys")
            api_keys = sqlite_cursor.fetchall()
            
            for key in api_keys:
                await postgres_conn.execute("""
                    INSERT INTO api_keys (
                        api_key_hash, wallet_address, name, is_active,
                        last_used, created_at, expires_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (api_key_hash) DO NOTHING
                """,
                    key['api_key_hash'],
                    key['wallet_address'],
                    key['name'],
                    bool(key['is_active']),
                    key['last_used'],
                    key['created_at'],
                    key['expires_at']
                )
            
            print(f"✅ Migrated {len(api_keys)} API keys")
            
            # Migrate mining_stats table
            print("⛏️  Migrating mining stats table...")
            sqlite_cursor.execute("SELECT * FROM mining_stats")
            mining_stats = sqlite_cursor.fetchall()
            
            for stat in mining_stats:
                await postgres_conn.execute("""
                    INSERT INTO mining_stats (
                        wallet_address, hashrate, gpu_temperature, blocks_found,
                        uptime_percentage, network_difficulty, recorded_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                    stat['wallet_address'],
                    stat['hashrate'],
                    stat['gpu_temperature'],
                    stat['blocks_found'],
                    stat['uptime_percentage'],
                    stat['network_difficulty'],
                    stat['recorded_at']
                )
            
            print(f"✅ Migrated {len(mining_stats)} mining stats")
            
            # Migrate earnings_history table
            print("💰 Migrating earnings history table...")
            sqlite_cursor.execute("SELECT * FROM earnings_history")
            earnings = sqlite_cursor.fetchall()
            
            for earning in earnings:
                await postgres_conn.execute("""
                    INSERT INTO earnings_history (wallet_address, earnings, recorded_at)
                    VALUES ($1, $2, $3)
                """,
                    earning['wallet_address'],
                    earning['earnings'],
                    earning['recorded_at']
                )
            
            print(f"✅ Migrated {len(earnings)} earnings records")
            
            # Migrate hashrate_history table
            print("📈 Migrating hashrate history table...")
            sqlite_cursor.execute("SELECT * FROM hashrate_history")
            hashrates = sqlite_cursor.fetchall()
            
            for hashrate in hashrates:
                await postgres_conn.execute("""
                    INSERT INTO hashrate_history (wallet_address, hashrate, recorded_at)
                    VALUES ($1, $2, $3)
                """,
                    hashrate['wallet_address'],
                    hashrate['hashrate'],
                    hashrate['recorded_at']
                )
            
            print(f"✅ Migrated {len(hashrates)} hashrate records")
    
    finally:
        sqlite_conn.close()
        await postgres_pool.close()
    
    print("✅ Migration completed successfully!")


async def main():
    """Main migration function."""
    sqlite_path = os.getenv("SQLITE_DB_PATH", "backend/database.db")
    postgres_url = os.getenv("DATABASE_URL")
    
    if not postgres_url:
        print("❌ Error: DATABASE_URL environment variable is required")
        sys.exit(1)
    
    if not Path(sqlite_path).exists():
        print(f"❌ Error: SQLite database not found at {sqlite_path}")
        sys.exit(1)
    
    await migrate_sqlite_to_postgres(sqlite_path, postgres_url)


if __name__ == "__main__":
    asyncio.run(main())


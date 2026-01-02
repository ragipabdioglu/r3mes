#!/usr/bin/env python3
"""
API Key Migration Script - Plain Text to Hashed Storage

This script migrates existing API keys from plain text storage to hashed storage
for improved security. This is a one-time migration.

CRITICAL: This migration will make existing API keys unusable since we cannot
reverse the hash. Users will need to create new API keys after migration.

Usage:
    python migrate_api_keys_to_hash.py [--db-path /path/to/database.db] [--backup]
"""

import sqlite3
import hashlib
import argparse
import shutil
import os
from datetime import datetime
from pathlib import Path


def backup_database(db_path: str) -> str:
    """Create a backup of the database before migration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{db_path}.backup_{timestamp}"
    shutil.copy2(db_path, backup_path)
    print(f"✅ Database backed up to: {backup_path}")
    return backup_path


def check_existing_schema(cursor) -> dict:
    """Check the current schema of api_keys table."""
    cursor.execute("PRAGMA table_info(api_keys)")
    columns = cursor.fetchall()
    
    schema = {}
    for col in columns:
        schema[col[1]] = col[2]  # column_name: data_type
    
    return schema


def migrate_api_keys_table(db_path: str, create_backup: bool = True):
    """
    Migrate API keys table from plain text to hashed storage.
    
    WARNING: This will invalidate all existing API keys!
    """
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return False
    
    # Create backup
    if create_backup:
        backup_path = backup_database(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check current schema
        schema = check_existing_schema(cursor)
        print(f"📋 Current schema: {list(schema.keys())}")
        
        if 'api_key_hash' in schema:
            print("✅ API keys table already uses hashed storage. No migration needed.")
            return True
        
        if 'api_key' not in schema:
            print("❌ No api_key column found. Cannot migrate.")
            return False
        
        # Get existing API keys (WARNING: This exposes plain text keys temporarily)
        cursor.execute("SELECT id, api_key, wallet_address, name, is_active, last_used, created_at, expires_at FROM api_keys")
        existing_keys = cursor.fetchall()
        
        print(f"📊 Found {len(existing_keys)} existing API keys")
        
        if len(existing_keys) > 0:
            print("⚠️  WARNING: All existing API keys will become invalid after migration!")
            print("⚠️  Users will need to create new API keys.")
            
            response = input("Continue with migration? (yes/no): ").lower().strip()
            if response != 'yes':
                print("❌ Migration cancelled by user")
                return False
        
        # Step 1: Create new table with hashed keys
        cursor.execute("""
            CREATE TABLE api_keys_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                api_key_hash TEXT UNIQUE NOT NULL,
                wallet_address TEXT NOT NULL,
                name TEXT,
                is_active BOOLEAN DEFAULT 1,
                last_used TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                FOREIGN KEY (wallet_address) REFERENCES users(wallet_address)
            )
        """)
        
        # Step 2: Create index
        cursor.execute("""
            CREATE INDEX idx_api_key_hash ON api_keys_new(api_key_hash)
        """)
        
        # Step 3: Migrate data (hash existing keys)
        migrated_count = 0
        for row in existing_keys:
            id_val, api_key, wallet_address, name, is_active, last_used, created_at, expires_at = row
            
            # Hash the existing API key
            api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            try:
                cursor.execute("""
                    INSERT INTO api_keys_new (api_key_hash, wallet_address, name, is_active, last_used, created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (api_key_hash, wallet_address, name, is_active, last_used, created_at, expires_at))
                migrated_count += 1
            except sqlite3.IntegrityError as e:
                print(f"⚠️  Skipping duplicate key for wallet {wallet_address}: {e}")
                continue
        
        # Step 4: Drop old table and rename new table
        cursor.execute("DROP TABLE api_keys")
        cursor.execute("ALTER TABLE api_keys_new RENAME TO api_keys")
        
        # Step 5: Commit changes
        conn.commit()
        
        print(f"✅ Migration completed successfully!")
        print(f"📊 Migrated {migrated_count} API keys")
        print(f"⚠️  All existing API keys are now invalid - users must create new ones")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Migrate API keys to hashed storage")
    parser.add_argument("--db-path", default="./data/database.db", help="Path to SQLite database")
    parser.add_argument("--backup", action="store_true", default=True, help="Create backup before migration")
    parser.add_argument("--no-backup", action="store_false", dest="backup", help="Skip backup creation")
    
    args = parser.parse_args()
    
    print("🔐 API Key Migration Script")
    print("=" * 50)
    print(f"Database: {args.db_path}")
    print(f"Backup: {'Yes' if args.backup else 'No'}")
    print()
    
    success = migrate_api_keys_table(args.db_path, args.backup)
    
    if success:
        print("\n✅ Migration completed successfully!")
        print("\n📋 Next steps:")
        print("1. Restart the backend service")
        print("2. Notify users to create new API keys")
        print("3. Update any automated systems using API keys")
    else:
        print("\n❌ Migration failed!")
        print("Check the error messages above and try again.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
#!/usr/bin/env python3
"""
Database Query Analysis Script

Analyzes slow queries and suggests optimizations.

Usage:
    python scripts/analyze_queries.py
"""

import sys
import os
import asyncio
import asyncpg
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from app.database_optimization import QueryAnalyzer, IndexAuditor


async def main():
    """Main analysis function."""
    database_url = os.getenv("DATABASE_URL")
    
    if not database_url:
        print("❌ DATABASE_URL environment variable is not set")
        return 1
    
    print("🔍 Analyzing database queries...")
    print()
    
    try:
        conn = await asyncpg.connect(database_url)
        
        # Analyze slow queries
        print("📊 Slow Queries:")
        print("-" * 60)
        slow_queries = await QueryAnalyzer.get_slow_queries(conn, limit=10)
        
        if slow_queries:
            for i, query in enumerate(slow_queries, 1):
                print(f"\n{i}. Mean execution time: {query['mean_exec_time']:.2f}ms")
                print(f"   Calls: {query['calls']}")
                print(f"   Query: {query['query'][:100]}...")
        else:
            print("No slow queries found (pg_stat_statements extension may not be enabled)")
        
        print()
        print("📊 Unused Indexes:")
        print("-" * 60)
        unused_indexes = await IndexAuditor.get_unused_indexes(conn)
        
        if unused_indexes:
            for index in unused_indexes:
                print(f"  - {index['schemaname']}.{index['tablename']}.{index['indexname']}")
                print(f"    Size: {index['index_size']}, Scans: {index['index_scans']}")
        else:
            print("No unused indexes found")
        
        print()
        print("📊 Missing Indexes (Potential):")
        print("-" * 60)
        missing_indexes = await IndexAuditor.get_missing_indexes(conn)
        
        if missing_indexes:
            for table in missing_indexes:
                print(f"  - {table['schemaname']}.{table['tablename']}")
                print(f"    Sequential scans: {table['seq_scan']}")
                print(f"    Avg seq read: {table['avg_seq_read']:.0f}")
        else:
            print("No tables with potential missing indexes")
        
        await conn.close()
        
        print()
        print("✅ Analysis completed")
        return 0
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)


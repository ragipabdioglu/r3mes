"""
Database Performance Optimizer

Comprehensive database performance optimization including query optimization,
connection pooling, caching, and monitoring.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from contextlib import asynccontextmanager
import asyncpg
import aiosqlite

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """Query performance metrics."""
    query_hash: str
    query: str
    execution_count: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    last_executed: datetime


@dataclass
class ConnectionPoolMetrics:
    """Connection pool metrics."""
    pool_size: int
    active_connections: int
    idle_connections: int
    waiting_connections: int
    total_connections_created: int
    total_connections_closed: int
    avg_connection_lifetime: float


class DatabasePerformanceOptimizer:
    """
    Comprehensive database performance optimizer.
    
    Features:
    - Query performance monitoring and optimization
    - Connection pool optimization
    - Index analysis and recommendations
    - Cache integration
    - Performance metrics collection
    """
    
    def __init__(self, database_instance):
        """
        Initialize performance optimizer.
        
        Args:
            database_instance: Database instance to optimize
        """
        self.db = database_instance
        self.query_metrics: Dict[str, QueryMetrics] = {}
        self.pool_metrics: Optional[ConnectionPoolMetrics] = None
        self.optimization_enabled = True
        self.monitoring_enabled = True
        
        # Performance thresholds
        self.slow_query_threshold = 0.1  # 100ms
        self.connection_timeout = 30.0
        self.max_pool_size = 20
        self.min_pool_size = 5
        
        # Query cache
        self.query_cache: Dict[str, Tuple[Any, datetime]] = {}
        self.cache_ttl = 300  # 5 minutes
        
        logger.info("Database Performance Optimizer initialized")
    
    async def optimize_connection_pool(self) -> Dict[str, Any]:
        """
        Optimize database connection pool settings.
        
        Returns:
            Optimization results and recommendations
        """
        if not hasattr(self.db, '_db') or not self.db._db:
            return {"status": "skipped", "reason": "No PostgreSQL connection pool"}
        
        try:
            pool = self.db._db.pool
            if not pool:
                return {"status": "skipped", "reason": "No connection pool available"}
            
            # Collect current pool metrics
            current_metrics = ConnectionPoolMetrics(
                pool_size=pool.get_size(),
                active_connections=len([c for c in pool._holders if c._con is not None]),
                idle_connections=len([c for c in pool._holders if c._con is None]),
                waiting_connections=len(pool._queue._queue) if hasattr(pool._queue, '_queue') else 0,
                total_connections_created=getattr(pool, '_created_connections', 0),
                total_connections_closed=getattr(pool, '_closed_connections', 0),
                avg_connection_lifetime=0.0  # Would need tracking
            )
            
            self.pool_metrics = current_metrics
            
            # Analyze pool performance
            recommendations = []
            
            # Check if pool is undersized
            if current_metrics.waiting_connections > 0:
                recommendations.append({
                    "type": "pool_size",
                    "issue": "Connection pool undersized",
                    "current": current_metrics.pool_size,
                    "recommended": min(current_metrics.pool_size * 2, self.max_pool_size),
                    "reason": f"{current_metrics.waiting_connections} connections waiting"
                })
            
            # Check if pool is oversized
            if (current_metrics.idle_connections > current_metrics.active_connections * 2 and
                current_metrics.pool_size > self.min_pool_size):
                recommendations.append({
                    "type": "pool_size",
                    "issue": "Connection pool oversized",
                    "current": current_metrics.pool_size,
                    "recommended": max(current_metrics.active_connections + 2, self.min_pool_size),
                    "reason": f"Too many idle connections: {current_metrics.idle_connections}"
                })
            
            return {
                "status": "completed",
                "metrics": current_metrics,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize connection pool: {e}")
            return {"status": "error", "error": str(e)}
    
    async def analyze_slow_queries(self) -> List[Dict[str, Any]]:
        """
        Analyze slow queries and provide optimization recommendations.
        
        Returns:
            List of slow queries with optimization suggestions
        """
        slow_queries = []
        
        for query_hash, metrics in self.query_metrics.items():
            if metrics.avg_time > self.slow_query_threshold:
                # Analyze query for optimization opportunities
                suggestions = await self._analyze_query_for_optimization(metrics.query)
                
                slow_queries.append({
                    "query_hash": query_hash,
                    "query": metrics.query[:200] + "..." if len(metrics.query) > 200 else metrics.query,
                    "execution_count": metrics.execution_count,
                    "avg_time": metrics.avg_time,
                    "max_time": metrics.max_time,
                    "total_time": metrics.total_time,
                    "suggestions": suggestions
                })
        
        # Sort by total time impact (avg_time * execution_count)
        slow_queries.sort(key=lambda x: x["avg_time"] * x["execution_count"], reverse=True)
        
        return slow_queries
    
    async def _analyze_query_for_optimization(self, query: str) -> List[str]:
        """
        Analyze a query and provide optimization suggestions.
        
        Args:
            query: SQL query to analyze
            
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        query_lower = query.lower()
        
        # Check for common performance issues
        if "select *" in query_lower:
            suggestions.append("Avoid SELECT * - specify only needed columns")
        
        if "where" not in query_lower and ("select" in query_lower and "from" in query_lower):
            suggestions.append("Consider adding WHERE clause to limit results")
        
        if query_lower.count("join") > 3:
            suggestions.append("Complex joins detected - consider query restructuring")
        
        if "order by" in query_lower and "limit" not in query_lower:
            suggestions.append("ORDER BY without LIMIT can be expensive - consider pagination")
        
        if "like '%%" in query_lower:
            suggestions.append("Leading wildcard LIKE patterns cannot use indexes efficiently")
        
        if "or" in query_lower:
            suggestions.append("OR conditions may prevent index usage - consider UNION")
        
        # Check for N+1 patterns
        if "= ?" in query and "select" in query_lower and "from users" in query_lower:
            suggestions.append("Potential N+1 query - consider batch loading or JOIN")
        
        return suggestions
    
    @asynccontextmanager
    async def monitor_query(self, query: str):
        """
        Context manager to monitor query performance.
        
        Args:
            query: SQL query being executed
        """
        if not self.monitoring_enabled:
            yield
            return
        
        query_hash = str(hash(query))
        start_time = time.time()
        
        try:
            yield
        finally:
            execution_time = time.time() - start_time
            await self._record_query_metrics(query_hash, query, execution_time)
    
    async def _record_query_metrics(self, query_hash: str, query: str, execution_time: float):
        """
        Record query performance metrics.
        
        Args:
            query_hash: Hash of the query
            query: SQL query
            execution_time: Query execution time in seconds
        """
        now = datetime.now()
        
        if query_hash in self.query_metrics:
            metrics = self.query_metrics[query_hash]
            metrics.execution_count += 1
            metrics.total_time += execution_time
            metrics.avg_time = metrics.total_time / metrics.execution_count
            metrics.min_time = min(metrics.min_time, execution_time)
            metrics.max_time = max(metrics.max_time, execution_time)
            metrics.last_executed = now
        else:
            self.query_metrics[query_hash] = QueryMetrics(
                query_hash=query_hash,
                query=query,
                execution_count=1,
                total_time=execution_time,
                avg_time=execution_time,
                min_time=execution_time,
                max_time=execution_time,
                last_executed=now
            )
        
        # Log slow queries
        if execution_time > self.slow_query_threshold:
            logger.warning(
                f"Slow query detected: {execution_time:.3f}s - {query[:100]}..."
            )
    
    async def get_cached_query_result(self, cache_key: str) -> Optional[Any]:
        """
        Get cached query result if available and not expired.
        
        Args:
            cache_key: Cache key for the query
            
        Returns:
            Cached result or None if not available/expired
        """
        if cache_key not in self.query_cache:
            return None
        
        result, cached_at = self.query_cache[cache_key]
        
        # Check if cache is expired
        if datetime.now() - cached_at > timedelta(seconds=self.cache_ttl):
            del self.query_cache[cache_key]
            return None
        
        return result
    
    async def cache_query_result(self, cache_key: str, result: Any):
        """
        Cache query result.
        
        Args:
            cache_key: Cache key for the query
            result: Query result to cache
        """
        self.query_cache[cache_key] = (result, datetime.now())
        
        # Clean up expired cache entries periodically
        if len(self.query_cache) > 1000:  # Arbitrary limit
            await self._cleanup_expired_cache()
    
    async def _cleanup_expired_cache(self):
        """Clean up expired cache entries."""
        now = datetime.now()
        expired_keys = [
            key for key, (_, cached_at) in self.query_cache.items()
            if now - cached_at > timedelta(seconds=self.cache_ttl)
        ]
        
        for key in expired_keys:
            del self.query_cache[key]
        
        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    async def optimize_indexes(self) -> Dict[str, Any]:
        """
        Analyze and optimize database indexes.
        
        Returns:
            Index optimization results and recommendations
        """
        if not hasattr(self.db, '_db') or not self.db._db:
            return await self._optimize_sqlite_indexes()
        
        return await self._optimize_postgresql_indexes()
    
    async def _optimize_postgresql_indexes(self) -> Dict[str, Any]:
        """Optimize PostgreSQL indexes."""
        try:
            async with self.db._db.pool.acquire() as conn:
                # Get unused indexes
                unused_indexes = await conn.fetch("""
                    SELECT
                        schemaname,
                        tablename,
                        indexname,
                        idx_scan as index_scans,
                        pg_size_pretty(pg_relation_size(indexrelid)) as index_size
                    FROM pg_stat_user_indexes
                    WHERE idx_scan = 0
                    AND indexrelid NOT IN (
                        SELECT conindid FROM pg_constraint WHERE conindid IS NOT NULL
                    )
                    ORDER BY pg_relation_size(indexrelid) DESC
                """)
                
                # Get missing indexes (tables with sequential scans)
                missing_indexes = await conn.fetch("""
                    SELECT
                        schemaname,
                        tablename,
                        seq_scan,
                        seq_tup_read,
                        idx_scan,
                        idx_tup_fetch,
                        seq_tup_read / GREATEST(seq_scan, 1) as avg_seq_read
                    FROM pg_stat_user_tables
                    WHERE seq_scan > idx_scan
                    AND seq_tup_read > 1000
                    ORDER BY seq_tup_read DESC
                """)
                
                # Get duplicate indexes
                duplicate_indexes = await conn.fetch("""
                    SELECT
                        t.tablename,
                        array_agg(t.indexname) as duplicate_indexes
                    FROM (
                        SELECT
                            tablename,
                            indexname,
                            array_to_string(array_agg(attname ORDER BY attnum), ',') as columns
                        FROM pg_index i
                        JOIN pg_class c ON c.oid = i.indexrelid
                        JOIN pg_class t ON t.oid = i.indrelid
                        JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(i.indkey)
                        WHERE c.relkind = 'i'
                        AND t.relname NOT LIKE 'pg_%'
                        GROUP BY tablename, indexname
                    ) t
                    GROUP BY tablename, columns
                    HAVING count(*) > 1
                """)
                
                return {
                    "status": "completed",
                    "unused_indexes": [dict(row) for row in unused_indexes],
                    "missing_indexes": [dict(row) for row in missing_indexes],
                    "duplicate_indexes": [dict(row) for row in duplicate_indexes],
                    "recommendations": self._generate_index_recommendations(
                        unused_indexes, missing_indexes, duplicate_indexes
                    )
                }
                
        except Exception as e:
            logger.error(f"Failed to optimize PostgreSQL indexes: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _optimize_sqlite_indexes(self) -> Dict[str, Any]:
        """Optimize SQLite indexes."""
        try:
            if not self.db._connection:
                await self.db.connect()
            
            cursor = await self.db._connection.cursor()
            
            # Get table statistics
            await cursor.execute("ANALYZE")
            
            # Get index usage statistics (limited in SQLite)
            await cursor.execute("""
                SELECT name, sql FROM sqlite_master 
                WHERE type = 'index' AND name NOT LIKE 'sqlite_%'
            """)
            indexes = await cursor.fetchall()
            
            # Check for missing indexes on foreign keys
            await cursor.execute("""
                SELECT DISTINCT m.name as table_name, p.from as column_name
                FROM sqlite_master m
                JOIN pragma_foreign_key_list(m.name) p
                WHERE m.type = 'table'
            """)
            foreign_keys = await cursor.fetchall()
            
            recommendations = []
            
            # Recommend indexes for foreign keys without indexes
            for table, column in foreign_keys:
                index_name = f"idx_{table}_{column}"
                if not any(index_name in idx[1] for idx in indexes if idx[1]):
                    recommendations.append({
                        "type": "missing_index",
                        "table": table,
                        "column": column,
                        "suggested_index": f"CREATE INDEX {index_name} ON {table}({column})"
                    })
            
            return {
                "status": "completed",
                "existing_indexes": len(indexes),
                "foreign_key_tables": len(set(fk[0] for fk in foreign_keys)),
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize SQLite indexes: {e}")
            return {"status": "error", "error": str(e)}
    
    def _generate_index_recommendations(self, unused, missing, duplicates) -> List[Dict[str, Any]]:
        """Generate index optimization recommendations."""
        recommendations = []
        
        # Recommend dropping unused indexes
        for idx in unused:
            if idx['index_scans'] == 0:
                recommendations.append({
                    "type": "drop_unused",
                    "action": f"DROP INDEX {idx['indexname']}",
                    "reason": f"Index never used, saves {idx['index_size']}",
                    "table": idx['tablename']
                })
        
        # Recommend creating missing indexes
        for table in missing:
            if table['seq_tup_read'] > 10000:  # High sequential scan activity
                recommendations.append({
                    "type": "create_missing",
                    "action": f"Consider adding index to {table['tablename']}",
                    "reason": f"High sequential scan activity: {table['seq_tup_read']} rows",
                    "table": table['tablename']
                })
        
        # Recommend removing duplicate indexes
        for dup in duplicates:
            if len(dup['duplicate_indexes']) > 1:
                indexes_to_keep = dup['duplicate_indexes'][0]
                indexes_to_drop = dup['duplicate_indexes'][1:]
                recommendations.append({
                    "type": "remove_duplicate",
                    "action": f"DROP INDEX {', '.join(indexes_to_drop)}",
                    "reason": f"Duplicate of {indexes_to_keep}",
                    "table": dup['tablename']
                })
        
        return recommendations
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Returns:
            Performance analysis report
        """
        # Analyze queries
        slow_queries = await self.analyze_slow_queries()
        
        # Optimize connection pool
        pool_analysis = await self.optimize_connection_pool()
        
        # Optimize indexes
        index_analysis = await self.optimize_indexes()
        
        # Calculate cache statistics
        cache_stats = {
            "total_entries": len(self.query_cache),
            "cache_hit_ratio": self._calculate_cache_hit_ratio(),
            "memory_usage_mb": self._estimate_cache_memory_usage()
        }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "slow_queries": slow_queries,
            "connection_pool": pool_analysis,
            "indexes": index_analysis,
            "cache": cache_stats,
            "recommendations": self._generate_performance_recommendations(
                slow_queries, pool_analysis, index_analysis
            )
        }
    
    def _calculate_cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio (placeholder - would need hit/miss tracking)."""
        # This would require actual hit/miss tracking
        return 0.75  # Placeholder
    
    def _estimate_cache_memory_usage(self) -> float:
        """Estimate cache memory usage in MB."""
        # Rough estimation - would need more sophisticated calculation
        return len(self.query_cache) * 0.001  # ~1KB per entry estimate
    
    def _generate_performance_recommendations(self, slow_queries, pool_analysis, index_analysis) -> List[str]:
        """Generate overall performance recommendations."""
        recommendations = []
        
        if len(slow_queries) > 0:
            recommendations.append(f"Optimize {len(slow_queries)} slow queries detected")
        
        if pool_analysis.get("recommendations"):
            recommendations.append("Adjust connection pool settings")
        
        if index_analysis.get("recommendations"):
            recommendations.append(f"Apply {len(index_analysis['recommendations'])} index optimizations")
        
        if len(self.query_cache) < 100:
            recommendations.append("Increase query result caching")
        
        return recommendations


class OptimizedDatabaseWrapper:
    """
    Wrapper for database operations with performance optimization.
    
    Provides transparent performance monitoring and optimization
    for database operations.
    """
    
    def __init__(self, database_instance):
        """
        Initialize optimized database wrapper.
        
        Args:
            database_instance: Original database instance
        """
        self.db = database_instance
        self.optimizer = DatabasePerformanceOptimizer(database_instance)
        
    async def execute_optimized_query(self, query: str, params: tuple = None) -> Any:
        """
        Execute query with performance monitoring and caching.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            Query result
        """
        # Generate cache key
        cache_key = f"{hash(query)}_{hash(params) if params else 'no_params'}"
        
        # Check cache first
        cached_result = await self.optimizer.get_cached_query_result(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Execute query with monitoring
        async with self.optimizer.monitor_query(query):
            if hasattr(self.db, '_db') and self.db._db:
                # PostgreSQL
                async with self.db._db.pool.acquire() as conn:
                    if params:
                        result = await conn.fetch(query, *params)
                    else:
                        result = await conn.fetch(query)
            else:
                # SQLite
                if not self.db._connection:
                    await self.db.connect()
                cursor = await self.db._connection.cursor()
                if params:
                    await cursor.execute(query, params)
                else:
                    await cursor.execute(query)
                result = await cursor.fetchall()
        
        # Cache result for SELECT queries
        if query.strip().upper().startswith('SELECT'):
            await self.optimizer.cache_query_result(cache_key, result)
        
        return result
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Get performance analysis report."""
        return await self.optimizer.get_performance_report()
"""
Performance Manager - Central Performance Optimization Coordinator

Coordinates all performance optimization components and provides
unified interface for performance monitoring and tuning.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import os

from .database_optimizer import DatabasePerformanceOptimizer, OptimizedDatabaseWrapper
from .batch_loader import DatabaseBatchLoader, BatchOperationManager
from .cache_optimizer import CacheManager, create_default_cache_manager
from .response_optimizer import ResponseOptimizationMiddleware, OptimizedResponseFactory

logger = logging.getLogger(__name__)


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    
    # Database optimization
    enable_query_optimization: bool = True
    enable_connection_pooling: bool = True
    slow_query_threshold: float = 0.1  # 100ms
    
    # Caching configuration
    enable_caching: bool = True
    redis_url: Optional[str] = None
    cache_ttl: int = 300  # 5 minutes
    memory_cache_size: int = 1000
    memory_cache_mb: int = 100
    
    # Batch loading
    enable_batch_loading: bool = True
    batch_size: int = 100
    batch_delay: float = 0.1  # 100ms
    
    # Response optimization
    enable_compression: bool = True
    enable_streaming: bool = True
    compression_min_size: int = 1024
    compression_level: int = 6
    
    # Monitoring
    enable_monitoring: bool = True
    metrics_collection_interval: int = 60  # seconds
    performance_report_interval: int = 300  # 5 minutes
    
    # Performance targets
    target_response_time: float = 0.1  # 100ms
    target_cache_hit_ratio: float = 0.8  # 80%
    target_compression_ratio: float = 0.3  # 70% size reduction
    
    @classmethod
    def from_environment(cls) -> 'PerformanceConfig':
        """Create configuration from environment variables."""
        return cls(
            enable_query_optimization=os.getenv('ENABLE_QUERY_OPTIMIZATION', 'true').lower() == 'true',
            enable_connection_pooling=os.getenv('ENABLE_CONNECTION_POOLING', 'true').lower() == 'true',
            slow_query_threshold=float(os.getenv('SLOW_QUERY_THRESHOLD', '0.1')),
            
            enable_caching=os.getenv('ENABLE_CACHING', 'true').lower() == 'true',
            redis_url=os.getenv('REDIS_URL'),
            cache_ttl=int(os.getenv('CACHE_TTL', '300')),
            memory_cache_size=int(os.getenv('MEMORY_CACHE_SIZE', '1000')),
            memory_cache_mb=int(os.getenv('MEMORY_CACHE_MB', '100')),
            
            enable_batch_loading=os.getenv('ENABLE_BATCH_LOADING', 'true').lower() == 'true',
            batch_size=int(os.getenv('BATCH_SIZE', '100')),
            batch_delay=float(os.getenv('BATCH_DELAY', '0.1')),
            
            enable_compression=os.getenv('ENABLE_COMPRESSION', 'true').lower() == 'true',
            enable_streaming=os.getenv('ENABLE_STREAMING', 'true').lower() == 'true',
            compression_min_size=int(os.getenv('COMPRESSION_MIN_SIZE', '1024')),
            compression_level=int(os.getenv('COMPRESSION_LEVEL', '6')),
            
            enable_monitoring=os.getenv('ENABLE_PERFORMANCE_MONITORING', 'true').lower() == 'true',
            metrics_collection_interval=int(os.getenv('METRICS_COLLECTION_INTERVAL', '60')),
            performance_report_interval=int(os.getenv('PERFORMANCE_REPORT_INTERVAL', '300')),
            
            target_response_time=float(os.getenv('TARGET_RESPONSE_TIME', '0.1')),
            target_cache_hit_ratio=float(os.getenv('TARGET_CACHE_HIT_RATIO', '0.8')),
            target_compression_ratio=float(os.getenv('TARGET_COMPRESSION_RATIO', '0.3'))
        )


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Database metrics
    avg_query_time: float = 0.0
    slow_queries_count: int = 0
    connection_pool_usage: float = 0.0
    
    # Cache metrics
    cache_hit_ratio: float = 0.0
    cache_memory_usage_mb: float = 0.0
    cache_entries_count: int = 0
    
    # Response metrics
    avg_response_time: float = 0.0
    compression_ratio: float = 0.0
    bytes_saved: int = 0
    
    # Batch loading metrics
    batch_efficiency: float = 0.0
    n1_queries_eliminated: int = 0
    
    # Overall performance score (0-100)
    performance_score: float = 0.0


class PerformanceManager:
    """
    Central performance optimization manager.
    
    Coordinates all performance optimization components and provides
    unified monitoring and control interface.
    """
    
    def __init__(self, config: PerformanceConfig, database_instance=None):
        """
        Initialize performance manager.
        
        Args:
            config: Performance configuration
            database_instance: Database instance to optimize
        """
        self.config = config
        self.database = database_instance
        self.metrics_history: List[PerformanceMetrics] = []
        self.optimization_tasks: List[asyncio.Task] = []
        
        # Initialize components
        self._initialize_components()
        
        # Performance monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.last_optimization_run = datetime.now()
        
        logger.info("Performance Manager initialized with config: %s", config)
    
    def _initialize_components(self):
        """Initialize performance optimization components."""
        # Database optimizer
        if self.config.enable_query_optimization and self.database:
            self.db_optimizer = DatabasePerformanceOptimizer(self.database)
            self.db_wrapper = OptimizedDatabaseWrapper(self.database)
            logger.info("Database optimization enabled")
        else:
            self.db_optimizer = None
            self.db_wrapper = None
        
        # Batch loader
        if self.config.enable_batch_loading and self.database:
            self.batch_loader = DatabaseBatchLoader(self.database)
            self.batch_manager = BatchOperationManager(self.database)
            logger.info("Batch loading enabled")
        else:
            self.batch_loader = None
            self.batch_manager = None
        
        # Cache manager
        if self.config.enable_caching:
            self.cache_manager = create_default_cache_manager(self.config.redis_url)
            logger.info("Caching enabled with Redis: %s", bool(self.config.redis_url))
        else:
            self.cache_manager = None
        
        # Response optimizer
        self.response_factory = OptimizedResponseFactory()
        logger.info("Response optimization enabled")
    
    async def start_monitoring(self):
        """Start performance monitoring."""
        if not self.config.enable_monitoring:
            return
        
        if self.monitoring_task and not self.monitoring_task.done():
            logger.warning("Performance monitoring already running")
            return
        
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop performance monitoring."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
        
        # Cancel optimization tasks
        for task in self.optimization_tasks:
            if not task.done():
                task.cancel()
        
        logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        try:
            while True:
                # Collect metrics
                metrics = await self.collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only recent metrics (last 24 hours)
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.metrics_history = [
                    m for m in self.metrics_history 
                    if m.timestamp > cutoff_time
                ]
                
                # Check if optimization is needed
                if await self._should_run_optimization(metrics):
                    optimization_task = asyncio.create_task(self._run_optimization())
                    self.optimization_tasks.append(optimization_task)
                
                # Clean up completed tasks
                self.optimization_tasks = [
                    task for task in self.optimization_tasks 
                    if not task.done()
                ]
                
                # Wait for next collection
                await asyncio.sleep(self.config.metrics_collection_interval)
                
        except asyncio.CancelledError:
            logger.info("Performance monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Performance monitoring error: {e}")
    
    async def collect_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive performance metrics."""
        metrics = PerformanceMetrics()
        
        try:
            # Database metrics
            if self.db_optimizer:
                slow_queries = await self.db_optimizer.analyze_slow_queries()
                pool_analysis = await self.db_optimizer.optimize_connection_pool()
                
                metrics.slow_queries_count = len(slow_queries)
                if slow_queries:
                    metrics.avg_query_time = sum(q["avg_time"] for q in slow_queries) / len(slow_queries)
                
                if pool_analysis.get("metrics"):
                    pool_metrics = pool_analysis["metrics"]
                    total_connections = pool_metrics.pool_size
                    active_connections = pool_metrics.active_connections
                    metrics.connection_pool_usage = active_connections / total_connections if total_connections > 0 else 0
            
            # Cache metrics
            if self.cache_manager:
                cache_stats = self.cache_manager.cache.get_stats()
                
                if "memory" in cache_stats:
                    memory_stats = cache_stats["memory"]
                    metrics.cache_hit_ratio = memory_stats.hit_ratio
                    metrics.cache_memory_usage_mb = memory_stats.memory_usage_bytes / (1024 * 1024)
                    metrics.cache_entries_count = memory_stats.total_entries
            
            # Batch loading metrics
            if self.batch_loader:
                loader_stats = self.batch_loader.get_loader_stats()
                
                total_requests = sum(stats.get("cache_hits", 0) + stats.get("cache_misses", 0) 
                                   for stats in loader_stats.values())
                total_batches = sum(stats.get("batch_loads", 0) for stats in loader_stats.values())
                
                if total_batches > 0:
                    metrics.batch_efficiency = total_requests / total_batches
                    metrics.n1_queries_eliminated = total_requests - total_batches
            
            # Calculate overall performance score
            metrics.performance_score = self._calculate_performance_score(metrics)
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
        
        return metrics
    
    def _calculate_performance_score(self, metrics: PerformanceMetrics) -> float:
        """
        Calculate overall performance score (0-100).
        
        Args:
            metrics: Performance metrics
            
        Returns:
            Performance score
        """
        score = 100.0
        
        # Query performance (30% weight)
        if metrics.avg_query_time > self.config.target_response_time:
            query_penalty = min(50, (metrics.avg_query_time / self.config.target_response_time - 1) * 30)
            score -= query_penalty
        
        # Cache performance (25% weight)
        if metrics.cache_hit_ratio < self.config.target_cache_hit_ratio:
            cache_penalty = (self.config.target_cache_hit_ratio - metrics.cache_hit_ratio) * 25
            score -= cache_penalty
        
        # Connection pool usage (20% weight)
        if metrics.connection_pool_usage > 0.8:  # Over 80% usage
            pool_penalty = (metrics.connection_pool_usage - 0.8) * 20 / 0.2
            score -= pool_penalty
        
        # Batch efficiency (15% weight)
        if metrics.batch_efficiency < 5:  # Less than 5 requests per batch
            batch_penalty = (5 - metrics.batch_efficiency) * 15 / 5
            score -= batch_penalty
        
        # Slow queries (10% weight)
        if metrics.slow_queries_count > 0:
            slow_query_penalty = min(10, metrics.slow_queries_count)
            score -= slow_query_penalty
        
        return max(0, score)
    
    async def _should_run_optimization(self, metrics: PerformanceMetrics) -> bool:
        """Check if optimization should be run."""
        # Don't run too frequently
        if datetime.now() - self.last_optimization_run < timedelta(minutes=5):
            return False
        
        # Run if performance score is low
        if metrics.performance_score < 70:
            return True
        
        # Run if there are slow queries
        if metrics.slow_queries_count > 5:
            return True
        
        # Run if cache hit ratio is low
        if metrics.cache_hit_ratio < 0.5:
            return True
        
        return False
    
    async def _run_optimization(self):
        """Run performance optimization."""
        try:
            logger.info("Running performance optimization")
            
            optimization_results = {}
            
            # Database optimization
            if self.db_optimizer:
                db_results = await self.db_optimizer.get_performance_report()
                optimization_results["database"] = db_results
            
            # Cache optimization
            if self.cache_manager:
                # Clear expired entries
                await self.cache_manager.cache.memory_cache.clear()
                optimization_results["cache"] = {"action": "cleared_expired_entries"}
            
            self.last_optimization_run = datetime.now()
            
            logger.info("Performance optimization completed: %s", optimization_results)
            
        except Exception as e:
            logger.error(f"Performance optimization failed: {e}")
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Returns:
            Performance analysis report
        """
        current_metrics = await self.collect_metrics()
        
        # Calculate trends from recent metrics
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp > datetime.now() - timedelta(hours=1)
        ]
        
        trends = {}
        if len(recent_metrics) > 1:
            first_metrics = recent_metrics[0]
            last_metrics = recent_metrics[-1]
            
            trends = {
                "performance_score_trend": last_metrics.performance_score - first_metrics.performance_score,
                "cache_hit_ratio_trend": last_metrics.cache_hit_ratio - first_metrics.cache_hit_ratio,
                "avg_query_time_trend": last_metrics.avg_query_time - first_metrics.avg_query_time
            }
        
        # Get component reports
        component_reports = {}
        
        if self.db_optimizer:
            component_reports["database"] = await self.db_optimizer.get_performance_report()
        
        if self.cache_manager:
            component_reports["cache"] = self.cache_manager.cache.get_stats()
        
        if self.batch_loader:
            component_reports["batch_loading"] = self.batch_loader.get_loader_stats()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "current_metrics": current_metrics,
            "trends": trends,
            "component_reports": component_reports,
            "recommendations": self._generate_recommendations(current_metrics),
            "config": self.config
        }
    
    def _generate_recommendations(self, metrics: PerformanceMetrics) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        if metrics.performance_score < 70:
            recommendations.append("Overall performance is below target - consider comprehensive optimization")
        
        if metrics.avg_query_time > self.config.target_response_time:
            recommendations.append("Query performance is slow - review slow queries and add indexes")
        
        if metrics.cache_hit_ratio < self.config.target_cache_hit_ratio:
            recommendations.append("Cache hit ratio is low - review caching strategy and TTL settings")
        
        if metrics.connection_pool_usage > 0.8:
            recommendations.append("Connection pool usage is high - consider increasing pool size")
        
        if metrics.slow_queries_count > 5:
            recommendations.append("Multiple slow queries detected - optimize database queries")
        
        if metrics.batch_efficiency < 3:
            recommendations.append("Batch loading efficiency is low - review batch size and timing")
        
        return recommendations
    
    def get_middleware(self) -> ResponseOptimizationMiddleware:
        """Get response optimization middleware."""
        return ResponseOptimizationMiddleware(
            app=None,  # Will be set by FastAPI
            enable_compression=self.config.enable_compression,
            enable_caching=True
        )


def create_performance_manager(
    database_instance=None,
    config: Optional[PerformanceConfig] = None
) -> PerformanceManager:
    """
    Create performance manager with default configuration.
    
    Args:
        database_instance: Database instance to optimize
        config: Performance configuration (uses environment if None)
        
    Returns:
        Configured performance manager
    """
    if config is None:
        config = PerformanceConfig.from_environment()
    
    return PerformanceManager(config, database_instance)
"""
Performance Optimization Package

Comprehensive performance optimization suite including:
- Database query optimization and N+1 elimination
- Multi-level caching with Redis integration
- Response compression and streaming
- Batch loading and operation optimization
- Performance monitoring and metrics
"""

from .database_optimizer import (
    DatabasePerformanceOptimizer,
    OptimizedDatabaseWrapper,
    QueryMetrics,
    ConnectionPoolMetrics
)

from .batch_loader import (
    DataLoader,
    DatabaseBatchLoader,
    BatchOperationManager
)

from .cache_optimizer import (
    MultiLevelCache,
    CacheManager,
    MemoryCache,
    RedisCache,
    create_default_cache_manager
)

from .response_optimizer import (
    ResponseOptimizationMiddleware,
    OptimizedResponseFactory,
    ResponseCompressor,
    PaginationOptimizer,
    StreamingResponseGenerator
)

from .performance_manager import (
    PerformanceManager,
    PerformanceConfig,
    create_performance_manager
)

__all__ = [
    # Database optimization
    'DatabasePerformanceOptimizer',
    'OptimizedDatabaseWrapper',
    'QueryMetrics',
    'ConnectionPoolMetrics',
    
    # Batch loading
    'DataLoader',
    'DatabaseBatchLoader', 
    'BatchOperationManager',
    
    # Caching
    'MultiLevelCache',
    'CacheManager',
    'MemoryCache',
    'RedisCache',
    'create_default_cache_manager',
    
    # Response optimization
    'ResponseOptimizationMiddleware',
    'OptimizedResponseFactory',
    'ResponseCompressor',
    'PaginationOptimizer',
    'StreamingResponseGenerator',
    
    # Main manager
    'PerformanceManager',
    'PerformanceConfig',
    'create_performance_manager'
]
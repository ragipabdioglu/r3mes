"""
Startup Tasks for R3MES Backend

Handles all startup initialization tasks in a structured way.
"""

import logging
import os
from typing import Dict, Any, Optional

from ..config_manager import get_config_manager
from ..database_async import AsyncDatabase
from ..cache import get_cache_manager
from ..notifications import get_notification_service, NotificationPriority
from ..serving_node_registry import ServingNodeRegistry
from ..inference_mode import should_load_ai_libraries, get_inference_mode

logger = logging.getLogger(__name__)


class StartupOrchestrator:
    """Orchestrates all startup tasks in the correct order."""
    
    def __init__(self):
        self.components: Dict[str, Any] = {}
        self.startup_errors: list = []
    
    async def initialize_all(self) -> Dict[str, Any]:
        """
        Initialize all components in the correct order.
        
        Returns:
            Dictionary of initialized components
            
        Raises:
            RuntimeError: If critical components fail to initialize
        """
        logger.info("🚀 Starting R3MES Backend initialization...")
        
        # Phase 1: Core Infrastructure
        await self._initialize_config()
        await self._initialize_database()
        await self._initialize_cache()
        
        # Phase 2: Optional Components (fail-soft)
        await self._initialize_notifications()
        await self._initialize_serving_registry()
        
        # Phase 3: AI/ML Components (conditional)
        if should_load_ai_libraries():
            await self._initialize_ai_components()
        
        # Phase 4: Background Services (optional)
        await self._initialize_background_services()
        
        # Phase 5: Final Validation
        await self._validate_startup()
        
        logger.info("✅ R3MES Backend initialization completed successfully")
        return self.components
    
    async def _initialize_config(self):
        """Initialize configuration manager."""
        try:
            logger.info("📋 Initializing configuration...")
            config_manager = get_config_manager()
            config = config_manager.load()
            
            self.components['config_manager'] = config_manager
            self.components['config'] = config
            
            logger.info(f"✅ Configuration loaded (env: {config.ENV})")
        except Exception as e:
            logger.critical(f"❌ Failed to initialize configuration: {e}")
            raise RuntimeError(f"Configuration initialization failed: {e}") from e
    
    async def _initialize_database(self):
        """Initialize database connection."""
        try:
            logger.info("🗄️ Initializing database...")
            database = AsyncDatabase()
            await database.connect()
            
            self.components['database'] = database
            
            logger.info("✅ Database connection established")
        except Exception as e:
            logger.critical(f"❌ Failed to initialize database: {e}")
            raise RuntimeError(f"Database initialization failed: {e}") from e
    
    async def _initialize_cache(self):
        """Initialize cache manager."""
        try:
            logger.info("🗂️ Initializing cache...")
            cache_manager = get_cache_manager()
            await cache_manager.connect()
            
            self.components['cache_manager'] = cache_manager
            
            logger.info("✅ Cache manager initialized")
        except Exception as e:
            logger.error(f"⚠️ Failed to initialize cache: {e}")
            self.startup_errors.append(f"Cache initialization failed: {e}")
            # Cache is optional - continue without it
            self.components['cache_manager'] = None
    
    async def _initialize_notifications(self):
        """Initialize notification service."""
        try:
            logger.info("📢 Initializing notifications...")
            notification_service = get_notification_service()
            
            self.components['notification_service'] = notification_service
            
            # Send startup notification
            await notification_service.send_notification(
                title="R3MES Backend Starting",
                message="Backend initialization in progress...",
                priority=NotificationPriority.INFO
            )
            
            logger.info("✅ Notification service initialized")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize notifications: {e}")
            self.startup_errors.append(f"Notification service failed: {e}")
            # Notifications are optional - continue without them
            self.components['notification_service'] = None
    
    async def _initialize_serving_registry(self):
        """Initialize serving node registry."""
        try:
            logger.info("🌐 Initializing serving node registry...")
            serving_registry = ServingNodeRegistry()
            
            self.components['serving_node_registry'] = serving_registry
            
            logger.info("✅ Serving node registry initialized")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize serving registry: {e}")
            self.startup_errors.append(f"Serving registry failed: {e}")
            # Serving registry is optional - continue without it
            self.components['serving_node_registry'] = None
    
    async def _initialize_ai_components(self):
        """Initialize AI/ML components (conditional)."""
        try:
            logger.info("🤖 Initializing AI components...")
            
            # Import AI modules only when needed
            from ..model_manager import AIModelManager
            from ..semantic_router import SemanticRouter
            from ..task_queue import TaskQueue
            
            # Initialize model manager
            config = self.components['config']
            model_manager = AIModelManager(base_model_path=config.base_model_path)
            self.components['model_manager'] = model_manager
            
            # Initialize semantic router
            similarity_threshold = float(os.getenv("SEMANTIC_ROUTER_THRESHOLD", "0.7"))
            semantic_router = SemanticRouter(
                similarity_threshold=similarity_threshold,
                use_semantic=True
            )
            self.components['semantic_router'] = semantic_router
            
            # Initialize task queue
            max_workers = int(os.getenv("MAX_WORKERS", "1"))
            task_queue = TaskQueue(max_workers=max_workers)
            self.components['task_queue'] = task_queue
            
            logger.info("✅ AI components initialized")
        except ImportError as e:
            logger.warning(f"⚠️ AI libraries not available: {e}")
            self.startup_errors.append(f"AI components not available: {e}")
            # AI components are optional in some deployment modes
            self.components['model_manager'] = None
            self.components['semantic_router'] = None
            self.components['task_queue'] = None
        except Exception as e:
            logger.error(f"⚠️ Failed to initialize AI components: {e}")
            self.startup_errors.append(f"AI components failed: {e}")
            # AI components are optional - continue without them
            self.components['model_manager'] = None
            self.components['semantic_router'] = None
            self.components['task_queue'] = None
    
    async def _initialize_background_services(self):
        """Initialize background services."""
        try:
            logger.info("⚙️ Initializing background services...")
            
            # Initialize blockchain indexer (optional)
            await self._initialize_blockchain_indexer()
            
            # Initialize cache warming (optional)
            await self._initialize_cache_warming()
            
            # Initialize system metrics (optional)
            await self._initialize_system_metrics()
            
            logger.info("✅ Background services initialized")
        except Exception as e:
            logger.warning(f"⚠️ Some background services failed: {e}")
            self.startup_errors.append(f"Background services partial failure: {e}")
    
    async def _initialize_blockchain_indexer(self):
        """Initialize blockchain indexer (optional)."""
        try:
            # Import only when needed
            from ..blockchain_indexer import BlockchainIndexer
            
            database = self.components['database']
            indexer = BlockchainIndexer(database)
            await indexer.start()
            
            self.components['blockchain_indexer'] = indexer
            logger.info("✅ Blockchain indexer started")
        except ImportError:
            logger.info("ℹ️ Blockchain indexer not available (optional)")
            self.components['blockchain_indexer'] = None
        except Exception as e:
            logger.warning(f"⚠️ Failed to start blockchain indexer: {e} (continuing without indexer)")
            self.components['blockchain_indexer'] = None
    
    async def _initialize_cache_warming(self):
        """Initialize cache warming (optional)."""
        try:
            cache_manager = self.components.get('cache_manager')
            if cache_manager:
                # Import only when needed
                from ..cache_warming import warm_cache
                
                await warm_cache(cache_manager)
                logger.info("✅ Cache warming completed")
            else:
                logger.info("ℹ️ Cache warming skipped (cache not available)")
        except Exception as e:
            logger.warning(f"⚠️ Failed to warm cache on startup: {e} (continuing without cache warming)")
    
    async def _initialize_system_metrics(self):
        """Initialize system metrics collector (optional)."""
        try:
            # Import only when needed
            from ..system_metrics import SystemMetricsCollector
            
            metrics_collector = SystemMetricsCollector()
            await metrics_collector.start()
            
            self.components['system_metrics'] = metrics_collector
            logger.info("✅ System metrics collector started")
        except ImportError:
            logger.info("ℹ️ System metrics collector not available (optional)")
            self.components['system_metrics'] = None
        except Exception as e:
            logger.warning(f"⚠️ Failed to start system metrics collector: {e} (continuing without metrics)")
            self.components['system_metrics'] = None
    
    async def _validate_startup(self):
        """Validate that critical components are initialized."""
        critical_components = ['config', 'database']
        missing_components = []
        
        for component in critical_components:
            if component not in self.components or self.components[component] is None:
                missing_components.append(component)
        
        if missing_components:
            error_msg = f"Critical components failed to initialize: {missing_components}"
            logger.critical(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        # Log startup summary
        if self.startup_errors:
            logger.warning(f"⚠️ Startup completed with {len(self.startup_errors)} non-critical errors:")
            for error in self.startup_errors:
                logger.warning(f"  - {error}")
        
        # Send success notification
        notification_service = self.components.get('notification_service')
        if notification_service:
            try:
                await notification_service.send_notification(
                    title="R3MES Backend Started",
                    message=f"Backend initialization completed successfully. "
                           f"Inference mode: {get_inference_mode().value}",
                    priority=NotificationPriority.INFO
                )
            except Exception as e:
                logger.warning(f"Failed to send startup notification: {e}")
        
        logger.info("🎉 All critical components initialized successfully")


async def initialize_startup_components() -> Dict[str, Any]:
    """
    Initialize all startup components.
    
    Returns:
        Dictionary of initialized components
        
    Raises:
        RuntimeError: If critical components fail to initialize
    """
    orchestrator = StartupOrchestrator()
    return await orchestrator.initialize_all()
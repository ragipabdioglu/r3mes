"""
Shutdown Tasks for R3MES Backend

Handles graceful shutdown of all components.
"""

import logging
import asyncio
from typing import Dict, Any

from ..notifications import NotificationPriority

logger = logging.getLogger(__name__)


class ShutdownOrchestrator:
    """Orchestrates graceful shutdown of all components."""
    
    def __init__(self, components: Dict[str, Any]):
        self.components = components
        self.shutdown_errors: list = []
    
    async def shutdown_all(self):
        """
        Shutdown all components in reverse order of initialization.
        """
        logger.info("🛑 Starting R3MES Backend shutdown...")
        
        # Send shutdown notification first
        await self._send_shutdown_notification()
        
        # Phase 1: Stop background services
        await self._shutdown_background_services()
        
        # Phase 2: Stop AI components
        await self._shutdown_ai_components()
        
        # Phase 3: Stop serving registry
        await self._shutdown_serving_registry()
        
        # Phase 4: Stop notifications
        await self._shutdown_notifications()
        
        # Phase 5: Stop cache
        await self._shutdown_cache()
        
        # Phase 6: Stop database (last)
        await self._shutdown_database()
        
        # Final cleanup
        await self._final_cleanup()
        
        if self.shutdown_errors:
            logger.warning(f"⚠️ Shutdown completed with {len(self.shutdown_errors)} errors:")
            for error in self.shutdown_errors:
                logger.warning(f"  - {error}")
        else:
            logger.info("✅ R3MES Backend shutdown completed successfully")
    
    async def _send_shutdown_notification(self):
        """Send shutdown notification."""
        try:
            notification_service = self.components.get('notification_service')
            if notification_service:
                await notification_service.send_notification(
                    title="R3MES Backend Shutting Down",
                    message="Backend shutdown initiated...",
                    priority=NotificationPriority.WARNING
                )
                logger.info("📢 Shutdown notification sent")
        except Exception as e:
            logger.warning(f"Failed to send shutdown notification: {e}")
    
    async def _shutdown_background_services(self):
        """Shutdown background services."""
        logger.info("⚙️ Shutting down background services...")
        
        # Stop system metrics collector
        await self._shutdown_system_metrics()
        
        # Stop blockchain indexer
        await self._shutdown_blockchain_indexer()
        
        logger.info("✅ Background services shutdown completed")
    
    async def _shutdown_system_metrics(self):
        """Shutdown system metrics collector."""
        try:
            system_metrics = self.components.get('system_metrics')
            if system_metrics:
                await system_metrics.stop()
                logger.info("✅ System metrics collector stopped")
        except Exception as e:
            logger.warning(f"⚠️ Error stopping system metrics: {e}")
            self.shutdown_errors.append(f"System metrics shutdown error: {e}")
    
    async def _shutdown_blockchain_indexer(self):
        """Shutdown blockchain indexer."""
        try:
            blockchain_indexer = self.components.get('blockchain_indexer')
            if blockchain_indexer:
                await blockchain_indexer.stop()
                logger.info("✅ Blockchain indexer stopped")
        except Exception as e:
            logger.warning(f"⚠️ Error stopping blockchain indexer: {e}")
            self.shutdown_errors.append(f"Blockchain indexer shutdown error: {e}")
    
    async def _shutdown_ai_components(self):
        """Shutdown AI/ML components."""
        logger.info("🤖 Shutting down AI components...")
        
        # Stop task queue
        await self._shutdown_task_queue()
        
        # Stop semantic router
        await self._shutdown_semantic_router()
        
        # Stop model manager
        await self._shutdown_model_manager()
        
        logger.info("✅ AI components shutdown completed")
    
    async def _shutdown_task_queue(self):
        """Shutdown task queue."""
        try:
            task_queue = self.components.get('task_queue')
            if task_queue:
                await task_queue.shutdown()
                logger.info("✅ Task queue stopped")
        except Exception as e:
            logger.warning(f"⚠️ Error stopping task queue: {e}")
            self.shutdown_errors.append(f"Task queue shutdown error: {e}")
    
    async def _shutdown_semantic_router(self):
        """Shutdown semantic router."""
        try:
            semantic_router = self.components.get('semantic_router')
            if semantic_router:
                # Semantic router might not have explicit shutdown
                logger.info("✅ Semantic router stopped")
        except Exception as e:
            logger.warning(f"⚠️ Error stopping semantic router: {e}")
            self.shutdown_errors.append(f"Semantic router shutdown error: {e}")
    
    async def _shutdown_model_manager(self):
        """Shutdown model manager."""
        try:
            model_manager = self.components.get('model_manager')
            if model_manager:
                await model_manager.cleanup()
                logger.info("✅ Model manager stopped")
        except Exception as e:
            logger.warning(f"⚠️ Error stopping model manager: {e}")
            self.shutdown_errors.append(f"Model manager shutdown error: {e}")
    
    async def _shutdown_serving_registry(self):
        """Shutdown serving node registry."""
        try:
            serving_registry = self.components.get('serving_node_registry')
            if serving_registry:
                await serving_registry.cleanup()
                logger.info("✅ Serving node registry stopped")
        except Exception as e:
            logger.warning(f"⚠️ Error stopping serving registry: {e}")
            self.shutdown_errors.append(f"Serving registry shutdown error: {e}")
    
    async def _shutdown_notifications(self):
        """Shutdown notification service."""
        try:
            notification_service = self.components.get('notification_service')
            if notification_service:
                # Send final notification
                await notification_service.send_notification(
                    title="R3MES Backend Stopped",
                    message="Backend shutdown completed successfully",
                    priority=NotificationPriority.INFO
                )
                
                # Give time for notification to be sent
                await asyncio.sleep(0.5)
                
                await notification_service.cleanup()
                logger.info("✅ Notification service stopped")
        except Exception as e:
            logger.warning(f"⚠️ Error stopping notifications: {e}")
            self.shutdown_errors.append(f"Notification service shutdown error: {e}")
    
    async def _shutdown_cache(self):
        """Shutdown cache manager."""
        try:
            cache_manager = self.components.get('cache_manager')
            if cache_manager:
                await cache_manager.disconnect()
                logger.info("✅ Cache manager stopped")
        except Exception as e:
            logger.warning(f"⚠️ Error stopping cache: {e}")
            self.shutdown_errors.append(f"Cache shutdown error: {e}")
    
    async def _shutdown_database(self):
        """Shutdown database connection."""
        try:
            database = self.components.get('database')
            if database:
                await database.disconnect()
                logger.info("✅ Database connection closed")
        except Exception as e:
            logger.error(f"❌ Error closing database: {e}")
            self.shutdown_errors.append(f"Database shutdown error: {e}")
    
    async def _final_cleanup(self):
        """Perform final cleanup tasks."""
        try:
            # Clear component references
            self.components.clear()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info("✅ Final cleanup completed")
        except Exception as e:
            logger.warning(f"⚠️ Error during final cleanup: {e}")
            self.shutdown_errors.append(f"Final cleanup error: {e}")


async def shutdown_components(components: Dict[str, Any]):
    """
    Shutdown all components gracefully.
    
    Args:
        components: Dictionary of components to shutdown
    """
    orchestrator = ShutdownOrchestrator(components)
    await orchestrator.shutdown_all()
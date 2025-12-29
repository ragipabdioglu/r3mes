"""
Graceful Shutdown Handler for R3MES Backend

Handles SIGTERM and SIGINT signals for graceful shutdown.
"""

import asyncio
import signal
import logging
from typing import Optional, Callable
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class GracefulShutdown:
    """
    Manages graceful shutdown of the application.
    
    Handles SIGTERM and SIGINT signals, allowing the application
    to finish current requests before shutting down.
    """
    
    def __init__(self):
        self.shutdown_event: Optional[asyncio.Event] = None
        self.shutdown_handlers: list[Callable] = []
        self._shutdown_in_progress = False
    
    def register_shutdown_handler(self, handler: Callable):
        """Register a function to be called during shutdown."""
        self.shutdown_handlers.append(handler)
    
    def setup_signal_handlers(self):
        """Setup signal handlers for SIGTERM and SIGINT."""
        if self.shutdown_event is None:
            self.shutdown_event = asyncio.Event()
        
        def signal_handler(signum, frame):
            signal_name = signal.Signals(signum).name
            logger.warning(f"Received {signal_name} signal. Initiating graceful shutdown...")
            if not self._shutdown_in_progress:
                self._shutdown_in_progress = True
                self.shutdown_event.set()
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        logger.info("Signal handlers registered (SIGTERM, SIGINT)")
    
    async def wait_for_shutdown(self):
        """Wait for shutdown signal."""
        if self.shutdown_event is None:
            self.shutdown_event = asyncio.Event()
        await self.shutdown_event.wait()
    
    async def execute_shutdown_handlers(self, timeout: float = 30.0):
        """
        Execute all registered shutdown handlers with timeout.
        
        Args:
            timeout: Maximum time to wait for shutdown handlers (seconds)
        """
        if not self.shutdown_handlers:
            return
        
        logger.info(f"Executing {len(self.shutdown_handlers)} shutdown handlers...")
        
        # Execute handlers concurrently with timeout
        tasks = []
        for handler in self.shutdown_handlers:
            if asyncio.iscoroutinefunction(handler):
                tasks.append(asyncio.create_task(handler()))
            else:
                # Synchronous handler - run in executor
                tasks.append(asyncio.create_task(
                    asyncio.to_thread(handler)
                ))
        
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
            logger.info("All shutdown handlers completed")
        except asyncio.TimeoutError:
            logger.error(f"Shutdown handlers did not complete within {timeout} seconds")
        except Exception as e:
            logger.error(f"Error executing shutdown handlers: {e}", exc_info=True)
    
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_in_progress or (
            self.shutdown_event is not None and self.shutdown_event.is_set()
        )


# Global graceful shutdown instance
_graceful_shutdown: Optional[GracefulShutdown] = None


def get_graceful_shutdown() -> GracefulShutdown:
    """Get or create the global graceful shutdown instance."""
    global _graceful_shutdown
    if _graceful_shutdown is None:
        _graceful_shutdown = GracefulShutdown()
    return _graceful_shutdown


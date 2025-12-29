"""
Inference Executor - ProcessPoolExecutor wrapper for CPU/GPU intensive inference

Moves model inference to separate processes to prevent blocking the FastAPI event loop.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Optional, AsyncIterator
import functools

logger = logging.getLogger(__name__)


class InferenceExecutor:
    """
    Manages inference execution in separate threads.
    
    This prevents blocking the FastAPI event loop during CPU/GPU intensive operations.
    Note: Using ThreadPoolExecutor instead of ProcessPoolExecutor because PyTorch models
    cannot be easily pickled for process-based execution.
    """
    
    def __init__(self, max_workers: int = 1):
        """
        Initialize inference executor.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self.max_workers = max_workers
        self.executor: Optional[ThreadPoolExecutor] = None
        self._lock = asyncio.Lock()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()
    
    async def start(self):
        """Start the executor."""
        async with self._lock:
            if self.executor is None:
                self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
                logger.info(f"InferenceExecutor started with {self.max_workers} workers")
    
    async def shutdown(self):
        """Shutdown the executor."""
        async with self._lock:
            if self.executor:
                self.executor.shutdown(wait=True)
                self.executor = None
                logger.info("InferenceExecutor shut down")
    
    async def run_inference(
        self,
        message: str,
        adapter_name: str,
        model_manager
    ) -> Iterator[str]:
        """
        Run inference in a separate process.
        
        Args:
            message: User message
            adapter_name: Selected adapter name
            model_manager: Model manager instance (will be pickled)
        
        Yields:
            Response tokens
        """
        if not self.executor:
            await self.start()
        
        # Run inference in executor
        loop = asyncio.get_event_loop()
        
        # Since model_manager contains PyTorch models which can't be pickled easily,
        # we'll use a different approach: run in thread pool for now
        # In production, consider using a separate inference service (gRPC/HTTP)
        
        # For now, use run_in_executor with thread pool
        # This is a compromise - full process isolation would require model serialization
        def _run_inference_sync(msg: str, adapter: str):
            """Synchronous inference wrapper."""
            try:
                # Call the model manager's generate_response
                # This will run in a thread, not blocking the main event loop
                tokens = []
                for token in model_manager.generate_response(msg, adapter):
                    tokens.append(token)
                return tokens
            except Exception as e:
                logger.error(f"Inference error: {e}", exc_info=True)
                raise
        
        # Run in thread pool (better than process pool for PyTorch models)
        tokens = await loop.run_in_executor(
            self.executor,
            _run_inference_sync,
            message,
            adapter_name
        )
        
        # Yield tokens
        for token in tokens:
            yield token
    
    async def run_inference_streaming(
        self,
        message: str,
        adapter_name: str,
        model_manager
    ) -> AsyncIterator[str]:
        """
        Run inference with streaming (async generator).
        
        This streams tokens as they're generated in a background thread.
        """
        if not self.executor:
            await self.start()
        
        loop = asyncio.get_event_loop()
        
        # Create a queue for streaming tokens
        token_queue = asyncio.Queue()
        done = asyncio.Event()
        error_occurred = asyncio.Event()
        error_message = [None]
        
        def _generate_tokens():
            """Generate tokens in background thread."""
            try:
                for token in model_manager.generate_response(message, adapter_name):
                    # Put token in queue (non-blocking)
                    asyncio.run_coroutine_threadsafe(
                        token_queue.put(token),
                        loop
                    )
                # Signal completion
                asyncio.run_coroutine_threadsafe(done.set(), loop)
            except Exception as e:
                logger.error(f"Inference error: {e}", exc_info=True)
                error_message[0] = str(e)
                asyncio.run_coroutine_threadsafe(error_occurred.set(), loop)
                asyncio.run_coroutine_threadsafe(done.set(), loop)
        
        # Start generation in thread
        loop.run_in_executor(self.executor, _generate_tokens)
        
        # Stream tokens from queue
        while not done.is_set() or not token_queue.empty():
            try:
                # Wait for token with timeout
                token = await asyncio.wait_for(token_queue.get(), timeout=0.1)
                yield token
            except asyncio.TimeoutError:
                # Check if done
                if done.is_set() and token_queue.empty():
                    break
                continue
        
        # Check for errors
        if error_occurred.is_set():
            raise RuntimeError(f"Inference failed: {error_message[0]}")


# Global executor instance
_inference_executor: Optional[InferenceExecutor] = None


async def get_inference_executor(max_workers: int = 1) -> InferenceExecutor:
    """Get or create global inference executor."""
    global _inference_executor
    if _inference_executor is None:
        _inference_executor = InferenceExecutor(max_workers=max_workers)
        await _inference_executor.start()
    return _inference_executor


async def shutdown_inference_executor():
    """Shutdown global inference executor."""
    global _inference_executor
    if _inference_executor:
        await _inference_executor.shutdown()
        _inference_executor = None


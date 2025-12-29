"""
Debug Tracer for R3MES Miner Engine

Provides training loop tracing, gradient flow tracing, memory usage tracking, and GPU utilization tracking.
"""

import time
import logging
import threading
from typing import Dict, Optional, Any
from datetime import datetime

try:
    from r3mes.debug_config import get_debug_config
    from r3mes.tracing import get_trace_collector
    from r3mes.performance_profiler import get_profiler
except ImportError:
    # Fallback if modules are not available
    def get_debug_config():
        class DummyConfig:
            enabled = False
        return DummyConfig()
    
    def get_trace_collector():
        return None
    
    def get_profiler():
        return None

logger = logging.getLogger(__name__)


class DebugTracer:
    """Debug tracer for miner engine operations"""
    
    def __init__(self):
        """Initialize debug tracer"""
        self.debug_config = get_debug_config()
        self.enabled = self.debug_config.enabled and self.debug_config.is_miner_enabled()
        self.trace_collector = get_trace_collector() if self.enabled else None
        self.profiler = get_profiler() if self.enabled else None
    
    def trace_training_iteration(self, iteration: int, loss: float, **fields):
        """
        Trace a training iteration.
        
        Args:
            iteration: Iteration number
            loss: Loss value
            **fields: Additional fields
        """
        if not self.enabled or not self.debug_config.training_loop_debug:
            return
        
        trace_id = self.trace_collector.generate_trace_id() if self.trace_collector else None
        entry = None
        
        if self.trace_collector:
            entry = self.trace_collector.start_trace(
                trace_id,
                "training_iteration",
                component="miner",
                iteration=iteration,
                loss=loss,
                **fields
            )
        
        if logger.isEnabledFor(logging.TRACE):
            logger.trace(
                f"Training iteration {iteration}",
                extra={
                    "trace_id": trace_id,
                    "iteration": iteration,
                    "loss": loss,
                    **fields
                }
            )
        
        return entry
    
    def trace_gradient_computation(self, gradient_hash: str, size_bytes: int, **fields):
        """
        Trace gradient computation.
        
        Args:
            gradient_hash: Gradient hash
            size_bytes: Gradient size in bytes
            **fields: Additional fields
        """
        if not self.enabled or not self.debug_config.gradient_debug:
            return
        
        trace_id = self.trace_collector.generate_trace_id() if self.trace_collector else None
        entry = None
        
        if self.trace_collector:
            entry = self.trace_collector.start_trace(
                trace_id,
                "gradient_computation",
                component="miner",
                gradient_hash=gradient_hash,
                size_bytes=size_bytes,
                **fields
            )
        
        if logger.isEnabledFor(logging.TRACE):
            logger.trace(
                f"Gradient computed: {gradient_hash[:16]}...",
                extra={
                    "trace_id": trace_id,
                    "gradient_hash": gradient_hash,
                    "size_bytes": size_bytes,
                    **fields
                }
            )
        
        return entry
    
    def trace_blockchain_interaction(self, operation: str, **fields):
        """
        Trace blockchain interaction.
        
        Args:
            operation: Operation name (e.g., "submit_gradient", "claim_task")
            **fields: Additional fields
        """
        if not self.enabled or not self.debug_config.blockchain_interaction_debug:
            return
        
        trace_id = self.trace_collector.generate_trace_id() if self.trace_collector else None
        entry = None
        
        if self.trace_collector:
            entry = self.trace_collector.start_trace(
                trace_id,
                f"blockchain_{operation}",
                component="miner",
                operation=operation,
                **fields
            )
        
        if logger.isEnabledFor(logging.TRACE):
            logger.trace(
                f"Blockchain interaction: {operation}",
                extra={
                    "trace_id": trace_id,
                    "operation": operation,
                    **fields
                }
            )
        
        return entry
    
    def trace_ipfs_operation(self, operation: str, cid: Optional[str] = None, **fields):
        """
        Trace IPFS operation.
        
        Args:
            operation: Operation name (e.g., "upload", "download")
            cid: IPFS content identifier
            **fields: Additional fields
        """
        if not self.enabled or not self.debug_config.ipfs_debug:
            return
        
        trace_id = self.trace_collector.generate_trace_id() if self.trace_collector else None
        entry = None
        
        if self.trace_collector:
            entry = self.trace_collector.start_trace(
                trace_id,
                f"ipfs_{operation}",
                component="miner",
                operation=operation,
                cid=cid,
                **fields
            )
        
        if logger.isEnabledFor(logging.TRACE):
            logger.trace(
                f"IPFS operation: {operation}",
                extra={
                    "trace_id": trace_id,
                    "operation": operation,
                    "cid": cid,
                    **fields
                }
            )
        
        return entry
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get current memory statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
            }
        except Exception:
            return {}
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """
        Get GPU statistics.
        
        Returns:
            Dictionary with GPU statistics
        """
        stats = {}
        try:
            import torch
            if torch.cuda.is_available():
                stats = {
                    "cuda_available": True,
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "allocated_memory_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                    "reserved_memory_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                }
            else:
                stats = {"cuda_available": False}
        except ImportError:
            stats = {"cuda_available": False, "error": "torch not available"}
        except Exception as e:
            stats = {"cuda_available": False, "error": str(e)}
        
        return stats


# Global debug tracer instance
_global_debug_tracer: Optional[DebugTracer] = None


def get_debug_tracer() -> DebugTracer:
    """
    Get the global debug tracer (cached).
    
    Returns:
        DebugTracer instance
    """
    global _global_debug_tracer
    if _global_debug_tracer is None:
        _global_debug_tracer = DebugTracer()
    return _global_debug_tracer

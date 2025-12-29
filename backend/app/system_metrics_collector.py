"""
System Metrics Collector

Periodically collects system metrics (CPU, memory, GPU) and updates Prometheus metrics.
"""

import asyncio
import logging
import psutil
import time
from typing import Optional

from .metrics import (
    update_system_metrics,
    update_gpu_metrics,
    system_memory_usage_bytes,
    system_cpu_usage_percent,
    gpu_utilization_percent,
    gpu_memory_usage_bytes,
    gpu_temperature_celsius,
)

logger = logging.getLogger(__name__)


class SystemMetricsCollector:
    """
    Collects system metrics periodically and updates Prometheus metrics.
    """
    
    def __init__(self, interval: float = 10.0):
        """
        Initialize system metrics collector.
        
        Args:
            interval: Collection interval in seconds (default: 10s)
        """
        self.interval = interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def collect_metrics(self):
        """Collect and update system metrics."""
        try:
            # Collect CPU and memory metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_bytes = memory.used
            
            # Update Prometheus metrics
            update_system_metrics(memory_bytes, cpu_percent)
            
            # Collect GPU metrics if available
            try:
                import pynvml
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # GPU utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    utilization = util.gpu
                    
                    # GPU memory
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    memory_used = mem_info.used
                    
                    # GPU temperature
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    
                    # Update Prometheus metrics
                    update_gpu_metrics(i, utilization, memory_used, temp)
                
                pynvml.nvmlShutdown()
            except ImportError:
                # pynvml not available, skip GPU metrics
                pass
            except Exception as e:
                logger.debug(f"Failed to collect GPU metrics: {e}")
        
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _collect_loop(self):
        """Main collection loop."""
        while self._running:
            try:
                await self.collect_metrics()
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(self.interval)
    
    def start(self):
        """Start metrics collection."""
        if self._running:
            logger.warning("Metrics collector is already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._collect_loop())
        logger.info(f"System metrics collector started (interval: {self.interval}s)")
    
    def stop(self):
        """Stop metrics collection."""
        if not self._running:
            return
        
        self._running = False
        if self._task:
            self._task.cancel()
        logger.info("System metrics collector stopped")


# Global collector instance
_collector: Optional[SystemMetricsCollector] = None


def get_system_metrics_collector(interval: float = 10.0) -> SystemMetricsCollector:
    """Get or create global system metrics collector."""
    global _collector
    if _collector is None:
        _collector = SystemMetricsCollector(interval=interval)
    return _collector


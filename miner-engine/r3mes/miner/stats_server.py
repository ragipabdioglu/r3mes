"""
Stats Server for Python Miner

Exposes miner statistics via gRPC or HTTP for WebSocket integration.
Stats include GPU metrics, training metrics, and miner status.
"""

import logging
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class MinerStats:
    """Miner statistics for WebSocket streaming."""
    gpu_temp: float = 0.0
    fan_speed: int = 0
    vram_usage: int = 0  # MB
    power_draw: float = 0.0  # Watts
    hash_rate: float = 0.0  # Gradients/hour
    uptime: int = 0  # Seconds
    timestamp: int = 0


@dataclass
class TrainingMetrics:
    """Training metrics for WebSocket streaming."""
    epoch: int = 0
    loss: float = 0.0
    accuracy: float = 0.0
    gradient_norm: float = 0.0
    timestamp: int = 0


class StatsCollector:
    """
    Collects miner statistics from various sources.
    
    This class aggregates stats from:
    - GPU monitoring (temperature, fan, VRAM, power)
    - Training metrics (loss, accuracy, gradient norm)
    - Miner status (uptime, hash rate)
    """
    
    def __init__(self, miner_engine=None):
        """
        Initialize stats collector.
        
        Args:
            miner_engine: Reference to MinerEngine instance for training metrics
        """
        self.miner_engine = miner_engine
        self.start_time = time.time()
        self.last_training_metrics: Optional[TrainingMetrics] = None
        
        # Try to import GPU monitoring libraries
        self.gpu_available = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.gpu_available = True
            self.pynvml = pynvml
            logger.info("GPU monitoring enabled (pynvml)")
        except ImportError:
            logger.warning("pynvml not available, GPU stats will be limited")
        except Exception as e:
            logger.warning(f"Failed to initialize GPU monitoring: {e}")
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """
        Get GPU statistics.
        
        Returns:
            Dictionary with GPU stats (temp, fan, VRAM, power)
        """
        stats = {
            "gpu_temp": 0.0,
            "fan_speed": 0,
            "vram_usage": 0,
            "power_draw": 0.0,
        }
        
        if not self.gpu_available:
            return stats
        
        try:
            # Try pynvml first (full stats)
            if hasattr(self, 'pynvml'):
                handle = self.pynvml.nvmlDeviceGetHandleByIndex(0)
                
                # Temperature
                temp = self.pynvml.nvmlDeviceGetTemperature(handle, self.pynvml.NVML_TEMPERATURE_GPU)
                stats["gpu_temp"] = float(temp)
                
                # Fan speed
                try:
                    fan_speed = self.pynvml.nvmlDeviceGetFanSpeed(handle)
                    stats["fan_speed"] = int(fan_speed)
                except self.pynvml.NVMLError:
                    pass  # Some GPUs don't support fan speed query
                
                # VRAM usage
                mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                stats["vram_usage"] = int(mem_info.used / 1024 / 1024)  # Convert to MB
                
                # Power draw
                try:
                    power = self.pynvml.nvmlDeviceGetPowerUsage(handle)
                    stats["power_draw"] = float(power) / 1000.0  # Convert mW to W
                except self.pynvml.NVMLError:
                    pass  # Some GPUs don't support power query
            # Fallback to PyTorch (limited stats)
            elif hasattr(self, 'torch') and self.torch.cuda.is_available():
                # PyTorch can only provide VRAM usage
                stats["vram_usage"] = int(self.torch.cuda.memory_allocated(0) / 1024 / 1024)  # MB
                # Other stats not available via PyTorch
                
        except Exception as e:
            logger.debug(f"Failed to get GPU stats: {e}")
        
        return stats
    
    def get_training_metrics(self) -> Optional[TrainingMetrics]:
        """
        Get current training metrics from miner engine.
        
        Returns:
            TrainingMetrics object or None if not available
        """
        if self.miner_engine is None:
            return None
        
        # Get latest training metrics from miner engine
        # This would need to be implemented in MinerEngine to track metrics
        # For now, return None or use cached metrics
        
        return self.last_training_metrics
    
    def get_miner_stats(self) -> MinerStats:
        """
        Get complete miner statistics.
        
        Returns:
            MinerStats object with all current stats
        """
        gpu_stats = self.get_gpu_stats()
        uptime = int(time.time() - self.start_time)
        
        # Calculate hash rate (gradients per hour)
        hash_rate = 0.0
        if self.miner_engine and uptime > 0:
            # Calculate gradients per hour based on successful submissions and uptime
            try:
                successful_submissions = getattr(self.miner_engine, 'successful_submissions', 0)
                if successful_submissions > 0:
                    # Convert uptime from seconds to hours
                    uptime_hours = uptime / 3600.0
                    if uptime_hours > 0:
                        hash_rate = successful_submissions / uptime_hours
            except AttributeError:
                pass
        
        return MinerStats(
            gpu_temp=gpu_stats["gpu_temp"],
            fan_speed=gpu_stats["fan_speed"],
            vram_usage=gpu_stats["vram_usage"],
            power_draw=gpu_stats["power_draw"],
            hash_rate=hash_rate,
            uptime=uptime,
            timestamp=int(time.time()),
        )
    
    def update_training_metrics(self, epoch: int, loss: float, accuracy: float, gradient_norm: float):
        """
        Update training metrics.
        
        Args:
            epoch: Current training epoch
            loss: Current loss value
            accuracy: Current accuracy
            gradient_norm: Current gradient norm
        """
        self.last_training_metrics = TrainingMetrics(
            epoch=epoch,
            loss=loss,
            accuracy=accuracy,
            gradient_norm=gradient_norm,
            timestamp=int(time.time()),
        )


# Global stats collector instance (will be initialized by miner engine)
_stats_collector: Optional[StatsCollector] = None


def get_stats_collector() -> Optional[StatsCollector]:
    """Get global stats collector instance."""
    return _stats_collector


def initialize_stats_collector(miner_engine=None):
    """Initialize global stats collector."""
    global _stats_collector
    _stats_collector = StatsCollector(miner_engine)
    return _stats_collector


#!/usr/bin/env python3
"""
R3MES Performance Monitor

Advanced performance monitoring and optimization utilities.
"""

import time
import psutil
import threading
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from collections import deque
import torch
import json
from pathlib import Path


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    gpu_memory_used_mb: float = 0.0
    gpu_utilization: float = 0.0
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)


class PerformanceProfiler:
    """Context manager for profiling code blocks."""
    
    def __init__(self, name: str, monitor: 'PerformanceMonitor'):
        self.name = name
        self.monitor = monitor
        self.start_time = None
        self.start_metrics = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.start_metrics = self.monitor.get_current_metrics()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        end_metrics = self.monitor.get_current_metrics()
        
        duration = end_time - self.start_time
        
        # Calculate deltas
        cpu_delta = end_metrics.cpu_percent - self.start_metrics.cpu_percent
        memory_delta = end_metrics.memory_used_mb - self.start_metrics.memory_used_mb
        gpu_memory_delta = end_metrics.gpu_memory_used_mb - self.start_metrics.gpu_memory_used_mb
        
        profile_data = {
            "name": self.name,
            "duration_ms": duration * 1000,
            "cpu_delta": cpu_delta,
            "memory_delta_mb": memory_delta,
            "gpu_memory_delta_mb": gpu_memory_delta,
            "timestamp": end_time,
        }
        
        self.monitor.add_profile_data(profile_data)


class PerformanceMonitor:
    """Advanced performance monitoring system."""
    
    def __init__(
        self,
        collection_interval: float = 1.0,
        history_size: int = 1000,
        enable_gpu_monitoring: bool = True,
        log_level: str = "INFO",
    ):
        """
        Initialize performance monitor.
        
        Args:
            collection_interval: Metrics collection interval in seconds
            history_size: Number of metrics to keep in history
            enable_gpu_monitoring: Enable GPU monitoring
            log_level: Logging level
        """
        self.collection_interval = collection_interval
        self.history_size = history_size
        self.enable_gpu_monitoring = enable_gpu_monitoring
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Metrics storage
        self.metrics_history = deque(maxlen=history_size)
        self.profile_data = deque(maxlen=history_size)
        
        # Monitoring state
        self.monitoring = False
        self.monitor_thread = None
        self.lock = threading.Lock()
        
        # GPU monitoring setup
        self.gpu_available = False
        if enable_gpu_monitoring and torch.cuda.is_available():
            try:
                torch.cuda.init()
                self.gpu_available = True
                self.logger.info("GPU monitoring enabled")
            except Exception as e:
                self.logger.warning(f"GPU monitoring failed: {e}")
        
        # Process monitoring
        self.process = psutil.Process()
        self.initial_io_counters = None
        self.initial_net_counters = None
        
        try:
            self.initial_io_counters = self.process.io_counters()
            self.initial_net_counters = psutil.net_io_counters()
        except (psutil.AccessDenied, AttributeError):
            self.logger.warning("IO/Network monitoring not available")
        
        self.logger.info("Performance monitor initialized")
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current system metrics."""
        try:
            # CPU and memory
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            memory_used_mb = memory_info.rss / (1024 * 1024)
            
            # GPU metrics
            gpu_memory_used_mb = 0.0
            gpu_utilization = 0.0
            
            if self.gpu_available:
                try:
                    gpu_memory_used_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                    # GPU utilization requires nvidia-ml-py
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_utilization = util.gpu
                    except ImportError:
                        pass
                except Exception as e:
                    self.logger.debug(f"GPU metrics error: {e}")
            
            # IO metrics
            disk_io_read_mb = 0.0
            disk_io_write_mb = 0.0
            
            if self.initial_io_counters:
                try:
                    current_io = self.process.io_counters()
                    disk_io_read_mb = (current_io.read_bytes - self.initial_io_counters.read_bytes) / (1024 * 1024)
                    disk_io_write_mb = (current_io.write_bytes - self.initial_io_counters.write_bytes) / (1024 * 1024)
                except (psutil.AccessDenied, AttributeError):
                    pass
            
            # Network metrics
            network_sent_mb = 0.0
            network_recv_mb = 0.0
            
            if self.initial_net_counters:
                try:
                    current_net = psutil.net_io_counters()
                    network_sent_mb = (current_net.bytes_sent - self.initial_net_counters.bytes_sent) / (1024 * 1024)
                    network_recv_mb = (current_net.bytes_recv - self.initial_net_counters.bytes_recv) / (1024 * 1024)
                except AttributeError:
                    pass
            
            return PerformanceMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                gpu_memory_used_mb=gpu_memory_used_mb,
                gpu_utilization=gpu_utilization,
                disk_io_read_mb=disk_io_read_mb,
                disk_io_write_mb=disk_io_write_mb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
            return PerformanceMetrics(
                timestamp=time.time(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
            )
    
    def start_monitoring(self):
        """Start background monitoring."""
        if self.monitoring:
            self.logger.warning("Monitoring already started")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                metrics = self.get_current_metrics()
                
                with self.lock:
                    self.metrics_history.append(metrics)
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.collection_interval)
    
    def add_profile_data(self, profile_data: Dict[str, Any]):
        """Add profiling data."""
        with self.lock:
            self.profile_data.append(profile_data)
    
    def profile(self, name: str) -> PerformanceProfiler:
        """Create performance profiler context manager."""
        return PerformanceProfiler(name, self)
    
    def get_metrics_summary(self, last_n_seconds: Optional[float] = None) -> Dict[str, Any]:
        """Get metrics summary."""
        with self.lock:
            metrics_list = list(self.metrics_history)
        
        if not metrics_list:
            return {"error": "No metrics available"}
        
        # Filter by time if specified
        if last_n_seconds:
            cutoff_time = time.time() - last_n_seconds
            metrics_list = [m for m in metrics_list if m.timestamp >= cutoff_time]
        
        if not metrics_list:
            return {"error": "No metrics in specified time range"}
        
        # Calculate statistics
        cpu_values = [m.cpu_percent for m in metrics_list]
        memory_values = [m.memory_used_mb for m in metrics_list]
        gpu_memory_values = [m.gpu_memory_used_mb for m in metrics_list]
        
        return {
            "time_range_seconds": last_n_seconds or "all",
            "sample_count": len(metrics_list),
            "cpu_percent": {
                "current": cpu_values[-1] if cpu_values else 0,
                "avg": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                "max": max(cpu_values) if cpu_values else 0,
                "min": min(cpu_values) if cpu_values else 0,
            },
            "memory_mb": {
                "current": memory_values[-1] if memory_values else 0,
                "avg": sum(memory_values) / len(memory_values) if memory_values else 0,
                "max": max(memory_values) if memory_values else 0,
                "min": min(memory_values) if memory_values else 0,
            },
            "gpu_memory_mb": {
                "current": gpu_memory_values[-1] if gpu_memory_values else 0,
                "avg": sum(gpu_memory_values) / len(gpu_memory_values) if gpu_memory_values else 0,
                "max": max(gpu_memory_values) if gpu_memory_values else 0,
                "min": min(gpu_memory_values) if gpu_memory_values else 0,
            },
            "timestamp": time.time(),
        }
    
    def get_profile_summary(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get profiling summary."""
        with self.lock:
            profile_list = list(self.profile_data)
        
        if operation_name:
            profile_list = [p for p in profile_list if p.get("name") == operation_name]
        
        if not profile_list:
            return {"error": "No profile data available"}
        
        # Group by operation name
        operations = {}
        for profile in profile_list:
            name = profile.get("name", "unknown")
            if name not in operations:
                operations[name] = []
            operations[name].append(profile)
        
        # Calculate statistics for each operation
        summary = {}
        for name, profiles in operations.items():
            durations = [p.get("duration_ms", 0) for p in profiles]
            memory_deltas = [p.get("memory_delta_mb", 0) for p in profiles]
            
            summary[name] = {
                "count": len(profiles),
                "duration_ms": {
                    "avg": sum(durations) / len(durations) if durations else 0,
                    "max": max(durations) if durations else 0,
                    "min": min(durations) if durations else 0,
                    "total": sum(durations),
                },
                "memory_delta_mb": {
                    "avg": sum(memory_deltas) / len(memory_deltas) if memory_deltas else 0,
                    "max": max(memory_deltas) if memory_deltas else 0,
                    "min": min(memory_deltas) if memory_deltas else 0,
                },
                "last_execution": max(p.get("timestamp", 0) for p in profiles) if profiles else 0,
            }
        
        return {
            "operations": summary,
            "total_profiles": len(profile_list),
            "timestamp": time.time(),
        }
    
    def export_metrics(self, filepath: str, format: str = "json"):
        """Export metrics to file."""
        with self.lock:
            metrics_list = list(self.metrics_history)
            profile_list = list(self.profile_data)
        
        export_data = {
            "export_timestamp": time.time(),
            "metrics_count": len(metrics_list),
            "profiles_count": len(profile_list),
            "metrics": [
                {
                    "timestamp": m.timestamp,
                    "cpu_percent": m.cpu_percent,
                    "memory_percent": m.memory_percent,
                    "memory_used_mb": m.memory_used_mb,
                    "gpu_memory_used_mb": m.gpu_memory_used_mb,
                    "gpu_utilization": m.gpu_utilization,
                    "disk_io_read_mb": m.disk_io_read_mb,
                    "disk_io_write_mb": m.disk_io_write_mb,
                    "network_sent_mb": m.network_sent_mb,
                    "network_recv_mb": m.network_recv_mb,
                    "custom_metrics": m.custom_metrics,
                }
                for m in metrics_list
            ],
            "profiles": profile_list,
        }
        
        filepath = Path(filepath)
        
        if format.lower() == "json":
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Metrics exported to {filepath}")
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get performance optimization recommendations."""
        recommendations = []
        
        summary = self.get_metrics_summary(last_n_seconds=60)  # Last minute
        if "error" in summary:
            return ["No metrics available for recommendations"]
        
        # CPU recommendations
        cpu_avg = summary["cpu_percent"]["avg"]
        if cpu_avg > 80:
            recommendations.append("High CPU usage detected. Consider reducing batch size or using more efficient algorithms.")
        elif cpu_avg < 20:
            recommendations.append("Low CPU usage. Consider increasing batch size for better throughput.")
        
        # Memory recommendations
        memory_current = summary["memory_mb"]["current"]
        if memory_current > 8000:  # 8GB
            recommendations.append("High memory usage. Consider enabling gradient checkpointing or reducing model size.")
        
        # GPU memory recommendations
        gpu_memory_current = summary["gpu_memory_mb"]["current"]
        if gpu_memory_current > 0:
            if self.gpu_available:
                try:
                    total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                    gpu_usage_percent = (gpu_memory_current / total_gpu_memory) * 100
                    
                    if gpu_usage_percent > 90:
                        recommendations.append("GPU memory usage very high. Consider reducing batch size or using gradient checkpointing.")
                    elif gpu_usage_percent < 30:
                        recommendations.append("GPU memory underutilized. Consider increasing batch size for better efficiency.")
                except Exception:
                    pass
        
        # Profile-based recommendations
        profile_summary = self.get_profile_summary()
        if "operations" in profile_summary:
            for op_name, op_stats in profile_summary["operations"].items():
                avg_duration = op_stats["duration_ms"]["avg"]
                if avg_duration > 1000:  # 1 second
                    recommendations.append(f"Operation '{op_name}' is slow (avg: {avg_duration:.1f}ms). Consider optimization.")
        
        if not recommendations:
            recommendations.append("Performance looks good! No specific recommendations at this time.")
        
        return recommendations
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()


# Global performance monitor instance
_global_monitor = None

def get_global_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor

def profile(name: str):
    """Decorator for profiling functions."""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            monitor = get_global_monitor()
            with monitor.profile(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator
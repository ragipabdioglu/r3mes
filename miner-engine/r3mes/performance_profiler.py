"""
Performance Profiler for R3MES Miner Engine

Provides performance profiling capabilities for training iterations, GPU kernels, and IPFS operations.
"""

import time
import threading
import json
import os
from pathlib import Path
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import psutil


@dataclass
class ProfileEntry:
    """Represents a single profiling entry"""
    function: str
    call_count: int = 0
    total_duration: float = 0.0  # in seconds
    min_duration: float = float('inf')
    max_duration: float = 0.0
    avg_duration: float = 0.0
    last_call_time: Optional[datetime] = None
    
    def update(self, duration: float):
        """Update profile entry with a new duration measurement"""
        self.call_count += 1
        self.total_duration += duration
        self.min_duration = min(self.min_duration, duration)
        self.max_duration = max(self.max_duration, duration)
        self.avg_duration = self.total_duration / self.call_count
        self.last_call_time = datetime.now()


@dataclass
class ProfileStats:
    """Represents overall profiling statistics"""
    start_time: datetime
    end_time: datetime
    duration: float  # in seconds
    profiles: Dict[str, ProfileEntry] = field(default_factory=dict)
    memory_stats: Dict[str, Any] = field(default_factory=dict)
    gpu_stats: Dict[str, Any] = field(default_factory=dict)
    cpu_percent: float = 0.0
    thread_count: int = 0


class PerformanceProfiler:
    """Performance profiler for miner engine operations"""
    
    def __init__(self, enabled: bool = True, export_path: Optional[str] = None):
        """
        Initialize performance profiler.
        
        Args:
            enabled: Whether profiling is enabled
            export_path: Path to export profile data
        """
        self.enabled = enabled
        self.export_path = export_path or os.path.expanduser("~/.r3mes/profiles")
        self.profiles: Dict[str, ProfileEntry] = {}
        self.lock = threading.Lock()
        self.start_time = datetime.now()
    
    def start_timer(self, function_name: str) -> Callable:
        """
        Start a timer for a function/procedure.
        
        Args:
            function_name: Name of the function being profiled
            
        Returns:
            A callable that should be called when the operation completes
        """
        if not self.enabled:
            return lambda: None  # No-op if profiling is disabled
        
        start = time.perf_counter()
        
        def end_timer():
            duration = time.perf_counter() - start
            self._record_profile(function_name, duration)
        
        return end_timer
    
    def _record_profile(self, function_name: str, duration: float):
        """Record a profile entry"""
        with self.lock:
            if function_name not in self.profiles:
                self.profiles[function_name] = ProfileEntry(function=function_name)
            self.profiles[function_name].update(duration)
    
    def profile_function(self, function_name: str, fn: Callable) -> Any:
        """
        Profile a function execution.
        
        Args:
            function_name: Name of the function
            fn: Function to profile
            
        Returns:
            Result of function execution
        """
        if not self.enabled:
            return fn()
        
        end = self.start_timer(function_name)
        try:
            result = fn()
            return result
        finally:
            end()
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """
        Get GPU statistics (if available).
        
        Returns:
            Dictionary with GPU stats
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
    
    def get_stats(self) -> ProfileStats:
        """Get current profiling statistics"""
        with self.lock:
            profiles_copy = {k: ProfileEntry(**asdict(v)) for k, v in self.profiles.items()}
        
        # Get process stats
        process = psutil.Process()
        try:
            cpu_percent = process.cpu_percent(interval=0.1)
            thread_count = process.num_threads()
            memory_info = process.memory_info()
            memory_stats = {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
            }
        except Exception:
            cpu_percent = 0.0
            thread_count = 0
            memory_stats = {}
        
        # Get GPU stats
        gpu_stats = self.get_gpu_stats()
        
        return ProfileStats(
            start_time=self.start_time,
            end_time=datetime.now(),
            duration=(datetime.now() - self.start_time).total_seconds(),
            profiles=profiles_copy,
            memory_stats=memory_stats,
            gpu_stats=gpu_stats,
            cpu_percent=cpu_percent,
            thread_count=thread_count,
        )
    
    def export_stats(self, filename: Optional[str] = None) -> str:
        """
        Export profiling statistics to a file.
        
        Args:
            filename: Output filename (default: auto-generated)
            
        Returns:
            Path to exported file
        """
        if not self.enabled:
            return ""
        
        stats = self.get_stats()
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.export_path, f"profile_{timestamp}.json")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Convert to dict for JSON serialization
        stats_dict = {
            "start_time": stats.start_time.isoformat(),
            "end_time": stats.end_time.isoformat(),
            "duration": stats.duration,
            "profiles": {k: asdict(v) for k, v in stats.profiles.items()},
            "memory_stats": stats.memory_stats,
            "gpu_stats": stats.gpu_stats,
            "cpu_percent": stats.cpu_percent,
            "thread_count": stats.thread_count,
        }
        
        # Write to file
        with open(filename, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        
        return filename
    
    def reset(self):
        """Reset all profiling data"""
        with self.lock:
            self.profiles = {}
            self.start_time = datetime.now()


# Global profiler instance (lazy-loaded)
_global_profiler: Optional[PerformanceProfiler] = None


def get_profiler() -> PerformanceProfiler:
    """
    Get the global performance profiler (cached).
    
    Returns:
        PerformanceProfiler instance
    """
    global _global_profiler
    if _global_profiler is None:
        # Check debug config
        try:
            from r3mes.debug_config import get_debug_config
            debug_config = get_debug_config()
            enabled = debug_config.enabled and debug_config.is_miner_enabled() and debug_config.profiling
            export_path = debug_config.profile_output if debug_config.enabled else None
        except ImportError:
            enabled = False
            export_path = None
        
        _global_profiler = PerformanceProfiler(enabled=enabled, export_path=export_path)
    return _global_profiler

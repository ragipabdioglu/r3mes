"""
Performance Profiler for R3MES Backend

Provides performance profiling capabilities for requests, database queries, and model inference.
"""

import time
import threading
import json
import os
from pathlib import Path
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
import psutil
import tracemalloc


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
    cpu_percent: float = 0.0
    thread_count: int = 0


class PerformanceProfiler:
    """Performance profiler for backend operations"""
    
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
        self.memory_tracking = False
        
        # Start memory tracking if enabled
        if self.enabled:
            try:
                tracemalloc.start()
                self.memory_tracking = True
            except RuntimeError:
                # Already started, ignore
                pass
    
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
    
    def profile_async(self, function_name: str):
        """
        Decorator for profiling async functions.
        
        Usage:
            @profiler.profile_async("my_function")
            async def my_function():
                ...
        """
        def decorator(fn):
            async def wrapper(*args, **kwargs):
                if not self.enabled:
                    return await fn(*args, **kwargs)
                
                end = self.start_timer(function_name)
                try:
                    return await fn(*args, **kwargs)
                finally:
                    end()
            return wrapper
        return decorator
    
    def get_stats(self) -> ProfileStats:
        """Get current profiling statistics"""
        with self.lock:
            profiles_copy = {k: ProfileEntry(**asdict(v)) for k, v in self.profiles.items()}
        
        # Get memory stats
        memory_stats = {}
        if self.memory_tracking:
            try:
                current, peak = tracemalloc.get_traced_memory()
                memory_stats = {
                    "current_mb": current / 1024 / 1024,
                    "peak_mb": peak / 1024 / 1024,
                }
            except Exception:
                pass
        
        # Get process stats
        process = psutil.Process()
        try:
            cpu_percent = process.cpu_percent(interval=0.1)
            thread_count = process.num_threads()
        except Exception:
            cpu_percent = 0.0
            thread_count = 0
        
        return ProfileStats(
            start_time=self.start_time,
            end_time=datetime.now(),
            duration=(datetime.now() - self.start_time).total_seconds(),
            profiles=profiles_copy,
            memory_stats=memory_stats,
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
            from .debug_config import get_debug_config
            debug_config = get_debug_config()
            enabled = debug_config.enabled and debug_config.is_backend_enabled() and debug_config.profiling
            export_path = debug_config.profile_output if debug_config.enabled else None
        except ImportError:
            enabled = False
            export_path = None
        
        _global_profiler = PerformanceProfiler(enabled=enabled, export_path=export_path)
    return _global_profiler

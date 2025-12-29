"""
Distributed Tracing for R3MES Miner Engine

Provides training operation tracing, blockchain transaction tracing, and IPFS operation tracing.
"""

import uuid
import time
import logging
import threading
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field, asdict
from datetime import datetime

try:
    from r3mes.debug_config import get_debug_config
except ImportError:
    # Fallback if debug_config is not available
    def get_debug_config():
        class DummyConfig:
            enabled = False
            trace_enabled = False
            trace_buffer_size = 10000
        return DummyConfig()

logger = logging.getLogger(__name__)


@dataclass
class TraceEntry:
    """Represents a single trace entry"""
    trace_id: str
    component: str
    operation: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    fields: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class TraceCollector:
    """Collects trace entries"""
    
    def __init__(self, max_buffer: int = 10000):
        """
        Initialize trace collector.
        
        Args:
            max_buffer: Maximum number of traces to keep in memory
        """
        try:
            self.debug_config = get_debug_config()
            self.enabled = self.debug_config.enabled and self.debug_config.is_miner_enabled() and self.debug_config.trace_enabled
            self.max_buffer = self.debug_config.trace_buffer_size if self.debug_config.enabled else max_buffer
        except Exception:
            self.enabled = False
            self.max_buffer = max_buffer
        
        self.traces: Dict[str, List[TraceEntry]] = {}
        self.lock = threading.Lock()
    
    def generate_trace_id(self) -> str:
        """Generate a new trace ID"""
        return str(uuid.uuid4())
    
    def start_trace(self, trace_id: str, operation: str, component: str = "miner", **fields) -> TraceEntry:
        """
        Start a new trace entry.
        
        Args:
            trace_id: Trace ID (use generate_trace_id() if starting new trace)
            operation: Operation name
            component: Component name (default: "miner")
            **fields: Additional fields to include
            
        Returns:
            TraceEntry instance
        """
        if not self.enabled:
            return None
        
        entry = TraceEntry(
            trace_id=trace_id,
            component=component,
            operation=operation,
            start_time=datetime.now(),
            fields=fields,
        )
        
        with self.lock:
            if trace_id not in self.traces:
                self.traces[trace_id] = []
            self.traces[trace_id].append(entry)
            
            # Trim buffer if needed
            if len(self.traces[trace_id]) > self.max_buffer:
                self.traces[trace_id] = self.traces[trace_id][-self.max_buffer:]
        
        return entry
    
    def end_trace(self, entry: TraceEntry, error: Optional[Exception] = None):
        """
        End a trace entry.
        
        Args:
            entry: TraceEntry instance
            error: Optional error that occurred
        """
        if not self.enabled or entry is None:
            return
        
        entry.end_time = datetime.now()
        entry.duration_ms = (entry.end_time - entry.start_time).total_seconds() * 1000
        if error:
            entry.error = str(error)
    
    def get_traces(self, trace_id: str) -> List[TraceEntry]:
        """
        Get all traces for a trace ID.
        
        Args:
            trace_id: Trace ID
            
        Returns:
            List of TraceEntry instances
        """
        if not self.enabled:
            return []
        
        with self.lock:
            traces = self.traces.get(trace_id, [])
            # Return a copy
            return [TraceEntry(**asdict(t)) for t in traces]
    
    def get_all_trace_ids(self) -> List[str]:
        """Get all trace IDs"""
        if not self.enabled:
            return []
        
        with self.lock:
            return list(self.traces.keys())
    
    def clear_traces(self):
        """Clear all traces"""
        if not self.enabled:
            return
        
        with self.lock:
            self.traces = {}


# Global trace collector instance
_global_trace_collector: Optional[TraceCollector] = None


def get_trace_collector() -> TraceCollector:
    """
    Get the global trace collector (cached).
    
    Returns:
        TraceCollector instance
    """
    global _global_trace_collector
    if _global_trace_collector is None:
        _global_trace_collector = TraceCollector()
    return _global_trace_collector

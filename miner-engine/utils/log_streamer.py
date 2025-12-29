#!/usr/bin/env python3
"""
Log Streamer for Python Miner

Streams miner logs to Go node via WebSocket for dashboard display.
"""

import json
import logging
import time
from typing import Optional
from datetime import datetime
import websocket
import threading
from queue import Queue


class WebSocketLogHandler(logging.Handler):
    """
    Custom logging handler that streams logs to WebSocket.
    """
    
    def __init__(
        self,
        ws_url: str = "ws://localhost:1317/ws?topic=miner_logs",
        miner_address: Optional[str] = None,
    ):
        """
        Initialize WebSocket log handler.
        
        Args:
            ws_url: WebSocket URL for log streaming
            miner_address: Miner's blockchain address (optional)
        """
        super().__init__()
        self.ws_url = ws_url
        self.miner_address = miner_address
        self.ws: Optional[websocket.WebSocketApp] = None
        self.log_queue = Queue()
        self.connected = False
        self._connect()
    
    def _connect(self):
        """Connect to WebSocket server."""
        try:
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            
            # Start WebSocket in background thread
            def run_ws():
                self.ws.run_forever()
            
            ws_thread = threading.Thread(target=run_ws, daemon=True)
            ws_thread.start()
            
            # Wait for connection
            timeout = 5
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if not self.connected:
                print(f"Warning: Could not connect to WebSocket at {self.ws_url}")
        except Exception as e:
            print(f"Warning: Failed to initialize WebSocket log handler: {e}")
    
    def _on_open(self, ws):
        """WebSocket connection opened."""
        self.connected = True
        print(f"✅ Log streaming connected to {self.ws_url}")
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket message (not used for logs)."""
        pass
    
    def _on_error(self, ws, error):
        """Handle WebSocket error."""
        self.connected = False
        print(f"WebSocket log handler error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close."""
        self.connected = False
        print("WebSocket log stream closed")
    
    def emit(self, record: logging.LogRecord):
        """
        Emit a log record to WebSocket.
        
        Args:
            record: LogRecord to emit
        """
        if not self.connected or self.ws is None:
            return
        
        try:
            # Format log message
            log_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname.lower(),
                "message": self.format(record),
                "miner_address": self.miner_address,
            }
            
            # Send to WebSocket
            try:
                self.ws.send(json.dumps(log_data))
            except Exception as e:
                # If send fails, queue for retry (simple implementation)
                if self.log_queue.qsize() < 100:  # Limit queue size
                    self.log_queue.put(log_data)
        except Exception:
            self.handleError(record)
    
    def flush(self):
        """Flush queued logs."""
        while not self.log_queue.empty():
            try:
                log_data = self.log_queue.get_nowait()
                if self.connected and self.ws is not None:
                    self.ws.send(json.dumps(log_data))
            except Exception:
                pass


def setup_miner_logging(
    miner_address: Optional[str] = None,
    ws_url: str = "ws://localhost:1317/ws?topic=miner_logs",
    log_level: int = logging.INFO,
) -> logging.Logger:
    """
    Setup logging for Python miner with WebSocket streaming.
    
    Args:
        miner_address: Miner's blockchain address
        ws_url: WebSocket URL for log streaming
        log_level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("r3mes_miner")
    logger.setLevel(log_level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler (for terminal output)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    # Add sensitive data filter to console handler
    try:
        from utils.logger import SensitiveDataFilter
        console_handler.addFilter(SensitiveDataFilter())
    except ImportError:
        # SensitiveDataFilter not available, continue without it
        pass
    logger.addHandler(console_handler)
    
    # WebSocket handler (for dashboard streaming)
    try:
        ws_handler = WebSocketLogHandler(ws_url=ws_url, miner_address=miner_address)
        ws_handler.setLevel(log_level)
        ws_formatter = logging.Formatter('%(message)s')
        ws_handler.setFormatter(ws_formatter)
        # Add sensitive data filter to WebSocket handler
        try:
            from utils.logger import SensitiveDataFilter
            ws_handler.addFilter(SensitiveDataFilter())
        except ImportError:
            pass
        logger.addHandler(ws_handler)
    except Exception as e:
        print(f"Warning: Could not setup WebSocket log handler: {e}")
        print("Logs will only be displayed in console")
    
    return logger


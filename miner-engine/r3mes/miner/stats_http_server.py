"""
HTTP Server for Miner Stats

Exposes miner statistics via HTTP endpoint for Desktop Launcher and Go node to query.
This allows the Desktop Launcher and Go node's WebSocket handler to fetch real-time stats.
Also exposes Prometheus metrics endpoint for monitoring.

Endpoints:
- GET /stats - Desktop Launcher compatible stats (flat JSON format)
- GET /stats/full - Full nested stats format
- GET /health - Health check
- GET /metrics - Prometheus metrics
"""

import http.server
import socketserver
import json
import logging
import threading
import time
import os
from typing import Optional, Dict, Any, List
from dataclasses import asdict

logger = logging.getLogger(__name__)

# Prometheus client for metrics export
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not available, /metrics endpoint will not work")

if PROMETHEUS_AVAILABLE:
    # Miner metrics
    miner_gpu_temp = Gauge('r3mes_miner_gpu_temp_celsius', 'GPU temperature in Celsius', ['gpu_id'])
    miner_gpu_utilization = Gauge('r3mes_miner_gpu_utilization_percent', 'GPU utilization percentage', ['gpu_id'])
    miner_vram_usage = Gauge('r3mes_miner_vram_usage_bytes', 'VRAM usage in bytes', ['gpu_id'])
    miner_vram_total = Gauge('r3mes_miner_vram_total_bytes', 'Total VRAM in bytes', ['gpu_id'])
    miner_power_draw = Gauge('r3mes_miner_power_draw_watts', 'Power draw in watts', ['gpu_id'])
    miner_hashrate = Gauge('r3mes_miner_hashrate_gradients_per_hour', 'Hash rate in gradients per hour')
    miner_uptime = Gauge('r3mes_miner_uptime_seconds', 'Miner uptime in seconds')
    miner_submissions_total = Counter('r3mes_miner_submissions_total', 'Total number of gradient submissions', ['status'])
    miner_training_duration = Histogram('r3mes_miner_training_duration_seconds', 'Training duration in seconds', buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0])
    miner_training_loss = Gauge('r3mes_miner_training_loss', 'Current training loss')
    miner_training_accuracy = Gauge('r3mes_miner_training_accuracy', 'Current training accuracy')
    miner_gradient_norm = Gauge('r3mes_miner_gradient_norm', 'Current gradient norm')
    miner_training_epoch = Gauge('r3mes_miner_training_epoch', 'Current training epoch')


class LossTrendTracker:
    """Tracks loss values to determine trend."""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.loss_history: List[float] = []
    
    def add_loss(self, loss: float):
        """Add a loss value to history."""
        self.loss_history.append(loss)
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)
    
    def get_trend(self) -> str:
        """
        Calculate loss trend.
        
        Returns:
            "decreasing", "increasing", or "stable"
        """
        if len(self.loss_history) < 3:
            return "stable"
        
        # Calculate average of first half vs second half
        mid = len(self.loss_history) // 2
        first_half_avg = sum(self.loss_history[:mid]) / mid if mid > 0 else 0
        second_half_avg = sum(self.loss_history[mid:]) / (len(self.loss_history) - mid) if len(self.loss_history) > mid else 0
        
        # Determine trend with 5% threshold
        if second_half_avg < first_half_avg * 0.95:
            return "decreasing"
        elif second_half_avg > first_half_avg * 1.05:
            return "increasing"
        else:
            return "stable"


# Global loss trend tracker
_loss_trend_tracker = LossTrendTracker()


def get_vram_total_mb() -> int:
    """
    Get total VRAM in MB.
    
    Tries multiple methods:
    1. NVML (nvidia-ml-py)
    2. PyTorch CUDA
    3. Returns 0 if no GPU available
    """
    # Method 1: Try NVML
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_mb = int(mem_info.total / 1024 / 1024)
        logger.debug(f"Got VRAM via NVML: {vram_mb} MB")
        return vram_mb
    except ImportError:
        logger.debug("pynvml not available, trying PyTorch")
    except Exception as e:
        logger.debug(f"NVML error: {e}, trying PyTorch")
    
    # Method 2: Try PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_mb = int(props.total_memory / 1024 / 1024)
            logger.debug(f"Got VRAM via PyTorch: {vram_mb} MB")
            return vram_mb
        else:
            logger.debug("CUDA not available via PyTorch")
    except ImportError:
        logger.debug("PyTorch not available")
    except Exception as e:
        logger.warning(f"PyTorch CUDA error: {e}")
    
    # No GPU available
    logger.info("No GPU detected, returning 0 for VRAM")
    return 0


def estimate_earnings_per_day(hash_rate: float, successful_submissions: int, uptime_seconds: int) -> float:
    """
    Estimate daily earnings based on hash rate and submission success.
    
    This is a rough estimate based on:
    - Current hash rate (gradients/hour)
    - Historical success rate
    - Network reward rate (configurable)
    
    Returns:
        Estimated earnings per day in REMES tokens
    """
    # Get reward per gradient from environment or use default
    reward_per_gradient = float(os.getenv("R3MES_REWARD_PER_GRADIENT", "0.1"))
    
    if uptime_seconds < 60:  # Need at least 1 minute of data
        return 0.0
    
    # Calculate gradients per day based on hash rate
    gradients_per_day = hash_rate * 24  # hash_rate is gradients/hour
    
    # Estimate earnings
    estimated_earnings = gradients_per_day * reward_per_gradient
    
    return round(estimated_earnings, 4)


class StatsHTTPHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for miner stats endpoint."""
    
    def do_GET(self):
        """Handle GET requests for miner stats."""
        if self.path == "/stats":
            self._handle_stats_flat()
        elif self.path == "/stats/full":
            self._handle_stats_full()
        elif self.path == "/health":
            self._handle_health()
        elif self.path == "/metrics":
            self._handle_metrics()
        else:
            self._send_error(404, "Not Found")
    
    def _handle_stats_flat(self):
        """
        Handle /stats endpoint - Desktop Launcher compatible flat format.
        
        Returns JSON in format expected by Desktop Launcher:
        {
            "hashrate": float,
            "loss": float,
            "loss_trend": "decreasing" | "increasing" | "stable",
            "estimated_earnings_per_day": float,
            "current_balance": float,
            "gpu_temp": float,
            "vram_usage_mb": int,
            "vram_total_mb": int,
            "training_epoch": int,
            "gradient_norm": float,
            "uptime_seconds": int
        }
        """
        from r3mes.miner.stats_server import get_stats_collector
        
        stats_collector = get_stats_collector()
        
        if stats_collector is None:
            # Return default values if stats collector not initialized
            response = {
                "hashrate": 0.0,
                "loss": 0.0,
                "loss_trend": "stable",
                "estimated_earnings_per_day": 0.0,
                "current_balance": 0.0,
                "gpu_temp": 0.0,
                "vram_usage_mb": 0,
                "vram_total_mb": get_vram_total_mb(),
                "training_epoch": 0,
                "gradient_norm": 0.0,
                "uptime_seconds": 0
            }
            self._send_json(response)
            return
        
        try:
            miner_stats = stats_collector.get_miner_stats()
            training_metrics = stats_collector.get_training_metrics()
            
            # Get loss and update trend tracker
            loss = training_metrics.loss if training_metrics else 0.0
            if loss > 0:
                _loss_trend_tracker.add_loss(loss)
            
            # Get successful submissions and uptime for earnings estimate
            successful_submissions = 0
            if stats_collector.miner_engine:
                successful_submissions = getattr(stats_collector.miner_engine, 'successful_submissions', 0)
            
            # Calculate estimated earnings
            estimated_earnings = estimate_earnings_per_day(
                hash_rate=miner_stats.hash_rate,
                successful_submissions=successful_submissions,
                uptime_seconds=miner_stats.uptime
            )
            
            # Get current balance (would need blockchain query - placeholder for now)
            # In production, this would query the blockchain for wallet balance
            current_balance = 0.0
            
            response = {
                "hashrate": miner_stats.hash_rate,
                "loss": loss,
                "loss_trend": _loss_trend_tracker.get_trend(),
                "estimated_earnings_per_day": estimated_earnings,
                "current_balance": current_balance,
                "gpu_temp": miner_stats.gpu_temp,
                "vram_usage_mb": miner_stats.vram_usage,
                "vram_total_mb": get_vram_total_mb(),
                "training_epoch": training_metrics.epoch if training_metrics else 0,
                "gradient_norm": training_metrics.gradient_norm if training_metrics else 0.0,
                "uptime_seconds": miner_stats.uptime
            }
            
            self._send_json(response)
        except Exception as e:
            logger.error(f"Error getting stats: {e}", exc_info=True)
            self._send_error(500, f"Internal Server Error: {str(e)}")
    
    def _handle_stats_full(self):
        """Handle /stats/full endpoint - Full nested format."""
        from r3mes.miner.stats_server import get_stats_collector
        
        stats_collector = get_stats_collector()
        
        if stats_collector is None:
            self._send_error(503, "Stats collector not initialized")
            return
        
        try:
            miner_stats = stats_collector.get_miner_stats()
            training_metrics = stats_collector.get_training_metrics()
            
            response = {
                "miner_stats": {
                    "gpu_temp": miner_stats.gpu_temp,
                    "fan_speed": miner_stats.fan_speed,
                    "vram_usage": miner_stats.vram_usage,
                    "vram_total": get_vram_total_mb(),
                    "power_draw": miner_stats.power_draw,
                    "hash_rate": miner_stats.hash_rate,
                    "uptime": miner_stats.uptime,
                    "timestamp": miner_stats.timestamp,
                },
                "training_metrics": {
                    "epoch": training_metrics.epoch if training_metrics else 0,
                    "loss": training_metrics.loss if training_metrics else 0.0,
                    "loss_trend": _loss_trend_tracker.get_trend(),
                    "accuracy": training_metrics.accuracy if training_metrics else 0.0,
                    "gradient_norm": training_metrics.gradient_norm if training_metrics else 0.0,
                    "timestamp": training_metrics.timestamp if training_metrics else 0,
                },
            }
            
            self._send_json(response)
        except Exception as e:
            logger.error(f"Error getting stats: {e}", exc_info=True)
            self._send_error(500, f"Internal Server Error: {str(e)}")
    
    def _handle_health(self):
        """Handle /health endpoint."""
        from r3mes.miner.stats_server import get_stats_collector
        
        stats_collector = get_stats_collector()
        
        health = {
            "status": "ok" if stats_collector is not None else "not_initialized",
            "timestamp": int(time.time()),
        }
        
        self._send_json(health)
    
    def _handle_metrics(self):
        """Handle /metrics endpoint for Prometheus."""
        if not PROMETHEUS_AVAILABLE:
            self._send_error(503, "Prometheus client not available")
            return
        
        from r3mes.miner.stats_server import get_stats_collector
        
        stats_collector = get_stats_collector()
        if stats_collector is None:
            self._send_error(503, "Stats collector not initialized")
            return
        
        try:
            # Update Prometheus metrics from stats collector
            miner_stats = stats_collector.get_miner_stats()
            training_metrics = stats_collector.get_training_metrics()
            
            # Update GPU metrics
            miner_gpu_temp.labels(gpu_id="0").set(miner_stats.gpu_temp)
            miner_vram_usage.labels(gpu_id="0").set(miner_stats.vram_usage * 1024 * 1024)  # Convert MB to bytes
            miner_vram_total.labels(gpu_id="0").set(get_vram_total_mb() * 1024 * 1024)
            miner_power_draw.labels(gpu_id="0").set(miner_stats.power_draw)
            miner_hashrate.set(miner_stats.hash_rate)
            miner_uptime.set(miner_stats.uptime)
            
            # Update training metrics if available
            if training_metrics:
                miner_training_loss.set(training_metrics.loss)
                miner_training_accuracy.set(training_metrics.accuracy)
                miner_gradient_norm.set(training_metrics.gradient_norm)
                miner_training_epoch.set(training_metrics.epoch)
            
            # Return Prometheus metrics
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPE_LATEST)
            self.end_headers()
            self.wfile.write(generate_latest())
        except Exception as e:
            logger.error(f"Error generating metrics: {e}", exc_info=True)
            self._send_error(500, f"Internal Server Error: {str(e)}")
    
    def _send_json(self, data: dict):
        """Send JSON response."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))
    
    def _send_error(self, code: int, message: str):
        """Send error response."""
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        error_response = {"error": message}
        self.wfile.write(json.dumps(error_response).encode("utf-8"))
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    
    def log_message(self, format, *args):
        """Override to use logger instead of stderr."""
        logger.debug(f"{self.address_string()} - {format % args}")


class ThreadedHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Threaded HTTP server for handling concurrent requests."""
    allow_reuse_address = True
    daemon_threads = True


# Global server instance for graceful shutdown
_http_server: Optional[ThreadedHTTPServer] = None
_server_thread: Optional[threading.Thread] = None


def start_stats_server(port: Optional[int] = None, host: Optional[str] = None, blocking: bool = True):
    """
    Start HTTP server for miner stats.
    
    Args:
        port: Port to listen on (default: from R3MES_STATS_PORT env var or 8080)
        host: Host to bind to (default: from R3MES_STATS_HOST env var or 0.0.0.0)
        blocking: If True, block until server is stopped. If False, run in background thread.
    
    Returns:
        If blocking=False, returns the server thread
    """
    global _http_server, _server_thread
    
    # Get port from parameter, environment variable, or default
    if port is None:
        port = int(os.getenv("R3MES_STATS_PORT", "8080"))
    
    # Get host from parameter, environment variable, or default
    # Default to 0.0.0.0 to allow external connections (Desktop Launcher needs this)
    if host is None:
        host = os.getenv("R3MES_STATS_HOST", "0.0.0.0")
    
    # Production localhost validation
    is_production = os.getenv("R3MES_ENV", "development").lower() == "production"
    if is_production:
        if host.lower() in ("localhost", "127.0.0.1", "::1") or host.startswith("127."):
            raise ValueError(
                f"R3MES_STATS_HOST cannot use localhost in production: {host}. "
                "Please set R3MES_STATS_HOST to a production hostname or IP address."
            )
    
    try:
        _http_server = ThreadedHTTPServer((host, port), StatsHTTPHandler)
        
        logger.info(f"Miner stats HTTP server starting on {host}:{port}")
        logger.info(f"Stats endpoint: http://{host}:{port}/stats")
        logger.info(f"Full stats endpoint: http://{host}:{port}/stats/full")
        logger.info(f"Health endpoint: http://{host}:{port}/health")
        if PROMETHEUS_AVAILABLE:
            logger.info(f"Metrics endpoint: http://{host}:{port}/metrics")
        
        print(f"✅ Stats HTTP server started on http://{host}:{port}/stats")
        
        if blocking:
            _http_server.serve_forever()
        else:
            _server_thread = threading.Thread(target=_http_server.serve_forever, daemon=True)
            _server_thread.start()
            return _server_thread
            
    except OSError as e:
        if "Address already in use" in str(e):
            logger.warning(f"Port {port} already in use, stats server may already be running")
            print(f"⚠️  Port {port} already in use")
        else:
            logger.error(f"Failed to start stats server: {e}")
            raise


def stop_stats_server():
    """Stop the HTTP stats server gracefully."""
    global _http_server, _server_thread
    
    if _http_server is not None:
        logger.info("Stopping stats HTTP server...")
        _http_server.shutdown()
        _http_server.server_close()
        _http_server = None
        
        if _server_thread is not None:
            _server_thread.join(timeout=5.0)
            _server_thread = None
        
        logger.info("Stats HTTP server stopped")


def is_server_running() -> bool:
    """Check if stats server is running."""
    return _http_server is not None


if __name__ == "__main__":
    # For testing
    logging.basicConfig(level=logging.INFO)
    start_stats_server()

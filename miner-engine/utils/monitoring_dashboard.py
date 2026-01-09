#!/usr/bin/env python3
"""
R3MES Monitoring Dashboard

Web-based monitoring dashboard for miner engine performance and status.
"""

import asyncio
import json
import time
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import threading

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from utils.performance_monitor import PerformanceMonitor, get_global_monitor


class MonitoringDashboard:
    """Web-based monitoring dashboard."""
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        performance_monitor: Optional[PerformanceMonitor] = None,
        title: str = "R3MES Miner Engine Monitor",
    ):
        """
        Initialize monitoring dashboard.
        
        Args:
            host: Dashboard host
            port: Dashboard port
            performance_monitor: Performance monitor instance
            title: Dashboard title
        """
        if not HAS_FASTAPI:
            raise ImportError("FastAPI is required for monitoring dashboard. Install with: pip install fastapi uvicorn jinja2")
        
        self.host = host
        self.port = port
        self.title = title
        self.performance_monitor = performance_monitor or get_global_monitor()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # WebSocket connections
        self.active_connections: List[WebSocket] = []
        
        # Create FastAPI app
        self.app = FastAPI(
            title="R3MES Monitoring Dashboard",
            description="Real-time monitoring dashboard for R3MES Miner Engine",
            version="1.0.0",
        )
        
        # Setup routes
        self._setup_routes()
        
        # Background task for broadcasting metrics
        self.broadcasting = False
        self.broadcast_task = None
        
        self.logger.info(f"Monitoring dashboard initialized on {host}:{port}")
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home(request: Request):
            """Main dashboard page."""
            return HTMLResponse(content=self._get_dashboard_html())
        
        @self.app.get("/api/metrics")
        async def get_metrics():
            """Get current metrics."""
            summary = self.performance_monitor.get_metrics_summary(last_n_seconds=60)
            return JSONResponse(content=summary)
        
        @self.app.get("/api/profiles")
        async def get_profiles():
            """Get profiling data."""
            summary = self.performance_monitor.get_profile_summary()
            return JSONResponse(content=summary)
        
        @self.app.get("/api/recommendations")
        async def get_recommendations():
            """Get optimization recommendations."""
            recommendations = self.performance_monitor.get_optimization_recommendations()
            return JSONResponse(content={"recommendations": recommendations})
        
        @self.app.get("/api/status")
        async def get_status():
            """Get system status."""
            return JSONResponse(content={
                "status": "running",
                "timestamp": time.time(),
                "monitoring_active": self.performance_monitor.monitoring,
                "connections": len(self.active_connections),
            })
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await websocket.accept()
            self.active_connections.append(websocket)
            
            try:
                while True:
                    # Keep connection alive
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
    
    def _get_dashboard_html(self) -> str:
        """Generate dashboard HTML."""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .metric-title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-unit {{
            font-size: 14px;
            color: #666;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .recommendations {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .recommendation {{
            padding: 10px;
            margin: 5px 0;
            background: #e8f4fd;
            border-left: 4px solid #667eea;
            border-radius: 5px;
        }}
        .status-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        .status-online {{ background-color: #4CAF50; }}
        .status-offline {{ background-color: #f44336; }}
        .profiles-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        .profiles-table th, .profiles-table td {{
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .profiles-table th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{self.title}</h1>
        <p>Real-time Performance Monitoring</p>
        <div>
            <span class="status-indicator status-online" id="status-indicator"></span>
            <span id="status-text">Connected</span>
        </div>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-title">CPU Usage</div>
            <div class="metric-value" id="cpu-value">0</div>
            <div class="metric-unit">%</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Memory Usage</div>
            <div class="metric-value" id="memory-value">0</div>
            <div class="metric-unit">MB</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">GPU Memory</div>
            <div class="metric-value" id="gpu-memory-value">0</div>
            <div class="metric-unit">MB</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Active Connections</div>
            <div class="metric-value" id="connections-value">0</div>
            <div class="metric-unit">clients</div>
        </div>
    </div>

    <div class="chart-container">
        <h3>Performance Trends</h3>
        <canvas id="performance-chart" width="400" height="200"></canvas>
    </div>

    <div class="chart-container">
        <h3>Operation Profiles</h3>
        <table class="profiles-table" id="profiles-table">
            <thead>
                <tr>
                    <th>Operation</th>
                    <th>Count</th>
                    <th>Avg Duration (ms)</th>
                    <th>Max Duration (ms)</th>
                    <th>Memory Delta (MB)</th>
                </tr>
            </thead>
            <tbody id="profiles-tbody">
            </tbody>
        </table>
    </div>

    <div class="recommendations">
        <h3>Optimization Recommendations</h3>
        <div id="recommendations-list">
            <p>Loading recommendations...</p>
        </div>
    </div>

    <script>
        // WebSocket connection
        const ws = new WebSocket(`ws://${{window.location.host}}/ws`);
        const statusIndicator = document.getElementById('status-indicator');
        const statusText = document.getElementById('status-text');

        // Chart setup
        const ctx = document.getElementById('performance-chart').getContext('2d');
        const chart = new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: [],
                datasets: [{{
                    label: 'CPU %',
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }}, {{
                    label: 'Memory MB',
                    data: [],
                    borderColor: 'rgb(255, 99, 132)',
                    tension: 0.1,
                    yAxisID: 'y1'
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {{ display: true, text: 'CPU %' }}
                    }},
                    y1: {{
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {{ display: true, text: 'Memory MB' }},
                        grid: {{ drawOnChartArea: false }}
                    }}
                }}
            }}
        }});

        // WebSocket event handlers
        ws.onopen = function(event) {{
            statusIndicator.className = 'status-indicator status-online';
            statusText.textContent = 'Connected';
        }};

        ws.onclose = function(event) {{
            statusIndicator.className = 'status-indicator status-offline';
            statusText.textContent = 'Disconnected';
        }};

        ws.onerror = function(error) {{
            statusIndicator.className = 'status-indicator status-offline';
            statusText.textContent = 'Error';
        }};

        // Update functions
        function updateMetrics() {{
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => {{
                    if (data.error) return;
                    
                    document.getElementById('cpu-value').textContent = data.cpu_percent.current.toFixed(1);
                    document.getElementById('memory-value').textContent = data.memory_mb.current.toFixed(0);
                    document.getElementById('gpu-memory-value').textContent = data.gpu_memory_mb.current.toFixed(0);
                    
                    // Update chart
                    const now = new Date().toLocaleTimeString();
                    chart.data.labels.push(now);
                    chart.data.datasets[0].data.push(data.cpu_percent.current);
                    chart.data.datasets[1].data.push(data.memory_mb.current);
                    
                    // Keep only last 20 points
                    if (chart.data.labels.length > 20) {{
                        chart.data.labels.shift();
                        chart.data.datasets[0].data.shift();
                        chart.data.datasets[1].data.shift();
                    }}
                    
                    chart.update('none');
                }})
                .catch(error => console.error('Error fetching metrics:', error));
        }}

        function updateProfiles() {{
            fetch('/api/profiles')
                .then(response => response.json())
                .then(data => {{
                    if (data.error) return;
                    
                    const tbody = document.getElementById('profiles-tbody');
                    tbody.innerHTML = '';
                    
                    for (const [opName, opStats] of Object.entries(data.operations || {{}})) {{
                        const row = tbody.insertRow();
                        row.insertCell(0).textContent = opName;
                        row.insertCell(1).textContent = opStats.count;
                        row.insertCell(2).textContent = opStats.duration_ms.avg.toFixed(1);
                        row.insertCell(3).textContent = opStats.duration_ms.max.toFixed(1);
                        row.insertCell(4).textContent = opStats.memory_delta_mb.avg.toFixed(2);
                    }}
                }})
                .catch(error => console.error('Error fetching profiles:', error));
        }}

        function updateRecommendations() {{
            fetch('/api/recommendations')
                .then(response => response.json())
                .then(data => {{
                    const list = document.getElementById('recommendations-list');
                    list.innerHTML = '';
                    
                    data.recommendations.forEach(rec => {{
                        const div = document.createElement('div');
                        div.className = 'recommendation';
                        div.textContent = rec;
                        list.appendChild(div);
                    }});
                }})
                .catch(error => console.error('Error fetching recommendations:', error));
        }}

        function updateStatus() {{
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {{
                    document.getElementById('connections-value').textContent = data.connections;
                }})
                .catch(error => console.error('Error fetching status:', error));
        }}

        // Update intervals
        setInterval(updateMetrics, 2000);      // Every 2 seconds
        setInterval(updateProfiles, 5000);     // Every 5 seconds
        setInterval(updateRecommendations, 10000); // Every 10 seconds
        setInterval(updateStatus, 3000);       // Every 3 seconds

        // Initial updates
        updateMetrics();
        updateProfiles();
        updateRecommendations();
        updateStatus();
    </script>
</body>
</html>
        """
    
    async def broadcast_metrics(self):
        """Broadcast metrics to all connected WebSocket clients."""
        while self.broadcasting:
            try:
                if self.active_connections:
                    metrics = self.performance_monitor.get_metrics_summary(last_n_seconds=10)
                    message = json.dumps({
                        "type": "metrics_update",
                        "data": metrics,
                        "timestamp": time.time(),
                    })
                    
                    # Send to all connections
                    disconnected = []
                    for connection in self.active_connections:
                        try:
                            await connection.send_text(message)
                        except Exception:
                            disconnected.append(connection)
                    
                    # Remove disconnected clients
                    for connection in disconnected:
                        if connection in self.active_connections:
                            self.active_connections.remove(connection)
                
                await asyncio.sleep(2.0)  # Broadcast every 2 seconds
                
            except Exception as e:
                self.logger.error(f"Error broadcasting metrics: {e}")
                await asyncio.sleep(2.0)
    
    async def start_async(self):
        """Start dashboard server (async)."""
        try:
            # Start performance monitoring if not already started
            if not self.performance_monitor.monitoring:
                self.performance_monitor.start_monitoring()
            
            # Start broadcasting
            self.broadcasting = True
            self.broadcast_task = asyncio.create_task(self.broadcast_metrics())
            
            self.logger.info(f"Starting monitoring dashboard on {self.host}:{self.port}")
            
            # Configure uvicorn
            config = uvicorn.Config(
                app=self.app,
                host=self.host,
                port=self.port,
                log_level="info",
                access_log=False,
            )
            
            server = uvicorn.Server(config)
            await server.serve()
            
        except Exception as e:
            self.logger.error(f"Error starting dashboard: {e}")
            raise
        finally:
            self.broadcasting = False
            if self.broadcast_task:
                self.broadcast_task.cancel()
    
    def start(self):
        """Start dashboard server (sync wrapper)."""
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, create task
                return asyncio.create_task(self.start_async())
            else:
                # Run async start
                return asyncio.run(self.start_async())
        except RuntimeError:
            # No event loop, create new one
            return asyncio.run(self.start_async())
    
    def stop(self):
        """Stop dashboard server."""
        self.broadcasting = False
        if self.broadcast_task:
            self.broadcast_task.cancel()
        self.logger.info("Monitoring dashboard stopped")


def create_dashboard(
    host: str = "127.0.0.1",
    port: int = 8080,
    performance_monitor: Optional[PerformanceMonitor] = None,
    title: str = "R3MES Miner Engine Monitor",
) -> MonitoringDashboard:
    """
    Create monitoring dashboard.
    
    Args:
        host: Dashboard host
        port: Dashboard port
        performance_monitor: Performance monitor instance
        title: Dashboard title
        
    Returns:
        MonitoringDashboard instance
    """
    return MonitoringDashboard(
        host=host,
        port=port,
        performance_monitor=performance_monitor,
        title=title,
    )


if __name__ == "__main__":
    # Run dashboard directly
    import argparse
    
    parser = argparse.ArgumentParser(description="R3MES Monitoring Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Dashboard host")
    parser.add_argument("--port", type=int, default=8080, help="Dashboard port")
    parser.add_argument("--title", default="R3MES Miner Engine Monitor", help="Dashboard title")
    
    args = parser.parse_args()
    
    dashboard = create_dashboard(
        host=args.host,
        port=args.port,
        title=args.title,
    )
    
    try:
        dashboard.start()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        dashboard.stop()
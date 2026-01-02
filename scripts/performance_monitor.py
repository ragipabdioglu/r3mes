#!/usr/bin/env python3
"""
R3MES Performance Monitoring Script

Comprehensive performance monitoring and optimization for R3MES system.
Monitors backend, database, Redis, and blockchain components.
"""

import asyncio
import aiohttp
import psutil
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import subprocess
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Comprehensive performance monitoring for R3MES system."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            "backend_url": "http://localhost:8000",
            "redis_host": "localhost",
            "redis_port": 6379,
            "postgres_host": "localhost",
            "postgres_port": 5432,
            "monitoring_interval": 30,  # seconds
            "alert_thresholds": {
                "cpu_percent": 80,
                "memory_percent": 85,
                "disk_percent": 90,
                "response_time_ms": 500,
                "error_rate_percent": 5
            }
        }
        self.metrics_history = []
        self.alerts = []
    
    async def check_backend_health(self) -> Dict:
        """Check backend API health and performance."""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "service": "backend",
            "status": "unknown",
            "response_time_ms": None,
            "error": None
        }
        
        try:
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config['backend_url']}/health",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    end_time = time.time()
                    response_time_ms = (end_time - start_time) * 1000
                    
                    metrics.update({
                        "status": "healthy" if response.status == 200 else "unhealthy",
                        "response_time_ms": response_time_ms,
                        "status_code": response.status
                    })
                    
                    if response.status == 200:
                        data = await response.json()
                        metrics["health_data"] = data
                    
                    # Check response time threshold
                    if response_time_ms > self.config["alert_thresholds"]["response_time_ms"]:
                        self.alerts.append({
                            "timestamp": datetime.now().isoformat(),
                            "service": "backend",
                            "alert": "high_response_time",
                            "value": response_time_ms,
                            "threshold": self.config["alert_thresholds"]["response_time_ms"]
                        })
        
        except asyncio.TimeoutError:
            metrics.update({
                "status": "timeout",
                "error": "Request timeout"
            })
        except Exception as e:
            metrics.update({
                "status": "error",
                "error": str(e)
            })
        
        return metrics
    
    async def check_database_performance(self) -> Dict:
        """Check database connection and performance."""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "service": "database",
            "status": "unknown",
            "connection_count": None,
            "query_time_ms": None
        }
        
        try:
            # Try to connect and run a simple query
            start_time = time.time()
            
            # This would need actual database connection
            # For now, simulate with a health check
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(
                        f"{self.config['backend_url']}/health/database",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        end_time = time.time()
                        query_time_ms = (end_time - start_time) * 1000
                        
                        metrics.update({
                            "status": "healthy" if response.status == 200 else "unhealthy",
                            "query_time_ms": query_time_ms
                        })
                        
                        if response.status == 200:
                            data = await response.json()
                            metrics.update(data.get("database_metrics", {}))
                
                except Exception:
                    # Fallback to system-level checks
                    metrics["status"] = "unknown"
        
        except Exception as e:
            metrics.update({
                "status": "error",
                "error": str(e)
            })
        
        return metrics
    
    async def check_redis_performance(self) -> Dict:
        """Check Redis performance and memory usage."""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "service": "redis",
            "status": "unknown",
            "memory_usage_mb": None,
            "connected_clients": None,
            "hit_rate_percent": None
        }
        
        try:
            # Try Redis health check via backend
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(
                        f"{self.config['backend_url']}/health/redis",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            metrics.update({
                                "status": "healthy",
                                **data.get("redis_metrics", {})
                            })
                        else:
                            metrics["status"] = "unhealthy"
                
                except Exception:
                    metrics["status"] = "unknown"
        
        except Exception as e:
            metrics.update({
                "status": "error",
                "error": str(e)
            })
        
        return metrics
    
    def check_system_resources(self) -> Dict:
        """Check system resource usage."""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "service": "system",
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
            "network_io": dict(psutil.net_io_counters()._asdict()),
            "disk_io": dict(psutil.disk_io_counters()._asdict())
        }
        
        # Check thresholds and generate alerts
        thresholds = self.config["alert_thresholds"]
        
        if metrics["cpu_percent"] > thresholds["cpu_percent"]:
            self.alerts.append({
                "timestamp": datetime.now().isoformat(),
                "service": "system",
                "alert": "high_cpu_usage",
                "value": metrics["cpu_percent"],
                "threshold": thresholds["cpu_percent"]
            })
        
        if metrics["memory_percent"] > thresholds["memory_percent"]:
            self.alerts.append({
                "timestamp": datetime.now().isoformat(),
                "service": "system",
                "alert": "high_memory_usage",
                "value": metrics["memory_percent"],
                "threshold": thresholds["memory_percent"]
            })
        
        if metrics["disk_percent"] > thresholds["disk_percent"]:
            self.alerts.append({
                "timestamp": datetime.now().isoformat(),
                "service": "system",
                "alert": "high_disk_usage",
                "value": metrics["disk_percent"],
                "threshold": thresholds["disk_percent"]
            })
        
        return metrics
    
    async def check_blockchain_connectivity(self) -> Dict:
        """Check blockchain node connectivity and sync status."""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "service": "blockchain",
            "status": "unknown",
            "block_height": None,
            "sync_status": None,
            "peer_count": None
        }
        
        try:
            # Check blockchain health via backend
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(
                        f"{self.config['backend_url']}/health/blockchain",
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            metrics.update({
                                "status": "healthy",
                                **data.get("blockchain_metrics", {})
                            })
                        else:
                            metrics["status"] = "unhealthy"
                
                except Exception:
                    metrics["status"] = "unknown"
        
        except Exception as e:
            metrics.update({
                "status": "error",
                "error": str(e)
            })
        
        return metrics
    
    async def run_comprehensive_check(self) -> Dict:
        """Run comprehensive performance check across all services."""
        logger.info("Running comprehensive performance check...")
        
        # Run all checks concurrently
        tasks = [
            self.check_backend_health(),
            self.check_database_performance(),
            self.check_redis_performance(),
            self.check_blockchain_connectivity()
        ]
        
        # Add system check (synchronous)
        system_metrics = self.check_system_resources()
        
        # Wait for async tasks
        async_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        results = {
            "timestamp": datetime.now().isoformat(),
            "system": system_metrics,
            "backend": async_results[0] if not isinstance(async_results[0], Exception) else {"error": str(async_results[0])},
            "database": async_results[1] if not isinstance(async_results[1], Exception) else {"error": str(async_results[1])},
            "redis": async_results[2] if not isinstance(async_results[2], Exception) else {"error": str(async_results[2])},
            "blockchain": async_results[3] if not isinstance(async_results[3], Exception) else {"error": str(async_results[3])},
            "alerts": self.alerts[-10:],  # Last 10 alerts
            "summary": self._generate_summary(async_results + [system_metrics])
        }
        
        # Store in history
        self.metrics_history.append(results)
        
        # Keep only last 100 entries
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
        
        return results
    
    def _generate_summary(self, metrics_list: List[Dict]) -> Dict:
        """Generate performance summary."""
        healthy_services = 0
        total_services = 0
        avg_response_time = 0
        response_time_count = 0
        
        for metric in metrics_list:
            if isinstance(metric, dict):
                total_services += 1
                if metric.get("status") == "healthy":
                    healthy_services += 1
                
                if "response_time_ms" in metric and metric["response_time_ms"]:
                    avg_response_time += metric["response_time_ms"]
                    response_time_count += 1
        
        return {
            "health_percentage": (healthy_services / total_services * 100) if total_services > 0 else 0,
            "healthy_services": healthy_services,
            "total_services": total_services,
            "avg_response_time_ms": avg_response_time / response_time_count if response_time_count > 0 else None,
            "active_alerts": len([a for a in self.alerts if 
                                datetime.fromisoformat(a["timestamp"]) > datetime.now() - timedelta(minutes=5)])
        }
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        if not self.metrics_history:
            return "No performance data available."
        
        latest = self.metrics_history[-1]
        
        report = f"""
# R3MES Performance Report

**Generated**: {datetime.now().isoformat()}

## System Overview

- **Health**: {latest['summary']['health_percentage']:.1f}% ({latest['summary']['healthy_services']}/{latest['summary']['total_services']} services healthy)
- **Average Response Time**: {latest['summary']['avg_response_time_ms']:.1f}ms
- **Active Alerts**: {latest['summary']['active_alerts']}

## Service Status

### Backend API
- **Status**: {latest['backend'].get('status', 'unknown')}
- **Response Time**: {latest['backend'].get('response_time_ms', 'N/A')}ms

### Database
- **Status**: {latest['database'].get('status', 'unknown')}
- **Query Time**: {latest['database'].get('query_time_ms', 'N/A')}ms

### Redis Cache
- **Status**: {latest['redis'].get('status', 'unknown')}
- **Memory Usage**: {latest['redis'].get('memory_usage_mb', 'N/A')}MB
- **Hit Rate**: {latest['redis'].get('hit_rate_percent', 'N/A')}%

### Blockchain
- **Status**: {latest['blockchain'].get('status', 'unknown')}
- **Block Height**: {latest['blockchain'].get('block_height', 'N/A')}
- **Sync Status**: {latest['blockchain'].get('sync_status', 'N/A')}

## System Resources

- **CPU Usage**: {latest['system']['cpu_percent']:.1f}%
- **Memory Usage**: {latest['system']['memory_percent']:.1f}%
- **Disk Usage**: {latest['system']['disk_percent']:.1f}%

## Recent Alerts

"""
        
        # Add recent alerts
        recent_alerts = [a for a in self.alerts if 
                        datetime.fromisoformat(a["timestamp"]) > datetime.now() - timedelta(hours=1)]
        
        if recent_alerts:
            for alert in recent_alerts[-5:]:  # Last 5 alerts
                report += f"- **{alert['alert']}** ({alert['service']}): {alert['value']} > {alert['threshold']} at {alert['timestamp']}\n"
        else:
            report += "No recent alerts.\n"
        
        report += f"""

## Recommendations

"""
        
        # Generate recommendations based on metrics
        recommendations = self._generate_recommendations(latest)
        for rec in recommendations:
            report += f"- {rec}\n"
        
        return report
    
    def _generate_recommendations(self, metrics: Dict) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        # System resource recommendations
        if metrics['system']['cpu_percent'] > 70:
            recommendations.append("Consider scaling backend instances or optimizing CPU-intensive operations")
        
        if metrics['system']['memory_percent'] > 80:
            recommendations.append("Monitor memory usage and consider increasing available RAM")
        
        if metrics['system']['disk_percent'] > 85:
            recommendations.append("Clean up disk space or expand storage capacity")
        
        # Backend recommendations
        if metrics['backend'].get('response_time_ms', 0) > 300:
            recommendations.append("Backend response time is high - check database queries and caching")
        
        # Database recommendations
        if metrics['database'].get('query_time_ms', 0) > 100:
            recommendations.append("Database queries are slow - consider indexing or query optimization")
        
        # Redis recommendations
        if metrics['redis'].get('hit_rate_percent', 100) < 80:
            recommendations.append("Redis cache hit rate is low - review caching strategy")
        
        if not recommendations:
            recommendations.append("System performance is within acceptable ranges")
        
        return recommendations
    
    async def continuous_monitoring(self, duration_minutes: int = 60):
        """Run continuous monitoring for specified duration."""
        logger.info(f"Starting continuous monitoring for {duration_minutes} minutes...")
        
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        
        while datetime.now() < end_time:
            try:
                results = await self.run_comprehensive_check()
                
                # Log summary
                summary = results['summary']
                logger.info(
                    f"Health: {summary['health_percentage']:.1f}%, "
                    f"Response: {summary['avg_response_time_ms']:.1f}ms, "
                    f"Alerts: {summary['active_alerts']}"
                )
                
                # Save results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_file = Path(f"performance_results_{timestamp}.json")
                with open(results_file, "w") as f:
                    json.dump(results, f, indent=2)
                
                # Wait for next check
                await asyncio.sleep(self.config["monitoring_interval"])
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error during monitoring: {e}")
                await asyncio.sleep(5)  # Short delay before retry
        
        # Generate final report
        report = self.generate_performance_report()
        report_file = Path("performance_report.md")
        with open(report_file, "w") as f:
            f.write(report)
        
        logger.info(f"Performance report saved to {report_file}")


async def main():
    """Main performance monitoring workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description="R3MES Performance Monitor")
    parser.add_argument("--duration", type=int, default=10, help="Monitoring duration in minutes")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds")
    parser.add_argument("--backend-url", default="http://localhost:8000", help="Backend URL")
    parser.add_argument("--single-check", action="store_true", help="Run single check and exit")
    
    args = parser.parse_args()
    
    config = {
        "backend_url": args.backend_url,
        "monitoring_interval": args.interval,
        "alert_thresholds": {
            "cpu_percent": 80,
            "memory_percent": 85,
            "disk_percent": 90,
            "response_time_ms": 500,
            "error_rate_percent": 5
        }
    }
    
    monitor = PerformanceMonitor(config)
    
    if args.single_check:
        print("🔍 Running single performance check...")
        results = await monitor.run_comprehensive_check()
        
        print(f"📊 Results:")
        print(f"  Health: {results['summary']['health_percentage']:.1f}%")
        print(f"  Response Time: {results['summary']['avg_response_time_ms']:.1f}ms")
        print(f"  Active Alerts: {results['summary']['active_alerts']}")
        
        # Save results
        with open("performance_check.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("📋 Detailed results saved to performance_check.json")
    else:
        await monitor.continuous_monitoring(args.duration)


if __name__ == "__main__":
    asyncio.run(main())
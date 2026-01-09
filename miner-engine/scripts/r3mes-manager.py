#!/usr/bin/env python3
"""
R3MES Miner Engine Management Tool

Comprehensive management tool for R3MES Miner Engine operations.
"""

import os
import sys
import json
import time
import signal
import subprocess
import psutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import argparse
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from utils.performance_monitor import get_global_monitor
    from utils.monitoring_dashboard import create_dashboard
    from utils.advanced_config import create_config_manager
    from r3mes.miner.engine import MinerEngine
    from bridge.blockchain_client import BlockchainClient
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the miner-engine directory")
    sys.exit(1)


class R3MESManager:
    """R3MES Miner Engine management tool."""
    
    def __init__(self):
        """Initialize manager."""
        self.project_root = project_root
        self.pid_file = self.project_root / "r3mes-miner.pid"
        self.log_file = self.project_root / "logs" / "manager.log"
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        try:
            self.config = create_config_manager(
                config_files=[
                    str(self.project_root / "config" / "default.yaml"),
                    str(self.project_root / "config" / "local.yaml"),
                ],
                enable_hot_reload=False
            )
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            self.config = None
    
    def start_miner(self, daemon: bool = False) -> bool:
        """Start the miner engine."""
        if self.is_running():
            print("❌ Miner is already running")
            return False
        
        print("🚀 Starting R3MES Miner Engine...")
        
        try:
            # Prepare command
            cmd = [sys.executable, "miner_engine.py"]
            
            if daemon:
                # Start as daemon
                process = subprocess.Popen(
                    cmd,
                    cwd=self.project_root,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True
                )
                
                # Save PID
                with open(self.pid_file, 'w') as f:
                    f.write(str(process.pid))
                
                print(f"✅ Miner started as daemon (PID: {process.pid})")
                self.logger.info(f"Miner started as daemon (PID: {process.pid})")
                
            else:
                # Start in foreground
                process = subprocess.run(cmd, cwd=self.project_root)
                return process.returncode == 0
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start miner: {e}")
            self.logger.error(f"Failed to start miner: {e}")
            return False
    
    def stop_miner(self) -> bool:
        """Stop the miner engine."""
        if not self.is_running():
            print("❌ Miner is not running")
            return False
        
        print("🛑 Stopping R3MES Miner Engine...")
        
        try:
            pid = self.get_miner_pid()
            if pid:
                # Try graceful shutdown first
                os.kill(pid, signal.SIGTERM)
                
                # Wait for graceful shutdown
                for _ in range(10):
                    if not psutil.pid_exists(pid):
                        break
                    time.sleep(1)
                
                # Force kill if still running
                if psutil.pid_exists(pid):
                    os.kill(pid, signal.SIGKILL)
                    print("⚠️  Force killed miner process")
                
                # Remove PID file
                if self.pid_file.exists():
                    self.pid_file.unlink()
                
                print("✅ Miner stopped")
                self.logger.info("Miner stopped")
                return True
            
        except Exception as e:
            print(f"❌ Failed to stop miner: {e}")
            self.logger.error(f"Failed to stop miner: {e}")
            return False
        
        return False
    
    def restart_miner(self, daemon: bool = False) -> bool:
        """Restart the miner engine."""
        print("🔄 Restarting R3MES Miner Engine...")
        
        if self.is_running():
            if not self.stop_miner():
                return False
        
        time.sleep(2)  # Brief pause
        return self.start_miner(daemon=daemon)
    
    def is_running(self) -> bool:
        """Check if miner is running."""
        pid = self.get_miner_pid()
        return pid is not None and psutil.pid_exists(pid)
    
    def get_miner_pid(self) -> Optional[int]:
        """Get miner process PID."""
        if not self.pid_file.exists():
            return None
        
        try:
            with open(self.pid_file, 'r') as f:
                return int(f.read().strip())
        except (ValueError, FileNotFoundError):
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get miner status."""
        status = {
            "running": self.is_running(),
            "pid": self.get_miner_pid(),
            "uptime": None,
            "memory_usage": None,
            "cpu_usage": None,
        }
        
        if status["running"] and status["pid"]:
            try:
                process = psutil.Process(status["pid"])
                status["uptime"] = time.time() - process.create_time()
                status["memory_usage"] = process.memory_info().rss / (1024 * 1024)  # MB
                status["cpu_usage"] = process.cpu_percent()
            except psutil.NoSuchProcess:
                status["running"] = False
                status["pid"] = None
        
        return status
    
    def show_status(self):
        """Show miner status."""
        print("📊 R3MES Miner Engine Status")
        print("=" * 40)
        
        status = self.get_status()
        
        # Basic status
        status_icon = "🟢" if status["running"] else "🔴"
        status_text = "Running" if status["running"] else "Stopped"
        print(f"Status: {status_icon} {status_text}")
        
        if status["running"]:
            print(f"PID: {status['pid']}")
            
            if status["uptime"]:
                uptime_hours = status["uptime"] / 3600
                print(f"Uptime: {uptime_hours:.1f} hours")
            
            if status["memory_usage"]:
                print(f"Memory: {status['memory_usage']:.1f} MB")
            
            if status["cpu_usage"]:
                print(f"CPU: {status['cpu_usage']:.1f}%")
        
        # Configuration status
        if self.config:
            print(f"\nConfiguration:")
            print(f"  Environment: {self.config.get('development.test_mode', 'Unknown')}")
            print(f"  Blockchain: {self.config.get('blockchain.url', 'Not configured')}")
            print(f"  Model Size: {self.config.get('model.hidden_size', 'Unknown')}")
            print(f"  Batch Size: {self.config.get('mining.batch_size', 'Unknown')}")
        
        # Log file status
        if self.log_file.exists():
            log_size = self.log_file.stat().st_size / (1024 * 1024)
            print(f"\nLog File: {self.log_file} ({log_size:.1f} MB)")
    
    def show_logs(self, lines: int = 50, follow: bool = False):
        """Show miner logs."""
        log_files = [
            self.project_root / "logs" / "miner-engine.log",
            self.project_root / "logs" / "manager.log",
        ]
        
        for log_file in log_files:
            if log_file.exists():
                print(f"\n📄 {log_file.name}")
                print("-" * 40)
                
                if follow:
                    # Follow log file
                    subprocess.run(["tail", "-f", str(log_file)])
                else:
                    # Show last N lines
                    try:
                        with open(log_file, 'r') as f:
                            all_lines = f.readlines()
                            for line in all_lines[-lines:]:
                                print(line.rstrip())
                    except Exception as e:
                        print(f"Error reading log: {e}")
    
    def start_monitoring(self):
        """Start monitoring dashboard."""
        print("📊 Starting monitoring dashboard...")
        
        try:
            dashboard = create_dashboard(
                host="127.0.0.1",
                port=8080,
                title="R3MES Miner Engine Monitor"
            )
            
            print("✅ Monitoring dashboard started at http://127.0.0.1:8080")
            dashboard.start()
            
        except Exception as e:
            print(f"❌ Failed to start monitoring: {e}")
            self.logger.error(f"Failed to start monitoring: {e}")
    
    def run_diagnostics(self):
        """Run system diagnostics."""
        print("🔍 Running R3MES Diagnostics")
        print("=" * 40)
        
        # System information
        print(f"Python Version: {sys.version}")
        print(f"Platform: {sys.platform}")
        print(f"Working Directory: {os.getcwd()}")
        print(f"Project Root: {self.project_root}")
        
        # Check dependencies
        print(f"\n📦 Dependencies:")
        dependencies = [
            "torch", "numpy", "grpc", "click", "yaml", 
            "fastapi", "uvicorn", "psutil", "cryptography"
        ]
        
        for dep in dependencies:
            try:
                __import__(dep)
                print(f"  ✅ {dep}")
            except ImportError:
                print(f"  ❌ {dep}")
        
        # Check GPU
        print(f"\n🖥️  GPU Status:")
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                print(f"  ✅ CUDA available ({gpu_count} GPUs)")
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    print(f"    GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            else:
                print(f"  ❌ CUDA not available")
        except Exception as e:
            print(f"  ❌ GPU check failed: {e}")
        
        # Check configuration
        print(f"\n⚙️  Configuration:")
        if self.config:
            print(f"  ✅ Configuration loaded")
            summary = self.config.get_config_summary()
            print(f"    Total keys: {summary['total_keys']}")
            print(f"    Config files: {summary['config_files']}")
        else:
            print(f"  ❌ Configuration not loaded")
        
        # Check blockchain connection
        print(f"\n🔗 Blockchain Connection:")
        if self.config:
            try:
                blockchain_url = self.config.get("blockchain.url")
                chain_id = self.config.get("blockchain.chain_id")
                
                # Try to create client (won't actually connect without private key)
                print(f"  URL: {blockchain_url}")
                print(f"  Chain ID: {chain_id}")
                print(f"  ⚠️  Connection test requires private key")
                
            except Exception as e:
                print(f"  ❌ Blockchain check failed: {e}")
        
        # Check file permissions
        print(f"\n📁 File Permissions:")
        important_dirs = [
            self.project_root / "cache",
            self.project_root / "logs",
            self.project_root / "config",
        ]
        
        for directory in important_dirs:
            if directory.exists():
                if os.access(directory, os.R_OK | os.W_OK):
                    print(f"  ✅ {directory}")
                else:
                    print(f"  ❌ {directory} (no read/write access)")
            else:
                print(f"  ⚠️  {directory} (does not exist)")
    
    def cleanup(self):
        """Cleanup temporary files and caches."""
        print("🧹 Cleaning up R3MES files...")
        
        cleanup_paths = [
            self.project_root / "__pycache__",
            self.project_root / "*.pyc",
            self.project_root / "cache" / "*.tmp",
            self.project_root / "logs" / "*.old",
        ]
        
        cleaned_count = 0
        
        for path_pattern in cleanup_paths:
            if "*" in str(path_pattern):
                # Handle glob patterns
                from glob import glob
                for file_path in glob(str(path_pattern)):
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            import shutil
                            shutil.rmtree(file_path)
                        cleaned_count += 1
                    except Exception as e:
                        print(f"  ⚠️  Failed to remove {file_path}: {e}")
            else:
                # Handle direct paths
                if path_pattern.exists():
                    try:
                        if path_pattern.is_file():
                            path_pattern.unlink()
                        elif path_pattern.is_dir():
                            import shutil
                            shutil.rmtree(path_pattern)
                        cleaned_count += 1
                    except Exception as e:
                        print(f"  ⚠️  Failed to remove {path_pattern}: {e}")
        
        print(f"✅ Cleaned up {cleaned_count} files/directories")
    
    def backup_config(self, backup_dir: Optional[str] = None):
        """Backup configuration files."""
        if not backup_dir:
            backup_dir = self.project_root / "backups" / f"config-{int(time.time())}"
        
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Backing up configuration to {backup_path}...")
        
        # Files to backup
        backup_files = [
            self.project_root / ".env",
            self.project_root / "config",
            self.project_root / "requirements.txt",
            self.project_root / "pyproject.toml",
        ]
        
        backed_up_count = 0
        
        for source in backup_files:
            if source.exists():
                try:
                    if source.is_file():
                        import shutil
                        shutil.copy2(source, backup_path / source.name)
                    elif source.is_dir():
                        import shutil
                        shutil.copytree(source, backup_path / source.name)
                    
                    backed_up_count += 1
                    print(f"  ✅ {source.name}")
                    
                except Exception as e:
                    print(f"  ❌ {source.name}: {e}")
        
        print(f"✅ Backed up {backed_up_count} items to {backup_path}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="R3MES Miner Engine Manager")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start the miner")
    start_parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    
    # Stop command
    subparsers.add_parser("stop", help="Stop the miner")
    
    # Restart command
    restart_parser = subparsers.add_parser("restart", help="Restart the miner")
    restart_parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    
    # Status command
    subparsers.add_parser("status", help="Show miner status")
    
    # Logs command
    logs_parser = subparsers.add_parser("logs", help="Show miner logs")
    logs_parser.add_argument("--lines", type=int, default=50, help="Number of lines to show")
    logs_parser.add_argument("--follow", action="store_true", help="Follow log output")
    
    # Monitor command
    subparsers.add_parser("monitor", help="Start monitoring dashboard")
    
    # Diagnostics command
    subparsers.add_parser("diagnostics", help="Run system diagnostics")
    
    # Cleanup command
    subparsers.add_parser("cleanup", help="Cleanup temporary files")
    
    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Backup configuration")
    backup_parser.add_argument("--dir", help="Backup directory")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize manager
    manager = R3MESManager()
    
    # Execute command
    try:
        if args.command == "start":
            manager.start_miner(daemon=args.daemon)
        elif args.command == "stop":
            manager.stop_miner()
        elif args.command == "restart":
            manager.restart_miner(daemon=args.daemon)
        elif args.command == "status":
            manager.show_status()
        elif args.command == "logs":
            manager.show_logs(lines=args.lines, follow=args.follow)
        elif args.command == "monitor":
            manager.start_monitoring()
        elif args.command == "diagnostics":
            manager.run_diagnostics()
        elif args.command == "cleanup":
            manager.cleanup()
        elif args.command == "backup":
            manager.backup_config(backup_dir=args.dir)
        else:
            print(f"❌ Unknown command: {args.command}")
            
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        logging.exception("Command execution failed")


if __name__ == "__main__":
    main()
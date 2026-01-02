#!/usr/bin/env python3
"""
R3MES Miner Engine Setup Script

Comprehensive setup script for R3MES Miner Engine installation and configuration.
"""

import os
import sys
import subprocess
import shutil
import json
import secrets
from pathlib import Path
from typing import Dict, Any, Optional, List
import argparse


class R3MESSetup:
    """R3MES Miner Engine setup manager."""
    
    def __init__(self, project_root: Optional[str] = None):
        """Initialize setup manager."""
        self.project_root = Path(project_root or Path(__file__).parent.parent)
        self.config_dir = self.project_root / "config"
        self.cache_dir = self.project_root / "cache"
        self.logs_dir = self.project_root / "logs"
        
        print(f"🚀 R3MES Miner Engine Setup")
        print(f"📁 Project root: {self.project_root}")
    
    def check_system_requirements(self) -> Dict[str, bool]:
        """Check system requirements."""
        print("\n🔍 Checking system requirements...")
        
        requirements = {
            "python": self._check_python_version(),
            "pip": self._check_command("pip"),
            "git": self._check_command("git"),
            "cuda": self._check_cuda(),
            "docker": self._check_command("docker"),
        }
        
        for req, status in requirements.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {req}")
        
        return requirements
    
    def _check_python_version(self) -> bool:
        """Check Python version."""
        return sys.version_info >= (3, 8)
    
    def _check_command(self, command: str) -> bool:
        """Check if command is available."""
        return shutil.which(command) is not None
    
    def _check_cuda(self) -> bool:
        """Check CUDA availability."""
        try:
            result = subprocess.run(
                ["nvidia-smi"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def create_directories(self):
        """Create necessary directories."""
        print("\n📁 Creating directories...")
        
        directories = [
            self.config_dir,
            self.cache_dir,
            self.cache_dir / "lora",
            self.cache_dir / "models",
            self.logs_dir,
            self.project_root / "privacy" / "enclave",
            self.project_root / "docs" / "api",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ {directory}")
    
    def install_dependencies(self, install_optional: bool = False):
        """Install Python dependencies."""
        print("\n📦 Installing dependencies...")
        
        # Core dependencies
        core_deps = [
            "torch>=2.0.0",
            "numpy>=1.24.0",
            "grpcio>=1.50.0",
            "protobuf>=4.21.0",
            "click>=8.0.0",
            "rich>=13.0.0",
            "pyyaml>=6.0",
            "watchdog>=3.0.0",
            "psutil>=5.9.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            "jinja2>=3.1.0",
            "cryptography>=41.0.0",
            "ipfshttpclient>=0.8.0",
        ]
        
        # Optional dependencies
        optional_deps = [
            "transformers>=4.30.0",
            "peft>=0.4.0",
            "bitsandbytes>=0.41.0",
            "llama-cpp-python>=0.2.0",
            "pynvml>=11.5.0",
        ]
        
        # Install core dependencies
        self._install_packages(core_deps, "Core")
        
        # Install optional dependencies if requested
        if install_optional:
            self._install_packages(optional_deps, "Optional")
    
    def _install_packages(self, packages: List[str], category: str):
        """Install list of packages."""
        print(f"\n   Installing {category} packages...")
        
        for package in packages:
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", package],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"   ✅ {package}")
            except subprocess.CalledProcessError as e:
                print(f"   ❌ {package} - {e}")
    
    def generate_private_key(self) -> str:
        """Generate a new private key."""
        return secrets.token_hex(32)
    
    def create_environment_file(self, 
                              environment: str = "development",
                              private_key: Optional[str] = None,
                              blockchain_url: str = "localhost:9090",
                              chain_id: str = "remes-test") -> str:
        """Create .env file."""
        print(f"\n🔐 Creating environment file for {environment}...")
        
        if not private_key:
            private_key = self.generate_private_key()
            print(f"   🔑 Generated new private key: {private_key[:16]}...")
        
        env_content = f"""# R3MES Miner Engine Environment Configuration
# Generated by setup script

# Environment
R3MES_ENV={environment}
R3MES_TEST_MODE={'true' if environment == 'development' else 'false'}

# Blockchain
R3MES_BLOCKCHAIN_URL={blockchain_url}
R3MES_CHAIN_ID={chain_id}
R3MES_PRIVATE_KEY={private_key}

# Node Roles
R3MES_ENABLE_MINER=true
R3MES_ENABLE_SERVING_NODE=false
R3MES_ENABLE_PROPOSER_NODE=false

# Model Configuration
R3MES_MODEL_HIDDEN_SIZE={'256' if environment == 'development' else '768'}
R3MES_LORA_RANK=8
R3MES_BATCH_SIZE={'2' if environment == 'development' else '4'}

# Performance
R3MES_MAX_MEMORY_MB={'4096' if environment == 'development' else '8192'}
R3MES_GPU_MEMORY_FRACTION=0.8

# Network
R3MES_IPFS_URL=http://localhost:5001
R3MES_STATS_PORT=8080
R3MES_MONITORING_PORT=8081

# Logging
R3MES_LOG_LEVEL={'DEBUG' if environment == 'development' else 'INFO'}
R3MES_LOG_OUTPUT=console

# Monitoring
R3MES_ENABLE_DASHBOARD={'true' if environment == 'development' else 'false'}
R3MES_DASHBOARD_HOST=127.0.0.1
R3MES_DASHBOARD_PORT=8080

# Privacy
R3MES_ENABLE_SGX=false
R3MES_ENABLE_GRADIENT_ENCRYPTION=false

# Cache
R3MES_LORA_CACHE_DIR=cache/lora
R3MES_MODEL_CACHE_DIR=cache/models
R3MES_MAX_CACHE_SIZE_MB={'256' if environment == 'development' else '1024'}
"""
        
        env_file = self.project_root / ".env"
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print(f"   ✅ Environment file created: {env_file}")
        return private_key
    
    def setup_git_hooks(self):
        """Setup Git hooks for development."""
        print("\n🪝 Setting up Git hooks...")
        
        hooks_dir = self.project_root / ".git" / "hooks"
        if not hooks_dir.exists():
            print("   ⚠️  Git repository not found, skipping hooks")
            return
        
        # Pre-commit hook
        pre_commit_hook = hooks_dir / "pre-commit"
        pre_commit_content = """#!/bin/bash
# R3MES pre-commit hook

echo "🔍 Running pre-commit checks..."

# Check Python syntax
python -m py_compile miner_engine.py
if [ $? -ne 0 ]; then
    echo "❌ Python syntax error in miner_engine.py"
    exit 1
fi

# Run basic tests
python -m pytest tests/ -x --tb=short
if [ $? -ne 0 ]; then
    echo "❌ Tests failed"
    exit 1
fi

echo "✅ Pre-commit checks passed"
"""
        
        with open(pre_commit_hook, 'w') as f:
            f.write(pre_commit_content)
        
        pre_commit_hook.chmod(0o755)
        print(f"   ✅ Pre-commit hook installed")
    
    def create_systemd_service(self, 
                              user: str = "r3mes",
                              working_dir: Optional[str] = None) -> str:
        """Create systemd service file."""
        print("\n🔧 Creating systemd service...")
        
        working_dir = working_dir or str(self.project_root)
        
        service_content = f"""[Unit]
Description=R3MES Miner Engine
After=network.target
Wants=network.target

[Service]
Type=simple
User={user}
Group={user}
WorkingDirectory={working_dir}
Environment=PATH=/usr/local/bin:/usr/bin:/bin
Environment=PYTHONPATH={working_dir}
ExecStart=/usr/bin/python3 miner_engine.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=r3mes-miner

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths={working_dir}

[Install]
WantedBy=multi-user.target
"""
        
        service_file = self.project_root / "r3mes-miner.service"
        with open(service_file, 'w') as f:
            f.write(service_content)
        
        print(f"   ✅ Service file created: {service_file}")
        print(f"   📝 To install: sudo cp {service_file} /etc/systemd/system/")
        print(f"   📝 To enable: sudo systemctl enable r3mes-miner")
        print(f"   📝 To start: sudo systemctl start r3mes-miner")
        
        return str(service_file)
    
    def create_docker_compose(self):
        """Create Docker Compose configuration."""
        print("\n🐳 Creating Docker Compose configuration...")
        
        compose_content = """version: '3.8'

services:
  r3mes-miner:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: r3mes-miner
    restart: unless-stopped
    environment:
      - R3MES_ENV=production
      - R3MES_BLOCKCHAIN_URL=blockchain:9090
      - R3MES_IPFS_URL=http://ipfs:5001
    volumes:
      - ./cache:/app/cache
      - ./logs:/app/logs
      - ./config:/app/config
    ports:
      - "8080:8080"  # Stats server
      - "8081:8081"  # Monitoring dashboard
    depends_on:
      - ipfs
      - blockchain
    networks:
      - r3mes-network

  ipfs:
    image: ipfs/go-ipfs:latest
    container_name: r3mes-ipfs
    restart: unless-stopped
    ports:
      - "4001:4001"  # P2P
      - "5001:5001"  # API
      - "8080:8080"  # Gateway
    volumes:
      - ipfs-data:/data/ipfs
    networks:
      - r3mes-network

  blockchain:
    image: r3mes/blockchain:latest
    container_name: r3mes-blockchain
    restart: unless-stopped
    ports:
      - "9090:9090"  # gRPC
      - "26657:26657"  # Tendermint RPC
    volumes:
      - blockchain-data:/root/.remes
    networks:
      - r3mes-network

  monitoring:
    image: grafana/grafana:latest
    container_name: r3mes-monitoring
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
    networks:
      - r3mes-network

volumes:
  ipfs-data:
  blockchain-data:
  grafana-data:

networks:
  r3mes-network:
    driver: bridge
"""
        
        compose_file = self.project_root / "docker-compose.yml"
        with open(compose_file, 'w') as f:
            f.write(compose_content)
        
        print(f"   ✅ Docker Compose file created: {compose_file}")
    
    def create_dockerfile(self):
        """Create Dockerfile."""
        print("\n🐳 Creating Dockerfile...")
        
        dockerfile_content = """FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    git \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 r3mes && \\
    chown -R r3mes:r3mes /app
USER r3mes

# Create necessary directories
RUN mkdir -p cache/lora cache/models logs

# Expose ports
EXPOSE 8080 8081 8082

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["python", "miner_engine.py"]
"""
        
        dockerfile = self.project_root / "Dockerfile"
        with open(dockerfile, 'w') as f:
            f.write(dockerfile_content)
        
        print(f"   ✅ Dockerfile created: {dockerfile}")
    
    def run_initial_tests(self):
        """Run initial tests to verify setup."""
        print("\n🧪 Running initial tests...")
        
        try:
            # Test imports
            print("   🔍 Testing imports...")
            test_imports = [
                "import torch",
                "import numpy",
                "import grpc",
                "import click",
                "import yaml",
                "from core.bitlinear import BitLinear",
                "from utils.logger import setup_logger",
            ]
            
            for import_stmt in test_imports:
                try:
                    exec(import_stmt)
                    print(f"   ✅ {import_stmt}")
                except ImportError as e:
                    print(f"   ❌ {import_stmt} - {e}")
            
            # Test basic functionality
            print("   🔍 Testing basic functionality...")
            
            # Test BitLinear layer
            exec("""
import torch
from core.bitlinear import BitLinear

layer = BitLinear(256, 256, lora_rank=8)
x = torch.randn(2, 256)
output = layer(x)
assert output.shape == (2, 256)
print("   ✅ BitLinear layer test")
""")
            
            # Test configuration loading
            exec("""
from utils.advanced_config import create_config_manager

config = create_config_manager(
    config_files=["config/default.yaml"],
    enable_hot_reload=False
)
assert config.get("app.name") == "R3MES Miner Engine"
print("   ✅ Configuration loading test")
""")
            
            print("   ✅ All tests passed!")
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
    
    def print_setup_summary(self, private_key: str):
        """Print setup summary."""
        print("\n" + "="*60)
        print("🎉 R3MES Miner Engine Setup Complete!")
        print("="*60)
        
        print(f"\n📁 Project Directory: {self.project_root}")
        print(f"🔑 Private Key: {private_key[:16]}... (saved in .env)")
        
        print(f"\n🚀 Quick Start:")
        print(f"   1. Activate environment: source .env")
        print(f"   2. Start miner: python miner_engine.py")
        print(f"   3. View dashboard: http://localhost:8080")
        
        print(f"\n📚 Documentation:")
        print(f"   - Main docs: README.md")
        print(f"   - API docs: docs/api/")
        print(f"   - Configuration: config/")
        
        print(f"\n🔧 Management:")
        print(f"   - View logs: tail -f logs/miner-engine.log")
        print(f"   - Monitor: python -m utils.monitoring_dashboard")
        print(f"   - CLI: python -m r3mes.cli.main --help")
        
        print(f"\n🐳 Docker (optional):")
        print(f"   - Build: docker-compose build")
        print(f"   - Start: docker-compose up -d")
        
        print(f"\n⚠️  Next Steps:")
        print(f"   1. Configure blockchain connection in .env")
        print(f"   2. Set up IPFS node")
        print(f"   3. Review configuration files in config/")
        print(f"   4. Run tests: python -m pytest tests/")
        
        print("\n" + "="*60)


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="R3MES Miner Engine Setup")
    parser.add_argument("--environment", default="development", 
                       choices=["development", "staging", "production"],
                       help="Environment to set up")
    parser.add_argument("--private-key", help="Private key (generates new if not provided)")
    parser.add_argument("--blockchain-url", default="localhost:9090", 
                       help="Blockchain gRPC URL")
    parser.add_argument("--chain-id", default="remes-test", help="Blockchain chain ID")
    parser.add_argument("--install-optional", action="store_true", 
                       help="Install optional dependencies")
    parser.add_argument("--skip-tests", action="store_true", help="Skip initial tests")
    parser.add_argument("--docker", action="store_true", help="Create Docker configuration")
    parser.add_argument("--systemd", action="store_true", help="Create systemd service")
    
    args = parser.parse_args()
    
    # Initialize setup
    setup = R3MESSetup()
    
    # Check requirements
    requirements = setup.check_system_requirements()
    if not requirements["python"]:
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    
    # Create directories
    setup.create_directories()
    
    # Install dependencies
    setup.install_dependencies(install_optional=args.install_optional)
    
    # Create environment file
    private_key = setup.create_environment_file(
        environment=args.environment,
        private_key=args.private_key,
        blockchain_url=args.blockchain_url,
        chain_id=args.chain_id
    )
    
    # Setup Git hooks
    setup.setup_git_hooks()
    
    # Create Docker configuration if requested
    if args.docker:
        setup.create_dockerfile()
        setup.create_docker_compose()
    
    # Create systemd service if requested
    if args.systemd:
        setup.create_systemd_service()
    
    # Run tests unless skipped
    if not args.skip_tests:
        setup.run_initial_tests()
    
    # Print summary
    setup.print_setup_summary(private_key)


if __name__ == "__main__":
    main()
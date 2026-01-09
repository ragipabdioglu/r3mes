#!/usr/bin/env python3
"""
R3MES Miner CLI Commands

Main entry point for r3mes-miner command-line interface.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
miner_engine_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(miner_engine_dir))

from r3mes.cli.wizard import run_setup_wizard
from r3mes.cli.wallet import WalletManager
from r3mes.cli.config import ConfigManager

# Import MinerEngine with backward compatibility
try:
    from r3mes.miner.engine import MinerEngine
except ImportError:
    # Fallback to old location for development
    sys.path.insert(0, str(miner_engine_dir))
    from miner_engine import MinerEngine


def cmd_start(args):
    """Start mining operation."""
    config = ConfigManager()
    config_path = args.config or config.get_default_config_path()
    
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        print("💡 Run 'r3mes-miner setup' to create a configuration file.")
        sys.exit(1)
    
    # Load configuration
    cfg = config.load_config(config_path)
    
    # Override with command-line arguments
    if args.private_key:
        cfg['private_key'] = args.private_key
    if args.blockchain_url:
        cfg['blockchain_url'] = args.blockchain_url
    if args.chain_id:
        cfg['chain_id'] = args.chain_id
    if args.max_iterations:
        cfg['max_iterations'] = args.max_iterations
    if args.model_size:
        cfg['model_size'] = args.model_size
    if args.lora_rank:
        cfg['lora_rank'] = args.lora_rank
    if getattr(args, 'use_tls', False):
        cfg['use_tls'] = True
    if getattr(args, 'tls_cert_file', None):
        cfg['tls_cert_file'] = args.tls_cert_file
    if getattr(args, 'tls_key_file', None):
        cfg['tls_key_file'] = args.tls_key_file
    if getattr(args, 'tls_ca_file', None):
        cfg['tls_ca_file'] = args.tls_ca_file
    if getattr(args, 'tls_server_name', None):
        cfg['tls_server_name'] = args.tls_server_name
    
    # Validate required fields
    if not cfg.get('private_key'):
        print("❌ Private key is required. Use --private-key or run 'r3mes-miner setup'")
        sys.exit(1)
    
    print("🚀 Starting R3MES Miner...")
    print(f"📍 Blockchain URL: {cfg['blockchain_url']}")
    print(f"🔗 Chain ID: {cfg['chain_id']}")
    print(f"🔐 TLS Enabled: {cfg.get('use_tls', False)}")
    print(f"🎯 Model Size: {cfg.get('model_size', 768)}")
    print(f"📊 LoRA Rank: {cfg.get('lora_rank', 8)}")
    print()
    
    # Pre-flight checks: Version and time sync
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    
    # Version check
    try:
        from r3mes.utils.version_checker import check_version_or_exit
        print("🔍 Checking version...")
        check_version_or_exit(
            api_url=backend_url,
            critical_only=False  # Warn on any update, exit only on critical
        )
        print("✅ Version check passed")
    except Exception as e:
        print(f"⚠️  Version check failed: {e}")
        # Continue anyway (non-blocking)
    
    # Time sync check
    try:
        from r3mes.utils.time_sync import check_time_sync_or_warn
        print("🔍 Checking time synchronization...")
        is_synced = check_time_sync_or_warn(
            api_url=backend_url,
            max_drift_seconds=5,
            critical=False  # Warn only, don't exit
        )
        if is_synced:
            print("✅ Time synchronization check passed")
    except Exception as e:
        print(f"⚠️  Time sync check failed: {e}")
        # Continue anyway (non-blocking)
    
    print()
    
    # Create miner engine
    try:
        miner = MinerEngine(
            private_key=cfg['private_key'],
            blockchain_url=cfg['blockchain_url'],
            chain_id=cfg['chain_id'],
            model_hidden_size=cfg.get('model_size', 768),
            lora_rank=cfg.get('lora_rank', 8),
            gradient_accumulation_steps=cfg.get('gradient_accumulation_steps', 4),
            top_k_compression=cfg.get('top_k_compression', 0.1),
            use_tls=cfg.get('use_tls', False),
            tls_cert_file=cfg.get('tls_cert_file'),
            tls_key_file=cfg.get('tls_key_file'),
            tls_ca_file=cfg.get('tls_ca_file'),
            tls_server_name=cfg.get('tls_server_name'),
        )
        
        # Run mining loop
        miner.train_and_submit(num_iterations=cfg.get('max_iterations', 5))
        
        # Print statistics
        stats = miner.get_statistics()
        print("\n📊 Final Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
    except KeyboardInterrupt:
        print("\n⚠️  Mining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error during mining: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cmd_setup(args):
    """Run interactive setup wizard."""
    print("🎯 R3MES Miner Setup Wizard")
    print("=" * 50)
    print()
    
    try:
        run_setup_wizard()
        print("\n✅ Setup complete! Run 'r3mes-miner start' to begin mining.")
    except KeyboardInterrupt:
        print("\n⚠️  Setup cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cmd_status(args):
    """Show miner status and configuration."""
    config = ConfigManager()
    config_path = args.config or config.get_default_config_path()
    
    if not os.path.exists(config_path):
        print("❌ Configuration file not found. Run 'r3mes-miner setup' first.")
        sys.exit(1)
    
    cfg = config.load_config(config_path)
    
    print("📋 R3MES Miner Status")
    print("=" * 50)
    print(f"Configuration: {config_path}")
    print()
    print("🔧 Configuration:")
    print(f"  Blockchain URL: {cfg.get('blockchain_url', 'N/A')}")
    print(f"  Chain ID: {cfg.get('chain_id', 'N/A')}")
    print(f"  Model Size: {cfg.get('model_size', 768)}")
    print(f"  LoRA Rank: {cfg.get('lora_rank', 8)}")
    print(f"  Gradient Accumulation Steps: {cfg.get('gradient_accumulation_steps', 4)}")
    print(f"  Top-K Compression: {cfg.get('top_k_compression', 0.1)}")
    print(f"  TLS Enabled: {cfg.get('use_tls', False)}")
    if cfg.get('use_tls', False):
        print(f"  TLS Cert File: {cfg.get('tls_cert_file', 'N/A')}")
        print(f"  TLS Key File: {cfg.get('tls_key_file', 'N/A')}")
        print(f"  TLS CA File: {cfg.get('tls_ca_file', 'N/A')}")
        print(f"  TLS Server Name: {cfg.get('tls_server_name', 'auto')}")
    print()
    
    # Check wallet
    wallet_path = config.get_wallet_path()
    if os.path.exists(wallet_path):
        wallet = WalletManager()
        try:
            address = wallet.get_address(wallet_path)
            print(f"💼 Wallet: {address}")
        except Exception as e:
            print(f"⚠️  Wallet file exists but could not read: {e}")
    else:
        print("⚠️  No wallet found. Run 'r3mes-miner setup' to create one.")
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            print(f"🎮 GPU: {props.name} (Compute Capability: {props.major}.{props.minor})")
        else:
            print("⚠️  No GPU detected. Mining will use CPU (slower).")
    except ImportError:
        print("⚠️  PyTorch not installed. Cannot detect GPU.")


def cmd_wallet(args):
    """Wallet management commands."""
    wallet = WalletManager()
    
    if args.action == 'create':
        print("🔐 Creating new wallet...")
        wallet_path = wallet.create_wallet()
        address = wallet.get_address(wallet_path)
        print(f"✅ Wallet created: {wallet_path}")
        print(f"📍 Address: {address}")
        print("\n⚠️  IMPORTANT: Save your private key securely!")
        print("   Do not share your private key with anyone.")
        
    elif args.action == 'show':
        wallet_path = args.wallet_path or wallet.get_default_wallet_path()
        if not os.path.exists(wallet_path):
            print(f"❌ Wallet not found: {wallet_path}")
            sys.exit(1)
        
        address = wallet.get_address(wallet_path)
        print(f"📍 Address: {address}")
        print(f"📁 Wallet Path: {wallet_path}")
        
    else:
        print("❌ Unknown wallet action. Use 'create' or 'show'")
        sys.exit(1)


def cmd_version(args):
    """Show version information."""
    try:
        from r3mes import __version__
        version = __version__
    except ImportError:
        version = "0.1.0"
    
    print(f"R3MES Miner Engine v{version}")
    print(f"Python {sys.version.split()[0]}")
    
    # Check PyTorch version
    try:
        import torch
        print(f"PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch: Not installed")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="R3MES Proof of Useful Work (PoUW) Miner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  r3mes-miner setup              # Run interactive setup wizard
  r3mes-miner start              # Start mining with default config
  r3mes-miner start --config /path/to/config.json
  r3mes-miner status             # Show current status
  r3mes-miner wallet create      # Create new wallet
  r3mes-miner version            # Show version information
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start mining operation')
    start_parser.add_argument('--config', type=str, help='Path to configuration file')
    start_parser.add_argument('--private-key', type=str, help='Private key (overrides config)')
    start_parser.add_argument('--blockchain-url', type=str, help='Blockchain gRPC URL')
    start_parser.add_argument('--chain-id', type=str, help='Chain ID')
    start_parser.add_argument('--max-iterations', type=int, help='Maximum training iterations')
    start_parser.add_argument('--model-size', type=int, help='Model hidden size')
    start_parser.add_argument('--lora-rank', type=int, help='LoRA rank')
    start_parser.add_argument('--use-tls', action='store_true', help='Enable TLS/mTLS for gRPC connection')
    start_parser.add_argument('--tls-cert-file', type=str, help='Client TLS certificate file (PEM)')
    start_parser.add_argument('--tls-key-file', type=str, help='Client TLS private key file (PEM)')
    start_parser.add_argument('--tls-ca-file', type=str, help='CA certificate file for server verification (PEM)')
    start_parser.add_argument('--tls-server-name', type=str, help='Expected TLS server name (overrides hostname detection)')
    start_parser.set_defaults(func=cmd_start)
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Run interactive setup wizard')
    setup_parser.set_defaults(func=cmd_setup)
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show miner status')
    status_parser.add_argument('--config', type=str, help='Path to configuration file')
    status_parser.set_defaults(func=cmd_status)
    
    # Wallet command
    wallet_parser = subparsers.add_parser('wallet', help='Wallet management')
    wallet_subparsers = wallet_parser.add_subparsers(dest='action', help='Wallet action')
    
    wallet_create = wallet_subparsers.add_parser('create', help='Create new wallet')
    wallet_create.set_defaults(func=cmd_wallet)
    
    wallet_show = wallet_subparsers.add_parser('show', help='Show wallet address')
    wallet_show.add_argument('--wallet-path', type=str, help='Path to wallet file')
    wallet_show.set_defaults(func=cmd_wallet)
    
    # Version command
    version_parser = subparsers.add_parser('version', help='Show version information')
    version_parser.set_defaults(func=cmd_version)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Call appropriate command function
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()


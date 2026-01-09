#!/usr/bin/env python3
"""
R3MES Serving CLI Commands

Command-line interface for serving node operations.
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
miner_engine_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(miner_engine_dir))

from r3mes.cli.wizard import run_setup_wizard
from r3mes.cli.wallet import WalletManager
from r3mes.cli.config import ConfigManager
from r3mes.serving.engine import ServingEngine


def cmd_setup(args):
    """Run setup wizard for serving node."""
    print("🚀 R3MES Serving Node Setup Wizard")
    print("=" * 50)
    
    # Use similar wizard structure as miner
    config = ConfigManager()
    config_path = args.config or config.get_default_config_path()
    
    # Run wizard (simplified for serving)
    print("\n📋 Serving Node Setup")
    print("This wizard will guide you through serving node configuration.")
    
    # Wallet setup
    wallet_manager = WalletManager()
    if not wallet_manager.wallet_exists():
        print("\n💼 Wallet Setup")
        wallet_manager.create_wallet()
    else:
        print("✅ Wallet found")
    
    # Blockchain configuration
    print("\n🌐 Blockchain Configuration")
    blockchain_url = input("Blockchain gRPC URL [localhost:9090]: ").strip() or "localhost:9090"
    chain_id = input("Chain ID [remes-1]: ").strip() or "remes-1"
    
    # Model configuration
    print("\n🤖 Model Configuration")
    model_ipfs_hash = input("Model IPFS hash (optional, will query from blockchain if empty): ").strip() or ""
    model_version = input("Model version [v1.0.0]: ").strip() or "v1.0.0"
    
    # Save configuration
    config_data = {
        "private_key": wallet_manager.get_private_key(),
        "blockchain_url": blockchain_url,
        "chain_id": chain_id,
        "model_ipfs_hash": model_ipfs_hash,
        "model_version": model_version,
        "node_type": "serving",
    }
    
    config.save_config(config_path, config_data)
    print(f"\n✅ Configuration saved to: {config_path}")
    print("\n🎉 Setup complete! Run 'r3mes-serving start' to start the serving node.")


def cmd_start(args):
    """Start serving node."""
    config = ConfigManager()
    config_path = args.config or config.get_default_config_path()
    
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        print("💡 Run 'r3mes-serving setup' to create a configuration file.")
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
    if args.model_ipfs_hash:
        cfg['model_ipfs_hash'] = args.model_ipfs_hash
    if args.model_version:
        cfg['model_version'] = args.model_version
    
    # Validate required fields
    if not cfg.get('private_key'):
        print("❌ Private key is required. Use --private-key or run 'r3mes-serving setup'")
        sys.exit(1)
    
    print("🚀 Starting R3MES Serving Node...")
    print(f"📍 Blockchain URL: {cfg['blockchain_url']}")
    print(f"🔗 Chain ID: {cfg['chain_id']}")
    print(f"🤖 Model Version: {cfg.get('model_version', 'v1.0.0')}")
    if cfg.get('model_ipfs_hash'):
        print(f"📦 Model IPFS Hash: {cfg['model_ipfs_hash']}")
    print()
    
    # Create and start serving engine
    engine = ServingEngine(
        private_key=cfg['private_key'],
        blockchain_url=cfg['blockchain_url'],
        chain_id=cfg['chain_id'],
        model_ipfs_hash=cfg.get('model_ipfs_hash'),
        model_version=cfg.get('model_version', 'v1.0.0'),
        log_level=args.log_level or "INFO",
        use_json_logs=args.json_logs,
    )
    
    try:
        engine.start()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        engine.stop()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def cmd_stop(args):
    """Stop serving node."""
    print("🛑 Stopping serving node...")
    
    try:
        import psutil
        
        # Find serving node processes
        serving_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and any('serving' in arg.lower() for arg in cmdline):
                    if any('r3mes' in arg.lower() for arg in cmdline):
                        serving_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if not serving_processes:
            print("❌ No serving node processes found")
            return
        
        # Stop processes
        for proc in serving_processes:
            try:
                print(f"Stopping serving node process: PID {proc.pid}")
                proc.terminate()
                
                # Wait for graceful shutdown
                try:
                    proc.wait(timeout=10)
                    print(f"✅ Process {proc.pid} stopped gracefully")
                except psutil.TimeoutExpired:
                    # Force kill if not stopped
                    proc.kill()
                    print(f"⚠️  Process {proc.pid} force killed")
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                print(f"❌ Error stopping process {proc.pid}: {e}")
        
        print("✅ Serving node stop command completed")
        
    except ImportError:
        print("❌ psutil not available. Install with: pip install psutil")
        print("⚠️  Use Ctrl+C to stop the serving node manually.")
    except Exception as e:
        print(f"❌ Error stopping serving node: {e}")
        print("⚠️  Use Ctrl+C to stop the serving node manually.")


def cmd_status(args):
    """Check serving node status."""
    config = ConfigManager()
    config_path = args.config or config.get_default_config_path()
    
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)
    
    cfg = config.load_config(config_path)
    
    print("📊 Serving Node Status")
    print("=" * 50)
    print(f"Blockchain URL: {cfg.get('blockchain_url', 'N/A')}")
    print(f"Chain ID: {cfg.get('chain_id', 'N/A')}")
    print(f"Model Version: {cfg.get('model_version', 'N/A')}")
    print(f"Model IPFS Hash: {cfg.get('model_ipfs_hash', 'Not set')}")
    print("\n⚠️  Detailed status query not yet implemented.")


def cmd_register(args):
    """Register as serving node on blockchain."""
    config = ConfigManager()
    config_path = args.config or config.get_default_config_path()
    
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)
    
    cfg = config.load_config(config_path)
    
    print("📝 Registering serving node on blockchain...")
    print(f"📍 Blockchain URL: {cfg['blockchain_url']}")
    print(f"🔗 Chain ID: {cfg['chain_id']}")
    
    try:
        from bridge.blockchain_client import BlockchainClient
        from bridge.crypto import derive_address_from_public_key, hex_to_private_key
        
        # Create blockchain client
        blockchain_client = BlockchainClient(
            node_url=cfg['blockchain_url'],
            chain_id=cfg['chain_id'],
            private_key=cfg['private_key'],
        )
        
        # Get node address
        node_address = derive_address_from_public_key(
            hex_to_private_key(cfg['private_key']).public_key()
        )
        
        print(f"🏷️  Node Address: {node_address}")
        
        # Register node (NODE_TYPE_SERVING = 3)
        result = blockchain_client.register_node(
            node_address=node_address,
            node_type=3,  # SERVING node type
            stake="0",    # No stake required for serving nodes
            roles=[3],    # SERVING role
        )
        
        if result.get("success", False):
            registration_id = result.get("registration_id", 0)
            tx_hash = result.get("tx_hash", "")
            print(f"✅ Registration successful!")
            print(f"📋 Registration ID: {registration_id}")
            print(f"🔗 Transaction Hash: {tx_hash}")
        else:
            error = result.get("error", "Unknown error")
            print(f"❌ Registration failed: {error}")
            sys.exit(1)
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure blockchain client dependencies are installed")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Registration error: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point for serving node."""
    parser = argparse.ArgumentParser(
        description="R3MES Serving Node - AI Model Inference Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  r3mes-serving setup              # Run interactive setup wizard
  r3mes-serving start              # Start serving node with default config
  r3mes-serving start --config /path/to/config.json
  r3mes-serving status             # Show current status
  r3mes-serving register           # Register as serving node on blockchain
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start serving node')
    start_parser.add_argument('--config', type=str, help='Path to configuration file')
    start_parser.add_argument('--private-key', type=str, help='Private key (overrides config)')
    start_parser.add_argument('--blockchain-url', type=str, help='Blockchain gRPC URL')
    start_parser.add_argument('--chain-id', type=str, help='Chain ID')
    start_parser.add_argument('--model-ipfs-hash', type=str, help='Model IPFS hash')
    start_parser.add_argument('--model-version', type=str, help='Model version')
    start_parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Log level')
    start_parser.add_argument('--json-logs', action='store_true', help='Use JSON log format')
    start_parser.set_defaults(func=cmd_start)
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Run interactive setup wizard')
    setup_parser.add_argument('--config', type=str, help='Path to configuration file')
    setup_parser.set_defaults(func=cmd_setup)
    
    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop serving node')
    stop_parser.set_defaults(func=cmd_stop)
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show serving node status')
    status_parser.add_argument('--config', type=str, help='Path to configuration file')
    status_parser.set_defaults(func=cmd_status)
    
    # Register command
    register_parser = subparsers.add_parser('register', help='Register as serving node on blockchain')
    register_parser.add_argument('--config', type=str, help='Path to configuration file')
    register_parser.set_defaults(func=cmd_register)
    
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


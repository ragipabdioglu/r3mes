#!/usr/bin/env python3
"""
R3MES Proposer CLI Commands

Command-line interface for proposer node operations.
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
miner_engine_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(miner_engine_dir))

from r3mes.cli.wallet import WalletManager
from r3mes.cli.config import ConfigManager
from r3mes.proposer.aggregator import ProposerAggregator


def cmd_setup(args):
    """Run setup wizard for proposer."""
    print("🚀 R3MES Proposer Setup Wizard")
    print("=" * 50)
    
    config = ConfigManager()
    config_path = args.config or config.get_default_config_path()
    
    print("\n📋 Proposer Setup")
    print("This wizard will guide you through proposer configuration.")
    
    # Wallet setup
    wallet_manager = WalletManager()
    if not wallet_manager.wallet_exists():
        print("\n💼 Wallet Setup")
        wallet_manager.create_wallet()
    else:
        print("✅ Wallet found")
    
    # Blockchain configuration
    print("\n🌐 Blockchain Configuration")
    
    # Production localhost validation
    import os
    is_production = os.getenv("R3MES_ENV", "development").lower() == "production"
    
    if is_production:
        # In production, require environment variable or explicit input (no localhost default)
        default_blockchain_url = os.getenv("R3MES_NODE_GRPC_URL")
        if not default_blockchain_url:
            print("⚠️  Production mode: R3MES_NODE_GRPC_URL environment variable must be set")
            blockchain_url = input("Blockchain gRPC URL (required): ").strip()
            if not blockchain_url:
                raise ValueError("Blockchain gRPC URL is required in production. Set R3MES_NODE_GRPC_URL or provide URL.")
            # Validate no localhost
            blockchain_host = blockchain_url.split(":")[0] if ":" in blockchain_url else blockchain_url
            if blockchain_host.lower() in ("localhost", "127.0.0.1", "::1") or blockchain_host.startswith("127."):
                raise ValueError(
                    f"Blockchain gRPC URL cannot use localhost in production: {blockchain_url}. "
                    "Please provide a production gRPC endpoint."
                )
        else:
            blockchain_url = input(f"Blockchain gRPC URL [{default_blockchain_url}]: ").strip() or default_blockchain_url
            # Validate no localhost even if from env var
            blockchain_host = blockchain_url.split(":")[0] if ":" in blockchain_url else blockchain_url
            if blockchain_host.lower() in ("localhost", "127.0.0.1", "::1") or blockchain_host.startswith("127."):
                raise ValueError(
                    f"R3MES_NODE_GRPC_URL cannot use localhost in production: {blockchain_url}. "
                    "Please set R3MES_NODE_GRPC_URL to a production gRPC endpoint."
                )
    else:
        # Development: allow localhost default
        blockchain_url = input("Blockchain gRPC URL [localhost:9090]: ").strip() or "localhost:9090"
    
    chain_id = input("Chain ID [remes-1]: ").strip() or "remes-1"
    
    # Save configuration
    config_data = {
        "private_key": wallet_manager.get_private_key(),
        "blockchain_url": blockchain_url,
        "chain_id": chain_id,
        "node_type": "proposer",
    }
    
    config.save_config(config_path, config_data)
    print(f"\n✅ Configuration saved to: {config_path}")
    print("\n🎉 Setup complete! Run 'r3mes-proposer start' to start the proposer.")


def cmd_start(args):
    """Start proposer aggregation service."""
    config = ConfigManager()
    config_path = args.config or config.get_default_config_path()
    
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        print("💡 Run 'r3mes-proposer setup' to create a configuration file.")
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
    
    # Validate required fields
    if not cfg.get('private_key'):
        print("❌ Private key is required. Use --private-key or run 'r3mes-proposer setup'")
        sys.exit(1)
    
    print("🚀 Starting R3MES Proposer...")
    print(f"📍 Blockchain URL: {cfg['blockchain_url']}")
    print(f"🔗 Chain ID: {cfg['chain_id']}")
    print()
    
    # Create proposer aggregator
    aggregator = ProposerAggregator(
        private_key=cfg['private_key'],
        blockchain_url=cfg['blockchain_url'],
        chain_id=cfg['chain_id'],
        log_level=args.log_level or "INFO",
        use_json_logs=args.json_logs,
    )
    
    # Start aggregation service
    # Note: Continuous aggregation polling is a future enhancement
    # For now, use 'r3mes-proposer aggregate' to manually trigger aggregation
    # or implement a custom polling loop using aggregator.query_pending_gradients()
    # and aggregator.aggregate_gradients() methods
    print("ℹ️  Continuous aggregation polling not yet implemented.")
    print("Use 'r3mes-proposer aggregate' to manually trigger aggregation.")
    print("Future: Implement background polling loop in aggregator.start() method.")


def cmd_pool(args):
    """View pending gradients pool."""
    config = ConfigManager()
    config_path = args.config or config.get_default_config_path()
    
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)
    
    cfg = config.load_config(config_path)
    
    print("📊 Pending Gradients Pool")
    print("=" * 50)
    
    aggregator = ProposerAggregator(
        private_key=cfg['private_key'],
        blockchain_url=cfg['blockchain_url'],
        chain_id=cfg['chain_id'],
    )
    
    # Query pending gradients
    pending = aggregator.query_pending_gradients(limit=args.limit or 100)
    
    print(f"Total pending gradients: {len(pending)}")
    if pending:
        print("\nGradient ID | Status | IPFS Hash")
        print("-" * 50)
        for grad in pending[:10]:  # Show first 10
            grad_id = grad.get("id", "N/A")
            status = grad.get("status", "N/A")
            ipfs_hash = grad.get("ipfs_hash", "N/A")[:20] + "..."
            print(f"{grad_id:11} | {status:6} | {ipfs_hash}")
    else:
        print("No pending gradients found.")


def cmd_aggregate(args):
    """Manually trigger aggregation."""
    config = ConfigManager()
    config_path = args.config or config.get_default_config_path()
    
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)
    
    cfg = config.load_config(config_path)
    
    if not args.gradient_ids:
        print("❌ --gradient-ids required. Example: --gradient-ids 1,2,3")
        sys.exit(1)
    
    gradient_ids = [int(id.strip()) for id in args.gradient_ids.split(',')]
    training_round_id = args.training_round_id or 1
    
    print(f"🔄 Starting aggregation: {len(gradient_ids)} gradients, round={training_round_id}")
    
    aggregator = ProposerAggregator(
        private_key=cfg['private_key'],
        blockchain_url=cfg['blockchain_url'],
        chain_id=cfg['chain_id'],
    )
    
    success = aggregator.aggregate_and_submit(gradient_ids, training_round_id)
    
    if success:
        print("✅ Aggregation completed successfully")
    else:
        print("❌ Aggregation failed")
        sys.exit(1)


def cmd_register(args):
    """Register as proposer on blockchain."""
    config = ConfigManager()
    config_path = args.config or config.get_default_config_path()
    
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)
    
    cfg = config.load_config(config_path)
    
    print("📝 Registering proposer node on blockchain...")
    print(f"📍 Blockchain URL: {cfg['blockchain_url']}")
    print(f"🔗 Chain ID: {cfg['chain_id']}")
    
    # TODO: Implement blockchain registration
    # This would use BlockchainClient.register_node with NODE_TYPE_PROPOSER
    print("⚠️  Registration not yet implemented. Use blockchain client directly.")


def main():
    """Main CLI entry point for proposer."""
    parser = argparse.ArgumentParser(
        description="R3MES Proposer - Gradient Aggregation Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  r3mes-proposer setup              # Run interactive setup wizard
  r3mes-proposer start              # Start proposer service
  r3mes-proposer pool               # View pending gradients pool
  r3mes-proposer aggregate --gradient-ids 1,2,3  # Manually trigger aggregation
  r3mes-proposer register           # Register as proposer on blockchain
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start proposer aggregation service')
    start_parser.add_argument('--config', type=str, help='Path to configuration file')
    start_parser.add_argument('--private-key', type=str, help='Private key (overrides config)')
    start_parser.add_argument('--blockchain-url', type=str, help='Blockchain gRPC URL')
    start_parser.add_argument('--chain-id', type=str, help='Chain ID')
    start_parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Log level')
    start_parser.add_argument('--json-logs', action='store_true', help='Use JSON log format')
    start_parser.set_defaults(func=cmd_start)
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Run interactive setup wizard')
    setup_parser.add_argument('--config', type=str, help='Path to configuration file')
    setup_parser.set_defaults(func=cmd_setup)
    
    # Pool command
    pool_parser = subparsers.add_parser('pool', help='View pending gradients pool')
    pool_parser.add_argument('--config', type=str, help='Path to configuration file')
    pool_parser.add_argument('--limit', type=int, help='Maximum number of gradients to show')
    pool_parser.set_defaults(func=cmd_pool)
    
    # Aggregate command
    aggregate_parser = subparsers.add_parser('aggregate', help='Manually trigger aggregation')
    aggregate_parser.add_argument('--config', type=str, help='Path to configuration file')
    aggregate_parser.add_argument('--gradient-ids', type=str, required=True, help='Comma-separated gradient IDs')
    aggregate_parser.add_argument('--training-round-id', type=int, help='Training round ID')
    aggregate_parser.set_defaults(func=cmd_aggregate)
    
    # Register command
    register_parser = subparsers.add_parser('register', help='Register as proposer on blockchain')
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


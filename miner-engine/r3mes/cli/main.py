#!/usr/bin/env python3
"""
R3MES Unified CLI

Main entry point for all r3mes commands.
Usage: r3mes <command> [options]
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
miner_engine_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(miner_engine_dir))

try:
    import click
    from rich.console import Console
    from rich.table import Table
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    click = None

console = Console() if HAS_RICH else None


def print_banner():
    """Print R3MES banner."""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   ██████╗ ██████╗ ███╗   ███╗███████╗███████╗            ║
║   ██╔══██╗╚════██╗████╗ ████║██╔════╝██╔════╝            ║
║   ██████╔╝ █████╔╝██╔████╔██║█████╗  ███████╗            ║
║   ██╔══██╗ ╚═══██╗██║╚██╔╝██║██╔══╝  ╚════██║            ║
║   ██║  ██║██████╔╝██║ ╚═╝ ██║███████╗███████║            ║
║   ╚═╝  ╚═╝╚═════╝ ╚═╝     ╚═╝╚══════╝╚══════╝            ║
║                                                           ║
║   Decentralized AI Training Network                       ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
"""
    print(banner)


def main_argparse():
    """Fallback main using argparse (when click not available)."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="R3MES - Decentralized AI Training Network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Miner command
    miner_parser = subparsers.add_parser('miner', help='Miner operations')
    miner_parser.add_argument('action', choices=['start', 'stop', 'status', 'setup'])
    
    # Serving command
    serving_parser = subparsers.add_parser('serving', help='Serving node operations')
    serving_parser.add_argument('action', choices=['start', 'stop', 'status'])
    
    # Proposer command
    proposer_parser = subparsers.add_parser('proposer', help='Proposer operations')
    proposer_parser.add_argument('action', choices=['start', 'stop', 'status'])
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Interactive setup wizard')
    
    # Version command
    version_parser = subparsers.add_parser('version', help='Show version')
    
    args = parser.parse_args()
    
    if not args.command:
        print_banner()
        parser.print_help()
        return
    
    if args.command == 'miner':
        from r3mes.cli.commands import main as miner_main
        sys.argv = ['r3mes-miner', args.action]
        miner_main()
    elif args.command == 'serving':
        from r3mes.cli.serving_commands import main as serving_main
        sys.argv = ['r3mes-serving', args.action]
        serving_main()
    elif args.command == 'proposer':
        from r3mes.cli.proposer_commands import main as proposer_main
        sys.argv = ['r3mes-proposer', args.action]
        proposer_main()
    elif args.command == 'setup':
        from r3mes.cli.setup import run_full_setup
        run_full_setup()
    elif args.command == 'version':
        show_version()


def show_version():
    """Show version information."""
    try:
        from r3mes import __version__
        version = __version__
    except ImportError:
        version = "0.1.0"
    
    print(f"R3MES v{version}")
    print(f"Python {sys.version.split()[0]}")
    
    try:
        import torch
        print(f"PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch: Not installed")


# Click-based CLI (preferred when available)
if click:
    @click.group()
    @click.version_option(version="0.1.0", prog_name="r3mes")
    def cli():
        """R3MES - Decentralized AI Training Network"""
        pass
    
    @cli.command()
    @click.argument('action', type=click.Choice(['start', 'stop', 'status', 'setup']))
    @click.option('--config', '-c', help='Configuration file path')
    @click.option('--private-key', '-k', help='Private key')
    def miner(action, config, private_key):
        """Miner node operations."""
        from r3mes.cli.commands import main as miner_main
        args = ['r3mes-miner', action]
        if config:
            args.extend(['--config', config])
        if private_key:
            args.extend(['--private-key', private_key])
        sys.argv = args
        miner_main()
    
    @cli.command()
    @click.argument('action', type=click.Choice(['start', 'stop', 'status']))
    @click.option('--config', '-c', help='Configuration file path')
    def serving(action, config):
        """Serving node operations."""
        from r3mes.cli.serving_commands import main as serving_main
        args = ['r3mes-serving', action]
        if config:
            args.extend(['--config', config])
        sys.argv = args
        serving_main()
    
    @cli.command()
    @click.argument('action', type=click.Choice(['start', 'stop', 'status']))
    @click.option('--config', '-c', help='Configuration file path')
    def proposer(action, config):
        """Proposer node operations."""
        from r3mes.cli.proposer_commands import main as proposer_main
        args = ['r3mes-proposer', action]
        if config:
            args.extend(['--config', config])
        sys.argv = args
        proposer_main()
    
    @cli.command()
    def setup():
        """Interactive setup wizard."""
        from r3mes.cli.setup import run_full_setup
        run_full_setup()
    
    @cli.command()
    def version():
        """Show version information."""
        show_version()
    
    @cli.command()
    def info():
        """Show system information."""
        print_banner()
        show_version()
        print()
        
        # Hardware info
        print("Hardware:")
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    vram_gb = props.total_memory / (1024**3)
                    print(f"  GPU {i}: {props.name} ({vram_gb:.1f} GB VRAM)")
            else:
                print("  No GPU detected")
        except ImportError:
            print("  PyTorch not installed")
    
    def main():
        """Main entry point."""
        cli()

else:
    def main():
        """Main entry point (argparse fallback)."""
        main_argparse()


if __name__ == '__main__':
    main()

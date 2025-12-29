#!/usr/bin/env python3
"""
Continuous Mining Wrapper for R3MES Miner

This script runs the miner in a continuous loop, restarting after each batch of iterations.
"""

import sys
import os
import time
import signal
import threading
from pathlib import Path

# Add parent directory to path
miner_engine_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(miner_engine_dir))

from r3mes.cli.config import ConfigManager
from r3mes.miner.engine import MinerEngine
from r3mes.miner.stats_http_server import start_stats_server
import argparse

# Global flag for graceful shutdown
running = True

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    global running
    print("\n⚠️  Stopping continuous mining...")
    running = False
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def continuous_mining(args):
    """Run miner in continuous loop."""
    global running
    
    # Load config if provided
    config = ConfigManager()
    config_path = args.config or config.get_default_config_path()
    
    if os.path.exists(config_path):
        cfg = config.load_config(config_path)
    else:
        cfg = {}
    
    # Override with command-line arguments
    if args.private_key:
        cfg['private_key'] = args.private_key
    if args.blockchain_url:
        cfg['blockchain_url'] = args.blockchain_url
    if args.chain_id:
        cfg['chain_id'] = args.chain_id
    if args.model_size:
        cfg['model_size'] = args.model_size
    if args.lora_rank:
        cfg['lora_rank'] = args.lora_rank
    
    # Validate required fields
    if not cfg.get('private_key'):
        print("❌ Private key is required. Use --private-key or run 'r3mes-miner setup'")
        sys.exit(1)
    
    batch_count = 0
    stats_server_thread = None
    miner_engine_created = False
    
    print("🔄 Starting continuous mining mode...")
    print(f"📦 Batch size: {args.batch_size} iterations per batch")
    print(f"⏸️  Pause between batches: {args.pause_seconds} seconds")
    print("Press Ctrl+C to stop\n")
    
    while running:
        batch_count += 1
        print(f"\n{'='*60}")
        print(f"Batch #{batch_count}")
        print(f"{'='*60}\n")
        
        try:
            # Create miner engine (this initializes stats collector)
            miner = MinerEngine(
                private_key=cfg['private_key'],
                blockchain_url=cfg.get('blockchain_url', 'localhost:9090'),
                chain_id=cfg.get('chain_id', 'remes-test'),
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
            
            # Start stats HTTP server after miner engine is created (stats collector is now initialized)
            if not miner_engine_created:
                try:
                    stats_server_thread = threading.Thread(
                        target=start_stats_server,
                        args=(8080, "localhost"),
                        daemon=True
                    )
                    stats_server_thread.start()
                    print("✅ Stats HTTP server started on http://localhost:8080/stats")
                    miner_engine_created = True
                except Exception as e:
                    print(f"⚠️  Failed to start stats server: {e}")
                    print("   LoRA adapters info will not be available")
            
            # Run one batch
            miner.train_and_submit(num_iterations=args.batch_size)
            
            if not running:
                break
            
            # Pause between batches
            if args.pause_seconds > 0:
                print(f"\n⏸️  Pausing for {args.pause_seconds} seconds before next batch...")
                for _ in range(args.pause_seconds):
                    if not running:
                        break
                    time.sleep(1)
            
        except KeyboardInterrupt:
            print("\n⚠️  Mining interrupted by user")
            running = False
            break
        except Exception as e:
            print(f"\n❌ Error in batch #{batch_count}: {e}")
            import traceback
            traceback.print_exc()
            if args.stop_on_error:
                print("Stopping due to error...")
                break
            
            # Wait before retrying
            if args.pause_seconds > 0:
                print(f"Waiting {args.pause_seconds} seconds before retry...")
                time.sleep(args.pause_seconds)
    
    print(f"\n✅ Continuous mining stopped. Completed {batch_count} batch(es).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="R3MES Continuous Mining")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--private-key", type=str, help="Private key")
    parser.add_argument("--blockchain-url", type=str, help="Blockchain gRPC URL")
    parser.add_argument("--chain-id", type=str, help="Chain ID")
    parser.add_argument("--batch-size", type=int, default=5, help="Iterations per batch (default: 5)")
    parser.add_argument("--pause-seconds", type=int, default=5, help="Pause between batches in seconds (default: 5)")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop on error instead of retrying")
    parser.add_argument("--model-size", type=int, help="Model hidden size")
    parser.add_argument("--lora-rank", type=int, help="LoRA rank")
    
    args = parser.parse_args()
    
    continuous_mining(args)


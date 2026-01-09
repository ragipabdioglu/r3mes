#!/usr/bin/env python3
"""
Interactive Setup Wizard for R3MES Miner

Guides users through the setup process:
1. System requirements check
2. Wallet creation
3. Blockchain configuration
4. Mining parameters
5. Connection test
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import sys
from pathlib import Path

# Add parent directory to path for imports
miner_engine_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(miner_engine_dir))

from r3mes.cli.wallet import WalletManager
from r3mes.cli.config import ConfigManager

# Import CUDA check utilities
try:
    from r3mes.utils.cuda_check import check_cuda_availability, check_pytorch_cuda
except ImportError:
    # Fallback for development
    from utils.cuda_check import check_cuda_availability, check_pytorch_cuda

# Import faucet client
try:
    from r3mes.utils.faucet import request_faucet, check_faucet_availability
except ImportError:
    # Fallback for development
    from utils.faucet import request_faucet, check_faucet_availability


def print_step(step_num: int, total_steps: int, title: str):
    """Print step header."""
    print(f"\n[{step_num}/{total_steps}] {title}")
    print("-" * 50)


def check_system_requirements() -> Dict[str, Any]:
    """Check system requirements and return status."""
    results = {
        'python_version': sys.version_info >= (3, 10),
        'pytorch_installed': False,
        'cuda_available': False,
        'gpu_detected': False,
        'ipfs_available': False,
        'firewall_warning': False,
    }
    
    # Check Python version
    if results['python_version']:
        print("✅ Python 3.10+ detected")
    else:
        print(f"❌ Python 3.10+ required (found {sys.version_info.major}.{sys.version_info.minor})")
    
    # Check PyTorch
    try:
        import torch
        results['pytorch_installed'] = True
        print("✅ PyTorch installed")
        
        # Check CUDA
        if torch.cuda.is_available():
            results['cuda_available'] = True
            results['gpu_detected'] = True
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            print(f"✅ GPU detected: {props.name} (Compute Capability: {props.major}.{props.minor})")
        else:
            print("⚠️  No GPU detected. Mining will use CPU (slower).")
    except ImportError:
        print("❌ PyTorch not installed. Install with: pip install torch")
    
    # Check IPFS (optional, just warn if not available)
    try:
        import ipfshttpclient
        results['ipfs_available'] = True
        print("✅ IPFS client available")
    except ImportError:
        print("⚠️  IPFS client not installed. Install with: pip install ipfshttpclient")
    
    # Check firewall ports
    try:
        from r3mes.utils.firewall_check import check_firewall_and_warn
        print("\n🔒 Checking firewall ports...")
        firewall_ok = check_firewall_and_warn()
        results['firewall_warning'] = not firewall_ok
        if firewall_ok:
            print("✅ Firewall ports check passed")
    except ImportError:
        print("⚠️  Firewall check utility not available")
    except Exception as e:
        print(f"⚠️  Firewall check failed: {e}")
    
    return results


def create_wallet_interactive() -> Optional[str]:
    """Create wallet interactively."""
    wallet = WalletManager()
    
    print("\n💼 Wallet Setup")
    print("You can either:")
    print("  1. Create a new wallet (recommended for first-time users)")
    print("  2. Use an existing private key")
    
    choice = input("\nChoose option (1 or 2): ").strip()
    
    if choice == '1':
        # Create new wallet
        wallet_path = wallet.create_wallet()
        address = wallet.get_address(wallet_path)
        print(f"\n✅ Wallet created: {wallet_path}")
        print(f"📍 Address: {address}")
        print("\n⚠️  IMPORTANT: Save your private key securely!")
        print("   Do not share your private key with anyone.")
        # Return wallet path and address for faucet
        return (wallet_path, address)
    elif choice == '2':
        # Use existing private key
        private_key = input("Enter your private key (hex string): ").strip()
        if not private_key:
            print("❌ Private key cannot be empty")
            return None
        
        # Validate private key format
        try:
            bytes.fromhex(private_key)
            # Save to config
            return private_key
        except ValueError:
            print("❌ Invalid private key format (must be hex string)")
            return None
    else:
        print("❌ Invalid choice")
        return None


def configure_blockchain() -> Dict[str, Any]:
    """Configure blockchain connection (including optional TLS/mTLS)."""
    print("\n🔗 Blockchain Configuration")
    
    # Import endpoint checker
    try:
        from r3mes.utils.endpoint_checker import get_best_endpoint, resolve_dns
    except ImportError:
        from utils.endpoint_checker import get_best_endpoint, resolve_dns
    
    # Check available endpoints
    print("Checking available endpoints...")
    best_endpoint = get_best_endpoint("mainnet")
    
    if best_endpoint:
        print(f"✅ Found available endpoints:")
        if "grpc" in best_endpoint:
            print(f"   gRPC: {best_endpoint['grpc']}")
        if "rest" in best_endpoint:
            print(f"   REST: {best_endpoint['rest']}")
    else:
        print("⚠️  No remote endpoints available. Using localhost...")
        best_endpoint = {"grpc": "localhost:9090", "rest": "http://localhost:1317"}
    
    default_url = best_endpoint.get("grpc", "localhost:9090") if best_endpoint else "localhost:9090"
    blockchain_url = input(f"Blockchain gRPC URL [{default_url}]: ").strip()
    if not blockchain_url:
        blockchain_url = default_url
    
    # Check endpoint connectivity
    print(f"\n🔍 Checking endpoint connectivity: {blockchain_url}...")
    host = blockchain_url.split(":")[0]
    dns_resolved = resolve_dns(host)
    if dns_resolved:
        print(f"✅ DNS resolved: {host} -> {dns_resolved}")
    else:
        print(f"⚠️  DNS resolution failed for {host}")
        print("   This endpoint may not be reachable.")
        proceed = input("   Continue anyway? (y/N): ").strip().lower()
        if proceed != 'y':
            print("Setup cancelled.")
            sys.exit(0)
    
    chain_id = input("Chain ID [remes-test]: ").strip()
    if not chain_id:
        chain_id = "remes-test"

    print("\n🔐 TLS / mTLS (optional)")
    print("You can enable TLS mutual authentication for the gRPC connection.")
    print("If you are running a secured remesd node with TLS enabled, you should enable this.")
    enable_tls = input("Enable TLS/mTLS? [y/N]: ").strip().lower() == 'y'

    use_tls = False
    tls_cert_file: Optional[str] = None
    tls_key_file: Optional[str] = None
    tls_ca_file: Optional[str] = None
    tls_server_name: Optional[str] = None

    if enable_tls:
        use_tls = True
        default_client_cert = str(Path.home() / ".r3mes" / "tls" / "client-cert.pem")
        default_client_key = str(Path.home() / ".r3mes" / "tls" / "client-key.pem")
        default_ca_cert = str(Path.home() / ".r3mes" / "tls" / "ca-cert.pem")

        tls_cert_file_input = input(f"Client TLS certificate path [{default_client_cert}]: ").strip()
        tls_key_file_input = input(f"Client TLS private key path [{default_client_key}]: ").strip()
        tls_ca_file_input = input(f"CA certificate path for server verification [{default_ca_cert}]: ").strip()
        tls_server_name_input = input("Expected TLS server name [auto-detect from URL]: ").strip()

        tls_cert_file = tls_cert_file_input or default_client_cert
        tls_key_file = tls_key_file_input or default_client_key
        tls_ca_file = tls_ca_file_input or default_ca_cert
        tls_server_name = tls_server_name_input or None
        
        # Check if TLS certificate files exist
        missing_files = []
        if not os.path.exists(tls_cert_file):
            missing_files.append(f"Client cert: {tls_cert_file}")
        if not os.path.exists(tls_key_file):
            missing_files.append(f"Client key: {tls_key_file}")
        if not os.path.exists(tls_ca_file):
            missing_files.append(f"CA cert: {tls_ca_file}")
        
        if missing_files:
            print("\n⚠️  WARNING: TLS certificate files not found:")
            for f in missing_files:
                print(f"   - {f}")
            print("\nTLS will be disabled. You can:")
            print("  1. Generate TLS certificates (see documentation)")
            print("  2. Disable TLS and use plain gRPC connection")
            print("  3. Continue anyway (miner will fail to start with TLS enabled)")
            
            choice = input("\nDisable TLS and continue? [Y/n]: ").strip().lower()
            if choice != 'n':
                use_tls = False
                tls_cert_file = None
                tls_key_file = None
                tls_ca_file = None
                tls_server_name = None
                print("✅ TLS disabled. Using plain gRPC connection.")
            else:
                print("⚠️  Continuing with TLS enabled. Make sure certificate files exist before starting miner.")
    
    return {
        'blockchain_url': blockchain_url,
        'chain_id': chain_id,
        'use_tls': use_tls,
        'tls_cert_file': tls_cert_file,
        'tls_key_file': tls_key_file,
        'tls_ca_file': tls_ca_file,
        'tls_server_name': tls_server_name,
    }


def configure_mining_params() -> Dict[str, Any]:
    """Configure mining parameters."""
    print("\n⚙️  Mining Parameters")
    
    model_size_input = input("Model hidden size [768]: ").strip()
    model_size = int(model_size_input) if model_size_input else 768
    
    lora_rank_input = input("LoRA rank [8]: ").strip()
    lora_rank = int(lora_rank_input) if lora_rank_input else 8
    
    gradient_accumulation_input = input("Gradient accumulation steps [4]: ").strip()
    gradient_accumulation_steps = int(gradient_accumulation_input) if gradient_accumulation_input else 4
    
    top_k_input = input("Top-k compression ratio (0.0-1.0) [0.1]: ").strip()
    top_k_compression = float(top_k_input) if top_k_input else 0.1
    
    max_iterations_input = input("Max training iterations per session [5]: ").strip()
    max_iterations = int(max_iterations_input) if max_iterations_input else 5
    
    return {
        'model_size': model_size,
        'lora_rank': lora_rank,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'top_k_compression': top_k_compression,
        'max_iterations': max_iterations,
    }


def test_connection(blockchain_url: str, chain_id: str) -> bool:
    """Test blockchain connection."""
    print("\n🔌 Testing Connection...")
    
    try:
        import grpc
        
        # Try to create a client (this will test the connection)
        # For now, just check if we can create the channel
        channel = grpc.insecure_channel(blockchain_url)
        
        # Try to wait for connection (with timeout)
        try:
            grpc.channel_ready_future(channel).result(timeout=5)
            print(f"✅ Successfully connected to {blockchain_url}")
            channel.close()
            return True
        except grpc.FutureTimeoutError:
            print(f"⚠️  Connection timeout to {blockchain_url}")
            print("   Make sure the blockchain node is running.")
            channel.close()
            return False
        except Exception as e:
            print(f"⚠️  Connection test failed: {e}")
            channel.close()
            return False
    except ImportError as e:
        print(f"⚠️  Connection test failed: Missing dependency ({e})")
        print("   Install with: pip install grpcio")
        return False
    except Exception as e:
        print(f"⚠️  Connection test failed: {e}")
        return False


def run_setup_wizard():
    """Run the complete setup wizard."""
    print("=" * 50)
    print("🎯 R3MES Miner Setup Wizard")
    print("=" * 50)
    print("\nThis wizard will guide you through the setup process.")
    print("You can press Ctrl+C at any time to cancel.")
    
    total_steps = 5
    
    # Step 1: System Requirements
    print_step(1, total_steps, "Checking System Requirements")
    requirements = check_system_requirements()
    
    if not requirements['pytorch_installed']:
        print("\n❌ PyTorch is required. Please install it first:")
        print("   pip install torch")
        sys.exit(1)
    
    # Step 2: Wallet Creation
    print_step(2, total_steps, "Creating Wallet")
    wallet_info = create_wallet_interactive()
    if not wallet_info:
        print("❌ Wallet setup failed")
        sys.exit(1)
    
    # Get wallet address for faucet
    wallet_address = None
    wallet_path = None
    
    if isinstance(wallet_info, tuple):
        # New wallet created (returns tuple: (path, address))
        wallet_path, wallet_address = wallet_info
    elif isinstance(wallet_info, str):
        if os.path.exists(wallet_info):
            # Wallet file path
            wallet = WalletManager()
            wallet_path = wallet_info
            wallet_address = wallet.get_address(wallet_info)
        else:
            # Private key directly (string but not a file path)
            print("⚠️  Cannot get address from private key directly. Skipping faucet.")
    else:
        print("⚠️  Cannot get address from wallet info. Skipping faucet.")
    
    # Step 2.5: Request Faucet (if address available)
    if wallet_address:
        print("\n🎁 Faucet Check...")
        
        # Check faucet availability
        if check_faucet_availability():
            faucet_result = request_faucet(wallet_address)
            if faucet_result['success']:
                print(f"✅ Welcome Bonus: {faucet_result.get('amount', '0.1')} REMES airdropped for gas fees! (Ready to mine)")
                if faucet_result.get('tx_hash'):
                    print(f"   Transaction: {faucet_result['tx_hash']}")
            else:
                print(f"⚠️  {faucet_result['message']}")
                print("   You can request tokens manually from the faucet later.")
        else:
            print("⚠️  Faucet is currently unavailable.")
            print("   You can request tokens manually from the faucet later.")
    
    # Update wallet_info to be just the path for config saving
    if wallet_path:
        wallet_info = wallet_path
    
    # Step 3: Blockchain Configuration
    print_step(3, total_steps, "Configuring Blockchain Connection")
    blockchain_config = configure_blockchain()
    
    # Step 4: Mining Parameters
    print_step(4, total_steps, "Setting Up Mining Parameters")
    mining_params = configure_mining_params()
    
    # Step 5: Connection Test
    print_step(5, total_steps, "Testing Connection")
    connection_ok = test_connection(
        blockchain_config['blockchain_url'],
        blockchain_config['chain_id']
    )
    
    if not connection_ok:
        print("\n⚠️  Connection test failed, but configuration will be saved.")
        print("   You can test the connection later with 'r3mes-miner status'")
        proceed = input("\nContinue anyway? (y/N): ").strip().lower()
        if proceed != 'y':
            print("Setup cancelled.")
            sys.exit(0)
    
    # Save configuration
    print("\n💾 Saving Configuration...")
    config = ConfigManager()
    
    # Prepare config dict
    config_dict = {
        **blockchain_config,
        **mining_params,
    }
    
    # Handle wallet (either path or private key)
    if isinstance(wallet_info, str) and os.path.exists(wallet_info):
        # Wallet file path
        wallet = WalletManager()
        private_key = wallet.get_private_key(wallet_info)
        config_dict['private_key'] = private_key
        config_dict['wallet_path'] = wallet_info
    else:
        # Private key directly
        config_dict['private_key'] = wallet_info
    
    # Save config
    config_path = config.save_config(config_dict)
    
    print(f"✅ Configuration saved to: {config_path}")
    print("\n🎉 Setup Complete!")
    print("\nNext steps:")
    print("  1. Make sure your blockchain node is running")
    print("  2. Run 'r3mes-miner start' to begin mining")
    print("  3. Run 'r3mes-miner status' to check your configuration")


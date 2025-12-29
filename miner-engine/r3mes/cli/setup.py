#!/usr/bin/env python3
"""
R3MES Setup Wizard

Interactive setup wizard for configuring R3MES nodes.
Handles hardware detection, wallet setup, and configuration.
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# Add parent directory to path
miner_engine_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(miner_engine_dir))


def detect_hardware() -> Dict[str, Any]:
    """Detect available hardware."""
    hardware = {
        "gpu_available": False,
        "gpu_count": 0,
        "gpus": [],
        "total_vram_gb": 0,
        "cpu_cores": os.cpu_count() or 1,
        "recommended_role": "miner",
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            hardware["gpu_available"] = True
            hardware["gpu_count"] = torch.cuda.device_count()
            
            for i in range(hardware["gpu_count"]):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / (1024**3)
                hardware["gpus"].append({
                    "index": i,
                    "name": props.name,
                    "vram_gb": round(vram_gb, 1),
                    "compute_capability": f"{props.major}.{props.minor}",
                })
                hardware["total_vram_gb"] += vram_gb
            
            hardware["total_vram_gb"] = round(hardware["total_vram_gb"], 1)
            
            # Recommend role based on VRAM
            if hardware["total_vram_gb"] >= 24:
                hardware["recommended_role"] = "multi_role"
            elif hardware["total_vram_gb"] >= 16:
                hardware["recommended_role"] = "serving"
            elif hardware["total_vram_gb"] >= 8:
                hardware["recommended_role"] = "miner"
            else:
                hardware["recommended_role"] = "miner"
    except ImportError:
        pass
    
    return hardware


def print_hardware_info(hardware: Dict[str, Any]) -> None:
    """Print detected hardware information."""
    print("\n🔍 Hardware Detection")
    print("=" * 50)
    
    if hardware["gpu_available"]:
        print(f"✅ GPU(s) detected: {hardware['gpu_count']}")
        for gpu in hardware["gpus"]:
            print(f"   GPU {gpu['index']}: {gpu['name']}")
            print(f"      VRAM: {gpu['vram_gb']} GB")
            print(f"      Compute: {gpu['compute_capability']}")
        print(f"   Total VRAM: {hardware['total_vram_gb']} GB")
    else:
        print("⚠️  No GPU detected. Mining will use CPU (slower).")
    
    print(f"   CPU Cores: {hardware['cpu_cores']}")
    print(f"   Recommended Role: {hardware['recommended_role']}")


def select_role(hardware: Dict[str, Any]) -> str:
    """Interactive role selection."""
    print("\n🎯 Role Selection")
    print("=" * 50)
    print("Available roles:")
    print("  1. miner    - Train AI models and earn rewards")
    print("  2. serving  - Serve inference requests")
    print("  3. proposer - Aggregate gradients and propose commits")
    print("  4. multi    - Run multiple roles (requires more resources)")
    print()
    print(f"Recommended for your hardware: {hardware['recommended_role']}")
    print()
    
    while True:
        choice = input("Select role [1-4] (default: 1): ").strip() or "1"
        role_map = {"1": "miner", "2": "serving", "3": "proposer", "4": "multi_role"}
        if choice in role_map:
            return role_map[choice]
        print("Invalid choice. Please enter 1, 2, 3, or 4.")


def setup_wallet() -> Tuple[str, str]:
    """Setup wallet (create new or import existing)."""
    print("\n💼 Wallet Setup")
    print("=" * 50)
    print("Options:")
    print("  1. Create new wallet")
    print("  2. Import existing private key")
    print()
    
    while True:
        choice = input("Select option [1-2] (default: 1): ").strip() or "1"
        
        if choice == "1":
            # Create new wallet
            from r3mes.cli.wallet import WalletManager
            wallet = WalletManager()
            wallet_path = wallet.create_wallet()
            address = wallet.get_address(wallet_path)
            private_key = wallet.get_private_key(wallet_path)
            
            print(f"\n✅ New wallet created!")
            print(f"   Address: {address}")
            print(f"   Wallet file: {wallet_path}")
            print("\n⚠️  IMPORTANT: Back up your wallet file securely!")
            
            return private_key, address
        
        elif choice == "2":
            # Import existing
            private_key = input("Enter private key (hex): ").strip()
            if private_key.startswith("0x"):
                private_key = private_key[2:]
            
            if len(private_key) != 64:
                print("❌ Invalid private key length. Must be 64 hex characters.")
                continue
            
            try:
                from bridge.crypto import private_key_to_address
                address = private_key_to_address(private_key)
                print(f"\n✅ Wallet imported!")
                print(f"   Address: {address}")
                return private_key, address
            except Exception as e:
                print(f"❌ Invalid private key: {e}")
                continue
        
        print("Invalid choice. Please enter 1 or 2.")


def setup_blockchain() -> Dict[str, str]:
    """Setup blockchain connection."""
    print("\n🔗 Blockchain Configuration")
    print("=" * 50)
    
    # gRPC URL
    default_grpc = os.getenv("R3MES_NODE_GRPC_URL", "localhost:9090")
    grpc_url = input(f"gRPC URL [{default_grpc}]: ").strip() or default_grpc
    
    # Chain ID
    default_chain = os.getenv("R3MES_CHAIN_ID", "remes-test")
    chain_id = input(f"Chain ID [{default_chain}]: ").strip() or default_chain
    
    # TLS
    use_tls = input("Enable TLS? [y/N]: ").strip().lower() == "y"
    
    config = {
        "blockchain_url": grpc_url,
        "chain_id": chain_id,
        "use_tls": use_tls,
    }
    
    if use_tls:
        config["tls_cert_file"] = input("TLS cert file (optional): ").strip() or None
        config["tls_key_file"] = input("TLS key file (optional): ").strip() or None
        config["tls_ca_file"] = input("TLS CA file (optional): ").strip() or None
    
    return config


def setup_model() -> Dict[str, Any]:
    """Setup model configuration."""
    print("\n🤖 Model Configuration")
    print("=" * 50)
    print("Model options:")
    print("  1. GGUF (recommended, lower memory)")
    print("  2. PyTorch (HuggingFace)")
    print("  3. Test model (SimpleBitNet)")
    print()
    
    choice = input("Select model type [1-3] (default: 1): ").strip() or "1"
    
    config = {}
    
    if choice == "1":
        config["use_gguf"] = True
        config["use_llama3"] = True
        model_path = input("GGUF model path (optional): ").strip()
        if model_path:
            config["gguf_model_path"] = model_path
    elif choice == "2":
        config["use_gguf"] = False
        config["use_llama3"] = True
        model_name = input("Model name [meta-llama/Meta-Llama-3-8B]: ").strip()
        config["model_name"] = model_name or "meta-llama/Meta-Llama-3-8B"
    else:
        config["use_gguf"] = False
        config["use_llama3"] = False
    
    # LoRA settings
    lora_rank = input("LoRA rank [8]: ").strip()
    config["lora_rank"] = int(lora_rank) if lora_rank else 8
    
    return config


def save_config(config: Dict[str, Any], config_path: Optional[str] = None) -> str:
    """Save configuration to file."""
    if not config_path:
        config_dir = Path.home() / ".r3mes"
        config_dir.mkdir(exist_ok=True)
        config_path = str(config_dir / "config.json")
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path


def run_full_setup() -> None:
    """Run the full interactive setup wizard."""
    print("\n" + "=" * 60)
    print("   R3MES Setup Wizard")
    print("=" * 60)
    
    # 1. Hardware detection
    hardware = detect_hardware()
    print_hardware_info(hardware)
    
    # 2. Role selection
    role = select_role(hardware)
    print(f"\n✅ Selected role: {role}")
    
    # 3. Wallet setup
    private_key, address = setup_wallet()
    
    # 4. Blockchain configuration
    blockchain_config = setup_blockchain()
    
    # 5. Model configuration (for miner/serving roles)
    model_config = {}
    if role in ["miner", "serving", "multi_role"]:
        model_config = setup_model()
    
    # 6. Build final config
    config = {
        "role": role,
        "private_key": private_key,
        "address": address,
        **blockchain_config,
        **model_config,
        "gradient_accumulation_steps": 4,
        "top_k_compression": 0.1,
    }
    
    # 7. Save config
    config_path = save_config(config)
    
    print("\n" + "=" * 60)
    print("   Setup Complete!")
    print("=" * 60)
    print(f"\n✅ Configuration saved to: {config_path}")
    print(f"   Address: {address}")
    print(f"   Role: {role}")
    print()
    print("Next steps:")
    print(f"  r3mes-{role.replace('_role', '')} start")
    print()


def main():
    """Main entry point for r3mes-setup."""
    try:
        run_full_setup()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

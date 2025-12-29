#!/usr/bin/env python3
"""
Generate Genesis Trap Jobs

This script generates trap jobs for the genesis state by:
1. Creating sample datasets
2. Training BitNet model on each dataset
3. Computing gradient hashes
4. Generating trap job records in genesis format

Usage:
    python scripts/generate_genesis_traps.py --output genesis_traps.json --count 50
"""

import argparse
import json
import sys
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Any
import torch
import torch.nn as nn
import numpy as np

# Add miner-engine to path
sys.path.insert(0, str(Path(__file__).parent.parent / "miner-engine"))

try:
    from core.bitlinear import BitLinear
    from core.trainer import LoRATrainer
    from core.deterministic import configure_deterministic_execution
except ImportError as e:
    print(f"Error importing miner-engine modules: {e}")
    print("Make sure miner-engine dependencies are installed.")
    print("Run: cd miner-engine && pip install -e .")
    sys.exit(1)


class SimpleTrapModel(nn.Module):
    """Simple BitNet model for trap job generation."""
    
    def __init__(self, hidden_size=768, num_layers=2, lora_rank=8):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(BitLinear(hidden_size, hidden_size, lora_rank=lora_rank, lora_alpha=16.0, deterministic=True))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(BitLinear(hidden_size, hidden_size, lora_rank=lora_rank, lora_alpha=16.0, deterministic=True))
    
    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


def generate_dataset(seed: int, size: int = 32, hidden_size: int = 768) -> torch.Tensor:
    """Generate a synthetic dataset with given seed for determinism."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate random input data
    inputs = torch.randn(size, hidden_size)
    return inputs


def compute_gradient_hash(gradients: Dict[str, torch.Tensor]) -> str:
    """
    Compute SHA256 hash of gradients for trap job verification.
    
    This matches the hash computation used in the blockchain verification.
    Uses the same method as LoRATrainer.compute_gradient_hash().
    """
    import numpy as np
    
    # Sort gradients by name for deterministic hashing
    sorted_names = sorted(gradients.keys())
    hash_input = b""
    
    for name in sorted_names:
        grad = gradients[name]
        if grad is not None:
            # Convert to numpy for consistent hashing
            grad_np = grad.detach().cpu().numpy()
            # Use fixed precision for deterministic hashing (float32)
            grad_bytes = grad_np.astype(np.float32).tobytes()
            hash_input += name.encode() + grad_bytes
    
    # Compute SHA256 hash
    hash_obj = hashlib.sha256(hash_input)
    gradient_hash = hash_obj.hexdigest()
    
    return gradient_hash


def generate_ipfs_hash(content: bytes, seed: int) -> str:
    """
    Generate a deterministic IPFS-like hash for dataset.
    
    In production, this would upload to IPFS and return the actual CID.
    For genesis generation, we use a deterministic hash based on content + seed.
    """
    # Simulate IPFS CIDv0 (starts with Qm)
    hash_obj = hashlib.sha256(content + str(seed).encode())
    # Take first 34 bytes and convert to base58-like representation (simplified)
    hash_hex = hash_obj.hexdigest()[:44]  # 44 hex chars = 22 bytes, enough for Qm prefix
    return f"Qm{hash_hex}"  # Simplified CID representation


def compute_top_k_fingerprint(gradients: Dict[str, torch.Tensor], top_k: int = 100) -> Dict[str, Any]:
    """
    Compute Top-K fingerprint from gradients for cosine similarity verification.
    
    Matches the Go implementation in gradient_utils.go:ExtractTopKFingerprint.
    Returns JSON-serializable fingerprint with top K indices and values (by absolute value).
    """
    import numpy as np
    
    # Flatten all gradients into a single vector (deterministic order)
    all_values = []
    
    for name in sorted(gradients.keys()):
        grad = gradients[name]
        if grad is not None:
            grad_flat = grad.detach().cpu().numpy().flatten()
            all_values.extend(grad_flat.tolist())
    
    if len(all_values) == 0:
        return {"top_k": top_k, "indices": [], "values": [], "shape": [0, 0]}
    
    # Convert to numpy
    values_arr = np.array(all_values)
    
    # Find top K by absolute value (descending order)
    top_k_actual = min(top_k, len(values_arr))
    abs_values = np.abs(values_arr)
    top_k_indices_abs = np.argsort(abs_values)[-top_k_actual:][::-1]  # Descending order
    
    # Get top K values and their indices
    top_values = values_arr[top_k_indices_abs].tolist()
    top_original_indices = top_k_indices_abs.tolist()
    
    # Shape is 1D array with total size (matching Go implementation)
    total_size = len(all_values)
    
    fingerprint = {
        "top_k": top_k_actual,
        "indices": top_original_indices,  # Positions in the flattened gradient vector
        "values": top_values,  # Values at those positions (same order as indices)
        "shape": [total_size],  # 1D shape (matching Go: []int64{int64(len(gradientTensor))})
    }
    
    return fingerprint


def generate_vault_entry(
    entry_id: int,
    dataset_seed: int,
    model_hidden_size: int = 768,
    lora_rank: int = 8,
    learning_rate: float = 1e-4,
    gpu_architecture: str = "Unknown",
) -> Dict[str, Any]:
    """
    Generate a single GenesisVaultEntry by training on a dataset and computing gradient hash.
    
    Returns a GenesisVaultEntry dictionary ready for genesis.json format.
    """
    # Configure deterministic execution
    configure_deterministic_execution(global_seed=dataset_seed)
    
    # Create model
    model = SimpleTrapModel(
        hidden_size=model_hidden_size,
        num_layers=2,
        lora_rank=lora_rank
    )
    
    # Create trainer
    trainer = LoRATrainer(
        model,
        learning_rate=learning_rate,
        deterministic=True
    )
    
    # Generate dataset
    inputs = generate_dataset(dataset_seed, size=32, hidden_size=model_hidden_size)
    targets = torch.randn(32, model_hidden_size)  # Random targets for training
    
    # Train one step to get gradients
    loss, gradients = trainer.train_step(inputs, targets)
    
    # Compute gradient hash
    gradient_hash = compute_gradient_hash(gradients)
    
    # Generate dataset IPFS hash (simulated)
    dataset_bytes = inputs.numpy().tobytes()
    dataset_ipfs_hash = generate_ipfs_hash(dataset_bytes, dataset_seed)
    
    # Compute Top-K fingerprint for cosine similarity verification
    fingerprint = compute_top_k_fingerprint(gradients, top_k=100)
    
    # Create GenesisVaultEntry record
    vault_entry = {
        "entry_id": entry_id,
        "data_hash": dataset_ipfs_hash,  # IPFS hash of input data (chunk data)
        "expected_gradient_hash": gradient_hash,  # SHA256 hash of expected gradient
        "expected_gradient_ipfs_hash": "",  # IPFS hash of gradient tensor (empty for genesis, can be added later)
        "expected_fingerprint": json.dumps(fingerprint),  # JSON string with Top-K fingerprint
        "gpu_architecture": gpu_architecture,  # GPU architecture that solved this correctly
        "created_height": 0,  # Genesis
        "usage_count": 0,  # Not used yet
        "last_used_height": 0,  # Not used yet
        "encrypted": False,  # Not encrypted by default
        # Additional metadata for generation
        "dataset_seed": dataset_seed,
        "model_hidden_size": model_hidden_size,
        "lora_rank": lora_rank,
        "training_loss": float(loss),
    }
    
    return vault_entry


def generate_genesis_vault_entries(count: int = 50, output_path: str = "genesis_vault_entries.json") -> None:
    """
    Generate multiple GenesisVaultEntry records for genesis state.
    
    Args:
        count: Number of vault entries to generate
        output_path: Path to output JSON file
    """
    print(f"Generating {count} genesis vault entries...")
    print("=" * 60)
    
    vault_entries = []
    
    # Detect GPU architecture if available
    gpu_architecture = "Unknown"
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            if "A100" in gpu_name or "H100" in gpu_name:
                gpu_architecture = "Ampere"
            elif "RTX 30" in gpu_name or "RTX 40" in gpu_name:
                gpu_architecture = "Ada"
            elif "V100" in gpu_name:
                gpu_architecture = "Volta"
            else:
                gpu_architecture = "Unknown"
        except Exception:
            pass
    
    for i in range(count):
        entry_id = i + 1
        dataset_seed = 1000 + i  # Deterministic seeds starting from 1000
        
        print(f"[{i+1}/{count}] Generating entry {entry_id}...", end=" ", flush=True)
        
        try:
            vault_entry = generate_vault_entry(
                entry_id=entry_id,
                dataset_seed=dataset_seed,
                gpu_architecture=gpu_architecture,
            )
            vault_entries.append(vault_entry)
            print(f"✅ (hash: {vault_entry['expected_gradient_hash'][:16]}...)")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Write to JSON file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            "genesis_vault_entries": vault_entries,
            "count": len(vault_entries),
            "gpu_architecture": gpu_architecture,
            "generated_at": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"),
        }, f, indent=2)
    
    print("=" * 60)
    print(f"✅ Generated {len(vault_entries)} genesis vault entries")
    print(f"📄 Saved to: {output_file.absolute()}")
    print(f"\nNext steps:")
    print(f"1. Review the generated vault entries: {output_file}")
    print(f"2. Run validation: python scripts/validate_genesis_traps.py {output_file}")
    print(f"3. Integrate into genesis.json (see remes/x/remes/keeper/genesis.go)")
    print(f"   Add 'genesis_vault_list' field to GenesisState in genesis.proto")


def main():
    parser = argparse.ArgumentParser(description="Generate genesis vault entries (trap jobs)")
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Number of vault entries to generate (default: 50)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="genesis_vault_entries.json",
        help="Output JSON file path (default: genesis_vault_entries.json)"
    )
    parser.add_argument(
        "--model-hidden-size",
        type=int,
        default=768,
        help="Model hidden size (default: 768)"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank (default: 8)"
    )
    
    args = parser.parse_args()
    
    try:
        generate_genesis_vault_entries(
            count=args.count,
            output_path=args.output,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


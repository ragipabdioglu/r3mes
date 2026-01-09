#!/usr/bin/env python3
"""
Create Base Model Checkpoint for R3MES

This script creates a base model checkpoint that can be uploaded to IPFS
and shared across miners for consistent starting point.
"""

import torch
import sys
from pathlib import Path

# Add to path
miner_engine_dir = Path(__file__).parent
sys.path.insert(0, str(miner_engine_dir))

try:
    from r3mes.miner.engine import SimpleBitNetModel
except ImportError:
    # Fallback for development
    sys.path.insert(0, str(miner_engine_dir.parent))
    from miner_engine import SimpleBitNetModel


def create_base_model(
    hidden_size: int = 768,
    num_layers: int = 2,
    lora_rank: int = 8,
    output_path: str = None
):
    """
    Create a base model checkpoint.
    
    Args:
        hidden_size: Model hidden size (default: 768)
        num_layers: Number of layers (default: 2)
        lora_rank: LoRA rank (default: 8)
        output_path: Output file path (default: base_model.pt in project root)
    """
    print("=" * 60)
    print("R3MES Base Model Checkpoint Creator")
    print("=" * 60)
    print(f"Hidden Size: {hidden_size}")
    print(f"Number of Layers: {num_layers}")
    print(f"LoRA Rank: {lora_rank}")
    print("-" * 60)
    
    # Create model
    print("Creating model...")
    model = SimpleBitNetModel(
        hidden_size=hidden_size,
        num_layers=num_layers,
        lora_rank=lora_rank,
    )
    
    # Create checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'lora_rank': lora_rank,
        'model_type': 'SimpleBitNetModel',
        'version': '1.0',
    }
    
    # Determine output path
    if output_path is None:
        project_root = Path(__file__).parent.parent
        output_path = project_root / 'base_model.pt'
    else:
        output_path = Path(output_path)
    
    # Save checkpoint
    print(f"Saving checkpoint to: {output_path}")
    torch.save(checkpoint, output_path)
    
    # Print info
    file_size_mb = output_path.stat().st_size / 1024 / 1024
    print("-" * 60)
    print(f"✅ Base model checkpoint oluşturuldu!")
    print(f"📦 Dosya: {output_path}")
    print(f"📊 Boyut: {file_size_mb:.2f} MB")
    print("-" * 60)
    print("\n💡 Sonraki adımlar:")
    print(f"1. Control Panel'de 'Select File' → {output_path} seç")
    print("2. 'Upload to IPFS' butonuna tıkla")
    print("3. CID'yi kaydet ve 'Create Task' için kullan")
    print("=" * 60)
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create R3MES base model checkpoint")
    parser.add_argument("--hidden-size", type=int, default=768, help="Model hidden size")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    
    args = parser.parse_args()
    
    create_base_model(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        lora_rank=args.lora_rank,
        output_path=args.output
    )


#!/usr/bin/env python3
"""
Simple Model Inference Test for R3MES

Tests a trained model by running inference on sample input.
This is a basic test - not a full chat interface (serving infrastructure not yet complete).
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add miner-engine directory to path
miner_engine_dir = Path(__file__).parent
sys.path.insert(0, str(miner_engine_dir))

from r3mes.miner.engine import SimpleBitNetModel
from core.bitlinear import BitLinear


def test_model_inference(
    model_path: str = None,
    hidden_size: int = 768,
    num_layers: int = 2,
    lora_rank: int = 8,
    input_text: str = "test input"
):
    """
    Test model inference with sample input.
    
    Args:
        model_path: Path to saved model checkpoint (optional)
        hidden_size: Model hidden size
        num_layers: Number of layers
        lora_rank: LoRA rank
        input_text: Input text for testing (will be converted to embeddings)
    
    Returns:
        Output tensor and statistics
    """
    print("=" * 60)
    print("R3MES Model Inference Test")
    print("=" * 60)
    
    # Create or load model
    if model_path and Path(model_path).exists():
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        model = SimpleBitNetModel(
            hidden_size=checkpoint.get('hidden_size', hidden_size),
            num_layers=checkpoint.get('num_layers', num_layers),
            lora_rank=checkpoint.get('lora_rank', lora_rank),
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Model loaded successfully")
    else:
        print("Creating new model (no checkpoint found)")
        model = SimpleBitNetModel(
            hidden_size=hidden_size,
            num_layers=num_layers,
            lora_rank=lora_rank,
        )
        print("✅ Model created")
    
    # Set to evaluation mode
    model.eval()
    
    # Create sample input (simple embedding-like tensor)
    # In a real scenario, this would be tokenized text
    batch_size = 1
    seq_length = 10  # Simple sequence length
    input_tensor = torch.randn(batch_size, seq_length, hidden_size)
    
    print(f"\nInput shape: {input_tensor.shape}")
    print(f"Input text (for reference): {input_text}")
    
    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"✅ Inference complete")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.6f}")
    print(f"Output std: {output.std().item():.6f}")
    
    # Get LoRA statistics
    lora_layers = [m for m in model.modules() if isinstance(m, BitLinear)]
    print(f"\nLoRA Layers: {len(lora_layers)}")
    for i, layer in enumerate(lora_layers):
        print(f"  Layer {i+1}: rank={layer.lora_rank}, alpha={layer.lora_alpha}")
    
    return {
        'output': output,
        'output_shape': output.shape,
        'output_mean': output.mean().item(),
        'output_std': output.std().item(),
        'lora_layers': len(lora_layers),
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test R3MES model inference")
    parser.add_argument("--model-path", type=str, help="Path to model checkpoint")
    parser.add_argument("--hidden-size", type=int, default=768, help="Model hidden size")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--input-text", type=str, default="test input", help="Input text for testing")
    
    args = parser.parse_args()
    
    result = test_model_inference(
        model_path=args.model_path,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        lora_rank=args.lora_rank,
        input_text=args.input_text,
    )
    
    print("\n" + "=" * 60)
    print("✅ Test completed successfully")
    print("=" * 60)


#!/usr/bin/env python3
"""
Basit miner engine test script'i

Bu script miner engine'in temel fonksiyonlarını test eder:
- BitLinear layer
- LoRA training
- Gradient computation
- Serialization
- IPFS upload
"""

import torch
import torch.nn as nn
from core.bitlinear import BitLinear
from core.trainer import LoRATrainer
from core.serialization import LoRASerializer
from utils.ipfs_client import IPFSClient
from utils.gpu_detection import GPUArchitectureDetector

def main():
    print("=" * 60)
    print("R3MES Miner Engine Test")
    print("=" * 60)
    
    # 1. GPU Detection
    print("\n1. GPU Detection")
    print("-" * 60)
    detector = GPUArchitectureDetector()
    print(f"GPU Architecture: {detector.get_architecture()}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {detector.get_device_name()}")
        print(f"Compute Capability: {detector.get_compute_capability()}")
    metadata = detector.get_metadata()
    print(f"Metadata: {metadata}")
    
    # 2. Model Creation
    print("\n2. Model Creation")
    print("-" * 60)
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = BitLinear(768, 768, lora_rank=8, deterministic=True)
            self.layer2 = BitLinear(768, 384, lora_rank=8, deterministic=True)
        
        def forward(self, x):
            x = self.layer1(x)
            x = torch.relu(x)
            x = self.layer2(x)
            return x
    
    model = SimpleModel()
    total_lora_size = sum(
        l.estimate_size_mb() 
        for l in model.modules() 
        if isinstance(l, BitLinear)
    )
    print(f"Model created with {sum(1 for _ in model.modules() if isinstance(_, BitLinear))} BitLinear layers")
    print(f"Total LoRA size: {total_lora_size:.4f} MB")
    
    # 3. Training Setup
    print("\n3. Training Setup")
    print("-" * 60)
    trainer = LoRATrainer(model, learning_rate=1e-4, deterministic=True)
    trainer.set_seed(42)
    print(f"Trainer created with {len(trainer.lora_params)} LoRA parameters")
    print(f"Learning rate: {trainer.learning_rate}")
    
    # 4. Training Step
    print("\n4. Training Step")
    print("-" * 60)
    # Inputs will be moved to device by trainer, but create on CPU first
    inputs = torch.randn(32, 768)
    targets = torch.randn(32, 384)
    
    loss, gradients = trainer.train_step(inputs, targets)
    print(f"Training loss: {loss:.6f}")
    print(f"Gradients computed for {len(gradients)} parameters")
    
    # 5. Gradient Hash
    print("\n5. Gradient Hash Computation")
    print("-" * 60)
    grad_hash = trainer.compute_gradient_hash(gradients)
    print(f"Gradient hash: {grad_hash}")
    print(f"Hash length: {len(grad_hash)} characters")
    
    # 6. LoRA State Serialization
    print("\n6. LoRA State Serialization")
    print("-" * 60)
    lora_state = trainer.get_lora_state_dict()
    serializer = LoRASerializer()
    serialized = serializer.serialize_lora_state(lora_state)
    serialized_size_mb = len(serialized) / (1024 * 1024)
    print(f"Serialized LoRA size: {serialized_size_mb:.4f} MB")
    print(f"Bandwidth reduction vs 28GB: {((28*1024 - serialized_size_mb) / (28*1024)) * 100:.2f}%")
    
    # 7. IPFS Upload Test
    print("\n7. IPFS Upload Test")
    print("-" * 60)
    ipfs_client = IPFSClient()
    if ipfs_client.is_connected():
        print("✅ IPFS daemon is connected")
        ipfs_hash = ipfs_client.upload_lora_state(serialized)
        print(f"IPFS hash: {ipfs_hash}")
        
        # Verify content
        verified = ipfs_client.verify_content(ipfs_hash, serialized)
        print(f"IPFS verification: {'✅ Verified' if verified else '❌ Failed'}")
        
        # Pin content
        pinned = ipfs_client.pin_content(ipfs_hash)
        print(f"Content pinned: {'✅ Yes' if pinned else '❌ No'}")
    else:
        print("⚠️  IPFS daemon is not connected (using simulated mode)")
        ipfs_hash = ipfs_client.upload_lora_state(serialized)
        print(f"Simulated IPFS hash: {ipfs_hash}")
        print("To enable real IPFS:")
        print("  1. Install IPFS: https://docs.ipfs.tech/install/")
        print("  2. Initialize: ipfs init")
        print("  3. Start daemon: ipfs daemon")
    
    # 8. Training Metadata
    print("\n8. Training Metadata")
    print("-" * 60)
    training_meta = trainer.get_training_metadata()
    for key, value in training_meta.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✅ Miner engine test completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Start blockchain: remesd start")
    print("2. Create wallet: remesd keys add miner1")
    print("3. Fund wallet and submit gradient to blockchain")
    print(f"4. Use IPFS hash: {ipfs_hash}")

if __name__ == "__main__":
    main()


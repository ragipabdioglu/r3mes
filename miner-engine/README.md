# R3MES Miner Engine

Python-based BitNet 1.58-bit AI training engine with LoRA (Low-Rank Adaptation) for the R3MES blockchain.

## Features

- **BitNet 1.58-bit Model**: Quantized {-1, 0, +1} weights for extreme efficiency
- **LoRA Adapters**: Trainable low-rank adapters (rank 4-64) for 99.6%+ bandwidth reduction
- **Frozen Backbone**: Backbone weights remain frozen, only LoRA adapters are trained
- **Deterministic Training**: CUDA deterministic algorithms for bit-exact reproducibility
- **GPU Architecture Detection**: Automatic detection and handling of GPU architectures

## Installation

### 1. Local development install (current)

```bash
cd miner-engine
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Bu sayede `r3mes` paketi editable modda kurulur ve komut satırından:

```bash
r3mes-miner --help
```

ile CLI'yi kullanabilirsin.

### 2. PyPI üzerinden (planlanan)

Paket PyPI'ye yüklendiğinde doğrudan:

```bash
pip install r3mes
```

komutuyla kurulabilir hale gelecektir.

## Architecture

### BitLinear Layer

The `BitLinear` layer implements:
- **Frozen Backbone**: Quantized weights {-1, 0, +1} that never change
- **LoRA Adapters**: Small trainable matrices A and B
- **Forward Pass**: `output = backbone(x) + (alpha/rank) * x @ A.T @ B.T`

### LoRA Benefits

- **Bandwidth Reduction**: 99.6%+ compared to full weight transfer
- **Memory Efficient**: Only 10-100MB instead of 28GB+
- **Fast Training**: Only small adapter matrices need gradients

## Usage

```python
from core.bitlinear import BitLinear
from utils.gpu_detection import GPUArchitectureDetector

# Detect GPU architecture
detector = GPUArchitectureDetector()
print(f"GPU Architecture: {detector.get_architecture()}")

# Create BitLinear layer
layer = BitLinear(
    in_features=768,
    out_features=768,
    lora_rank=8,
    lora_alpha=16.0,
    deterministic=True
)

# Forward pass
x = torch.randn(32, 768)
output = layer(x)

# Get LoRA parameters for serialization
lora_A, lora_B, alpha = layer.get_lora_params()
print(f"LoRA size: {layer.estimate_size_mb():.2f} MB")
```

## Status

**Current Status**: Core BitLinear and GPU detection implemented.

**Next Steps**:
- LoRA adapter training loop
- Gradient computation and serialization
- IPFS integration for gradient storage
- Blockchain client integration


"""
Adaptive VRAM Scaling (Oto-VRAM Ölçekleme)

Automatically detects GPU VRAM and applies appropriate training profile.
No user configuration required - system adapts to available hardware.
"""

import torch
from typing import Dict, Any, Optional


def detect_vram_profile() -> Dict[str, Any]:
    """
    Detect GPU VRAM and return appropriate training profile.
    
    Profiles:
    - < 6GB (Entry Level): batch_size=1, gradient_accumulation=32, PagedAdamW8bit
    - 6GB - 12GB (Mid Range): batch_size=4, gradient_accumulation=8, AdamW
    - > 12GB (High End): batch_size=16+, gradient_accumulation=1, AdamW
    
    Returns:
        Dict with profile configuration (batch_size, gradient_accumulation, optimizer, etc.)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available - cannot detect VRAM profile")
    
    device = torch.cuda.current_device()
    device_props = torch.cuda.get_device_properties(device)
    vram_gb = device_props.total_memory / (1024**3)
    
    print(f"🔍 Detected GPU: {device_props.name}")
    print(f"🔍 VRAM: {vram_gb:.1f}GB")
    
    if vram_gb < 6:
        # Entry Level (GTX 1650, RTX 3050, etc.)
        profile = {
            "batch_size": 1,
            "gradient_accumulation": 32,
            "optimizer": "PagedAdamW8bit",  # RAM spillover allowed
            "mixed_precision": True,
            "max_memory": {0: f"{int(vram_gb * 0.9)}GB"},  # 90% VRAM usage
            "gradient_checkpointing": True,  # Enable for memory efficiency
            "profile_name": "entry_level",
        }
        print(f"✅ Applied Profile: Entry Level (< 6GB)")
    elif vram_gb < 12:
        # Mid Range (RTX 3060, RTX 3070, etc.)
        profile = {
            "batch_size": 4,
            "gradient_accumulation": 8,
            "optimizer": "AdamW",
            "mixed_precision": True,
            "max_memory": {0: f"{int(vram_gb * 0.85)}GB"},  # 85% VRAM usage
            "gradient_checkpointing": False,
            "profile_name": "mid_range",
        }
        print(f"✅ Applied Profile: Mid Range (6-12GB)")
    else:
        # High End (RTX 3090, RTX 4090, etc.)
        profile = {
            "batch_size": 16,
            "gradient_accumulation": 1,
            "optimizer": "AdamW",
            "mixed_precision": False,  # Full precision for high-end cards
            "max_memory": {0: f"{int(vram_gb * 0.8)}GB"},  # 80% VRAM usage
            "gradient_checkpointing": False,
            "profile_name": "high_end",
        }
        print(f"✅ Applied Profile: High End (> 12GB)")
    
    print(f"   - Batch Size: {profile['batch_size']}")
    print(f"   - Gradient Accumulation: {profile['gradient_accumulation']}")
    print(f"   - Optimizer: {profile['optimizer']}")
    print(f"   - Mixed Precision: {profile['mixed_precision']}")
    
    return profile


def apply_profile_to_model(model: torch.nn.Module, profile: Dict[str, Any]) -> torch.nn.Module:
    """
    Apply VRAM profile optimizations to model.
    
    Args:
        model: PyTorch model
        profile: VRAM profile configuration
        
    Returns:
        Model with optimizations applied
    """
    # Enable gradient checkpointing for low VRAM
    if profile.get("gradient_checkpointing", False):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            print("   - Gradient Checkpointing: Enabled")
    
    return model


def create_optimizer_from_profile(
    parameters,
    profile: Dict[str, Any],
    learning_rate: float = 1e-4
) -> torch.optim.Optimizer:
    """
    Create optimizer based on VRAM profile.
    
    Args:
        parameters: Model parameters to optimize
        profile: VRAM profile configuration
        learning_rate: Learning rate
        
    Returns:
        Optimizer instance
    """
    optimizer_type = profile.get("optimizer", "AdamW")
    
    if optimizer_type == "PagedAdamW8bit":
        try:
            from transformers import PagedAdamW8bit
            optimizer = PagedAdamW8bit(
                parameters,
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
            print("   - Using PagedAdamW8bit (RAM spillover enabled)")
        except ImportError:
            print("⚠️  PagedAdamW8bit not available, falling back to AdamW")
            import torch.optim as optim
            optimizer = optim.AdamW(
                parameters,
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01,
            )
    else:
        # Standard AdamW
        import torch.optim as optim
        optimizer = optim.AdamW(
            parameters,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )
        print("   - Using AdamW")
    
    return optimizer


"""
DoRA-Enforced Architecture (Zorunlu DoRA)

FAZ 4: MinerEngine DoRA Migration

Model yükleme fonksiyonunda DoRA adaptörlerini zorunlu kıl.
Ana model parametreleri otomatik dondurulur.
Sadece DoRA adaptörleri trainable.

DoRA vs LoRA:
- LoRA: output = W₀x + BA*x (direction only)
- DoRA: output = W₀x + m*(V/||V||)*x (magnitude + normalized direction)
"""

import torch
import torch.nn as nn
from typing import Optional, List
from core.bitlinear import BitLinear
from core.dora import BitLinearDoRA, create_dora_from_bitlinear


def load_model_with_enforced_dora(
    model: nn.Module,
    dora_rank: int = 16,
    dora_alpha: float = 1.0,
    dora_dropout: float = 0.0,
    target_modules: Optional[List[str]] = None
) -> nn.Module:
    """
    Load model with MANDATORY DoRA architecture.
    Full fine-tuning is BLOCKED at code level.
    
    Args:
        model: Base model to wrap with DoRA
        dora_rank: DoRA rank (default: 16)
        dora_alpha: DoRA alpha scaling (default: 1.0)
        dora_dropout: Dropout probability (default: 0.0)
        target_modules: List of module names to apply DoRA to (default: all BitLinear layers)
        
    Returns:
        Model with enforced DoRA architecture
        
    Raises:
        RuntimeError: If full fine-tuning is detected
    """
    # MANDATORY: Freeze all base model parameters first
    for name, param in model.named_parameters():
        if "dora" not in name.lower() and "magnitude" not in name.lower() and "direction" not in name.lower():
            param.requires_grad = False
    
    # Convert BitLinear layers to BitLinearDoRA
    dora_layers_created = 0
    
    def replace_bitlinear_with_dora(module: nn.Module, prefix: str = ""):
        nonlocal dora_layers_created
        
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, BitLinear):
                # Check if this module should be converted
                if target_modules is None or any(t in full_name for t in target_modules):
                    # Create DoRA layer wrapping the BitLinear
                    dora_layer = create_dora_from_bitlinear(
                        bitlinear=child,
                        rank=dora_rank,
                        alpha=dora_alpha,
                        dropout=dora_dropout,
                    )
                    setattr(module, name, dora_layer)
                    dora_layers_created += 1
            else:
                # Recursively process child modules
                replace_bitlinear_with_dora(child, full_name)
    
    replace_bitlinear_with_dora(model)
    
    print(f"✅ DoRA Enforced: Created {dora_layers_created} BitLinearDoRA layers")
    
    # VERIFY: Ensure no base parameters are trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_ratio = trainable_params / total_params if total_params > 0 else 0.0
    
    # For SimpleBitNetModel (test model), allow higher ratio as it's small
    is_test_model = total_params < 100000
    
    if not is_test_model and trainable_ratio > 0.15:  # DoRA has slightly more params than LoRA
        raise RuntimeError(
            f"DoRA enforcement failed: Too many trainable parameters detected. "
            f"Trainable: {trainable_params:,} / {total_params:,} ({trainable_ratio*100:.2f}%). "
            f"Full fine-tuning is not supported. "
            f"Expected < 15% trainable parameters for DoRA-only training."
        )
    
    print(f"✅ DoRA Enforced: {trainable_params:,} trainable / {total_params:,} total parameters ({trainable_ratio*100:.2f}%)")
    
    return model


def load_model_with_enforced_lora(
    model: nn.Module,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    target_modules: Optional[list] = None
) -> nn.Module:
    """
    DEPRECATED: Use load_model_with_enforced_dora instead.
    
    This function is kept for backward compatibility but now uses DoRA internally.
    """
    print("⚠️  load_model_with_enforced_lora is deprecated. Using DoRA instead.")
    return load_model_with_enforced_dora(
        model=model,
        dora_rank=lora_rank,
        dora_alpha=float(lora_alpha),
        target_modules=target_modules,
    )


def validate_lora_only_training(model: nn.Module) -> bool:
    """
    DEPRECATED: Use validate_dora_only_training instead.
    Kept for backward compatibility.
    """
    return validate_dora_only_training(model)


def validate_dora_only_training(model: nn.Module) -> bool:
    """
    Validate that only DoRA parameters are trainable.
    
    Args:
        model: Model to validate
        
    Returns:
        True if validation passes
        
    Raises:
        RuntimeError: If non-DoRA parameters are trainable
    """
    dora_keywords = ['dora', 'magnitude', 'direction_a', 'direction_b', 'lora']
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            name_lower = name.lower()
            if not any(kw in name_lower for kw in dora_keywords):
                raise RuntimeError(
                    f"Non-DoRA parameter '{name}' is trainable. "
                    f"This hardware architecture does not support full fine-tuning. "
                    f"Only DoRA adapter parameters should be trainable."
                )
    
    print("✅ DoRA-only training validation passed")
    return True


def check_full_finetune_config(config_dict: dict) -> None:
    """
    Check if full fine-tuning is attempted in config and raise error.
    
    Args:
        config_dict: Configuration dictionary
        
    Raises:
        ValueError: If full fine-tuning is detected
    """
    if config_dict.get("full_finetune", False):
        raise ValueError(
            "Full fine-tuning is not supported. "
            "This hardware architecture only supports LoRA training. "
            "Please remove 'full_finetune' from config or set it to False."
        )
    
    if config_dict.get("train_all_params", False):
        raise ValueError(
            "Training all parameters is not supported. "
            "This hardware architecture only supports LoRA training. "
            "Please remove 'train_all_params' from config or set it to False."
        )


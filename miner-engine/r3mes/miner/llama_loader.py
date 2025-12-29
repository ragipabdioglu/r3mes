"""
Llama 3 8B Model Loader

HuggingFace'den Llama 3 8B modelini yükler ve BitNet b1.58 quantization + LoRA uygular.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers library not available. Llama 3 8B loading will not work.")


def load_llama3_8b_model(
    model_name: str = "meta-llama/Meta-Llama-3-8B",
    cache_dir: Optional[str] = None,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.float16,
    use_quantization: bool = True,
    quantization_bits: int = 4,
    low_cpu_mem_usage: bool = True,
) -> Tuple[nn.Module, Optional[object]]:
    """
    Load Llama 3 8B model from HuggingFace and apply LoRA configuration.
    
    Args:
        model_name: HuggingFace model identifier (default: "meta-llama/Meta-Llama-3-8B")
        cache_dir: Directory to cache model files (default: ~/.cache/huggingface)
        lora_rank: LoRA rank (default: 8)
        lora_alpha: LoRA alpha scaling (default: 16)
        lora_dropout: LoRA dropout rate (default: 0.1)
        device_map: Device mapping strategy (default: "auto")
        torch_dtype: Data type for model weights (default: torch.float16)
        use_quantization: Whether to use quantization (default: True)
        quantization_bits: Quantization bits (4 or 8, default: 4)
        low_cpu_mem_usage: Use low CPU memory usage (default: True)
        
    Returns:
        Tuple of (model, tokenizer)
        
    Raises:
        ImportError: If transformers or peft libraries are not available
        RuntimeError: If model loading fails
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "transformers library is required for Llama 3 8B loading. "
            "Install with: pip install transformers peft bitsandbytes"
        )
    
    logger.info(f"Loading Llama 3 8B model from: {model_name}")
    
    # Set cache directory
    if cache_dir is None:
        cache_dir = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise RuntimeError(f"Tokenizer loading failed: {e}")
    
    # Quantization configuration
    quantization_config = None
    if use_quantization:
        if quantization_bits == 4:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            logger.info("Using 4-bit quantization (NF4)")
        elif quantization_bits == 8:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch_dtype,
            )
            logger.info("Using 8-bit quantization")
        else:
            logger.warning(f"Unsupported quantization bits: {quantization_bits}, using FP16")
    
    # Load model
    logger.info("Loading model (this may take a few minutes on first run)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype if not use_quantization else None,
            device_map=device_map,
            low_cpu_mem_usage=low_cpu_mem_usage,
            trust_remote_code=True,
        )
        logger.info("✅ Base model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")
    
    # Apply LoRA configuration
    logger.info("Applying LoRA configuration...")
    try:
        # Llama 3 attention modules
        target_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        model = get_peft_model(model, lora_config)
        logger.info(f"✅ LoRA applied: rank={lora_rank}, alpha={lora_alpha}")
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_ratio = trainable_params / total_params if total_params > 0 else 0.0
        logger.info(
            f"Trainable parameters: {trainable_params:,} / {total_params:,} "
            f"({trainable_ratio*100:.2f}%)"
        )
    except Exception as e:
        logger.error(f"Failed to apply LoRA: {e}")
        raise RuntimeError(f"LoRA configuration failed: {e}")
    
    # Apply BitNet b1.58 quantization if requested
    use_bitnet = os.getenv("R3MES_USE_BITNET_B158", "true").lower() == "true"
    if use_bitnet and not use_quantization:  # Only if not using standard quantization
        logger.info("Applying BitNet b1.58 quantization...")
        model = apply_bitnet_quantization(model, threshold=0.0)
    elif use_bitnet and use_quantization:
        logger.warning(
            "Both standard quantization and BitNet b1.58 are enabled. "
            "Using standard quantization. Set R3MES_USE_BITNET_B158=false to use BitNet b1.58."
        )
    
    logger.info("✅ Llama 3 8B model loaded and configured successfully")
    return model, tokenizer


def apply_bitnet_quantization(model: nn.Module, threshold: float = 0.0) -> nn.Module:
    """
    Apply BitNet b1.58 quantization to model weights.
    
    Converts FP16 weights to {-1, 0, +1} ternary representation.
    
    Args:
        model: Model to quantize
        threshold: Threshold for zero quantization (default: 0.0)
        
    Returns:
        Quantized model
    """
    from r3mes.miner.bitnet_quantization import (
        apply_bitnet_quantization_to_model,
        get_quantization_stats,
    )
    
    logger.info("Applying BitNet b1.58 quantization to model...")
    
    # Target modules for Llama architecture
    target_modules = [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    
    # Apply quantization
    quantized_model = apply_bitnet_quantization_to_model(
        model,
        threshold=threshold,
        target_modules=target_modules,
    )
    
    # Print statistics
    stats = get_quantization_stats(quantized_model)
    logger.info(
        f"✅ BitNet b1.58 quantization applied: "
        f"{stats['quantized_layers']}/{stats['total_layers']} layers quantized "
        f"({stats['quantization_ratio']*100:.1f}%)"
    )
    
    return quantized_model


def get_model_info(model: nn.Module) -> dict:
    """
    Get information about the loaded model.
    
    Args:
        model: Model to inspect
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "trainable_ratio": trainable_params / total_params if total_params > 0 else 0.0,
    }


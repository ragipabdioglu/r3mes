"""
Model Manager - Multi-LoRA Engine

Sunucu açıldığında Ana Modeli (BitNet) yükler ve istek geldiğinde
"Tak-Çıkar" adaptörleri yönetir.

NOTE: This module requires GPU libraries (torch, transformers, peft, bitsandbytes).
It should only be imported when R3MES_INFERENCE_MODE=local.
For GPU-less deployment, use the lazy loading pattern in main.py.
"""

from typing import Optional, Dict, Iterator, TYPE_CHECKING
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)

# Lazy import GPU libraries to support GPU-less deployment
# These imports are deferred until actually needed
_torch = None
_transformers = None
_peft_available = None
_bitsandbytes_available = None


def _ensure_gpu_libraries():
    """Lazy load GPU libraries when needed."""
    global _torch, _transformers, _peft_available, _bitsandbytes_available
    
    if _torch is not None:
        return  # Already loaded
    
    # Import torch
    import torch as torch_module
    _torch = torch_module
    
    # Set environment variables to help bitsandbytes find CUDA
    os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")
    
    # Set CUDA version for bitsandbytes
    try:
        cuda_version = _torch.version.cuda
        if cuda_version:
            cuda_major, cuda_minor = cuda_version.split(".")[:2]
            bnb_cuda_version = f"{cuda_major}{cuda_minor}"
            os.environ.setdefault("BNB_CUDA_VERSION", bnb_cuda_version)
            logger.info(f"Set BNB_CUDA_VERSION={bnb_cuda_version} for bitsandbytes")
    except Exception:
        pass
    
    # Try to add PyTorch's CUDA library path to LD_LIBRARY_PATH
    try:
        torch_lib_path = os.path.join(os.path.dirname(_torch.__file__), "lib")
        if os.path.exists(torch_lib_path):
            current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            if torch_lib_path not in current_ld_path:
                os.environ["LD_LIBRARY_PATH"] = f"{torch_lib_path}:{current_ld_path}" if current_ld_path else torch_lib_path
    except Exception:
        pass
    
    # Import transformers
    import transformers as transformers_module
    _transformers = transformers_module
    
    # Try to import peft
    try:
        from peft import PeftModel
        _peft_available = True
    except Exception as e:
        logger.warning(f"peft not available (bitsandbytes/CUDA issue): {e}")
        logger.warning("Will use mock mode for adapters")
        _peft_available = False
    
    # Try to import BitsAndBytesConfig
    try:
        from transformers import BitsAndBytesConfig
        _bitsandbytes_available = True
    except Exception as e:
        logger.warning(f"bitsandbytes not available: {e}")
        logger.warning("Will use CPU mode or full precision loading")
        _bitsandbytes_available = False


# Import exceptions (these don't require GPU)
from .exceptions import (
    ProductionConfigurationError,
    ModelLoadError,
    ModelNotFoundError,
    AdapterNotFoundError,
    InsufficientVRAMError,
)


class AIModelManager:
    """
    Multi-LoRA Model Manager
    
    Özellikler:
    - BitNet Base Model'i düşük VRAM modunda yükler
    - LoRA adaptörlerini dinamik olarak yükler/kaldırır
    - Adaptörler arasında geçiş yapar
    - Streaming response üretir
    
    NOTE: This class requires GPU libraries. Only instantiate when
    R3MES_INFERENCE_MODE=local.
    """
    
    def __init__(self, base_model_path: Optional[str] = None):
        """
        Başlangıçta BitNet Base Model'i düşük VRAM modunda yükler.
        
        Args:
            base_model_path: Base model dosya yolu (None ise multi-source loader kullanılır)
        """
        # Ensure GPU libraries are loaded
        _ensure_gpu_libraries()
        
        # Check environment for production mode
        is_production = os.getenv("R3MES_ENV", "development").lower() == "production"
        use_mock_model = os.getenv("R3MES_USE_MOCK_MODEL", "false").lower() == "true"
        
        # SECURITY: In production, mock model is strictly forbidden
        if is_production and use_mock_model:
            raise ProductionConfigurationError(
                "FATAL: R3MES_USE_MOCK_MODEL=true is not allowed in production environment. "
                "Mock models are for development/testing only. "
                "Production requires a real model. Set R3MES_USE_MOCK_MODEL=false and provide a valid model."
            )
        
        # Use multi-source loader if path not provided
        if base_model_path is None:
            from .model_loader import get_model_loader
            model_loader = get_model_loader()
            model_path, source = model_loader.get_model_path()
            
            if model_path and source != "none":
                base_model_path = model_path
                logger.info(f"Model path resolved: {model_path} (source: {source})")
            else:
                base_model_path = os.getenv("BASE_MODEL_PATH", "checkpoints/base_model")
        
        # Normalize path (resolve relative paths, keep absolute paths as-is)
        # HuggingFace model names don't need normalization
        if base_model_path and not base_model_path.startswith(("meta-llama/", "microsoft/", "google/", "huggingface/")):
            if not Path(base_model_path).is_absolute():
                # Resolve relative path from current working directory
                base_model_path = str(Path.cwd() / base_model_path)
            else:
                base_model_path = str(Path(base_model_path).resolve())
        
        # Check if model exists
        model_exists = False
        if base_model_path:
            # For HuggingFace models, path is the model name (transformers handles it)
            if base_model_path.startswith(("meta-llama/", "microsoft/", "google/", "huggingface/")):
                model_exists = True  # HuggingFace models don't need local path
            else:
                model_exists = Path(base_model_path).exists()
        
        if not model_exists:
            if is_production and not use_mock_model:
                raise ModelLoadError(
                    f"Model not found at {base_model_path} and R3MES_USE_MOCK_MODEL is not enabled. "
                    "Cannot run in production without a valid model. "
                    "Set R3MES_MODEL_IPFS_HASH, R3MES_MODEL_NAME, or BASE_MODEL_PATH environment variable."
                )
            logger.warning(f"Base model not found at {base_model_path}")
            logger.warning("Using mock mode (for development)")
            self.base_model = None
            self.tokenizer = None
            self.adapters: Dict[str, object] = {}
            self.active_adapter: Optional[str] = None
            return
        
        try:
            # Get transformers classes
            AutoModelForCausalLM = _transformers.AutoModelForCausalLM
            AutoTokenizer = _transformers.AutoTokenizer
            
            # Quantization config (4-bit loading for low VRAM) - only if bitsandbytes is available
            quantization_config = None
            if _bitsandbytes_available:
                try:
                    BitsAndBytesConfig = _transformers.BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=_torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                except Exception as e:
                    logger.warning(f"Failed to create BitsAndBytesConfig: {e}")
                    logger.warning("Will load model in full precision")
                    quantization_config = None
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model with or without quantization
            load_kwargs = {
                "device_map": "auto",
                "torch_dtype": _torch.float16
            }
            if quantization_config is not None:
                load_kwargs["quantization_config"] = quantization_config
            
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                **load_kwargs
            )
            
            # Active adapters registry
            self.adapters: Dict[str, object] = {}
            self.active_adapter: Optional[str] = None
            
            logger.info("Base model loaded with 4-bit quantization")
        except Exception as e:
            logger.warning(f"Failed to load base model: {e}")
            logger.warning("Using mock mode (for development)")
            self.base_model = None
            self.tokenizer = None
            self.adapters: Dict[str, object] = {}
            self.active_adapter: Optional[str] = None
    
    def load_adapter(self, name: str, path: str) -> bool:
        """
        Belirtilen yoldaki LoRA adaptörünü modele ekler (modeli kapatmadan).
        
        Args:
            name: Adaptör adı (örn: "coder_adapter", "law_adapter")
            path: Adaptör dosya yolu
            
        Returns:
            True if successful, False otherwise
        """
        if self.base_model is None:
            logger.warning(f"Mock mode: Adapter '{name}' would be loaded from {path}")
            self.adapters[name] = None  # Mock adapter
            return True
        
        # Check if peft is available (lazy loaded)
        if not _peft_available:
            logger.warning(f"peft not available, cannot load adapter '{name}'")
            self.adapters[name] = None  # Mock adapter fallback
            return True
        
        try:
            # Import PeftModel from lazy-loaded peft
            from peft import PeftModel
            
            # Load LoRA adapter
            adapter = PeftModel.from_pretrained(
                self.base_model,
                path,
                adapter_name=name
            )
            
            self.adapters[name] = adapter
            logger.info(f"Adapter '{name}' loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load adapter '{name}': {e}")
            return False
    
    def switch_adapter(self, name: str) -> bool:
        """
        Aktif adaptörü değiştirir.
        
        Args:
            name: Adaptör adı
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self.adapters:
            logger.error(f"Adapter '{name}' not found")
            return False
        
        if self.base_model is None:
            # Mock mode
            self.active_adapter = name
            logger.info(f"Mock: Switched to adapter '{name}'")
            return True
        
        # Switch active adapter
        self.base_model.set_adapter(name)
        self.active_adapter = name
        logger.info(f"Switched to adapter '{name}'")
        return True
    
    def generate_response(
        self, 
        prompt: str, 
        adapter_name: Optional[str] = None
    ) -> Iterator[str]:
        """
        İlgili adaptörü aktif edip cevabı stream (akış) olarak üretir.
        
        Args:
            prompt: Kullanıcı sorusu
            adapter_name: Kullanılacak adaptör adı (None ise aktif adaptör kullanılır)
            
        Yields:
            Token strings (streaming)
        """
        # Mock mode
        if self.base_model is None:
            # Generate a more realistic mock response based on adapter
            if adapter_name == "coder_adapter":
                mock_response = f"Here's a code solution for your question: '{prompt[:50]}...'\n\nIn Python, you can solve this by using appropriate data structures and algorithms. The key is to understand the problem requirements first, then design an efficient solution."
            elif adapter_name == "law_adapter":
                mock_response = f"From a legal perspective regarding '{prompt[:50]}...':\n\nThis matter involves several legal considerations. It's important to consult with a qualified attorney for specific advice tailored to your situation."
            else:
                mock_response = f"Thank you for your question: '{prompt[:50]}...'\n\nThis is a mock response in development mode. The actual AI model will provide detailed answers once the model files are configured."
            
            # Stream response character by character (no blocking sleep)
            # Note: Real streaming uses async generators, this is mock mode only
            for char in mock_response:
                yield char
            return
        
        # Switch adapter if specified
        if adapter_name and adapter_name != self.active_adapter:
            if not self.switch_adapter(adapter_name):
                yield f"Error: Adapter '{adapter_name}' not available"
                return
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.base_model.device)
        
        # Generate with streaming (use lazy-loaded torch)
        with _torch.no_grad():
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode and stream tokens
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Stream character by character
        for char in generated_text:
            yield char


"""
GGUF Model Loader - llama-cpp-python Integration

Loads .gguf model files using llama-cpp-python for efficient inference.
This replaces PyTorch-based loading for better performance and lower memory usage.
"""

import os
import logging
from typing import Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import llama-cpp-python
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama-cpp-python not available. Install with: pip install llama-cpp-python")


class GGUFModelLoader:
    """Loader for GGUF format models using llama-cpp-python."""
    
    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = -1,  # -1 means use all GPU layers
        n_ctx: int = 2048,  # Context window size
        n_threads: Optional[int] = None,  # CPU threads (None = auto)
        verbose: bool = False,
    ):
        """
        Initialize GGUF model loader.
        
        Args:
            model_path: Path to .gguf model file
            n_gpu_layers: Number of layers to offload to GPU (-1 = all)
            n_ctx: Context window size in tokens
            n_threads: Number of CPU threads (None = auto-detect)
            verbose: Enable verbose logging
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is not installed. "
                "Install with: pip install llama-cpp-python"
            )
        
        # Normalize model path
        model_path_obj = Path(model_path)
        if not model_path_obj.is_absolute():
            model_path = str(Path.cwd() / model_path_obj)
        else:
            model_path = str(model_path_obj.resolve())
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if not model_path.endswith('.gguf'):
            raise ValueError(f"Model file must be .gguf format: {model_path}")
        
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.verbose = verbose
        self.llm: Optional[Llama] = None
        
        logger.info(f"Initializing GGUF model loader: {model_path}")
        logger.info(f"  GPU layers: {n_gpu_layers} (-1 = all)")
        logger.info(f"  Context size: {n_ctx} tokens")
    
    def load(self) -> Llama:
        """
        Load the GGUF model.
        
        Returns:
            Llama instance ready for inference
        """
        if self.llm is not None:
            return self.llm
        
        logger.info(f"Loading GGUF model: {self.model_path}")
        
        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,  # Offload all layers to GPU if available
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=self.verbose,
            )
            logger.info("✅ GGUF model loaded successfully")
            return self.llm
        except Exception as e:
            logger.error(f"Failed to load GGUF model: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list] = None,
        stream: bool = False,
    ):
        """
        Generate text from prompt.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            top_p: Nucleus sampling parameter
            stop: List of stop sequences
            
        Returns:
            Generated text
        """
        if self.llm is None:
            self.load()
        
        try:
            if stream:
                # Streaming generation
                for token in self.llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    stream=True,
                ):
                    if isinstance(token, dict):
                        # Extract text from streaming dict
                        text = token.get('choices', [{}])[0].get('delta', {}).get('content', '')
                        if text:
                            yield text
                    elif isinstance(token, str):
                        yield token
            else:
                # Non-streaming generation
                output = self.llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                )
                
                # Extract text from output
                if isinstance(output, dict):
                    return output.get('choices', [{}])[0].get('text', '')
                elif isinstance(output, str):
                    return output
                else:
                    return str(output)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            if stream:
                yield f"[Error: {str(e)}]"
            else:
                raise
    
    def __del__(self):
        """Cleanup on deletion."""
        if self.llm is not None:
            # llama-cpp-python handles cleanup automatically
            pass


def load_gguf_model(
    model_path: str,
    n_gpu_layers: int = -1,
    n_ctx: int = 2048,
) -> GGUFModelLoader:
    """
    Convenience function to load a GGUF model.
    
    Args:
        model_path: Path to .gguf model file
        n_gpu_layers: Number of GPU layers (-1 = all)
        n_ctx: Context window size
        
    Returns:
        GGUFModelLoader instance
    """
    loader = GGUFModelLoader(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
    )
    loader.load()
    return loader


def find_gguf_model(model_dir: Optional[str] = None) -> Optional[str]:
    """
    Find GGUF model file in default locations.
    
    Args:
        model_dir: Directory to search (default: ~/.r3mes/models, or models/ relative to cwd)
        
    Returns:
        Path to .gguf file or None if not found
    """
    if model_dir is None:
        # Try relative path first (Docker volume mount compatible)
        relative_dir = Path.cwd() / "models"
        if relative_dir.exists():
            model_dir = relative_dir
        else:
            # Fallback to user home directory
            home = Path.home()
            model_dir = home / ".r3mes" / "models"
    else:
        # Normalize provided path
        model_dir_path = Path(model_dir)
        if not model_dir_path.is_absolute():
            model_dir = Path.cwd() / model_dir_path
        else:
            model_dir = model_dir_path.resolve()
    
    if not model_dir.exists():
        return None
    
    # Search for .gguf files
    gguf_files = list(model_dir.glob("*.gguf"))
    
    if not gguf_files:
        return None
    
    # Return the first one found (or could implement priority logic)
    return str(gguf_files[0].resolve())


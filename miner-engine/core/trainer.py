"""
LoRA Adapter Training Module

Implements training loop for LoRA adapters only (backbone remains frozen).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, Tuple
import numpy as np

from core.bitlinear import BitLinear
from utils.gpu_detection import GPUArchitectureDetector

# Configurable training parameters via environment variables
DEFAULT_WEIGHT_DECAY = float(os.getenv("LORA_WEIGHT_DECAY", "0.01"))
DEFAULT_GRAD_CLIP_MAX_NORM = float(os.getenv("LORA_GRAD_CLIP_MAX_NORM", "1.0"))
DEFAULT_QUANTIZATION_SCALE = float(os.getenv("LORA_QUANTIZATION_SCALE", "127.0"))


class LoRATrainer:
    """
    Trainer for LoRA adapters with frozen backbone.
    
    Only LoRA adapter parameters (A, B, alpha) are trained.
    Backbone weights remain frozen throughout training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        deterministic: bool = True,
        device: Optional[torch.device] = None,
        custom_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        Initialize LoRA trainer.
        
        Args:
            model: Model with BitLinear layers
            learning_rate: Learning rate for LoRA adapters
            deterministic: Enable deterministic operations
            device: Device to use (auto-detect if None)
            custom_optimizer: Optional custom optimizer (e.g., PagedAdamW8bit for low VRAM)
        """
        self.model = model
        self.learning_rate = learning_rate
        self.deterministic = deterministic
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Freeze all backbone weights
        self._freeze_backbone()
        
        # Get only LoRA parameters for training
        self.lora_params = self._get_lora_parameters()
        
        # Initialize optimizer (only for LoRA parameters)
        if custom_optimizer is not None:
            self.optimizer = custom_optimizer
        else:
            self.optimizer = optim.AdamW(
                self.lora_params,
                lr=learning_rate,
                weight_decay=DEFAULT_WEIGHT_DECAY,
            )
        
        # GPU architecture detector
        self.gpu_detector = GPUArchitectureDetector()
        
        # Training state
        self.training_step = 0
        self.loss_history = []
        
        if deterministic:
            self._configure_deterministic()
    
    def _freeze_backbone(self):
        """Freeze all backbone weights in BitLinear layers."""
        for module in self.model.modules():
            if isinstance(module, BitLinear):
                # Freeze backbone weight
                if hasattr(module, 'backbone_weight'):
                    module.backbone_weight.requires_grad = False
                # Ensure LoRA parameters are trainable
                module.lora_A.requires_grad = True
                module.lora_B.requires_grad = True
    
    def _get_lora_parameters(self) -> list:
        """
        Get all LoRA adapter parameters (A and B matrices).
        
        Returns:
            List of LoRA parameters
        """
        lora_params = []
        for module in self.model.modules():
            if isinstance(module, BitLinear):
                lora_params.append(module.lora_A)
                lora_params.append(module.lora_B)
        return lora_params
    
    def _configure_deterministic(self):
        """Configure deterministic operations for reproducibility."""
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
    
    def set_seed(self, seed: int):
        """
        Set random seed for deterministic training.
        
        Seeds ALL random sources for complete reproducibility:
        - Python's random module
        - NumPy's random
        - PyTorch CPU and CUDA
        
        Args:
            seed: Random seed
        """
        import random
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
        # Set Python hash seed for deterministic hashing
        import os
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        if self.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: Optional[nn.Module] = None,
    ) -> Tuple[float, Dict[str, torch.Tensor]]:
        """
        Perform one training step (only LoRA adapters are updated).
        
        Args:
            inputs: Input tensor
            targets: Target tensor
            loss_fn: Loss function (default: MSELoss)
            
        Returns:
            Tuple of (loss_value, gradients_dict)
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        
        # Move inputs and targets to device
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = loss_fn(outputs, targets)
        
        # Backward pass (only LoRA parameters get gradients)
        loss.backward()
        
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.lora_params, max_norm=DEFAULT_GRAD_CLIP_MAX_NORM)
        
        # Optimizer step (only updates LoRA parameters)
        self.optimizer.step()
        
        # Collect gradients for LoRA parameters only
        gradients = {}
        for name, module in self.model.named_modules():
            if isinstance(module, BitLinear):
                if module.lora_A.grad is not None:
                    gradients[f"{name}.lora_A"] = module.lora_A.grad.clone()
                if module.lora_B.grad is not None:
                    gradients[f"{name}.lora_B"] = module.lora_B.grad.clone()
        
        self.training_step += 1
        self.loss_history.append(loss.item())
        
        return loss.item(), gradients
    
    def get_lora_gradients(self) -> Dict[str, torch.Tensor]:
        """
        Get current LoRA adapter gradients.
        
        Returns:
            Dictionary mapping parameter names to gradients
        """
        gradients = {}
        for name, module in self.model.named_modules():
            if isinstance(module, BitLinear):
                if module.lora_A.grad is not None:
                    gradients[f"{name}.lora_A"] = module.lora_A.grad.clone()
                if module.lora_B.grad is not None:
                    gradients[f"{name}.lora_B"] = module.lora_B.grad.clone()
        return gradients
    
    def compute_gradient_hash(self, gradients: Dict[str, torch.Tensor], precision: str = "int8") -> str:
        """
        Compute deterministic hash of gradients for verification.
        
        Uses quantized int8 representation for cross-platform determinism.
        Delegates to DeterministicHashVerifier for consistent hashing across the codebase.
        
        Args:
            gradients: Dictionary of gradients
            precision: Hash precision ("int8" for cross-platform, "float32" for same-platform)
            
        Returns:
            Hex string hash
        """
        from core.verification import DeterministicHashVerifier
        return DeterministicHashVerifier.compute_deterministic_hash(gradients, precision=precision)
    
    def get_lora_state_dict(self) -> Dict:
        """
        Get LoRA adapter state dict for serialization.
        
        Returns:
            Dictionary containing LoRA parameters
        """
        state_dict = {}
        for name, module in self.model.named_modules():
            if isinstance(module, BitLinear):
                state_dict[f"{name}.lora_A"] = module.lora_A.data.clone()
                state_dict[f"{name}.lora_B"] = module.lora_B.data.clone()
                state_dict[f"{name}.lora_alpha"] = module.lora_alpha
        return state_dict
    
    def estimate_lora_size_mb(self) -> float:
        """
        Estimate total size of LoRA adapters in MB.
        
        Returns:
            Size in megabytes
        """
        total_bytes = 0
        for module in self.model.modules():
            if isinstance(module, BitLinear):
                total_bytes += module.in_features * module.lora_rank * 4  # lora_A
                total_bytes += module.out_features * module.lora_rank * 4  # lora_B
        return total_bytes / (1024 * 1024)
    
    def get_training_metadata(self) -> Dict:
        """
        Get training metadata for gradient submission.
        
        Returns:
            Dictionary with training metadata
        """
        gpu_meta = self.gpu_detector.get_metadata()
        return {
            **gpu_meta,
            "training_step": self.training_step,
            "learning_rate": self.learning_rate,
            "loss": self.loss_history[-1] if self.loss_history else None,
            "lora_size_mb": self.estimate_lora_size_mb(),
        }


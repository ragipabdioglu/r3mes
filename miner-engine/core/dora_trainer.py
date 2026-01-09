"""
DoRA (Weight-Decomposed Low-Rank Adaptation) Trainer Module

FAZ 4: MinerEngine DoRA Migration

Implements training loop for DoRA adapters (backbone remains frozen).
Replaces LoRATrainer for BitNet + DoRA architecture.

DoRA Formula:
    output = W₀x + m * (V / ||V||) * x
    
Where:
    - W₀ = BitLinear backbone (frozen, {-1, 0, +1})
    - m  = magnitude (learnable scalar per output dim)
    - V  = direction matrix = B @ A (low-rank decomposition)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, Tuple, List
import numpy as np
import logging

from core.bitlinear import BitLinear
from core.dora import BitLinearDoRA, DoRAAdapter
from utils.gpu_detection import GPUArchitectureDetector

# Import submission pipeline (optional, for auto-submit)
try:
    from pipeline.gradient_submission import GradientSubmissionPipeline, SubmissionResult
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    GradientSubmissionPipeline = None
    SubmissionResult = None

logger = logging.getLogger(__name__)

# Configurable training parameters via environment variables
DEFAULT_WEIGHT_DECAY = float(os.getenv("DORA_WEIGHT_DECAY", "0.01"))
DEFAULT_GRAD_CLIP_MAX_NORM = float(os.getenv("DORA_GRAD_CLIP_MAX_NORM", "1.0"))
DEFAULT_MAGNITUDE_LR_SCALE = float(os.getenv("DORA_MAGNITUDE_LR_SCALE", "0.1"))


class DoRATrainer:
    """
    Trainer for DoRA adapters with frozen BitLinear backbone.
    
    Only DoRA adapter parameters (magnitude, direction_A, direction_B) are trained.
    Backbone weights remain frozen throughout training.
    
    Key differences from LoRATrainer:
    - Trains magnitude (m) in addition to direction (V = B @ A)
    - Uses separate learning rate for magnitude (typically lower)
    - Supports direction normalization during training
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        magnitude_lr_scale: float = DEFAULT_MAGNITUDE_LR_SCALE,
        deterministic: bool = True,
        device: Optional[torch.device] = None,
        custom_optimizer: Optional[torch.optim.Optimizer] = None,
        # Auto-submit parameters (KRİTİK EKSİKLİK #1 ÇÖZÜMÜ)
        auto_submit: bool = False,
        submission_pipeline: Optional["GradientSubmissionPipeline"] = None,
        training_round_id: int = 0,
        shard_id: int = 0,
        submit_interval: int = 1,  # Submit every N steps
    ):
        """
        Initialize DoRA trainer.
        
        Args:
            model: Model with BitLinearDoRA layers
            learning_rate: Learning rate for direction parameters (A, B)
            magnitude_lr_scale: Scale factor for magnitude learning rate (default: 0.1)
            deterministic: Enable deterministic operations
            device: Device to use (auto-detect if None)
            custom_optimizer: Optional custom optimizer (e.g., PagedAdamW8bit for low VRAM)
            auto_submit: Enable automatic gradient submission to IPFS + Blockchain
            submission_pipeline: GradientSubmissionPipeline instance (required if auto_submit=True)
            training_round_id: Training round ID for submissions
            shard_id: Shard ID for submissions
            submit_interval: Submit gradients every N training steps
        """
        self.model = model
        self.learning_rate = learning_rate
        self.magnitude_lr_scale = magnitude_lr_scale
        self.deterministic = deterministic
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Auto-submit configuration
        self.auto_submit = auto_submit
        self.submission_pipeline = submission_pipeline
        self.training_round_id = training_round_id
        self.shard_id = shard_id
        self.submit_interval = submit_interval
        self._submission_results: List["SubmissionResult"] = []
        
        if auto_submit and not PIPELINE_AVAILABLE:
            logger.warning(
                "auto_submit=True but GradientSubmissionPipeline not available. "
                "Install pipeline module or disable auto_submit."
            )
            self.auto_submit = False
        
        if auto_submit and submission_pipeline is None:
            logger.warning(
                "auto_submit=True but no submission_pipeline provided. "
                "Disabling auto_submit."
            )
            self.auto_submit = False
        
        # Move model to device
        self.model.to(self.device)
        
        # Freeze all backbone weights
        self._freeze_backbone()
        
        # Get DoRA parameters for training (with separate groups)
        self.direction_params, self.magnitude_params = self._get_dora_parameters()
        
        # Initialize optimizer with parameter groups
        if custom_optimizer is not None:
            self.optimizer = custom_optimizer
        else:
            param_groups = [
                {
                    'params': self.direction_params,
                    'lr': learning_rate,
                    'name': 'direction'
                },
                {
                    'params': self.magnitude_params,
                    'lr': learning_rate * magnitude_lr_scale,
                    'name': 'magnitude'
                },
            ]
            self.optimizer = optim.AdamW(
                param_groups,
                weight_decay=DEFAULT_WEIGHT_DECAY,
            )
        
        # GPU architecture detector
        self.gpu_detector = GPUArchitectureDetector()
        
        # Training state
        self.training_step = 0
        self.loss_history = []
        
        if deterministic:
            self._configure_deterministic()
        
        # Log trainer info
        total_params = sum(p.numel() for p in self.direction_params) + \
                       sum(p.numel() for p in self.magnitude_params)
        logger.info(f"DoRATrainer initialized: {total_params:,} trainable parameters")
        logger.info(f"  Direction LR: {learning_rate}, Magnitude LR: {learning_rate * magnitude_lr_scale}")
    
    def _freeze_backbone(self):
        """Freeze all backbone weights in BitLinear/BitLinearDoRA layers."""
        for module in self.model.modules():
            if isinstance(module, BitLinearDoRA):
                # Freeze backbone
                module.backbone.requires_grad_(False)
                for param in module.backbone.parameters():
                    param.requires_grad = False
                # Ensure DoRA parameters are trainable
                module.magnitude.requires_grad = True
                module.direction_A.requires_grad = True
                module.direction_B.requires_grad = True
            elif isinstance(module, BitLinear):
                # Freeze BitLinear backbone weight
                if hasattr(module, 'backbone_weight'):
                    module.backbone_weight.requires_grad = False
    
    def _get_dora_parameters(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        """
        Get DoRA adapter parameters separated by type.
        
        Returns:
            Tuple of (direction_params, magnitude_params)
        """
        direction_params = []
        magnitude_params = []
        
        for module in self.model.modules():
            if isinstance(module, BitLinearDoRA):
                direction_params.append(module.direction_A)
                direction_params.append(module.direction_B)
                magnitude_params.append(module.magnitude)
        
        return direction_params, magnitude_params
    
    def _configure_deterministic(self):
        """Configure deterministic operations for reproducibility."""
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
    
    def set_seed(self, seed: int):
        """
        Set random seed for deterministic training.
        
        Args:
            seed: Random seed
        """
        import random
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
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
        Perform one training step (only DoRA adapters are updated).
        
        If auto_submit is enabled, gradients are automatically submitted
        to IPFS and blockchain after training.
        
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
        
        # Backward pass (only DoRA parameters get gradients)
        loss.backward()
        
        # Clip gradients for stability
        all_params = self.direction_params + self.magnitude_params
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=DEFAULT_GRAD_CLIP_MAX_NORM)
        
        # Optimizer step (only updates DoRA parameters)
        self.optimizer.step()
        
        # Collect gradients for DoRA parameters
        gradients = self.get_dora_gradients()
        
        self.training_step += 1
        self.loss_history.append(loss.item())
        
        # Auto-submit gradients (KRİTİK EKSİKLİK #1 ÇÖZÜMÜ)
        if self.auto_submit and self.submission_pipeline:
            if self.training_step % self.submit_interval == 0:
                self._auto_submit_gradients(gradients, loss.item())
        
        return loss.item(), gradients
    
    def _auto_submit_gradients(
        self,
        gradients: Dict[str, torch.Tensor],
        loss: float,
    ):
        """
        Automatically submit gradients to IPFS and blockchain.
        
        Args:
            gradients: Gradient dictionary
            loss: Current loss value
        """
        if not self.submission_pipeline:
            return
        
        try:
            metadata = self.get_training_metadata()
            metadata["loss"] = loss
            
            result = self.submission_pipeline.submit_after_training(
                gradients=gradients,
                training_round_id=self.training_round_id,
                shard_id=self.shard_id,
                metadata=metadata,
            )
            
            self._submission_results.append(result)
            
            if result.success:
                logger.info(
                    f"Gradient auto-submitted: IPFS={result.ipfs_hash}, "
                    f"TX={result.tx_hash}"
                )
            else:
                logger.warning(f"Gradient auto-submit failed: {result.error}")
                
        except Exception as e:
            logger.error(f"Auto-submit error: {e}", exc_info=True)
    
    def get_submission_results(self) -> List["SubmissionResult"]:
        """Get list of submission results from auto-submit."""
        return self._submission_results
    
    def set_training_round(self, training_round_id: int, shard_id: int = 0):
        """
        Set training round and shard for submissions.
        
        Args:
            training_round_id: Training round ID
            shard_id: Shard ID
        """
        self.training_round_id = training_round_id
        self.shard_id = shard_id
        logger.info(f"Training round set: round={training_round_id}, shard={shard_id}")
    
    def get_dora_gradients(self) -> Dict[str, torch.Tensor]:
        """
        Get current DoRA adapter gradients.
        
        Returns:
            Dictionary mapping parameter names to gradients
        """
        gradients = {}
        for name, module in self.model.named_modules():
            if isinstance(module, BitLinearDoRA):
                if module.magnitude.grad is not None:
                    gradients[f"{name}.magnitude"] = module.magnitude.grad.clone()
                if module.direction_A.grad is not None:
                    gradients[f"{name}.direction_A"] = module.direction_A.grad.clone()
                if module.direction_B.grad is not None:
                    gradients[f"{name}.direction_B"] = module.direction_B.grad.clone()
        return gradients
    
    def compute_gradient_hash(self, gradients: Dict[str, torch.Tensor], precision: str = "int8") -> str:
        """
        Compute deterministic hash of gradients for verification.
        
        Args:
            gradients: Dictionary of gradients
            precision: Hash precision ("int8" for cross-platform, "float32" for same-platform)
            
        Returns:
            Hex string hash
        """
        from core.verification import DeterministicHashVerifier
        return DeterministicHashVerifier.compute_deterministic_hash(gradients, precision=precision)
    
    def get_dora_state_dict(self) -> Dict:
        """
        Get DoRA adapter state dict for serialization.
        
        Returns:
            Dictionary containing DoRA parameters
        """
        state_dict = {}
        for name, module in self.model.named_modules():
            if isinstance(module, BitLinearDoRA):
                state_dict[f"{name}.magnitude"] = module.magnitude.data.clone()
                state_dict[f"{name}.direction_A"] = module.direction_A.data.clone()
                state_dict[f"{name}.direction_B"] = module.direction_B.data.clone()
                state_dict[f"{name}.rank"] = module.rank
                state_dict[f"{name}.alpha"] = module.alpha
        return state_dict
    
    def load_dora_state_dict(self, state_dict: Dict):
        """
        Load DoRA adapter state dict.
        
        Args:
            state_dict: Dictionary containing DoRA parameters
        """
        for name, module in self.model.named_modules():
            if isinstance(module, BitLinearDoRA):
                if f"{name}.magnitude" in state_dict:
                    module.magnitude.data = state_dict[f"{name}.magnitude"]
                if f"{name}.direction_A" in state_dict:
                    module.direction_A.data = state_dict[f"{name}.direction_A"]
                if f"{name}.direction_B" in state_dict:
                    module.direction_B.data = state_dict[f"{name}.direction_B"]
    
    def estimate_dora_size_mb(self) -> float:
        """
        Estimate total size of DoRA adapters in MB.
        
        Returns:
            Size in megabytes
        """
        total_bytes = 0
        for module in self.model.modules():
            if isinstance(module, BitLinearDoRA):
                total_bytes += module.estimate_size_mb() * 1024 * 1024
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
            "magnitude_lr_scale": self.magnitude_lr_scale,
            "loss": self.loss_history[-1] if self.loss_history else None,
            "dora_size_mb": self.estimate_dora_size_mb(),
            "adapter_type": "dora",
        }
    
    def create_adapter(self, adapter_id: str, domain: str) -> DoRAAdapter:
        """
        Create DoRAAdapter from current model state.
        
        Args:
            adapter_id: Unique adapter identifier
            domain: Domain category
            
        Returns:
            DoRAAdapter with current parameters
        """
        # Get first DoRA layer for dimensions
        for module in self.model.modules():
            if isinstance(module, BitLinearDoRA):
                adapter = DoRAAdapter(
                    adapter_id=adapter_id,
                    domain=domain,
                    rank=module.rank,
                    alpha=module.alpha,
                    in_features=module.in_features,
                    out_features=module.out_features,
                    params=module.get_trainable_params(),
                    metadata=self.get_training_metadata(),
                )
                return adapter
        
        raise RuntimeError("No BitLinearDoRA layers found in model")
    
    def apply_adapter(self, adapter: DoRAAdapter):
        """
        Apply DoRAAdapter to model.
        
        Args:
            adapter: DoRAAdapter to apply
        """
        for module in self.model.modules():
            if isinstance(module, BitLinearDoRA):
                adapter.apply_to_layer(module)
                break


# Backward compatibility alias
LoRATrainer = DoRATrainer  # For gradual migration

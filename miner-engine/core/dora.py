"""
DoRA (Weight-Decomposed Low-Rank Adaptation) Layer for BitNet

This module implements custom DoRA layers that work with BitLinear backbone.
Unlike PEFT library, this is designed specifically for BitNet's quantized weights.

DoRA Formula:
    output = W₀x + m * (V / ||V||) * x
    
Where:
    - W₀ = BitLinear backbone (frozen, {-1, 0, +1})
    - m  = magnitude (learnable scalar per output dim)
    - V  = direction matrix = B @ A (low-rank decomposition)
    - ||V|| = column-wise L2 norm

Reference: https://arxiv.org/abs/2402.09353
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import hashlib
import logging

from .bitlinear import BitLinear

logger = logging.getLogger(__name__)


class BitLinearDoRA(nn.Module):
    """
    DoRA adapter layer wrapping a frozen BitLinear backbone.
    
    Architecture:
        - Backbone: Frozen BitLinear (quantized {-1, 0, +1})
        - Magnitude: Learnable scalar per output dimension
        - Direction: Low-rank matrices A and B (V = B @ A)
    
    Trainable Parameters:
        - magnitude: [out_features] - scales the direction
        - direction_A: [rank, in_features] - low-rank factor
        - direction_B: [out_features, rank] - low-rank factor
    """
    
    def __init__(
        self,
        backbone: BitLinear,
        rank: int = 16,
        alpha: float = 1.0,
        dropout: float = 0.0,
        init_magnitude_from_backbone: bool = True,
    ):
        """
        Initialize DoRA adapter on top of BitLinear backbone.
        
        Args:
            backbone: Frozen BitLinear layer
            rank: Low-rank dimension (default: 16, range: 4-64)
            alpha: Scaling factor for DoRA output (default: 1.0)
            dropout: Dropout probability for direction (default: 0.0)
            init_magnitude_from_backbone: Initialize magnitude from backbone weights norm
        """
        super().__init__()
        
        self.in_features = backbone.in_features
        self.out_features = backbone.out_features
        self.rank = rank
        self.alpha = alpha
        
        # Store backbone reference (frozen)
        self.backbone = backbone
        self._freeze_backbone()
        
        # DoRA magnitude: learnable scalar per output dimension
        # Initialized from backbone weight norms if requested
        if init_magnitude_from_backbone:
            with torch.no_grad():
                # Compute column-wise L2 norm of backbone weights
                backbone_norm = backbone.backbone_weight.norm(dim=1)
                # Clamp to avoid zero magnitudes
                backbone_norm = torch.clamp(backbone_norm, min=1e-8)
        else:
            backbone_norm = torch.ones(self.out_features)
        
        self.magnitude = nn.Parameter(backbone_norm.clone())
        
        # DoRA direction: low-rank decomposition V = B @ A
        # A: [rank, in_features] - projects input to low-rank space
        # B: [out_features, rank] - projects back to output space
        self.direction_A = nn.Parameter(
            torch.randn(rank, self.in_features) * (1.0 / rank)
        )
        self.direction_B = nn.Parameter(
            torch.zeros(self.out_features, rank)
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Scaling factor
        self.scaling = alpha / rank
        
        logger.debug(
            f"BitLinearDoRA initialized: in={self.in_features}, "
            f"out={self.out_features}, rank={rank}, alpha={alpha}"
        )
    
    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        self.backbone.requires_grad_(False)
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: backbone(x) + DoRA(x)
        
        DoRA Formula:
            output = W₀x + m * (V / ||V||) * x
        
        Args:
            x: Input tensor [batch_size, ..., in_features]
            
        Returns:
            Output tensor [batch_size, ..., out_features]
        """
        # Backbone forward (frozen BitNet)
        backbone_out = self.backbone(x)
        
        # Compute direction matrix V = B @ A
        # B: [out_features, rank], A: [rank, in_features]
        # V: [out_features, in_features]
        V = self.direction_B @ self.direction_A
        
        # Normalize direction (row-wise L2 norm for weight matrix)
        # Each row of V corresponds to one output neuron
        V_norm = V.norm(dim=1, keepdim=True).clamp(min=1e-8)
        V_normalized = V / V_norm
        
        # Apply dropout to normalized direction
        V_normalized = self.dropout(V_normalized)
        
        # DoRA output: m * (V / ||V||) * x
        # magnitude: [out_features] -> [out_features, 1] for broadcasting
        # F.linear(x, V_normalized): [batch, ..., out_features]
        dora_out = F.linear(x, V_normalized)
        
        # Scale by magnitude (per output dimension)
        # magnitude: [out_features] -> broadcast over batch dims
        dora_out = dora_out * self.magnitude
        
        # Apply scaling factor
        dora_out = dora_out * self.scaling
        
        # Combine backbone and DoRA outputs
        return backbone_out + dora_out
    
    def get_direction_matrix(self) -> torch.Tensor:
        """
        Get the full direction matrix V = B @ A.
        
        Returns:
            Direction matrix [out_features, in_features]
        """
        return self.direction_B @ self.direction_A
    
    def get_trainable_params(self) -> Dict[str, torch.Tensor]:
        """
        Get all trainable DoRA parameters.
        
        Returns:
            Dict with magnitude, direction_A, direction_B
        """
        return {
            'magnitude': self.magnitude.data,
            'direction_A': self.direction_A.data,
            'direction_B': self.direction_B.data,
        }
    
    def set_trainable_params(self, params: Dict[str, torch.Tensor]):
        """
        Set trainable DoRA parameters (for loading).
        
        Args:
            params: Dict with magnitude, direction_A, direction_B
        """
        if 'magnitude' in params:
            self.magnitude.data = params['magnitude']
        if 'direction_A' in params:
            self.direction_A.data = params['direction_A']
        if 'direction_B' in params:
            self.direction_B.data = params['direction_B']
    
    def get_adapter_hash(self) -> str:
        """
        Get hash of DoRA adapter for verification.
        
        Returns:
            SHA256 hash of adapter parameters
        """
        params = self.get_trainable_params()
        combined = torch.cat([
            params['magnitude'].flatten(),
            params['direction_A'].flatten(),
            params['direction_B'].flatten(),
        ])
        return hashlib.sha256(combined.cpu().numpy().tobytes()).hexdigest()
    
    def estimate_size_mb(self) -> float:
        """
        Estimate size of DoRA adapter in MB.
        
        Returns:
            Size in megabytes
        """
        # magnitude: [out_features] * 4 bytes
        # direction_A: [rank, in_features] * 4 bytes
        # direction_B: [out_features, rank] * 4 bytes
        magnitude_size = self.out_features * 4
        direction_A_size = self.rank * self.in_features * 4
        direction_B_size = self.out_features * self.rank * 4
        total_bytes = magnitude_size + direction_A_size + direction_B_size
        return total_bytes / (1024 * 1024)
    
    def num_trainable_params(self) -> int:
        """
        Count trainable parameters.
        
        Returns:
            Number of trainable parameters
        """
        return (
            self.out_features +  # magnitude
            self.rank * self.in_features +  # direction_A
            self.out_features * self.rank  # direction_B
        )
    
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, "
            f"trainable_params={self.num_trainable_params():,}"
        )



class DoRAAdapter:
    """
    Serializable DoRA adapter container.
    
    Used for saving/loading adapters to/from disk or IPFS.
    Contains metadata and parameter tensors.
    """
    
    def __init__(
        self,
        adapter_id: str,
        domain: str,
        rank: int,
        alpha: float,
        in_features: int,
        out_features: int,
        params: Optional[Dict[str, torch.Tensor]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize DoRA adapter container.
        
        Args:
            adapter_id: Unique identifier (e.g., "medical_dora")
            domain: Domain category (e.g., "medical", "turkish", "coding")
            rank: Low-rank dimension
            alpha: Scaling factor
            in_features: Input dimension
            out_features: Output dimension
            params: Pre-loaded parameters (optional)
            metadata: Additional metadata (version, training info, etc.)
        """
        self.adapter_id = adapter_id
        self.domain = domain
        self.rank = rank
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features
        self.params = params or {}
        self.metadata = metadata or {}
        
        # Computed on demand
        self._hash: Optional[str] = None
    
    @property
    def hash(self) -> str:
        """Get or compute adapter hash."""
        if self._hash is None and self.params:
            combined = torch.cat([
                self.params.get('magnitude', torch.tensor([])).flatten(),
                self.params.get('direction_A', torch.tensor([])).flatten(),
                self.params.get('direction_B', torch.tensor([])).flatten(),
            ])
            self._hash = hashlib.sha256(combined.cpu().numpy().tobytes()).hexdigest()
        return self._hash or ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize adapter to dictionary."""
        return {
            'adapter_id': self.adapter_id,
            'domain': self.domain,
            'rank': self.rank,
            'alpha': self.alpha,
            'in_features': self.in_features,
            'out_features': self.out_features,
            'params': {k: v.cpu().numpy().tolist() for k, v in self.params.items()},
            'metadata': self.metadata,
            'hash': self.hash,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DoRAAdapter':
        """Deserialize adapter from dictionary."""
        params = {}
        if 'params' in data and data['params']:
            params = {
                k: torch.tensor(v, dtype=torch.float32)
                for k, v in data['params'].items()
            }
        
        adapter = cls(
            adapter_id=data['adapter_id'],
            domain=data['domain'],
            rank=data['rank'],
            alpha=data['alpha'],
            in_features=data['in_features'],
            out_features=data['out_features'],
            params=params,
            metadata=data.get('metadata', {}),
        )
        adapter._hash = data.get('hash')
        return adapter
    
    def save(self, path: str):
        """Save adapter to file."""
        torch.save(self.to_dict(), path)
        logger.info(f"Saved adapter {self.adapter_id} to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'DoRAAdapter':
        """Load adapter from file."""
        data = torch.load(path, map_location='cpu')
        # Convert numpy arrays back to tensors
        if 'params' in data:
            data['params'] = {
                k: torch.tensor(v, dtype=torch.float32) if not isinstance(v, torch.Tensor) else v
                for k, v in data['params'].items()
            }
        adapter = cls(
            adapter_id=data['adapter_id'],
            domain=data['domain'],
            rank=data['rank'],
            alpha=data['alpha'],
            in_features=data['in_features'],
            out_features=data['out_features'],
            params=data.get('params', {}),
            metadata=data.get('metadata', {}),
        )
        adapter._hash = data.get('hash')
        logger.info(f"Loaded adapter {adapter.adapter_id} from {path}")
        return adapter
    
    def apply_to_layer(self, dora_layer: BitLinearDoRA):
        """Apply adapter parameters to a DoRA layer."""
        if self.params:
            dora_layer.set_trainable_params(self.params)
    
    def extract_from_layer(self, dora_layer: BitLinearDoRA):
        """Extract parameters from a DoRA layer."""
        self.params = dora_layer.get_trainable_params()
        self._hash = None  # Reset hash
    
    def estimate_size_mb(self) -> float:
        """Estimate adapter size in MB."""
        magnitude_size = self.out_features * 4
        direction_A_size = self.rank * self.in_features * 4
        direction_B_size = self.out_features * self.rank * 4
        return (magnitude_size + direction_A_size + direction_B_size) / (1024 * 1024)


class DoRAExpertRegistry:
    """
    Registry for managing DoRA expert adapters.
    
    Provides:
        - Expert registration and lookup
        - Domain-based categorization
        - Metadata management
    """
    
    # Predefined expert categories
    DOMAIN_EXPERTS = [
        'medical_dora', 'legal_dora', 'coding_dora', 'finance_dora',
        'science_dora', 'history_dora', 'education_dora',
    ]
    
    LANGUAGE_EXPERTS = [
        'turkish_dora', 'german_dora', 'french_dora', 'spanish_dora',
        'arabic_dora', 'chinese_dora', 'japanese_dora', 'korean_dora',
    ]
    
    TASK_EXPERTS = [
        'summarization_dora', 'translation_dora', 'qa_dora',
        'creative_dora', 'analysis_dora',
    ]
    
    GENERAL_EXPERT = 'general_dora'
    
    def __init__(self):
        """Initialize expert registry."""
        self._experts: Dict[str, DoRAAdapter] = {}
        self._domain_map: Dict[str, str] = {}  # adapter_id -> domain
        
        # Build domain map
        for expert in self.DOMAIN_EXPERTS:
            domain = expert.replace('_dora', '')
            self._domain_map[expert] = domain
        
        for expert in self.LANGUAGE_EXPERTS:
            self._domain_map[expert] = 'language'
        
        for expert in self.TASK_EXPERTS:
            self._domain_map[expert] = 'task'
        
        self._domain_map[self.GENERAL_EXPERT] = 'general'
    
    def register(self, adapter: DoRAAdapter):
        """Register an expert adapter."""
        self._experts[adapter.adapter_id] = adapter
        if adapter.adapter_id not in self._domain_map:
            self._domain_map[adapter.adapter_id] = adapter.domain
        logger.info(f"Registered expert: {adapter.adapter_id}")
    
    def get(self, adapter_id: str) -> Optional[DoRAAdapter]:
        """Get expert by ID."""
        return self._experts.get(adapter_id)
    
    def get_domain(self, adapter_id: str) -> str:
        """Get domain for an adapter."""
        return self._domain_map.get(adapter_id, 'unknown')
    
    def list_experts(self, domain: Optional[str] = None) -> list:
        """List registered experts, optionally filtered by domain."""
        if domain is None:
            return list(self._experts.keys())
        return [
            aid for aid, d in self._domain_map.items()
            if d == domain and aid in self._experts
        ]
    
    def is_registered(self, adapter_id: str) -> bool:
        """Check if expert is registered."""
        return adapter_id in self._experts
    
    def unregister(self, adapter_id: str) -> bool:
        """Unregister an expert."""
        if adapter_id in self._experts:
            del self._experts[adapter_id]
            logger.info(f"Unregistered expert: {adapter_id}")
            return True
        return False
    
    def get_all_known_experts(self) -> list:
        """Get all known expert IDs (registered or not)."""
        return (
            self.DOMAIN_EXPERTS +
            self.LANGUAGE_EXPERTS +
            self.TASK_EXPERTS +
            [self.GENERAL_EXPERT]
        )


def create_dora_from_bitlinear(
    bitlinear: BitLinear,
    rank: int = 16,
    alpha: float = 1.0,
    dropout: float = 0.0,
) -> BitLinearDoRA:
    """
    Factory function to create DoRA layer from BitLinear.
    
    Args:
        bitlinear: Source BitLinear layer
        rank: DoRA rank
        alpha: Scaling factor
        dropout: Dropout probability
        
    Returns:
        BitLinearDoRA layer wrapping the backbone
    """
    return BitLinearDoRA(
        backbone=bitlinear,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        init_magnitude_from_backbone=True,
    )


def merge_dora_experts(
    experts: list,
    weights: list,
    base_layer: BitLinearDoRA,
) -> torch.Tensor:
    """
    Merge multiple DoRA expert outputs with weights.
    
    Used for multi-expert inference where router selects multiple experts.
    
    Args:
        experts: List of DoRAAdapter objects
        weights: List of weights (should sum to 1.0)
        base_layer: Base DoRA layer to apply experts to
        
    Returns:
        Merged direction matrix
    """
    if len(experts) != len(weights):
        raise ValueError("Number of experts must match number of weights")
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
    
    # Merge direction matrices
    merged_direction = torch.zeros(
        base_layer.out_features,
        base_layer.in_features,
        device=base_layer.direction_A.device,
    )
    
    merged_magnitude = torch.zeros(
        base_layer.out_features,
        device=base_layer.magnitude.device,
    )
    
    for expert, weight in zip(experts, weights):
        if expert.params:
            direction_A = expert.params.get('direction_A')
            direction_B = expert.params.get('direction_B')
            magnitude = expert.params.get('magnitude')
            
            if direction_A is not None and direction_B is not None:
                V = direction_B @ direction_A
                merged_direction += weight * V
            
            if magnitude is not None:
                merged_magnitude += weight * magnitude
    
    return merged_direction, merged_magnitude

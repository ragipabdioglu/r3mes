"""
LoRA Adapter Serialization Module

Efficient serialization of LoRA adapters for network transmission.
Target: 10-100MB total instead of 28GB+ (99.6%+ bandwidth reduction).
"""

import torch
import pickle
import gzip
import numpy as np
from typing import Dict, Tuple, Optional
import hashlib
import json

from core.trainer import LoRATrainer


class LoRASerializer:
    """
    Serializer for LoRA adapters with compression.
    
    Serializes only LoRA adapter parameters (A, B, alpha),
    not the full model weights.
    """
    
    def __init__(self, compression_level: int = 6):
        """
        Initialize LoRA serializer.
        
        Args:
            compression_level: gzip compression level (0-9, default: 6)
        """
        self.compression_level = compression_level
    
    def serialize_lora_state(
        self,
        lora_state_dict: Dict,
        metadata: Optional[Dict] = None,
    ) -> bytes:
        """
        Serialize LoRA adapter state dict to bytes.
        
        Args:
            lora_state_dict: Dictionary of LoRA parameters
            metadata: Optional metadata to include
            
        Returns:
            Compressed bytes
        """
        # Convert tensors to numpy arrays for efficient serialization
        serialized_data = {}
        for key, value in lora_state_dict.items():
            if isinstance(value, torch.Tensor):
                serialized_data[key] = {
                    'data': value.detach().cpu().numpy(),
                    'shape': list(value.shape),
                    'dtype': str(value.dtype),
                }
            else:
                serialized_data[key] = value
        
        # Create serialization package
        package = {
            'lora_params': serialized_data,
            'metadata': metadata or {},
        }
        
        # Serialize to pickle
        pickled = pickle.dumps(package, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Compress with gzip
        compressed = gzip.compress(pickled, compresslevel=self.compression_level)
        
        return compressed
    
    def deserialize_lora_state(self, data: bytes) -> Tuple[Dict, Dict]:
        """
        Deserialize LoRA adapter state from bytes.
        
        Args:
            data: Compressed bytes
            
        Returns:
            Tuple of (lora_state_dict, metadata)
        """
        # Decompress
        decompressed = gzip.decompress(data)
        
        # Unpickle
        package = pickle.loads(decompressed)
        
        # Convert numpy arrays back to tensors
        lora_state_dict = {}
        for key, value in package['lora_params'].items():
            if isinstance(value, dict) and 'data' in value:
                # Reconstruct tensor from numpy
                tensor = torch.from_numpy(value['data'])
                lora_state_dict[key] = tensor
            else:
                lora_state_dict[key] = value
        
        metadata = package.get('metadata', {})
        
        return lora_state_dict, metadata
    
    def compute_hash(self, data: bytes) -> str:
        """
        Compute SHA256 hash of serialized data.
        
        Args:
            data: Serialized bytes
            
        Returns:
            Hex string hash
        """
        return hashlib.sha256(data).hexdigest()
    
    def estimate_size_mb(self, lora_state_dict: Dict) -> float:
        """
        Estimate serialized size in MB.
        
        Args:
            lora_state_dict: LoRA state dictionary
            
        Returns:
            Estimated size in megabytes
        """
        # Serialize without compression for size estimation
        serialized_data = {}
        for key, value in lora_state_dict.items():
            if isinstance(value, torch.Tensor):
                serialized_data[key] = {
                    'data': value.detach().cpu().numpy(),
                    'shape': list(value.shape),
                    'dtype': str(value.dtype),
                }
            else:
                serialized_data[key] = value
        
        package = {
            'lora_params': serialized_data,
            'metadata': {},
        }
        
        pickled = pickle.dumps(package, protocol=pickle.HIGHEST_PROTOCOL)
        compressed = gzip.compress(pickled, compresslevel=self.compression_level)
        
        return len(compressed) / (1024 * 1024)
    
    def serialize_gradients(
        self,
        gradients: Dict[str, torch.Tensor],
        metadata: Optional[Dict] = None,
    ) -> bytes:
        """
        Serialize LoRA gradients for transmission.
        
        Args:
            gradients: Dictionary of gradients
            metadata: Optional metadata
            
        Returns:
            Compressed bytes
        """
        # Convert gradients to numpy arrays
        serialized_grads = {}
        for key, value in gradients.items():
            if isinstance(value, torch.Tensor):
                serialized_grads[key] = {
                    'data': value.detach().cpu().numpy(),
                    'shape': list(value.shape),
                    'dtype': str(value.dtype),
                }
        
        package = {
            'gradients': serialized_grads,
            'metadata': metadata or {},
        }
        
        pickled = pickle.dumps(package, protocol=pickle.HIGHEST_PROTOCOL)
        compressed = gzip.compress(pickled, compresslevel=self.compression_level)
        
        return compressed
    
    def deserialize_gradients(self, data: bytes) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Deserialize LoRA gradients from bytes.
        
        Args:
            data: Compressed bytes
            
        Returns:
            Tuple of (gradients_dict, metadata)
        """
        decompressed = gzip.decompress(data)
        package = pickle.loads(decompressed)
        
        gradients = {}
        for key, value in package['gradients'].items():
            if isinstance(value, dict) and 'data' in value:
                gradients[key] = torch.from_numpy(value['data'])
        
        metadata = package.get('metadata', {})
        
        return gradients, metadata


class LoRAAggregator:
    """
    Aggregator for LoRA adapters from multiple miners.
    
    Performs weighted averaging of LoRA parameters.
    """
    
    @staticmethod
    def aggregate_lora_states(
        lora_states: list[Dict],
        weights: Optional[list[float]] = None,
    ) -> Dict:
        """
        Aggregate multiple LoRA state dicts using weighted average.
        
        Args:
            lora_states: List of LoRA state dictionaries
            weights: Optional weights for each state (default: uniform)
            
        Returns:
            Aggregated LoRA state dictionary
        """
        if not lora_states:
            raise ValueError("No LoRA states to aggregate")
        
        if weights is None:
            weights = [1.0 / len(lora_states)] * len(lora_states)
        
        if len(weights) != len(lora_states):
            raise ValueError("Number of weights must match number of states")
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Get all parameter names
        param_names = set()
        for state in lora_states:
            param_names.update(state.keys())
        
        # Aggregate each parameter
        aggregated = {}
        for param_name in param_names:
            # Collect all values for this parameter
            values = []
            for i, state in enumerate(lora_states):
                if param_name in state:
                    values.append((state[param_name], weights[i]))
            
            if not values:
                continue
            
            # Weighted average
            if isinstance(values[0][0], torch.Tensor):
                aggregated[param_name] = sum(
                    value * weight for value, weight in values
                )
            else:
                # For non-tensor values (like alpha), use simple average
                aggregated[param_name] = sum(
                    value * weight for value, weight in values
                ) / sum(weights[i] for i, (_, _) in enumerate(values))
        
        return aggregated


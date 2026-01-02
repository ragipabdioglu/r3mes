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



class LoRAAggregator:
    """
    LoRA gradient aggregator for proposer nodes.
    
    Aggregates multiple LoRA adapter gradients using weighted averaging.
    """
    
    def __init__(self):
        """Initialize LoRA aggregator."""
        pass
    
    def aggregate_lora_states(
        self,
        lora_states: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate multiple LoRA states using weighted averaging.
        
        Args:
            lora_states: List of LoRA state dictionaries
            weights: Optional weights for each state (default: equal weights)
            
        Returns:
            Aggregated LoRA state dictionary
        """
        if not lora_states:
            raise ValueError("No LoRA states to aggregate")
        
        # Use equal weights if not provided
        if weights is None:
            weights = [1.0 / len(lora_states)] * len(lora_states)
        else:
            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
        
        if len(weights) != len(lora_states):
            raise ValueError("Number of weights must match number of LoRA states")
        
        # Get all parameter names from first state
        param_names = set(lora_states[0].keys())
        
        # Verify all states have same parameters
        for i, state in enumerate(lora_states[1:], 1):
            if set(state.keys()) != param_names:
                raise ValueError(f"LoRA state {i} has different parameters than state 0")
        
        # Aggregate parameters
        aggregated_state = {}
        
        for param_name in param_names:
            # Get tensors for this parameter from all states
            tensors = [state[param_name] for state in lora_states]
            
            # Verify all tensors have same shape
            first_shape = tensors[0].shape
            for i, tensor in enumerate(tensors[1:], 1):
                if tensor.shape != first_shape:
                    raise ValueError(f"Parameter {param_name} has different shapes: {first_shape} vs {tensor.shape}")
            
            # Weighted average
            aggregated_tensor = torch.zeros_like(tensors[0])
            for tensor, weight in zip(tensors, weights):
                aggregated_tensor += tensor * weight
            
            aggregated_state[param_name] = aggregated_tensor
        
        return aggregated_state
    
    def compute_merkle_root(self, gradient_hashes: List[str]) -> str:
        """
        Compute Merkle root of gradient hashes.
        
        Args:
            gradient_hashes: List of gradient IPFS hashes
            
        Returns:
            Merkle root as hex string
        """
        if not gradient_hashes:
            return ""
        
        # Simple Merkle tree implementation
        import hashlib
        
        # Start with leaf nodes (hash each gradient hash)
        nodes = [hashlib.sha256(hash_str.encode()).hexdigest() for hash_str in gradient_hashes]
        
        # Build tree bottom-up
        while len(nodes) > 1:
            next_level = []
            
            # Process pairs
            for i in range(0, len(nodes), 2):
                if i + 1 < len(nodes):
                    # Hash pair
                    combined = nodes[i] + nodes[i + 1]
                    next_level.append(hashlib.sha256(combined.encode()).hexdigest())
                else:
                    # Odd number of nodes, promote last node
                    next_level.append(nodes[i])
            
            nodes = next_level
        
        return nodes[0] if nodes else ""
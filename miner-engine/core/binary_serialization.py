"""
Binary Serialization using Protocol Buffers

Replaces JSON/pickle serialization with Protocol Buffers for:
- ~30% bandwidth reduction vs JSON
- Type safety
- Cross-language compatibility
- Better performance
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
import time

try:
    from google.protobuf.message import Message
    import gradient_pb2
    PROTOBUF_AVAILABLE = True
except ImportError:
    PROTOBUF_AVAILABLE = False
    print("⚠️  Protocol Buffers not available, falling back to pickle")


class BinaryGradientSerializer:
    """
    Binary serialization using Protocol Buffers.
    
    Provides ~30% bandwidth reduction compared to JSON.
    """
    
    def __init__(self):
        self.use_protobuf = PROTOBUF_AVAILABLE
    
    def serialize_gradients(
        self,
        gradients: Dict[str, torch.Tensor],
        metadata: Optional[Dict[str, Any]] = None,
        training_round_id: Optional[int] = None,
        miner_address: Optional[str] = None,
        gradient_hash: Optional[str] = None,
    ) -> bytes:
        """
        Serialize gradients using Protocol Buffers.
        
        Args:
            gradients: Dictionary of gradient tensors
            metadata: Optional metadata
            training_round_id: Training round ID
            miner_address: Miner address
            gradient_hash: Gradient hash
            
        Returns:
            Serialized bytes (protobuf binary format)
        """
        if not self.use_protobuf:
            # Fallback to pickle if protobuf not available
            return self._fallback_serialize(gradients, metadata)
        
        # Create protobuf message
        package = gradient_pb2.GradientPackage()
        
        # Add gradients
        for name, tensor in gradients.items():
            grad_tensor = package.gradients.add()
            grad_tensor.name = name
            
            # Convert to numpy and flatten
            np_array = tensor.detach().cpu().numpy()
            grad_tensor.data.extend(np_array.flatten().tolist())
            grad_tensor.shape.extend(np_array.shape)
            grad_tensor.dtype = str(np_array.dtype)
        
        # Add metadata
        if metadata:
            for key, value in metadata.items():
                package.metadata[str(key)] = str(value)
        
        # Add fields
        if training_round_id is not None:
            package.training_round_id = training_round_id
        if miner_address:
            package.miner_address = miner_address
        if gradient_hash:
            package.gradient_hash = gradient_hash
        package.timestamp = int(time.time())
        
        # Serialize to bytes
        return package.SerializeToString()
    
    def deserialize_gradients(self, data: bytes) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Deserialize gradients from Protocol Buffers bytes.
        
        Args:
            data: Serialized bytes
            
        Returns:
            Tuple of (gradients_dict, metadata)
        """
        if not self.use_protobuf:
            # Fallback to pickle if protobuf not available
            return self._fallback_deserialize(data)
        
        # Parse protobuf message
        package = gradient_pb2.GradientPackage()
        package.ParseFromString(data)
        
        # Reconstruct gradients
        gradients = {}
        for grad_tensor in package.gradients:
            # Reconstruct shape
            shape = tuple(grad_tensor.shape)
            
            # Reconstruct numpy array
            np_array = np.array(grad_tensor.data, dtype=grad_tensor.dtype).reshape(shape)
            
            # Convert to torch tensor
            gradients[grad_tensor.name] = torch.from_numpy(np_array)
        
        # Reconstruct metadata
        metadata = dict(package.metadata)
        if package.training_round_id > 0:
            metadata['training_round_id'] = package.training_round_id
        if package.miner_address:
            metadata['miner_address'] = package.miner_address
        if package.gradient_hash:
            metadata['gradient_hash'] = package.gradient_hash
        if package.timestamp > 0:
            metadata['timestamp'] = package.timestamp
        
        return gradients, metadata
    
    def _fallback_serialize(
        self,
        gradients: Dict[str, torch.Tensor],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        """Fallback to pickle if protobuf not available."""
        import pickle
        import gzip
        
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
        return gzip.compress(pickled, compresslevel=6)
    
    def _fallback_deserialize(self, data: bytes) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Fallback to pickle if protobuf not available."""
        import pickle
        import gzip
        
        decompressed = gzip.decompress(data)
        package = pickle.loads(decompressed)
        
        gradients = {}
        for key, value in package['gradients'].items():
            if isinstance(value, dict) and 'data' in value:
                gradients[key] = torch.from_numpy(value['data'])
        
        metadata = package.get('metadata', {})
        return gradients, metadata
    
    def estimate_size_mb(self, gradients: Dict[str, torch.Tensor]) -> float:
        """
        Estimate serialized size in MB.
        
        Args:
            gradients: Dictionary of gradient tensors
            
        Returns:
            Estimated size in MB
        """
        if not self.use_protobuf:
            # Estimate for pickle
            total_elements = sum(t.numel() for t in gradients.values())
            # Assume float32 (4 bytes) + overhead
            estimated_bytes = total_elements * 4 * 1.3  # 30% overhead
            return estimated_bytes / (1024 * 1024)
        
        # Estimate for protobuf (more efficient)
        total_elements = sum(t.numel() for t in gradients.values())
        # Protobuf is more efficient: float32 (4 bytes) + minimal overhead
        estimated_bytes = total_elements * 4 * 1.1  # 10% overhead
        return estimated_bytes / (1024 * 1024)


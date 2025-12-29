"""
Fixed Chunk / Variable Speed Protocol (Sabit Veri, Değişken Hız)

Protocol Rules:
1. Server always sends 2048-token chunks (never split)
2. Miner receives full chunk (cannot request smaller chunks)
3. Miner can micro-batch locally for VRAM efficiency
4. Gradient must be computed on full chunk (for validation)

FAZ 5 Integration: Uses core.constants for centralized configuration.
"""

import torch
from typing import Dict, Any, List, Optional
from torch.nn import Module
from torch.optim import Optimizer

# Import from core constants (FAZ 1)
try:
    from core.constants import CHUNK_SIZE_TOKENS
    FIXED_CHUNK_SIZE_TOKENS = CHUNK_SIZE_TOKENS
except ImportError:
    # Fallback for backward compatibility
    FIXED_CHUNK_SIZE_TOKENS = 2048

# Import validation utilities (FAZ 1)
try:
    from core.validation import validate_chunk_size as core_validate_chunk_size
    from core.exceptions import ChunkSizeError
    HAS_CORE_VALIDATION = True
except ImportError:
    HAS_CORE_VALIDATION = False


class ChunkProcessor:
    """
    Process fixed-size chunks with local micro-batching.
    
    Protocol Rules:
    1. Server always sends 2048-token chunks (never split)
    2. Miner receives full chunk (cannot request smaller chunks)
    3. Miner can micro-batch locally for VRAM efficiency
    4. Gradient must be computed on full chunk (for validation)
    """
    
    def __init__(self, local_batch_size: int = 1):
        """
        Initialize chunk processor.
        
        Args:
            local_batch_size: Local micro-batch size (VRAM-dependent, default: 1)
        """
        self.local_batch_size = local_batch_size
        self.chunk_buffer: List[Dict[str, Any]] = []
        print(f"✅ ChunkProcessor initialized with local_batch_size={local_batch_size}")
        print(f"   Protocol chunk size: {FIXED_CHUNK_SIZE_TOKENS} tokens (fixed)")
    
    def validate_chunk_size(self, chunk_data: Dict[str, Any], strict: bool = True) -> bool:
        """
        Validate that chunk matches protocol-mandated size.
        
        Args:
            chunk_data: Chunk data dictionary with 'input_ids' or 'token_count'
            strict: If True, require exact CHUNK_SIZE_TOKENS (default: True)
            
        Returns:
            True if valid, raises ValueError/ChunkSizeError if invalid
        """
        if "input_ids" in chunk_data:
            token_count = chunk_data["input_ids"].shape[1] if hasattr(chunk_data["input_ids"], "shape") else len(chunk_data["input_ids"])
        elif "token_count" in chunk_data:
            token_count = chunk_data["token_count"]
        else:
            raise ValueError("Chunk data must contain 'input_ids' or 'token_count'")
        
        # Use core validation if available (FAZ 1)
        if HAS_CORE_VALIDATION:
            try:
                core_validate_chunk_size(token_count, strict=strict)
                return True
            except ChunkSizeError as e:
                raise ValueError(str(e))
        
        # Fallback validation
        if strict and token_count != FIXED_CHUNK_SIZE_TOKENS:
            raise ValueError(
                f"Invalid chunk size: {token_count} tokens "
                f"(expected {FIXED_CHUNK_SIZE_TOKENS} tokens per protocol)"
            )
        
        return True
    
    def process_chunk(
        self,
        chunk_data: Dict[str, Any],
        model: Module,
        optimizer: Optimizer,
        device: Optional[torch.device] = None
    ) -> float:
        """
        Process full chunk with local micro-batching.
        
        Args:
            chunk_data: Full 2048-token chunk (protocol-mandated)
            model: Training model
            optimizer: Optimizer instance
            device: Device to use (auto-detect if None)
            
        Returns:
            Total loss for the chunk
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Validate chunk size matches protocol
        self.validate_chunk_size(chunk_data)
        
        # Extract input data
        input_ids = chunk_data["input_ids"]
        labels = chunk_data.get("labels", input_ids)  # Default to input_ids if labels not provided
        
        # Move to device
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.to(device)
        if isinstance(labels, torch.Tensor):
            labels = labels.to(device)
        
        # Split into micro-batches for local processing
        # Protocol: Full chunk is 2048 tokens, but we can process in smaller batches
        if isinstance(input_ids, torch.Tensor):
            seq_len = input_ids.shape[1]
            micro_batch_size = min(self.local_batch_size, seq_len)
        else:
            seq_len = len(input_ids)
            micro_batch_size = min(self.local_batch_size, seq_len)
        
        # Calculate number of micro-batches
        num_micro_batches = (seq_len + micro_batch_size - 1) // micro_batch_size
        
        # Process micro-batches with gradient accumulation
        total_loss = 0.0
        model.train()
        optimizer.zero_grad()
        
        for i in range(0, seq_len, micro_batch_size):
            end_idx = min(i + micro_batch_size, seq_len)
            
            # Extract micro-batch
            if isinstance(input_ids, torch.Tensor):
                micro_input = input_ids[:, i:end_idx]
                micro_labels = labels[:, i:end_idx] if isinstance(labels, torch.Tensor) else labels
            else:
                micro_input = input_ids[i:end_idx]
                micro_labels = labels[i:end_idx] if isinstance(labels, list) else labels
            
            # Forward pass
            output = model(micro_input)
            
            # Compute loss (simplified - in production would use proper loss function)
            if isinstance(output, torch.Tensor) and isinstance(micro_labels, torch.Tensor):
                loss_fn = torch.nn.MSELoss()
                loss = loss_fn(output, micro_labels)
            else:
                # Fallback: simple mean squared error
                loss = torch.mean((output - micro_labels) ** 2) if isinstance(output, torch.Tensor) else torch.tensor(0.0)
            
            # Scale loss for gradient accumulation
            scaled_loss = loss / num_micro_batches
            scaled_loss.backward()
            
            total_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
        
        # Gradient is now computed on full chunk (via accumulation)
        # This ensures validation can verify gradient on complete chunk
        optimizer.step()
        
        return total_loss
    
    def get_chunk_info(self, chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get information about a chunk.
        
        Args:
            chunk_data: Chunk data dictionary
            
        Returns:
            Dictionary with chunk information
        """
        if "input_ids" in chunk_data:
            token_count = chunk_data["input_ids"].shape[1] if hasattr(chunk_data["input_ids"], "shape") else len(chunk_data["input_ids"])
        elif "token_count" in chunk_data:
            token_count = chunk_data["token_count"]
        else:
            token_count = 0
        
        return {
            "token_count": token_count,
            "protocol_size": FIXED_CHUNK_SIZE_TOKENS,
            "is_valid": token_count == FIXED_CHUNK_SIZE_TOKENS,
            "local_batch_size": self.local_batch_size,
        }


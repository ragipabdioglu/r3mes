"""
Gradient Accumulation for Bandwidth Optimization

Accumulates gradients over multiple training steps before submission,
reducing network load by 4x (if accumulation_steps=4).
"""

import torch
from typing import List, Optional, Dict, Any


class GradientAccumulator:
    """Accumulates gradients over multiple training steps."""
    
    def __init__(self, accumulation_steps: int = 4):
        """
        Initialize gradient accumulator.
        
        Args:
            accumulation_steps: Number of steps to accumulate before submission
        """
        self.accumulation_steps = accumulation_steps
        self.accumulated_gradients: Optional[List[torch.Tensor]] = None
        self.step_count = 0
    
    def accumulate(self, gradients: List[torch.Tensor]) -> Optional[List[torch.Tensor]]:
        """
        Accumulate gradients over multiple steps.
        
        Args:
            gradients: List of gradient tensors from current step
        
        Returns:
            Accumulated gradients if accumulation complete, None otherwise
        """
        if self.accumulated_gradients is None:
            # Initialize with zeros
            self.accumulated_gradients = [torch.zeros_like(g) for g in gradients]
        
        # Accumulate (average)
        for i, g in enumerate(gradients):
            self.accumulated_gradients[i] += g / self.accumulation_steps
        
        self.step_count += 1
        
        if self.step_count >= self.accumulation_steps:
            return self.flush()
        
        return None
    
    def flush(self) -> List[torch.Tensor]:
        """
        Flush accumulated gradients for submission.
        
        Returns:
            Accumulated gradients
        """
        result = self.accumulated_gradients.copy()
        self.accumulated_gradients = None
        self.step_count = 0
        return result
    
    def reset(self):
        """Reset accumulator state."""
        self.accumulated_gradients = None
        self.step_count = 0


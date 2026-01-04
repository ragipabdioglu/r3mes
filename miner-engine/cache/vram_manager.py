"""
VRAM Manager for DoRA Adapter Allocation

Manages GPU memory allocation for DoRA adapters with:
- VRAM-adaptive gating (Top-K based on available VRAM)
- Dynamic allocation/deallocation
- Memory pressure monitoring
"""

import torch
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class VRAMTier(Enum):
    """VRAM capacity tiers."""
    LOW = "low"       # < 8GB
    MEDIUM = "medium" # 8-16GB
    HIGH = "high"     # > 16GB


@dataclass
class VRAMAllocation:
    """Represents a VRAM allocation."""
    adapter_id: str
    size_mb: float
    priority: int  # Higher = more important
    allocated_at: float
    
    def __lt__(self, other):
        return self.priority < other.priority


class VRAMManager:
    """
    Manages VRAM allocation for DoRA adapters.
    
    Implements VRAM-adaptive gating:
    - VRAM < 8GB: Top-1 expert only
    - VRAM 8-16GB: Top-2 experts
    - VRAM > 16GB: Top-3 experts
    """
    
    # VRAM thresholds (MB)
    LOW_THRESHOLD_MB = 8 * 1024      # 8GB
    HIGH_THRESHOLD_MB = 16 * 1024    # 16GB
    
    # Reserved VRAM for model and overhead (MB)
    RESERVED_VRAM_MB = 2048  # 2GB reserved
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize VRAM manager.
        
        Args:
            device: CUDA device to manage
        """
        self.device = device
        self._allocations: Dict[str, VRAMAllocation] = {}
        self._total_allocated_mb: float = 0.0
        
        # Detect VRAM capacity
        self._total_vram_mb = self._get_total_vram()
        self._available_vram_mb = self._total_vram_mb - self.RESERVED_VRAM_MB
        self._tier = self._determine_tier()
        
        logger.info(
            f"VRAMManager initialized: {self._total_vram_mb:.0f}MB total, "
            f"{self._available_vram_mb:.0f}MB available, tier={self._tier.value}"
        )
    
    def _get_total_vram(self) -> float:
        """Get total VRAM in MB."""
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / (1024 * 1024)
        except Exception:
            return 0.0
    
    def _determine_tier(self) -> VRAMTier:
        """Determine VRAM tier based on capacity."""
        if self._total_vram_mb < self.LOW_THRESHOLD_MB:
            return VRAMTier.LOW
        elif self._total_vram_mb < self.HIGH_THRESHOLD_MB:
            return VRAMTier.MEDIUM
        else:
            return VRAMTier.HIGH
    
    @property
    def tier(self) -> VRAMTier:
        """Get current VRAM tier."""
        return self._tier
    
    @property
    def max_experts(self) -> int:
        """Get maximum number of experts based on VRAM tier."""
        if self._tier == VRAMTier.LOW:
            return 1
        elif self._tier == VRAMTier.MEDIUM:
            return 2
        else:
            return 3
    
    def get_free_vram(self) -> float:
        """Get free VRAM in MB."""
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            free, total = torch.cuda.mem_get_info()
            return free / (1024 * 1024)
        except Exception:
            return self._available_vram_mb - self._total_allocated_mb
    
    def get_used_vram(self) -> float:
        """Get used VRAM in MB."""
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            return torch.cuda.memory_allocated() / (1024 * 1024)
        except Exception:
            return self._total_allocated_mb
    
    def can_allocate(self, size_mb: float) -> bool:
        """Check if allocation is possible."""
        return self.get_free_vram() >= size_mb
    
    def allocate(
        self,
        adapter_id: str,
        size_mb: float,
        priority: int = 0,
    ) -> bool:
        """
        Allocate VRAM for adapter.
        
        Args:
            adapter_id: Adapter identifier
            size_mb: Size to allocate
            priority: Allocation priority (higher = more important)
            
        Returns:
            True if allocation successful
        """
        if adapter_id in self._allocations:
            logger.warning(f"Adapter {adapter_id} already allocated")
            return True
        
        if not self.can_allocate(size_mb):
            # Try to free space by evicting low-priority allocations
            freed = self._evict_for_space(size_mb, priority)
            if not freed:
                logger.warning(f"Cannot allocate {size_mb}MB for {adapter_id}")
                return False
        
        import time
        allocation = VRAMAllocation(
            adapter_id=adapter_id,
            size_mb=size_mb,
            priority=priority,
            allocated_at=time.time(),
        )
        
        self._allocations[adapter_id] = allocation
        self._total_allocated_mb += size_mb
        
        logger.debug(f"Allocated {size_mb}MB for {adapter_id}")
        return True
    
    def deallocate(self, adapter_id: str) -> bool:
        """
        Deallocate VRAM for adapter.
        
        Args:
            adapter_id: Adapter to deallocate
            
        Returns:
            True if deallocation successful
        """
        if adapter_id not in self._allocations:
            return False
        
        allocation = self._allocations.pop(adapter_id)
        self._total_allocated_mb -= allocation.size_mb
        
        logger.debug(f"Deallocated {allocation.size_mb}MB from {adapter_id}")
        return True
    
    def _evict_for_space(self, needed_mb: float, min_priority: int) -> bool:
        """
        Evict low-priority allocations to free space.
        
        Args:
            needed_mb: Space needed
            min_priority: Minimum priority to keep
            
        Returns:
            True if enough space was freed
        """
        # Sort by priority (lowest first)
        sorted_allocs = sorted(
            self._allocations.items(),
            key=lambda x: x[1].priority
        )
        
        freed_mb = 0.0
        to_evict = []
        
        for adapter_id, alloc in sorted_allocs:
            if alloc.priority >= min_priority:
                break
            
            to_evict.append(adapter_id)
            freed_mb += alloc.size_mb
            
            if self.get_free_vram() + freed_mb >= needed_mb:
                break
        
        if self.get_free_vram() + freed_mb < needed_mb:
            return False
        
        for adapter_id in to_evict:
            self.deallocate(adapter_id)
        
        return True
    
    def select_experts_for_vram(
        self,
        expert_scores: List[Tuple[str, float]],
        expert_sizes: Dict[str, float],
        fallback_expert: str = "general_dora",
        confidence_threshold: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """
        Select experts based on VRAM capacity and scores.
        
        Implements VRAM-adaptive gating:
        - Selects Top-K experts based on VRAM tier
        - Falls back to general_dora if confidence too low
        
        Args:
            expert_scores: List of (expert_id, score) sorted by score desc
            expert_sizes: Dict of expert_id -> size_mb
            fallback_expert: Fallback expert ID
            confidence_threshold: Minimum confidence to use expert
            
        Returns:
            List of (expert_id, weight) to use
        """
        max_k = self.max_experts
        
        # Filter out fallback from main selection
        candidates = [
            (eid, score) for eid, score in expert_scores
            if eid != fallback_expert
        ]
        
        # Select top-K that fit in VRAM
        selected = []
        total_size = 0.0
        
        for expert_id, score in candidates[:max_k]:
            size = expert_sizes.get(expert_id, 10.0)  # Default 10MB
            
            if total_size + size <= self._available_vram_mb:
                selected.append((expert_id, score))
                total_size += size
        
        # Check if we need fallback
        if not selected or max(s for _, s in selected) < confidence_threshold:
            fallback_size = expert_sizes.get(fallback_expert, 10.0)
            if total_size + fallback_size <= self._available_vram_mb:
                selected.append((fallback_expert, confidence_threshold))
        
        # Normalize weights
        total_score = sum(s for _, s in selected)
        if total_score > 0:
            selected = [(eid, s / total_score) for eid, s in selected]
        
        return selected
    
    def get_status(self) -> Dict[str, any]:
        """Get VRAM manager status."""
        return {
            'tier': self._tier.value,
            'max_experts': self.max_experts,
            'total_vram_mb': self._total_vram_mb,
            'available_vram_mb': self._available_vram_mb,
            'used_vram_mb': self.get_used_vram(),
            'free_vram_mb': self.get_free_vram(),
            'allocations': {
                aid: {
                    'size_mb': a.size_mb,
                    'priority': a.priority,
                }
                for aid, a in self._allocations.items()
            },
        }

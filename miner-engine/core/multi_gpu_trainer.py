"""
Multi-GPU Training Support for R3MES Miner Engine

Supports DataParallel and DistributedDataParallel for training across multiple GPUs.
Includes advanced features:
- Mixed precision training (AMP)
- Gradient accumulation
- Gradient checkpointing for memory optimization
- Multi-node distributed training
- Automatic batch size scaling
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler
from typing import Optional, List, Dict, Callable, Any, Tuple
import logging
import os
import subprocess
import json

from core.trainer import LoRATrainer

logger = logging.getLogger(__name__)


def get_gpu_memory_info(device_id: int) -> Dict[str, float]:
    """Get detailed GPU memory information using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free,utilization.gpu',
             '--format=csv,noheader,nounits', f'--id={device_id}'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            values = result.stdout.strip().split(', ')
            return {
                'total_mb': float(values[0]),
                'used_mb': float(values[1]),
                'free_mb': float(values[2]),
                'utilization_percent': float(values[3]),
            }
    except Exception as e:
        logger.debug(f"nvidia-smi query failed: {e}")
    return {'total_mb': 0, 'used_mb': 0, 'free_mb': 0, 'utilization_percent': 0}


class MultiGPUTrainer(LoRATrainer):
    """
    Multi-GPU trainer extending LoRATrainer.
    
    Supports:
    - DataParallel (single node, multiple GPUs)
    - DistributedDataParallel (multi-node, multi-GPU)
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Gradient checkpointing
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        deterministic: bool = True,
        devices: Optional[List[int]] = None,
        use_ddp: bool = False,
        ddp_backend: str = "nccl",
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        gradient_checkpointing: bool = False,
        find_unused_parameters: bool = True,
        bucket_cap_mb: int = 25,
        **kwargs
    ):
        """
        Initialize multi-GPU trainer.
        
        Args:
            model: Model with BitLinear layers
            learning_rate: Learning rate for LoRA adapters
            deterministic: Enable deterministic operations
            devices: List of GPU device IDs to use (None = all available)
            use_ddp: Use DistributedDataParallel instead of DataParallel
            ddp_backend: DDP backend ('nccl' for CUDA, 'gloo' for CPU)
            mixed_precision: Enable automatic mixed precision (AMP)
            gradient_accumulation_steps: Number of steps to accumulate gradients
            gradient_checkpointing: Enable gradient checkpointing for memory savings
            find_unused_parameters: DDP find_unused_parameters flag
            bucket_cap_mb: DDP gradient bucket size in MB
            **kwargs: Additional arguments for LoRATrainer
        """
        # Detect available GPUs
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available, cannot use MultiGPUTrainer")
        
        available_gpus = torch.cuda.device_count()
        if available_gpus < 2:
            logger.warning(f"Only {available_gpus} GPU(s) available, falling back to single GPU")
            super().__init__(model, learning_rate, deterministic, **kwargs)
            self.use_multi_gpu = False
            self.mixed_precision = mixed_precision
            self.gradient_accumulation_steps = gradient_accumulation_steps
            self.scaler = GradScaler() if mixed_precision else None
            self._accumulation_step = 0
            return
        
        # Select devices
        if devices is None:
            devices = list(range(available_gpus))
        else:
            # Validate device IDs
            devices = [d for d in devices if 0 <= d < available_gpus]
            if len(devices) < 2:
                logger.warning(f"Less than 2 valid devices, falling back to single GPU")
                super().__init__(model, learning_rate, deterministic, **kwargs)
                self.use_multi_gpu = False
                self.mixed_precision = mixed_precision
                self.gradient_accumulation_steps = gradient_accumulation_steps
                self.scaler = GradScaler() if mixed_precision else None
                self._accumulation_step = 0
                return
        
        self.devices = devices
        self.use_ddp = use_ddp
        self.ddp_backend = ddp_backend
        self.use_multi_gpu = True
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_checkpointing = gradient_checkpointing
        self.find_unused_parameters = find_unused_parameters
        self.bucket_cap_mb = bucket_cap_mb
        self._accumulation_step = 0
        
        # Initialize AMP scaler
        self.scaler = GradScaler() if mixed_precision else None
        
        # Initialize base trainer
        super().__init__(model, learning_rate, deterministic, device=torch.device(f'cuda:{devices[0]}'), **kwargs)
        
        # Enable gradient checkpointing if requested
        if gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        # Setup multi-GPU
        if use_ddp:
            self._setup_ddp()
        else:
            self._setup_data_parallel()
        
        logger.info(f"Multi-GPU training enabled: {len(devices)} GPUs (DDP={use_ddp}, AMP={mixed_precision})")
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory optimization."""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        else:
            # Manual checkpointing for custom models
            for module in self.model.modules():
                if hasattr(module, 'gradient_checkpointing'):
                    module.gradient_checkpointing = True
            logger.info("Manual gradient checkpointing enabled")
    
    def _setup_data_parallel(self):
        """Setup DataParallel for single-node multi-GPU."""
        if len(self.devices) > 1:
            self.model = nn.DataParallel(
                self.model,
                device_ids=self.devices,
                output_device=self.devices[0]
            )
            logger.info(f"DataParallel enabled on devices: {self.devices}")
    
    def _setup_ddp(self):
        """Setup DistributedDataParallel for multi-node multi-GPU."""
        # Check if DDP is already initialized
        if not dist.is_initialized():
            # Initialize process group
            dist.init_process_group(
                backend=self.ddp_backend,
                init_method='env://',
            )
        
        # Wrap model with DDP with optimizations
        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.devices[0]],  # DDP uses single device per process
            output_device=self.devices[0],
            find_unused_parameters=self.find_unused_parameters,
            bucket_cap_mb=self.bucket_cap_mb,
            gradient_as_bucket_view=True,  # Memory optimization
        )
        
        logger.info(f"DistributedDataParallel enabled on device: {self.devices[0]}")
    
    def train_step(self, batch: Dict, **kwargs) -> float:
        """
        Training step with multi-GPU support, AMP, and gradient accumulation.
        
        Args:
            batch: Training batch (will be split across GPUs automatically)
            **kwargs: Additional arguments
            
        Returns:
            Average loss across all GPUs
        """
        self._accumulation_step += 1
        
        # Mixed precision context
        with autocast(enabled=self.mixed_precision):
            if not self.use_multi_gpu:
                loss = super().train_step(batch, **kwargs)
            else:
                # DataParallel/DDP automatically handles batch splitting
                loss = self._compute_loss(batch, **kwargs)
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
        
        # Backward pass with AMP scaling
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Only update weights after accumulation steps
        if self._accumulation_step % self.gradient_accumulation_steps == 0:
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
        
        return loss.item() * self.gradient_accumulation_steps
    
    def _compute_loss(self, batch: Dict, **kwargs) -> torch.Tensor:
        """Compute loss for a batch."""
        # Move batch to device
        if isinstance(batch, dict):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(**batch) if isinstance(batch, dict) else self.model(batch)
        
        # Extract loss
        if isinstance(outputs, dict):
            loss = outputs.get('loss', outputs.get('losses', torch.tensor(0.0)))
        elif hasattr(outputs, 'loss'):
            loss = outputs.loss
        else:
            loss = outputs
        
        return loss
    
    def sync_gradients(self):
        """Explicit gradient synchronization for DDP."""
        if self.use_ddp and dist.is_initialized():
            for param in self.model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
    
    def get_gpu_utilization(self) -> Dict[int, Dict[str, float]]:
        """
        Get detailed GPU utilization for all devices.
        
        Returns:
            Dictionary mapping device ID to utilization metrics
        """
        if not self.use_multi_gpu:
            return {0: get_gpu_memory_info(0)}
        
        utilization = {}
        for device_id in self.devices:
            utilization[device_id] = get_gpu_memory_info(device_id)
        
        return utilization
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        stats = {
            'use_multi_gpu': self.use_multi_gpu,
            'use_ddp': self.use_ddp,
            'mixed_precision': self.mixed_precision,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'gradient_checkpointing': self.gradient_checkpointing,
            'devices': self.devices if self.use_multi_gpu else [0],
            'gpu_utilization': self.get_gpu_utilization(),
        }
        
        if self.use_ddp and dist.is_initialized():
            stats['world_size'] = dist.get_world_size()
            stats['rank'] = dist.get_rank()
        
        return stats
    
    def cleanup(self):
        """Cleanup multi-GPU resources."""
        if self.use_ddp and dist.is_initialized():
            dist.destroy_process_group()
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        if hasattr(super(), 'cleanup'):
            super().cleanup()


class DistributedTrainingLauncher:
    """
    Launcher for multi-node distributed training.
    
    Handles:
    - Process spawning across nodes
    - Environment variable setup
    - Distributed sampler creation
    """
    
    def __init__(
        self,
        num_nodes: int = 1,
        gpus_per_node: int = None,
        master_addr: str = "localhost",
        master_port: int = 29500,
        backend: str = "nccl",
    ):
        """
        Initialize distributed training launcher.
        
        Args:
            num_nodes: Number of nodes in the cluster
            gpus_per_node: GPUs per node (None = auto-detect)
            master_addr: Master node address
            master_port: Master node port
            backend: Distributed backend ('nccl' or 'gloo')
        """
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node or torch.cuda.device_count()
        self.master_addr = master_addr
        self.master_port = master_port
        self.backend = backend
        self.world_size = num_nodes * self.gpus_per_node
    
    def setup_environment(self, node_rank: int = 0):
        """Setup environment variables for distributed training."""
        os.environ['MASTER_ADDR'] = self.master_addr
        os.environ['MASTER_PORT'] = str(self.master_port)
        os.environ['WORLD_SIZE'] = str(self.world_size)
        os.environ['NODE_RANK'] = str(node_rank)
    
    def launch(
        self,
        train_fn: Callable,
        model: nn.Module,
        train_dataset,
        node_rank: int = 0,
        **kwargs
    ):
        """
        Launch distributed training.
        
        Args:
            train_fn: Training function to execute
            model: Model to train
            train_dataset: Training dataset
            node_rank: Rank of this node
            **kwargs: Additional arguments for train_fn
        """
        import torch.multiprocessing as mp
        
        self.setup_environment(node_rank)
        
        mp.spawn(
            self._worker,
            args=(train_fn, model, train_dataset, node_rank, kwargs),
            nprocs=self.gpus_per_node,
            join=True
        )
    
    def _worker(
        self,
        local_rank: int,
        train_fn: Callable,
        model: nn.Module,
        train_dataset,
        node_rank: int,
        kwargs: Dict
    ):
        """Worker function for each GPU process."""
        # Calculate global rank
        global_rank = node_rank * self.gpus_per_node + local_rank
        
        # Set environment variables
        os.environ['RANK'] = str(global_rank)
        os.environ['LOCAL_RANK'] = str(local_rank)
        
        # Set device
        torch.cuda.set_device(local_rank)
        
        # Initialize process group
        dist.init_process_group(
            backend=self.backend,
            init_method='env://',
            world_size=self.world_size,
            rank=global_rank
        )
        
        # Create distributed sampler
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=global_rank,
            shuffle=True
        )
        
        # Create trainer
        trainer = MultiGPUTrainer(
            model,
            devices=[local_rank],
            use_ddp=True,
            ddp_backend=self.backend,
            **kwargs
        )
        
        # Run training
        train_fn(trainer, train_dataset, sampler)
        
        # Cleanup
        trainer.cleanup()
    
    def create_distributed_dataloader(
        self,
        dataset,
        batch_size: int,
        rank: int,
        **kwargs
    ) -> DataLoader:
        """Create a DataLoader with DistributedSampler."""
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=rank,
            shuffle=True
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            **kwargs
        )


def create_multi_gpu_trainer(
    model: nn.Module,
    use_all_gpus: bool = True,
    device_ids: Optional[List[int]] = None,
    use_ddp: bool = False,
    **kwargs
) -> MultiGPUTrainer:
    """
    Factory function to create multi-GPU trainer.
    
    Args:
        model: Model to train
        use_all_gpus: Use all available GPUs
        device_ids: Specific GPU IDs to use (overrides use_all_gpus)
        use_ddp: Use DistributedDataParallel
        **kwargs: Additional arguments for MultiGPUTrainer
        
    Returns:
        MultiGPUTrainer instance
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, creating single-GPU trainer")
        return LoRATrainer(model, **kwargs)
    
    available_gpus = torch.cuda.device_count()
    if available_gpus < 2:
        logger.warning("Less than 2 GPUs available, creating single-GPU trainer")
        return LoRATrainer(model, **kwargs)
    
    if device_ids is None and use_all_gpus:
        device_ids = list(range(available_gpus))
    
    return MultiGPUTrainer(
        model,
        devices=device_ids,
        use_ddp=use_ddp,
        **kwargs
    )



def auto_scale_batch_size(
    model: nn.Module,
    initial_batch_size: int = 32,
    max_batch_size: int = 256,
    device_id: int = 0
) -> int:
    """
    Automatically find the optimal batch size for available GPU memory.
    
    Args:
        model: Model to test
        initial_batch_size: Starting batch size
        max_batch_size: Maximum batch size to try
        device_id: GPU device ID
        
    Returns:
        Optimal batch size
    """
    device = torch.device(f'cuda:{device_id}')
    model = model.to(device)
    
    batch_size = initial_batch_size
    
    while batch_size <= max_batch_size:
        try:
            # Create dummy input
            dummy_input = torch.randn(batch_size, 512, device=device)
            
            # Forward pass
            with torch.no_grad():
                _ = model(dummy_input)
            
            # Clear cache
            torch.cuda.empty_cache()
            
            # Try larger batch size
            batch_size *= 2
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                optimal_batch_size = batch_size // 2
                logger.info(f"Auto-scaled batch size: {optimal_batch_size}")
                return optimal_batch_size
            raise
    
    return max_batch_size


def launch_distributed_training(
    train_fn: Callable,
    model: nn.Module,
    train_dataset,
    num_nodes: int = 1,
    gpus_per_node: int = None,
    master_addr: str = "localhost",
    master_port: int = 29500,
    **kwargs
):
    """
    Convenience function to launch distributed training.
    
    Args:
        train_fn: Training function (receives trainer, dataset, sampler)
        model: Model to train
        train_dataset: Training dataset
        num_nodes: Number of nodes
        gpus_per_node: GPUs per node (None = auto-detect)
        master_addr: Master node address
        master_port: Master node port
        **kwargs: Additional arguments for MultiGPUTrainer
    """
    launcher = DistributedTrainingLauncher(
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        master_addr=master_addr,
        master_port=master_port,
    )
    
    launcher.launch(train_fn, model, train_dataset, **kwargs)

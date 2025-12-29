"""
Task Pool Client for Open Buffet / Asenkron İş Havuzu

Implements prefetching and task claiming for variable-speed mining.
RTX 4090 can claim 20 tasks while GTX 1650 claims 1 task.

FAZ 5 Integration: Blind delivery support for trap jobs.
"""

from typing import List, Optional, Dict, Any
import os
import time
import logging
import json
import pickle
import torch

logger = logging.getLogger(__name__)


class TaskPoolClient:
    """
    Client for interacting with Task Pool system.
    
    Features:
    - Prefetching: Claim multiple tasks ahead of time
    - Variable Speed: Fast GPUs can claim more tasks
    - Async Processing: Process tasks independently of block time
    - Blind Delivery: Trap jobs mixed with real tasks (FAZ 5)
    """
    
    def __init__(self, grpc_client, max_prefetch: int = 5):
        """
        Initialize Task Pool Client.
        
        Args:
            grpc_client: gRPC client for blockchain communication
            max_prefetch: Maximum number of tasks to prefetch (default: 5)
        """
        self.grpc_client = grpc_client
        self.prefetch_queue: List[Dict[str, Any]] = []
        self.max_prefetch = max_prefetch
        self.current_pool_id: Optional[int] = None
        self.claimed_tasks: Dict[int, Dict[str, Any]] = {}  # chunk_id -> task info
        
        # Blind delivery support (FAZ 5)
        self._trap_mapping: Dict[int, str] = {}  # chunk_id -> vault_entry_id
        self._blind_delivery_enabled = os.getenv("R3MES_BLIND_DELIVERY", "true").lower() == "true"
    
    def get_active_pool(self) -> Optional[int]:
        """
        Get the active task pool ID from blockchain.
        
        Returns:
            Pool ID if available, None otherwise
        """
        try:
            if self.grpc_client and hasattr(self.grpc_client, 'get_active_pool'):
                # Query blockchain for active pool using gRPC query
                active_pool_id = self.grpc_client.get_active_pool()
                if active_pool_id:
                    # Update current pool ID cache
                    self.current_pool_id = active_pool_id
                    return active_pool_id
            
            # Fallback: use current pool ID if set
            if self.current_pool_id:
                return self.current_pool_id
            
            # No active pool found
            return None
        except Exception as e:
            logger.error(f"Failed to get active pool: {e}", exc_info=True)
            # Return cached pool ID as fallback
            return self.current_pool_id
    
    def claim_and_prefetch_tasks(self, pool_id: int, count: int = 1) -> List[Dict[str, Any]]:
        """
        Claim current task(s) and prefetch next tasks.
        
        Args:
            pool_id: Task pool ID
            count: Number of tasks to claim (1 for slow GPUs, 20 for fast GPUs)
            
        Returns:
            List of claimed tasks
        """
        claimed_tasks = []
        
        # Claim tasks
        for _ in range(count):
            task = self._claim_task(pool_id)
            if task:
                claimed_tasks.append(task)
                self.claimed_tasks[task["chunk_id"]] = task
        
        # Prefetch additional tasks
        for _ in range(self.max_prefetch):
            task = self._claim_task(pool_id)
            if task:
                self.prefetch_queue.append(task)
        
        return claimed_tasks
    
    def _claim_task(self, pool_id: int) -> Optional[Dict[str, Any]]:
        """
        Claim a single task from the pool.
        
        Args:
            pool_id: Task pool ID
            
        Returns:
            Task info if available, None otherwise
        """
        try:
            if self.grpc_client and hasattr(self.grpc_client, 'send_claim_task'):
                # Use gRPC transaction to claim task
                # Get available chunks first
                claim_limit = int(os.getenv("R3MES_TASK_POOL_CLAIM_LIMIT", "1"))
                available_chunks = self.grpc_client.get_available_chunks(pool_id, limit=claim_limit)
                if not available_chunks:
                    return None
                
                chunk = available_chunks[0]
                
                # Handle both dict and list return types from get_available_chunks
                if isinstance(chunk, dict):
                    chunk_id = chunk.get("chunk_id")
                    data_hash = chunk.get("data_hash")
                    shard_id = chunk.get("shard_id")
                else:
                    # If it's a proto object, use attributes
                    chunk_id = getattr(chunk, 'chunk_id', None)
                    data_hash = getattr(chunk, 'data_hash', '')
                    shard_id = getattr(chunk, 'shard_id', 0)
                
                if chunk_id is None:
                    logger.warning("Invalid chunk data: missing chunk_id")
                    return None
                
                # Send ClaimTask transaction
                miner_address = self.grpc_client.get_miner_address()
                result = self.grpc_client.send_claim_task(
                    miner=miner_address,
                    pool_id=pool_id,
                    chunk_id=chunk_id
                )
                
                # Handle both bool (old) and dict (new) return types
                if isinstance(result, dict):
                    success = result.get("success", False)
                else:
                    success = bool(result)
                
                if success:
                    return {
                        "chunk_id": chunk_id,
                        "data_hash": data_hash,
                        "shard_id": shard_id,
                        "pool_id": pool_id,
                    }
                else:
                    error_msg = result.get("error", "Unknown error") if isinstance(result, dict) else "Failed to claim task"
                    logger.warning(f"Failed to claim task chunk {chunk_id}: {error_msg}")
                    return None
            else:
                # gRPC client not available, return None
                logger.warning("gRPC client not available for task claiming")
                return None
        except Exception as e:
            logger.error(f"Failed to claim task: {e}")
            return None
    
    def get_next_task(self) -> Optional[Dict[str, Any]]:
        """
        Get next task from prefetch queue or claim new one.
        
        Returns:
            Task info if available, None otherwise
        """
        if self.prefetch_queue:
            return self.prefetch_queue.pop(0)
        
        if self.current_pool_id:
            tasks = self.claim_and_prefetch_tasks(self.current_pool_id, count=1)
            if tasks:
                return tasks[0]
        
        return None
    
    def complete_task(self, chunk_id: int, gradient_hash: str) -> bool:
        """
        Mark a task as completed.
        
        Args:
            chunk_id: Chunk ID of completed task
            gradient_hash: IPFS hash of gradient result
            
        Returns:
            True if successful, False otherwise
        """
        if chunk_id not in self.claimed_tasks:
            logger.warning(f"Task {chunk_id} not found in claimed tasks")
            return False
        
        task = self.claimed_tasks[chunk_id]
        pool_id = task["pool_id"]
        
        try:
            if self.grpc_client and hasattr(self.grpc_client, 'send_complete_task'):
                # Use gRPC transaction to complete task
                miner_address = self.grpc_client.get_miner_address()
                result = self.grpc_client.send_complete_task(
                    miner=miner_address,
                    pool_id=pool_id,
                    chunk_id=chunk_id,
                    gradient_hash=gradient_hash
                )
                
                # Handle both bool (old) and dict (new) return types
                if isinstance(result, dict):
                    success = result.get("success", False)
                else:
                    success = bool(result)
                
                if success:
                    del self.claimed_tasks[chunk_id]
                    tx_hash = result.get("tx_hash", "unknown") if isinstance(result, dict) else None
                    logger.info(f"Task {chunk_id} completed (gradient: {gradient_hash[:16]}..., tx: {tx_hash})")
                    return True
                else:
                    error_msg = result.get("error", "Unknown error") if isinstance(result, dict) else "Failed to complete task"
                    logger.error(f"Failed to complete task {chunk_id}: {error_msg}")
                    return False
            else:
                # gRPC client not available, just remove from local cache
                logger.warning("gRPC client not available, removing task from local cache only")
                del self.claimed_tasks[chunk_id]
                return True
        except Exception as e:
            logger.error(f"Failed to complete task {chunk_id}: {e}")
            return False
    
    def get_available_tasks_count(self, pool_id: int) -> int:
        """
        Get count of available tasks in pool.
        
        Args:
            pool_id: Task pool ID
            
        Returns:
            Number of available tasks
        """
        try:
            if self.grpc_client and hasattr(self.grpc_client, 'get_available_chunks'):
                # Query blockchain for available chunks
                query_limit = int(os.getenv("R3MES_TASK_POOL_QUERY_LIMIT", "1000"))
                available_chunks = self.grpc_client.get_available_chunks(pool_id, limit=query_limit)
                return len(available_chunks) if available_chunks else 0
            else:
                # gRPC client not available
                logger.warning("gRPC client not available for available tasks count")
                return 0
        except Exception as e:
            logger.error(f"Failed to get available tasks count: {e}")
            return 0
    
    def set_pool_id(self, pool_id: int):
        """Set the current active pool ID."""
        self.current_pool_id = pool_id
    
    def download_chunk_data(self, data_hash: str, ipfs_client) -> Optional[Dict[str, Any]]:
        """
        Download chunk data from IPFS using data_hash.
        
        Args:
            data_hash: IPFS hash of chunk data (or data hash that maps to IPFS)
            ipfs_client: IPFSClient instance for downloading
            
        Returns:
            Chunk data dictionary with 'input_ids' and 'labels', or None if failed
        """
        if not ipfs_client:
            logger.error("IPFS client not provided for chunk data download")
            return None
        
        if not ipfs_client.is_connected():
            logger.error("IPFS client not connected - cannot download chunk data")
            return None
        
        try:
            logger.info(f"Downloading chunk data from IPFS: {data_hash}")
            
            # Retrieve content from IPFS
            # data_hash is the IPFS CID
            content = ipfs_client.retrieve_content(data_hash)
            
            if content is None:
                logger.error(f"Failed to retrieve chunk data from IPFS: {data_hash}")
                return None
            
            # Try to parse as JSON first (structured data)
            try:
                chunk_data = json.loads(content)
                logger.info("Chunk data parsed as JSON")
                
                # Convert to torch tensors if needed
                if "input_ids" in chunk_data:
                    if isinstance(chunk_data["input_ids"], list):
                        chunk_data["input_ids"] = torch.tensor(chunk_data["input_ids"], dtype=torch.long)
                    elif not isinstance(chunk_data["input_ids"], torch.Tensor):
                        # Try to convert numpy array
                        import numpy as np
                        if isinstance(chunk_data["input_ids"], np.ndarray):
                            chunk_data["input_ids"] = torch.from_numpy(chunk_data["input_ids"]).long()
                
                if "labels" in chunk_data:
                    if isinstance(chunk_data["labels"], list):
                        chunk_data["labels"] = torch.tensor(chunk_data["labels"], dtype=torch.long)
                    elif not isinstance(chunk_data["labels"], torch.Tensor):
                        import numpy as np
                        if isinstance(chunk_data["labels"], np.ndarray):
                            chunk_data["labels"] = torch.from_numpy(chunk_data["labels"]).long()
                else:
                    # Default labels to input_ids if not provided
                    chunk_data["labels"] = chunk_data.get("input_ids", chunk_data["input_ids"])
                
                logger.info(f"Successfully downloaded and parsed chunk data from IPFS")
                return chunk_data
                
            except json.JSONDecodeError:
                # Try pickle deserialization (for binary data)
                try:
                    chunk_data = pickle.loads(content)
                    logger.info("Chunk data parsed as pickle")
                    
                    # Ensure tensors are torch tensors
                    if isinstance(chunk_data, dict):
                        if "input_ids" in chunk_data and not isinstance(chunk_data["input_ids"], torch.Tensor):
                            chunk_data["input_ids"] = torch.tensor(chunk_data["input_ids"], dtype=torch.long)
                        if "labels" in chunk_data and not isinstance(chunk_data["labels"], torch.Tensor):
                            chunk_data["labels"] = torch.tensor(chunk_data["labels"], dtype=torch.long)
                        elif "labels" not in chunk_data:
                            chunk_data["labels"] = chunk_data.get("input_ids")
                    
                    logger.info(f"Successfully downloaded and parsed chunk data from IPFS (pickle)")
                    return chunk_data
                except Exception as e:
                    logger.error(f"Failed to parse chunk data (neither JSON nor pickle): {e}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error downloading chunk data from IPFS: {e}", exc_info=True)
            return None
    
    # =========================================================================
    # BLIND DELIVERY SUPPORT (FAZ 5)
    # =========================================================================
    
    def get_tasks_with_blind_delivery(
        self,
        pool_id: int,
        count: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Get tasks with blind trap injection.
        
        Tasks are returned with trap jobs mixed in. The miner cannot
        distinguish trap jobs from real tasks.
        
        Args:
            pool_id: Task pool ID
            count: Number of tasks to claim
        
        Returns:
            List of tasks (may include hidden traps)
        """
        if not self._blind_delivery_enabled:
            return self.claim_and_prefetch_tasks(pool_id, count)
        
        try:
            from core.trap_jobs import BlindDeliveryMixer, GenesisVaultManager
            from core.types import TaskChunk, TaskStatus
            
            # Load Genesis Vault
            vault_path = os.getenv("R3MES_GENESIS_VAULT_PATH", "genesis_vault_entries.json")
            if not os.path.exists(vault_path):
                logger.debug("Genesis Vault not found, using standard task claiming")
                return self.claim_and_prefetch_tasks(pool_id, count)
            
            vault_manager = GenesisVaultManager(vault_path)
            mixer = BlindDeliveryMixer(vault_manager)
            
            # Get real tasks
            real_tasks = self.claim_and_prefetch_tasks(pool_id, count)
            if not real_tasks:
                return []
            
            # Convert to TaskChunk objects
            real_chunks = []
            for task in real_tasks:
                chunk = TaskChunk(
                    chunk_id=task['chunk_id'],
                    pool_id=task['pool_id'],
                    data_hash=task['data_hash'],
                    shard_id=task.get('shard_id', 0),
                    status=TaskStatus.CLAIMED,
                )
                real_chunks.append(chunk)
            
            # Mix with traps
            batch = mixer.mix_chunks(real_chunks, pool_id)
            
            # Store trap mapping for later verification
            for trap_chunk in batch.trap_chunks:
                self._trap_mapping[trap_chunk.chunk_id] = f"trap_{abs(trap_chunk.chunk_id)}"
            
            # Convert back to task dicts (sanitized - no is_trap flag)
            mixed_tasks = []
            for chunk in batch.mixed_chunks:
                task = {
                    'chunk_id': chunk.chunk_id,
                    'pool_id': chunk.pool_id,
                    'data_hash': chunk.data_hash,
                    'shard_id': chunk.shard_id,
                }
                # Add trap entry ID if this is a trap (for internal use only)
                if chunk.chunk_id in self._trap_mapping:
                    task['trap_entry_id'] = self._trap_mapping[chunk.chunk_id]
                mixed_tasks.append(task)
            
            logger.debug(f"Blind delivery: {len(real_tasks)} real + {len(batch.trap_chunks)} traps")
            return mixed_tasks
            
        except ImportError:
            logger.debug("Trap job modules not available, using standard task claiming")
            return self.claim_and_prefetch_tasks(pool_id, count)
        except Exception as e:
            logger.warning(f"Blind delivery failed, using standard claiming: {e}")
            return self.claim_and_prefetch_tasks(pool_id, count)
    
    def is_trap_task(self, chunk_id: int) -> bool:
        """
        Check if a task is a trap job.
        
        Note: This should only be used internally for verification,
        not exposed to the miner logic.
        
        Args:
            chunk_id: Chunk ID to check
        
        Returns:
            True if this is a trap task
        """
        return chunk_id < 0 or chunk_id in self._trap_mapping
    
    def get_trap_entry_id(self, chunk_id: int) -> Optional[str]:
        """
        Get the vault entry ID for a trap task.
        
        Args:
            chunk_id: Chunk ID
        
        Returns:
            Vault entry ID if trap, None otherwise
        """
        return self._trap_mapping.get(chunk_id)


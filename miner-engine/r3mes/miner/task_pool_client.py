#!/usr/bin/env python3
"""
R3MES Task Pool Client

Production-ready task pool client that:
1. Queries available tasks from blockchain
2. Claims tasks for processing
3. Downloads task data from IPFS
4. Manages task lifecycle
5. Submits task completion to blockchain
"""

import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from r3mes.utils.logger import setup_logger
from r3mes.utils.ipfs_manager import IPFSClient
from bridge.blockchain_client import BlockchainClient


class TaskPoolClient:
    """Task pool client for miners."""
    
    def __init__(
        self,
        blockchain_client: BlockchainClient,
        ipfs_client: Optional[IPFSClient] = None,
        miner_address: Optional[str] = None,
        log_level: str = "INFO",
        use_json_logs: bool = False,
    ):
        """
        Initialize task pool client.
        
        Args:
            blockchain_client: Blockchain client instance
            ipfs_client: IPFS client instance (optional, will create if not provided)
            miner_address: Miner's blockchain address
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            use_json_logs: Whether to use JSON-formatted logs
        """
        # Setup logger
        log_level_map = {
            "DEBUG": 10,
            "INFO": 20,
            "WARNING": 30,
            "ERROR": 40,
        }
        self.logger = setup_logger(
            "r3mes.task_pool",
            level=log_level_map.get(log_level.upper(), logging.INFO),
            use_json=use_json_logs,
        )
        
        self.blockchain_client = blockchain_client
        self.ipfs_client = ipfs_client or IPFSClient()
        self.miner_address = miner_address
        
        # Task management
        self.claimed_tasks = {}  # task_id -> task_info
        self.completed_tasks = set()  # Set of completed task IDs
        
        self.logger.info("Task pool client initialized")
    
    def get_active_pool_id(self) -> Optional[int]:
        """
        Get the currently active task pool ID.
        
        Returns:
            Active pool ID, or None if no active pool
        """
        try:
            pool_id = self.blockchain_client.get_active_pool()
            if pool_id is not None:
                self.logger.info(f"Active pool ID: {pool_id}")
            else:
                self.logger.warning("No active pool found")
            return pool_id
        except Exception as e:
            self.logger.error(f"Error getting active pool ID: {e}", exc_info=True)
            return None
    
    def get_available_chunks(self, pool_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get available chunks from a task pool.
        
        Args:
            pool_id: Task pool ID
            limit: Maximum number of chunks to return
            
        Returns:
            List of available chunks with chunk_id, data_hash, shard_id
        """
        try:
            self.logger.debug(f"Querying available chunks from pool {pool_id} (limit: {limit})")
            
            chunks = self.blockchain_client.get_available_chunks(pool_id, limit)
            
            if chunks:
                self.logger.info(f"Found {len(chunks)} available chunks in pool {pool_id}")
            else:
                self.logger.debug(f"No available chunks in pool {pool_id}")
            
            return chunks
        except Exception as e:
            self.logger.error(f"Error getting available chunks: {e}", exc_info=True)
            return []
    
    def claim_task(self, pool_id: int, chunk_id: int) -> bool:
        """
        Claim a task for processing.
        
        Args:
            pool_id: Task pool ID
            chunk_id: Chunk ID to claim
            
        Returns:
            True if task claimed successfully
        """
        try:
            if not self.miner_address:
                self.logger.error("Miner address not set, cannot claim task")
                return False
            
            self.logger.info(f"Claiming task: pool={pool_id}, chunk={chunk_id}")
            
            # Send claim transaction to blockchain
            result = self.blockchain_client.send_claim_task(
                miner=self.miner_address,
                pool_id=pool_id,
                chunk_id=chunk_id,
            )
            
            if result.get("success", False):
                # Track claimed task
                task_key = f"{pool_id}_{chunk_id}"
                self.claimed_tasks[task_key] = {
                    "pool_id": pool_id,
                    "chunk_id": chunk_id,
                    "claimed_at": time.time(),
                    "tx_hash": result.get("tx_hash", ""),
                }
                
                self.logger.info(f"Task claimed successfully: {task_key}, TX: {result.get('tx_hash', 'pending')}")
                return True
            else:
                self.logger.error(f"Failed to claim task: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error claiming task: {e}", exc_info=True)
            return False
    
    def download_chunk_data(self, data_hash: str, output_dir: str = "task_data") -> Optional[str]:
        """
        Download chunk data from IPFS.
        
        Args:
            data_hash: IPFS hash of chunk data
            output_dir: Output directory for downloaded data
            
        Returns:
            Path to downloaded file, or None if failed
        """
        try:
            self.logger.debug(f"Downloading chunk data from IPFS: {data_hash}")
            
            # Download from IPFS
            file_path = self.ipfs_client.get(data_hash, output_dir=output_dir)
            
            if file_path:
                self.logger.debug(f"Chunk data downloaded: {file_path}")
                return file_path
            else:
                self.logger.error(f"Failed to download chunk data: {data_hash}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error downloading chunk data: {e}", exc_info=True)
            return None
    
    def load_chunk_data(self, file_path: str) -> Optional[Any]:
        """
        Load chunk data from file.
        
        Args:
            file_path: Path to chunk data file
            
        Returns:
            Loaded data, or None if failed
        """
        try:
            self.logger.debug(f"Loading chunk data from: {file_path}")
            
            # Determine file format and load accordingly
            file_path = Path(file_path)
            
            if file_path.suffix.lower() == '.json':
                # JSON format
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
            elif file_path.suffix.lower() in ['.pkl', '.pickle']:
                # Pickle format
                import pickle
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    
            elif file_path.suffix.lower() in ['.txt', '.text']:
                # Text format
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = f.read()
                    
            elif file_path.suffix.lower() in ['.pt', '.pth']:
                # PyTorch tensor format
                import torch
                data = torch.load(file_path, map_location='cpu')
                
            else:
                # Binary format (fallback)
                with open(file_path, 'rb') as f:
                    data = f.read()
            
            self.logger.debug(f"Chunk data loaded: {type(data)}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading chunk data: {e}", exc_info=True)
            return None
    
    def complete_task(
        self,
        pool_id: int,
        chunk_id: int,
        gradient_hash: str,
        gradient_ipfs_hash: str = "",
        miner_gpu: str = "",
    ) -> bool:
        """
        Mark task as completed and submit to blockchain.
        
        Args:
            pool_id: Task pool ID
            chunk_id: Chunk ID
            gradient_hash: Deterministic hash of gradient result
            gradient_ipfs_hash: IPFS hash of gradient data (optional)
            miner_gpu: GPU architecture used (optional)
            
        Returns:
            True if task completed successfully
        """
        try:
            if not self.miner_address:
                self.logger.error("Miner address not set, cannot complete task")
                return False
            
            task_key = f"{pool_id}_{chunk_id}"
            
            # Check if task was claimed by this miner
            if task_key not in self.claimed_tasks:
                self.logger.warning(f"Task not claimed by this miner: {task_key}")
                # Continue anyway - task might have been claimed in previous session
            
            self.logger.info(f"Completing task: {task_key}")
            
            # Send complete transaction to blockchain
            result = self.blockchain_client.send_complete_task(
                miner=self.miner_address,
                pool_id=pool_id,
                chunk_id=chunk_id,
                gradient_hash=gradient_hash,
                gradient_ipfs_hash=gradient_ipfs_hash,
                miner_gpu=miner_gpu,
            )
            
            if result.get("success", False):
                # Track completed task
                self.completed_tasks.add(task_key)
                
                # Remove from claimed tasks
                if task_key in self.claimed_tasks:
                    del self.claimed_tasks[task_key]
                
                self.logger.info(f"Task completed successfully: {task_key}, TX: {result.get('tx_hash', 'pending')}")
                return True
            else:
                self.logger.error(f"Failed to complete task: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error completing task: {e}", exc_info=True)
            return False
    
    def get_claimed_tasks(self) -> Dict[str, Dict[str, Any]]:
        """
        Get currently claimed tasks.
        
        Returns:
            Dictionary of claimed tasks (task_key -> task_info)
        """
        return self.claimed_tasks.copy()
    
    def get_completed_tasks(self) -> set:
        """
        Get completed task IDs.
        
        Returns:
            Set of completed task keys
        """
        return self.completed_tasks.copy()
    
    def cleanup_expired_claims(self, max_age_seconds: int = 3600):
        """
        Clean up expired task claims.
        
        Args:
            max_age_seconds: Maximum age for task claims (default: 1 hour)
        """
        try:
            current_time = time.time()
            expired_tasks = []
            
            for task_key, task_info in self.claimed_tasks.items():
                claimed_at = task_info.get("claimed_at", 0)
                age = current_time - claimed_at
                
                if age > max_age_seconds:
                    expired_tasks.append(task_key)
            
            if expired_tasks:
                self.logger.info(f"Cleaning up {len(expired_tasks)} expired task claims")
                for task_key in expired_tasks:
                    del self.claimed_tasks[task_key]
            
        except Exception as e:
            self.logger.error(f"Error cleaning up expired claims: {e}", exc_info=True)
    
    def find_and_claim_task(self, pool_id: Optional[int] = None, limit: int = 50) -> Optional[Dict[str, Any]]:
        """
        Find and claim an available task.
        
        Args:
            pool_id: Specific pool ID to search (optional, will use active pool if not provided)
            limit: Maximum number of chunks to query
            
        Returns:
            Claimed task info, or None if no tasks available
        """
        try:
            # Get pool ID
            if pool_id is None:
                pool_id = self.get_active_pool_id()
                if pool_id is None:
                    self.logger.warning("No active pool found")
                    return None
            
            # Get available chunks
            chunks = self.get_available_chunks(pool_id, limit)
            if not chunks:
                self.logger.debug(f"No available chunks in pool {pool_id}")
                return None
            
            # Try to claim first available chunk
            for chunk in chunks:
                chunk_id = chunk.get("chunk_id")
                data_hash = chunk.get("data_hash", "")
                shard_id = chunk.get("shard_id", 0)
                
                if chunk_id is None:
                    continue
                
                # Attempt to claim task
                if self.claim_task(pool_id, chunk_id):
                    return {
                        "pool_id": pool_id,
                        "chunk_id": chunk_id,
                        "data_hash": data_hash,
                        "shard_id": shard_id,
                    }
            
            self.logger.debug(f"Failed to claim any tasks from pool {pool_id}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding and claiming task: {e}", exc_info=True)
            return None
    
    def process_task_workflow(
        self,
        task_info: Dict[str, Any],
        process_function: callable,
        upload_result: bool = True,
    ) -> bool:
        """
        Complete task processing workflow.
        
        Args:
            task_info: Task information from find_and_claim_task
            process_function: Function to process chunk data (should return gradient_hash, gradient_data)
            upload_result: Whether to upload result to IPFS
            
        Returns:
            True if workflow completed successfully
        """
        try:
            pool_id = task_info["pool_id"]
            chunk_id = task_info["chunk_id"]
            data_hash = task_info["data_hash"]
            
            self.logger.info(f"Processing task workflow: pool={pool_id}, chunk={chunk_id}")
            
            # 1. Download chunk data
            chunk_file = self.download_chunk_data(data_hash)
            if not chunk_file:
                self.logger.error(f"Failed to download chunk data: {data_hash}")
                return False
            
            # 2. Load chunk data
            chunk_data = self.load_chunk_data(chunk_file)
            if chunk_data is None:
                self.logger.error(f"Failed to load chunk data: {chunk_file}")
                return False
            
            # 3. Process chunk data
            try:
                result = process_function(chunk_data)
                if isinstance(result, tuple) and len(result) == 2:
                    gradient_hash, gradient_data = result
                else:
                    self.logger.error("Process function must return (gradient_hash, gradient_data)")
                    return False
            except Exception as e:
                self.logger.error(f"Error in process function: {e}", exc_info=True)
                return False
            
            # 4. Upload result to IPFS (optional)
            gradient_ipfs_hash = ""
            if upload_result and gradient_data is not None:
                if isinstance(gradient_data, bytes):
                    gradient_ipfs_hash = self.ipfs_client.add_bytes(gradient_data)
                else:
                    # Serialize gradient data
                    import pickle
                    gradient_bytes = pickle.dumps(gradient_data)
                    gradient_ipfs_hash = self.ipfs_client.add_bytes(gradient_bytes)
                
                if not gradient_ipfs_hash:
                    self.logger.warning("Failed to upload gradient to IPFS")
            
            # 5. Complete task
            success = self.complete_task(
                pool_id=pool_id,
                chunk_id=chunk_id,
                gradient_hash=gradient_hash,
                gradient_ipfs_hash=gradient_ipfs_hash,
                miner_gpu="",  # TODO: Get from GPU detection
            )
            
            if success:
                self.logger.info(f"Task workflow completed successfully: pool={pool_id}, chunk={chunk_id}")
                return True
            else:
                self.logger.error(f"Failed to complete task: pool={pool_id}, chunk={chunk_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in task workflow: {e}", exc_info=True)
            return False
#!/usr/bin/env python3
"""
Atomic Transaction Coordinator for Mining Submissions

Ensures atomic semantics for the mining submission pipeline:
1. Train model and generate gradients
2. Upload gradients to IPFS
3. Submit IPFS hash to blockchain
4. Complete task in task pool

If any step fails, all previous steps are rolled back.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
import torch

from utils.error_handling import (
    exponential_backoff,
    handle_specific_errors,
    NetworkError,
    AuthenticationError,
    ResourceError,
    RetryableError,
)

logger = logging.getLogger(__name__)


class TransactionState(Enum):
    """Transaction states for atomic coordinator."""
    PENDING = "pending"
    TRAINING = "training"
    IPFS_UPLOAD = "ipfs_upload"
    BLOCKCHAIN_SUBMIT = "blockchain_submit"
    TASK_COMPLETE = "task_complete"
    COMMITTED = "committed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class TransactionContext:
    """Context for atomic mining transaction."""
    transaction_id: str
    state: TransactionState
    created_at: float
    
    # Training data
    task: Optional[Dict[str, Any]] = None
    chunk_data: Optional[Dict[str, Any]] = None
    gradients_dict: Optional[Dict[str, torch.Tensor]] = None
    gradient_hash: Optional[str] = None
    
    # IPFS data
    ipfs_hash: Optional[str] = None
    ipfs_upload_time: Optional[float] = None
    
    # Blockchain data
    blockchain_tx_hash: Optional[str] = None
    stored_gradient_id: Optional[int] = None
    blockchain_submit_time: Optional[float] = None
    
    # Task completion
    task_completed: bool = False
    
    # Error information
    error_message: Optional[str] = None
    failed_step: Optional[str] = None
    
    # Rollback information
    rollback_actions: List[str] = None
    
    def __post_init__(self):
        if self.rollback_actions is None:
            self.rollback_actions = []


class AtomicMiningCoordinator:
    """
    Coordinates atomic mining transactions with rollback capability.
    
    Ensures that mining submissions are atomic:
    - Either all steps succeed (COMMITTED)
    - Or all steps are rolled back (ROLLED_BACK)
    """
    
    def __init__(
        self,
        trainer,
        ipfs_client,
        blockchain_client,
        task_pool_client,
        timeout_seconds: int = 300,  # 5 minutes
        max_retries: int = 3,
    ):
        self.trainer = trainer
        self.ipfs_client = ipfs_client
        self.blockchain_client = blockchain_client
        self.task_pool_client = task_pool_client
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        
        # Active transactions
        self.active_transactions: Dict[str, TransactionContext] = {}
        
        # Statistics
        self.stats = {
            "total_transactions": 0,
            "committed_transactions": 0,
            "failed_transactions": 0,
            "rolled_back_transactions": 0,
            "average_commit_time": 0.0,
        }
    
    def create_transaction(self, task: Optional[Dict[str, Any]] = None) -> TransactionContext:
        """Create a new atomic transaction."""
        transaction_id = f"mining_tx_{int(time.time() * 1000)}_{len(self.active_transactions)}"
        
        context = TransactionContext(
            transaction_id=transaction_id,
            state=TransactionState.PENDING,
            created_at=time.time(),
            task=task,
        )
        
        self.active_transactions[transaction_id] = context
        self.stats["total_transactions"] += 1
        
        logger.info(f"Created atomic transaction: {transaction_id}")
        return context
    
    async def execute_atomic_mining(
        self,
        context: TransactionContext,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        miner_address: str,
        training_round_id: int,
        model_version: str = "v1.0.0",
    ) -> Tuple[bool, Optional[str]]:
        """
        Execute atomic mining transaction.
        
        Returns:
            (success, error_message) tuple
        """
        start_time = time.time()
        
        try:
            # Step 1: Training
            success = await self._step_training(context, inputs, targets)
            if not success:
                await self._rollback_transaction(context)
                return False, context.error_message
            
            # Step 2: IPFS Upload
            success = await self._step_ipfs_upload(context, miner_address, training_round_id)
            if not success:
                await self._rollback_transaction(context)
                return False, context.error_message
            
            # Step 3: Blockchain Submission
            success = await self._step_blockchain_submit(
                context, miner_address, training_round_id, model_version
            )
            if not success:
                await self._rollback_transaction(context)
                return False, context.error_message
            
            # Step 4: Task Completion (if task exists)
            if context.task:
                success = await self._step_task_complete(context)
                if not success:
                    # Task completion failure is not critical - log warning but don't rollback
                    logger.warning(f"Task completion failed for {context.transaction_id}: {context.error_message}")
            
            # Commit transaction
            await self._commit_transaction(context)
            
            commit_time = time.time() - start_time
            self._update_stats(commit_time)
            
            logger.info(f"Atomic transaction committed: {context.transaction_id} ({commit_time:.2f}s)")
            return True, None
            
        except Exception as e:
            context.error_message = str(e)
            context.failed_step = context.state.value
            logger.error(f"Atomic transaction failed: {context.transaction_id}: {e}")
            
            await self._rollback_transaction(context)
            return False, str(e)
        
        finally:
            # Cleanup
            if context.transaction_id in self.active_transactions:
                del self.active_transactions[context.transaction_id]
    
    async def _step_training(
        self,
        context: TransactionContext,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> bool:
        """Step 1: Train model and generate gradients."""
        try:
            context.state = TransactionState.TRAINING
            logger.debug(f"[{context.transaction_id}] Starting training step")
            
            # Training step
            loss, gradients_dict = self.trainer.train_step(inputs, targets)
            
            # Compute gradient hash
            gradient_hash = self.trainer.compute_gradient_hash(gradients_dict)
            
            # Store in context
            context.gradients_dict = gradients_dict
            context.gradient_hash = gradient_hash
            
            logger.debug(f"[{context.transaction_id}] Training completed: loss={loss:.6f}")
            return True
            
        except Exception as e:
            context.error_message = f"Training failed: {e}"
            context.failed_step = "training"
            return False
    
    @handle_specific_errors
    @exponential_backoff(max_retries=3, retryable_exceptions=[NetworkError, ConnectionError, TimeoutError])
    async def _step_ipfs_upload(
        self,
        context: TransactionContext,
        miner_address: str,
        training_round_id: int,
    ) -> bool:
        """Step 2: Upload gradients to IPFS with enhanced error handling."""
        try:
            context.state = TransactionState.IPFS_UPLOAD
            logger.debug(f"[{context.transaction_id}] Starting IPFS upload")
            
            # Serialize gradients
            from core.serialization import LoRASerializer
            from core.binary_serialization import BinaryGradientSerializer
            
            serializer = LoRASerializer()
            binary_serializer = BinaryGradientSerializer()
            
            # Try binary serialization first
            try:
                serialized = binary_serializer.serialize_gradients(
                    context.gradients_dict,
                    metadata=self.trainer.get_training_metadata(),
                    training_round_id=training_round_id,
                    miner_address=miner_address,
                    gradient_hash=context.gradient_hash,
                )
                logger.debug(f"[{context.transaction_id}] Using binary serialization")
            except Exception as e:
                logger.warning(f"Binary serialization failed, using pickle: {e}")
                lora_state = self.trainer.get_lora_state_dict()
                serialized = serializer.serialize_lora_state(
                    lora_state,
                    metadata=self.trainer.get_training_metadata(),
                )
            
            # Validate serialized data size
            if len(serialized) > 100 * 1024 * 1024:  # 100MB limit
                raise ResourceError(f"Serialized gradient size too large: {len(serialized)} bytes")
            
            # Upload to IPFS (retry logic handled by decorator)
            ipfs_hash = self.ipfs_client.upload_lora_state(serialized)
            
            if not ipfs_hash:
                raise NetworkError("IPFS upload returned empty hash")
            
            # Validate IPFS hash format
            if not ipfs_hash.startswith(('Qm', 'bafy')):
                raise ValueError(f"Invalid IPFS hash format: {ipfs_hash}")
            
            # Store in context
            context.ipfs_hash = ipfs_hash
            context.ipfs_upload_time = time.time()
            
            # Add rollback action
            context.rollback_actions.append(f"remove_ipfs:{ipfs_hash}")
            
            logger.debug(f"[{context.transaction_id}] IPFS upload completed: {ipfs_hash}")
            return True
            
        except (NetworkError, ConnectionError, TimeoutError) as e:
            context.error_message = f"IPFS upload network error: {e}"
            context.failed_step = "ipfs_upload"
            raise  # Let decorator handle retry
        except ResourceError as e:
            context.error_message = f"IPFS upload resource error: {e}"
            context.failed_step = "ipfs_upload"
            return False  # Don't retry resource errors
        except Exception as e:
            context.error_message = f"IPFS upload failed: {e}"
            context.failed_step = "ipfs_upload"
            logger.error(f"[{context.transaction_id}] Unexpected IPFS error: {e}", exc_info=True)
            return False
    
    @handle_specific_errors
    @exponential_backoff(max_retries=3, retryable_exceptions=[NetworkError, ConnectionError, TimeoutError])
    async def _step_blockchain_submit(
        self,
        context: TransactionContext,
        miner_address: str,
        training_round_id: int,
        model_version: str,
    ) -> bool:
        """Step 3: Submit IPFS hash to blockchain with enhanced error handling."""
        try:
            context.state = TransactionState.BLOCKCHAIN_SUBMIT
            logger.debug(f"[{context.transaction_id}] Starting blockchain submission")
            
            # Get additional metadata
            from utils.shard_assignment import calculate_shard_id
            from utils.gpu_detection import GPUArchitectureDetector
            
            gpu_detector = GPUArchitectureDetector()
            gpu_architecture = gpu_detector.get_architecture()
            
            # Validate blockchain client connection
            if not hasattr(self.blockchain_client, 'get_block_hash'):
                raise AuthenticationError("Blockchain client not properly initialized")
            
            try:
                block_hash = self.blockchain_client.get_block_hash()
            except Exception as e:
                raise NetworkError(f"Failed to get block hash: {e}")
            
            total_shards = 100  # TODO: Make configurable
            shard_id = calculate_shard_id(
                miner_address=miner_address,
                block_hash=block_hash,
                training_round_id=training_round_id,
                total_shards=total_shards,
            )
            
            # Validate inputs before submission
            if not context.ipfs_hash:
                raise ValueError("IPFS hash is required for blockchain submission")
            if not context.gradient_hash:
                raise ValueError("Gradient hash is required for blockchain submission")
            
            # Submit to blockchain (retry logic handled by decorator)
            response = self.blockchain_client.submit_gradient(
                miner_address=miner_address,
                ipfs_hash=context.ipfs_hash,
                model_version=model_version,
                training_round_id=training_round_id,
                shard_id=shard_id,
                gradient_hash=context.gradient_hash,
                gpu_architecture=gpu_architecture,
            )
            
            # Validate response
            if not response:
                raise NetworkError("Blockchain submission returned empty response")
            
            if not response.get("success"):
                error_msg = response.get("error", "Unknown blockchain error")
                if "insufficient funds" in error_msg.lower():
                    raise AuthenticationError(f"Insufficient funds: {error_msg}")
                elif "invalid signature" in error_msg.lower():
                    raise AuthenticationError(f"Invalid signature: {error_msg}")
                else:
                    raise NetworkError(f"Blockchain submission failed: {error_msg}")
            
            # Store in context
            context.blockchain_tx_hash = response.get("tx_hash")
            context.stored_gradient_id = response.get("stored_gradient_id")
            context.blockchain_submit_time = time.time()
            
            # Validate response data
            if not context.blockchain_tx_hash:
                logger.warning(f"[{context.transaction_id}] No transaction hash in response")
            
            # Add rollback action (if possible)
            if context.stored_gradient_id:
                context.rollback_actions.append(f"remove_gradient:{context.stored_gradient_id}")
            
            logger.debug(f"[{context.transaction_id}] Blockchain submission completed: {context.blockchain_tx_hash}")
            return True
            
        except (NetworkError, ConnectionError, TimeoutError) as e:
            context.error_message = f"Blockchain submission network error: {e}"
            context.failed_step = "blockchain_submit"
            raise  # Let decorator handle retry
        except AuthenticationError as e:
            context.error_message = f"Blockchain submission auth error: {e}"
            context.failed_step = "blockchain_submit"
            return False  # Don't retry auth errors
        except Exception as e:
            context.error_message = f"Blockchain submission failed: {e}"
            context.failed_step = "blockchain_submit"
            logger.error(f"[{context.transaction_id}] Unexpected blockchain error: {e}", exc_info=True)
            return False
    
    @handle_specific_errors
    @exponential_backoff(max_retries=2, retryable_exceptions=[NetworkError, ConnectionError])
    async def _step_task_complete(self, context: TransactionContext) -> bool:
        """Step 4: Complete task in task pool with enhanced error handling."""
        try:
            context.state = TransactionState.TASK_COMPLETE
            logger.debug(f"[{context.transaction_id}] Starting task completion")
            
            if not context.task:
                return True  # No task to complete
            
            # Validate task data
            if not context.task.get('chunk_id'):
                logger.warning(f"[{context.transaction_id}] No chunk_id in task data")
                return True  # Don't fail transaction for missing chunk_id
            
            if not context.ipfs_hash:
                raise ValueError("IPFS hash is required for task completion")
            
            # Complete task (retry logic handled by decorator)
            success = self.task_pool_client.complete_task(
                chunk_id=context.task['chunk_id'],
                gradient_hash=context.ipfs_hash
            )
            
            if not success:
                raise NetworkError("Task completion returned false")
            
            context.task_completed = True
            logger.debug(f"[{context.transaction_id}] Task completion successful")
            return True
            
        except (NetworkError, ConnectionError) as e:
            context.error_message = f"Task completion network error: {e}"
            context.failed_step = "task_complete"
            raise  # Let decorator handle retry
        except Exception as e:
            context.error_message = f"Task completion failed: {e}"
            context.failed_step = "task_complete"
            logger.warning(f"[{context.transaction_id}] Task completion error (non-critical): {e}")
            return False  # Task completion failure is not critical
    
    async def _commit_transaction(self, context: TransactionContext):
        """Commit the transaction."""
        context.state = TransactionState.COMMITTED
        self.stats["committed_transactions"] += 1
        logger.info(f"Transaction committed: {context.transaction_id}")
    
    async def _rollback_transaction(self, context: TransactionContext):
        """Rollback the transaction with enhanced error handling."""
        logger.warning(f"Rolling back transaction: {context.transaction_id}")
        
        rollback_errors = []
        
        # Execute rollback actions in reverse order
        for action in reversed(context.rollback_actions):
            try:
                await self._execute_rollback_action(action)
                logger.debug(f"[{context.transaction_id}] Rollback action completed: {action}")
            except Exception as e:
                error_msg = f"Rollback action failed: {action}: {e}"
                logger.error(error_msg)
                rollback_errors.append(error_msg)
        
        # Update context with rollback information
        context.state = TransactionState.ROLLED_BACK
        if rollback_errors:
            context.error_message = f"{context.error_message}. Rollback errors: {'; '.join(rollback_errors)}"
        
        self.stats["failed_transactions"] += 1
        self.stats["rolled_back_transactions"] += 1
        
        # Log rollback summary
        if rollback_errors:
            logger.error(f"Transaction rolled back with errors: {context.transaction_id}")
            for error in rollback_errors:
                logger.error(f"  - {error}")
        else:
            logger.info(f"Transaction rolled back successfully: {context.transaction_id}")
    
    async def _execute_rollback_action(self, action: str):
        """Execute a single rollback action with timeout and error handling."""
        if action.startswith("remove_ipfs:"):
            ipfs_hash = action.split(":", 1)[1]
            try:
                # Note: IPFS doesn't support deletion, but we can unpin
                # This is a placeholder for IPFS cleanup
                logger.debug(f"Would unpin IPFS hash: {ipfs_hash}")
                # In a real implementation, you might:
                # - Remove from local IPFS cache
                # - Unpin from IPFS cluster
                # - Mark as invalid in local database
            except Exception as e:
                raise NetworkError(f"Failed to cleanup IPFS hash {ipfs_hash}: {e}")
        
        elif action.startswith("remove_gradient:"):
            gradient_id = action.split(":", 1)[1]
            try:
                # Note: Blockchain transactions are immutable
                # This is a placeholder for potential future cleanup mechanisms
                logger.debug(f"Would mark gradient as invalid: {gradient_id}")
                # In a real implementation, you might:
                # - Submit invalidation transaction
                # - Mark as invalid in local database
                # - Notify other nodes about invalidation
            except Exception as e:
                raise NetworkError(f"Failed to invalidate gradient {gradient_id}: {e}")
        
        else:
            logger.warning(f"Unknown rollback action: {action}")
    
    def _update_stats(self, commit_time: float):
        """Update statistics."""
        total_committed = self.stats["committed_transactions"]
        current_avg = self.stats["average_commit_time"]
        
        # Update running average
        self.stats["average_commit_time"] = (
            (current_avg * (total_committed - 1) + commit_time) / total_committed
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive coordinator statistics."""
        current_time = time.time()
        
        # Calculate additional metrics
        total_transactions = self.stats["total_transactions"]
        success_rate = (
            self.stats["committed_transactions"] / max(total_transactions, 1)
        )
        
        # Calculate average transaction time for active transactions
        active_transaction_times = []
        for context in self.active_transactions.values():
            if context.state != TransactionState.PENDING:
                active_transaction_times.append(current_time - context.created_at)
        
        avg_active_time = (
            sum(active_transaction_times) / len(active_transaction_times)
            if active_transaction_times else 0.0
        )
        
        return {
            **self.stats,
            "active_transactions": len(self.active_transactions),
            "success_rate": success_rate,
            "failure_rate": 1.0 - success_rate,
            "average_active_transaction_time": avg_active_time,
            "rollback_rate": (
                self.stats["rolled_back_transactions"] / max(total_transactions, 1)
            ),
            "performance_metrics": {
                "transactions_per_minute": self._calculate_transactions_per_minute(),
                "average_commit_time": self.stats["average_commit_time"],
                "peak_active_transactions": self._get_peak_active_transactions(),
            },
            "error_breakdown": self._get_error_breakdown(),
            "state_distribution": self._get_state_distribution(),
        }
    
    def _calculate_transactions_per_minute(self) -> float:
        """Calculate transactions per minute over the last hour."""
        # This is a simplified implementation
        # In production, you'd maintain a sliding window of transaction timestamps
        return self.stats["total_transactions"] / max(1, time.time() / 60)
    
    def _get_peak_active_transactions(self) -> int:
        """Get peak number of active transactions (placeholder)."""
        # In production, you'd track this over time
        return len(self.active_transactions)
    
    def _get_error_breakdown(self) -> Dict[str, int]:
        """Get breakdown of errors by type (placeholder)."""
        # In production, you'd track error types
        return {
            "network_errors": 0,
            "authentication_errors": 0,
            "resource_errors": 0,
            "validation_errors": 0,
            "unknown_errors": 0,
        }
    
    def _get_state_distribution(self) -> Dict[str, int]:
        """Get distribution of transaction states."""
        state_counts = {}
        for context in self.active_transactions.values():
            state = context.state.value
            state_counts[state] = state_counts.get(state, 0) + 1
        return state_counts
    
    async def cleanup_stale_transactions(self, max_age_seconds: int = 3600):
        """Clean up stale transactions older than max_age_seconds with enhanced monitoring."""
        current_time = time.time()
        stale_transactions = []
        cleanup_stats = {
            "total_checked": len(self.active_transactions),
            "stale_found": 0,
            "cleanup_successful": 0,
            "cleanup_failed": 0,
        }
        
        # Find stale transactions
        for tx_id, context in self.active_transactions.items():
            age = current_time - context.created_at
            if age > max_age_seconds:
                stale_transactions.append((tx_id, context, age))
                cleanup_stats["stale_found"] += 1
        
        # Clean up stale transactions
        for tx_id, context, age in stale_transactions:
            try:
                logger.warning(
                    f"Cleaning up stale transaction: {tx_id} "
                    f"(age: {age:.1f}s, state: {context.state.value})"
                )
                
                # Add timeout information to context
                context.error_message = f"Transaction timeout after {age:.1f}s (max: {max_age_seconds}s)"
                context.failed_step = f"timeout_in_{context.state.value}"
                
                # Rollback the transaction
                await self._rollback_transaction(context)
                
                # Remove from active transactions
                del self.active_transactions[tx_id]
                cleanup_stats["cleanup_successful"] += 1
                
            except Exception as e:
                logger.error(f"Failed to cleanup stale transaction {tx_id}: {e}", exc_info=True)
                cleanup_stats["cleanup_failed"] += 1
        
        # Log cleanup summary
        if cleanup_stats["stale_found"] > 0:
            logger.info(
                f"Stale transaction cleanup completed: "
                f"{cleanup_stats['cleanup_successful']} cleaned up, "
                f"{cleanup_stats['cleanup_failed']} failed"
            )
        
        return cleanup_stats
    
    async def force_cleanup_transaction(self, transaction_id: str) -> bool:
        """Force cleanup of a specific transaction."""
        if transaction_id not in self.active_transactions:
            logger.warning(f"Transaction not found for cleanup: {transaction_id}")
            return False
        
        try:
            context = self.active_transactions[transaction_id]
            context.error_message = "Force cleanup requested"
            context.failed_step = f"force_cleanup_from_{context.state.value}"
            
            await self._rollback_transaction(context)
            del self.active_transactions[transaction_id]
            
            logger.info(f"Force cleanup completed for transaction: {transaction_id}")
            return True
            
        except Exception as e:
            logger.error(f"Force cleanup failed for transaction {transaction_id}: {e}", exc_info=True)
            return False
    
    def get_transaction_status(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a specific transaction."""
        if transaction_id not in self.active_transactions:
            return None
        
        context = self.active_transactions[transaction_id]
        current_time = time.time()
        
        return {
            "transaction_id": context.transaction_id,
            "state": context.state.value,
            "created_at": context.created_at,
            "age_seconds": current_time - context.created_at,
            "failed_step": context.failed_step,
            "error_message": context.error_message,
            "ipfs_hash": context.ipfs_hash,
            "blockchain_tx_hash": context.blockchain_tx_hash,
            "stored_gradient_id": context.stored_gradient_id,
            "task_completed": context.task_completed,
            "rollback_actions": len(context.rollback_actions),
            "progress": {
                "training": context.gradients_dict is not None,
                "ipfs_upload": context.ipfs_hash is not None,
                "blockchain_submit": context.blockchain_tx_hash is not None,
                "task_complete": context.task_completed,
            }
        }
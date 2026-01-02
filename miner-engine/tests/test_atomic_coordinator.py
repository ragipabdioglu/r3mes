#!/usr/bin/env python3
"""
Test suite for AtomicMiningCoordinator

Tests atomic transaction semantics, error handling, and rollback mechanisms.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.atomic_coordinator import (
    AtomicMiningCoordinator,
    TransactionContext,
    TransactionState,
)
from utils.error_handling import NetworkError, AuthenticationError, ResourceError


class TestAtomicMiningCoordinator:
    """Test cases for AtomicMiningCoordinator."""
    
    @pytest.fixture
    def mock_trainer(self):
        """Mock trainer for testing."""
        trainer = Mock()
        trainer.train_step.return_value = (0.5, {"layer1": torch.randn(10, 10)})
        trainer.compute_gradient_hash.return_value = "test_gradient_hash"
        trainer.get_training_metadata.return_value = {"epoch": 1, "batch": 1}
        trainer.get_lora_state_dict.return_value = {"lora_A": torch.randn(5, 5)}
        return trainer
    
    @pytest.fixture
    def mock_ipfs_client(self):
        """Mock IPFS client for testing."""
        client = Mock()
        client.upload_lora_state.return_value = "QmTestHash123"
        return client
    
    @pytest.fixture
    def mock_blockchain_client(self):
        """Mock blockchain client for testing."""
        client = Mock()
        client.get_block_hash.return_value = "test_block_hash"
        client.submit_gradient.return_value = {
            "success": True,
            "tx_hash": "test_tx_hash",
            "stored_gradient_id": 123
        }
        client.get_miner_address.return_value = "remes1testaddress"
        return client
    
    @pytest.fixture
    def mock_task_pool_client(self):
        """Mock task pool client for testing."""
        client = Mock()
        client.complete_task.return_value = True
        return client
    
    @pytest.fixture
    def coordinator(self, mock_trainer, mock_ipfs_client, mock_blockchain_client, mock_task_pool_client):
        """Create AtomicMiningCoordinator for testing."""
        return AtomicMiningCoordinator(
            trainer=mock_trainer,
            ipfs_client=mock_ipfs_client,
            blockchain_client=mock_blockchain_client,
            task_pool_client=mock_task_pool_client,
            timeout_seconds=30,
            max_retries=2,
        )
    
    def test_create_transaction(self, coordinator):
        """Test transaction creation."""
        task = {"chunk_id": 123, "data_hash": "test_hash"}
        context = coordinator.create_transaction(task=task)
        
        assert context.transaction_id.startswith("mining_tx_")
        assert context.state == TransactionState.PENDING
        assert context.task == task
        assert context.rollback_actions == []
        assert coordinator.stats["total_transactions"] == 1
    
    @pytest.mark.asyncio
    async def test_successful_atomic_mining(self, coordinator):
        """Test successful atomic mining transaction."""
        context = coordinator.create_transaction()
        inputs = torch.randn(32, 768)
        targets = torch.randn(32, 768)
        
        success, error_message = await coordinator.execute_atomic_mining(
            context=context,
            inputs=inputs,
            targets=targets,
            miner_address="remes1testaddress",
            training_round_id=1,
            model_version="v1.0.0",
        )
        
        assert success is True
        assert error_message is None
        assert context.state == TransactionState.COMMITTED
        assert context.ipfs_hash == "QmTestHash123"
        assert context.blockchain_tx_hash == "test_tx_hash"
        assert coordinator.stats["committed_transactions"] == 1
    
    @pytest.mark.asyncio
    async def test_ipfs_failure_rollback(self, coordinator, mock_ipfs_client):
        """Test rollback when IPFS upload fails."""
        # Make IPFS upload fail
        mock_ipfs_client.upload_lora_state.side_effect = NetworkError("IPFS connection failed")
        
        context = coordinator.create_transaction()
        inputs = torch.randn(32, 768)
        targets = torch.randn(32, 768)
        
        success, error_message = await coordinator.execute_atomic_mining(
            context=context,
            inputs=inputs,
            targets=targets,
            miner_address="remes1testaddress",
            training_round_id=1,
            model_version="v1.0.0",
        )
        
        assert success is False
        assert "IPFS upload network error" in error_message
        assert context.state == TransactionState.ROLLED_BACK
        assert context.failed_step == "ipfs_upload"
        assert coordinator.stats["failed_transactions"] == 1
        assert coordinator.stats["rolled_back_transactions"] == 1
    
    @pytest.mark.asyncio
    async def test_blockchain_failure_rollback(self, coordinator, mock_blockchain_client):
        """Test rollback when blockchain submission fails."""
        # Make blockchain submission fail
        mock_blockchain_client.submit_gradient.return_value = {
            "success": False,
            "error": "Insufficient funds"
        }
        
        context = coordinator.create_transaction()
        inputs = torch.randn(32, 768)
        targets = torch.randn(32, 768)
        
        success, error_message = await coordinator.execute_atomic_mining(
            context=context,
            inputs=inputs,
            targets=targets,
            miner_address="remes1testaddress",
            training_round_id=1,
            model_version="v1.0.0",
        )
        
        assert success is False
        assert "Insufficient funds" in error_message
        assert context.state == TransactionState.ROLLED_BACK
        assert context.failed_step == "blockchain_submit"
        # Should have IPFS hash but no blockchain tx
        assert context.ipfs_hash == "QmTestHash123"
        assert context.blockchain_tx_hash is None
    
    @pytest.mark.asyncio
    async def test_task_completion_failure_non_critical(self, coordinator, mock_task_pool_client):
        """Test that task completion failure doesn't fail the transaction."""
        # Make task completion fail
        mock_task_pool_client.complete_task.side_effect = NetworkError("Task pool unavailable")
        
        task = {"chunk_id": 123, "data_hash": "test_hash"}
        context = coordinator.create_transaction(task=task)
        inputs = torch.randn(32, 768)
        targets = torch.randn(32, 768)
        
        success, error_message = await coordinator.execute_atomic_mining(
            context=context,
            inputs=inputs,
            targets=targets,
            miner_address="remes1testaddress",
            training_round_id=1,
            model_version="v1.0.0",
        )
        
        # Transaction should still succeed even if task completion fails
        assert success is True
        assert error_message is None
        assert context.state == TransactionState.COMMITTED
        assert context.task_completed is False  # Task completion failed
    
    @pytest.mark.asyncio
    async def test_cleanup_stale_transactions(self, coordinator):
        """Test cleanup of stale transactions."""
        # Create a transaction and make it stale
        context = coordinator.create_transaction()
        context.created_at = time.time() - 3700  # 1 hour and 2 minutes ago
        context.state = TransactionState.IPFS_UPLOAD
        
        cleanup_stats = await coordinator.cleanup_stale_transactions(max_age_seconds=3600)
        
        assert cleanup_stats["stale_found"] == 1
        assert cleanup_stats["cleanup_successful"] == 1
        assert len(coordinator.active_transactions) == 0
    
    def test_get_statistics(self, coordinator):
        """Test statistics collection."""
        # Create some transactions
        coordinator.create_transaction()
        coordinator.create_transaction()
        coordinator.stats["committed_transactions"] = 1
        coordinator.stats["failed_transactions"] = 1
        
        stats = coordinator.get_statistics()
        
        assert stats["total_transactions"] == 2
        assert stats["active_transactions"] == 2
        assert stats["success_rate"] == 0.5
        assert stats["failure_rate"] == 0.5
        assert "performance_metrics" in stats
        assert "error_breakdown" in stats
        assert "state_distribution" in stats
    
    def test_get_transaction_status(self, coordinator):
        """Test getting transaction status."""
        context = coordinator.create_transaction()
        context.state = TransactionState.IPFS_UPLOAD
        context.ipfs_hash = "QmTestHash"
        
        status = coordinator.get_transaction_status(context.transaction_id)
        
        assert status is not None
        assert status["transaction_id"] == context.transaction_id
        assert status["state"] == "ipfs_upload"
        assert status["ipfs_hash"] == "QmTestHash"
        assert status["progress"]["training"] is False
        assert status["progress"]["ipfs_upload"] is True
    
    @pytest.mark.asyncio
    async def test_force_cleanup_transaction(self, coordinator):
        """Test force cleanup of specific transaction."""
        context = coordinator.create_transaction()
        tx_id = context.transaction_id
        
        success = await coordinator.force_cleanup_transaction(tx_id)
        
        assert success is True
        assert tx_id not in coordinator.active_transactions
        assert context.state == TransactionState.ROLLED_BACK


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
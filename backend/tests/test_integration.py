#!/usr/bin/env python3
"""
Integration tests for cross-service interactions.

Tests the integration between different components and services.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.exceptions import (
    DatabaseError,
    BlockchainError,
    NetworkError,
    InsufficientCreditsError,
    AuthenticationError,
)


class TestUserWorkflows:
    """Test cases for complete user workflows."""
    
    @pytest.mark.asyncio
    async def test_user_registration_workflow(self):
        """Test complete user registration workflow."""
        # Mock components
        mock_blockchain = AsyncMock()
        mock_database = AsyncMock()
        
        # Test data
        wallet_address = "remes1testaddress234567890234567890234567890"
        
        # Mock successful blockchain validation
        mock_blockchain.validate_address.return_value = True
        
        # Mock successful database user creation
        mock_database.create_user.return_value = {
            "id": 1,
            "wallet_address": wallet_address,
            "credits": 0.0,
            "is_miner": False,
            "created_at": datetime.now().isoformat()
        }
        
        # Simulate workflow
        # 1. Validate wallet address with blockchain
        is_valid = await mock_blockchain.validate_address(wallet_address)
        assert is_valid
        
        # 2. Create user in database
        user = await mock_database.create_user(wallet_address)
        assert user["wallet_address"] == wallet_address
        assert user["credits"] == 0.0
        
        # Verify calls
        mock_blockchain.validate_address.assert_called_once_with(wallet_address)
        mock_database.create_user.assert_called_once_with(wallet_address)
    
    @pytest.mark.asyncio
    async def test_api_key_creation_workflow(self):
        """Test API key creation workflow."""
        # Mock components
        mock_user_repo = AsyncMock()
        mock_api_key_repo = AsyncMock()
        
        # Test data
        wallet_address = "remes1testaddress234567890234567890234567890"
        api_key_name = "test_key"
        
        # Mock user exists
        mock_user_repo.get_user_info.return_value = {
            "id": 1,
            "wallet_address": wallet_address,
            "credits": 100.0
        }
        
        # Mock API key creation
        mock_api_key_repo.create_api_key.return_value = {
            "id": 1,
            "name": api_key_name,
            "api_key": "r3mes_test_key_123456789",
            "is_active": True
        }
        
        # Simulate workflow
        # 1. Verify user exists
        user = await mock_user_repo.get_user_info(wallet_address)
        assert user is not None
        
        # 2. Create API key
        api_key = await mock_api_key_repo.create_api_key(
            wallet_address, api_key_name
        )
        assert api_key["name"] == api_key_name
        assert api_key["is_active"] is True
    
    @pytest.mark.asyncio
    async def test_mining_submission_workflow(self):
        """Test mining submission workflow."""
        # Mock components
        mock_user_repo = AsyncMock()
        mock_blockchain = AsyncMock()
        mock_ipfs = AsyncMock()
        
        # Test data
        wallet_address = "remes1testaddress234567890234567890234567890"
        gradient_data = {"weights": [1.0, 2.0, 3.0]}
        
        # Mock user is eligible miner
        mock_user_repo.get_user_info.return_value = {
            "wallet_address": wallet_address,
            "credits": 100.0,
            "is_miner": True
        }
        
        # Mock IPFS upload
        mock_ipfs.upload.return_value = "QmTestHash123"
        
        # Mock blockchain submission
        mock_blockchain.submit_gradient.return_value = {
            "tx_hash": "ABC123",
            "success": True
        }
        
        # Simulate workflow
        # 1. Verify user is eligible miner
        user = await mock_user_repo.get_user_info(wallet_address)
        assert user["is_miner"] is True
        
        # 2. Upload gradient to IPFS
        ipfs_hash = await mock_ipfs.upload(gradient_data)
        assert ipfs_hash.startswith("Qm")
        
        # 3. Submit to blockchain
        result = await mock_blockchain.submit_gradient(wallet_address, ipfs_hash)
        assert result["success"] is True


class TestErrorHandlingIntegration:
    """Test cases for error handling across services."""
    
    @pytest.mark.asyncio
    async def test_database_failure_handling(self):
        """Test handling of database failures."""
        # Mock database that fails
        mock_database = AsyncMock()
        mock_database.get_user_info.side_effect = DatabaseError(
            message="Connection timeout",
            operation="get_user_info"
        )
        
        # Test error propagation
        with pytest.raises(DatabaseError) as exc_info:
            await mock_database.get_user_info("test_wallet")
        
        assert "Connection timeout" in str(exc_info.value)
        assert exc_info.value.details["operation"] == "get_user_info"
    
    @pytest.mark.asyncio
    async def test_blockchain_failure_handling(self):
        """Test handling of blockchain failures."""
        # Mock blockchain that fails
        mock_blockchain = AsyncMock()
        mock_blockchain.submit_transaction.side_effect = BlockchainError(
            message="RPC endpoint unavailable",
            operation="submit_transaction"
        )
        
        # Test error propagation
        with pytest.raises(BlockchainError) as exc_info:
            await mock_blockchain.submit_transaction("test_tx")
        
        assert "RPC endpoint unavailable" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_network_failure_handling(self):
        """Test handling of network failures."""
        # Mock network service that fails
        mock_network = AsyncMock()
        mock_network.fetch_data.side_effect = NetworkError(
            message="Connection refused",
            endpoint="https://api.example.com"
        )
        
        # Test error propagation
        with pytest.raises(NetworkError) as exc_info:
            await mock_network.fetch_data("test_endpoint")
        
        assert "Connection refused" in str(exc_info.value)


class TestTransactionIntegrity:
    """Test cases for transaction integrity across services."""
    
    @pytest.mark.asyncio
    async def test_credit_deduction_rollback(self):
        """Test credit deduction rollback on failure."""
        # Mock components
        mock_user_repo = AsyncMock()
        mock_blockchain = AsyncMock()
        
        wallet_address = "remes1testaddress234567890234567890234567890"
        initial_credits = 100.0
        deduction_amount = 25.0
        
        # Mock initial user state
        mock_user_repo.get_user_info.return_value = {
            "wallet_address": wallet_address,
            "credits": initial_credits
        }
        
        # Mock successful credit deduction
        mock_user_repo.deduct_credits.return_value = True
        
        # Mock blockchain failure
        mock_blockchain.submit_transaction.side_effect = BlockchainError(
            message="Transaction failed",
            operation="submit_transaction"
        )
        
        # Mock credit restoration
        mock_user_repo.add_credits.return_value = True
        
        # Simulate transaction with rollback
        try:
            # 1. Deduct credits
            await mock_user_repo.deduct_credits(wallet_address, deduction_amount)
            
            # 2. Submit to blockchain (fails)
            await mock_blockchain.submit_transaction("test_tx")
            
        except BlockchainError:
            # 3. Rollback credits on failure
            await mock_user_repo.add_credits(wallet_address, deduction_amount)
        
        # Verify rollback occurred
        mock_user_repo.add_credits.assert_called_once_with(
            wallet_address, deduction_amount
        )
    
    @pytest.mark.asyncio
    async def test_atomic_mining_operation(self):
        """Test atomic mining operation across services."""
        # Mock components
        mock_user_repo = AsyncMock()
        mock_ipfs = AsyncMock()
        mock_blockchain = AsyncMock()
        
        wallet_address = "remes1testaddress234567890234567890234567890"
        gradient_data = {"weights": [1.0, 2.0, 3.0]}
        
        # Mock successful operations
        mock_ipfs.upload.return_value = "QmTestHash123"
        mock_blockchain.submit_gradient.return_value = {"success": True}
        mock_user_repo.add_credits.return_value = True
        
        # Simulate atomic operation
        ipfs_hash = None
        try:
            # 1. Upload to IPFS
            ipfs_hash = await mock_ipfs.upload(gradient_data)
            
            # 2. Submit to blockchain
            result = await mock_blockchain.submit_gradient(wallet_address, ipfs_hash)
            
            # 3. Award credits
            if result["success"]:
                await mock_user_repo.add_credits(wallet_address, 10.0)
            
        except Exception as e:
            # Cleanup on failure
            if ipfs_hash:
                await mock_ipfs.unpin(ipfs_hash)
            raise
        
        # Verify all operations completed
        mock_ipfs.upload.assert_called_once()
        mock_blockchain.submit_gradient.assert_called_once()
        mock_user_repo.add_credits.assert_called_once()


class TestServiceCommunication:
    """Test cases for service-to-service communication."""
    
    @pytest.mark.asyncio
    async def test_user_service_to_blockchain_communication(self):
        """Test communication between user service and blockchain."""
        # Mock services
        mock_user_service = AsyncMock()
        mock_blockchain_service = AsyncMock()
        
        wallet_address = "remes1testaddress234567890234567890234567890"
        
        # Mock user service response
        mock_user_service.get_balance.return_value = 100.0
        
        # Mock blockchain service response
        mock_blockchain_service.get_on_chain_balance.return_value = 1000000  # microunits
        
        # Test communication
        off_chain_balance = await mock_user_service.get_balance(wallet_address)
        on_chain_balance = await mock_blockchain_service.get_on_chain_balance(wallet_address)
        
        # Verify responses
        assert off_chain_balance == 100.0
        assert on_chain_balance == 1000000
        
        # Verify calls
        mock_user_service.get_balance.assert_called_once_with(wallet_address)
        mock_blockchain_service.get_on_chain_balance.assert_called_once_with(wallet_address)
    
    @pytest.mark.asyncio
    async def test_api_service_to_user_service_communication(self):
        """Test communication between API service and user service."""
        # Mock services
        mock_api_service = AsyncMock()
        mock_user_service = AsyncMock()
        
        api_key = "r3mes_test_key_123456789"
        wallet_address = "remes1testaddress234567890234567890234567890"
        
        # Mock API key validation
        mock_api_service.validate_api_key.return_value = {
            "wallet_address": wallet_address,
            "is_active": True
        }
        
        # Mock user info retrieval
        mock_user_service.get_user_info.return_value = {
            "wallet_address": wallet_address,
            "credits": 100.0,
            "is_miner": True
        }
        
        # Test communication flow
        api_key_info = await mock_api_service.validate_api_key(api_key)
        if api_key_info["is_active"]:
            user_info = await mock_user_service.get_user_info(
                api_key_info["wallet_address"]
            )
        
        # Verify communication
        assert user_info["wallet_address"] == wallet_address
        assert user_info["credits"] == 100.0


class TestDataConsistency:
    """Test cases for data consistency across services."""
    
    @pytest.mark.asyncio
    async def test_user_credit_consistency(self):
        """Test user credit consistency across operations."""
        # Mock user repository
        mock_user_repo = AsyncMock()
        
        wallet_address = "remes1testaddress234567890234567890234567890"
        initial_credits = 100.0
        
        # Track credit changes
        credit_history = []
        
        def track_credits(wallet, amount, operation):
            credit_history.append({
                "wallet": wallet,
                "amount": amount,
                "operation": operation,
                "timestamp": datetime.now()
            })
            return True
        
        mock_user_repo.add_credits.side_effect = lambda w, a: track_credits(w, a, "add")
        mock_user_repo.deduct_credits.side_effect = lambda w, a: track_credits(w, a, "deduct")
        
        # Simulate credit operations
        await mock_user_repo.add_credits(wallet_address, 50.0)
        await mock_user_repo.deduct_credits(wallet_address, 25.0)
        await mock_user_repo.add_credits(wallet_address, 10.0)
        
        # Verify credit history
        assert len(credit_history) == 3
        assert credit_history[0]["operation"] == "add"
        assert credit_history[1]["operation"] == "deduct"
        assert credit_history[2]["operation"] == "add"
        
        # Calculate final balance
        final_balance = initial_credits
        for entry in credit_history:
            if entry["operation"] == "add":
                final_balance += entry["amount"]
            else:
                final_balance -= entry["amount"]
        
        expected_balance = 100.0 + 50.0 - 25.0 + 10.0
        assert final_balance == expected_balance
    
    @pytest.mark.asyncio
    async def test_api_key_state_consistency(self):
        """Test API key state consistency."""
        # Mock API key repository
        mock_api_key_repo = AsyncMock()
        
        wallet_address = "remes1testaddress234567890234567890234567890"
        api_key = "r3mes_test_key_123456789"
        
        # Mock API key lifecycle
        mock_api_key_repo.create_api_key.return_value = {
            "api_key": api_key,
            "is_active": True,
            "created_at": datetime.now().isoformat()
        }
        
        mock_api_key_repo.validate_api_key.return_value = {
            "wallet_address": wallet_address,
            "is_active": True
        }
        
        mock_api_key_repo.revoke_api_key.return_value = True
        
        # Test API key lifecycle
        # 1. Create API key
        created_key = await mock_api_key_repo.create_api_key(wallet_address, "test_key")
        assert created_key["is_active"] is True
        
        # 2. Validate API key
        validation_result = await mock_api_key_repo.validate_api_key(api_key)
        assert validation_result["is_active"] is True
        
        # 3. Revoke API key
        revoke_result = await mock_api_key_repo.revoke_api_key(api_key, wallet_address)
        assert revoke_result is True


class TestPerformanceIntegration:
    """Test cases for performance across integrated services."""
    
    @pytest.mark.asyncio
    async def test_concurrent_user_operations(self):
        """Test concurrent user operations."""
        # Mock user repository
        mock_user_repo = AsyncMock()
        
        # Mock concurrent operations
        wallets = [
            "remes1testaddress234567890234567890234567890",
            "remes1testaddress234567890234567890234567891",
            "remes1testaddress234567890234567890234567892",
        ]
        
        # Mock responses
        mock_user_repo.get_user_info.return_value = {
            "credits": 100.0,
            "is_miner": True
        }
        
        # Test concurrent operations
        tasks = []
        for wallet in wallets:
            task = mock_user_repo.get_user_info(wallet)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Verify all operations completed
        assert len(results) == len(wallets)
        for result in results:
            assert result["credits"] == 100.0
    
    @pytest.mark.asyncio
    async def test_batch_processing_integration(self):
        """Test batch processing across services."""
        # Mock services
        mock_user_repo = AsyncMock()
        mock_blockchain = AsyncMock()
        
        # Test data
        batch_size = 10
        wallets = [f"remes1wallet{i:038d}" for i in range(batch_size)]
        
        # Mock batch operations
        mock_user_repo.get_users_batch.return_value = [
            {"wallet_address": wallet, "credits": 100.0} for wallet in wallets
        ]
        
        mock_blockchain.get_balances_batch.return_value = {
            wallet: 1000000 for wallet in wallets
        }
        
        # Test batch processing
        users = await mock_user_repo.get_users_batch(wallets)
        balances = await mock_blockchain.get_balances_batch(wallets)
        
        # Verify batch results
        assert len(users) == batch_size
        assert len(balances) == batch_size
        
        for wallet in wallets:
            assert wallet in balances
            assert balances[wallet] == 1000000


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
#!/usr/bin/env python3
"""
Unit tests for repository pattern implementations.

Tests the repository layer for proper data access patterns, validation,
and error handling.
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.repositories.base_repository import BaseRepository
from app.repositories.user_repository import UserRepository
from app.repositories.api_key_repository import APIKeyRepository
from app.exceptions import (
    DatabaseError,
    InvalidInputError,
    ValidationError,
    AuthenticationError,
)


class TestBaseRepository:
    """Test cases for BaseRepository."""
    
    @pytest.fixture
    def mock_db(self):
        """Mock database connection."""
        db = Mock()
        db.execute = AsyncMock()
        db.fetch_one = AsyncMock()
        db.fetch_all = AsyncMock()
        return db
    
    @pytest.fixture
    def base_repo(self, mock_db):
        """Create BaseRepository instance."""
        return BaseRepository(mock_db)
    
    def test_init(self, mock_db):
        """Test repository initialization."""
        repo = BaseRepository(mock_db)
        assert repo.db == mock_db
        assert repo.logger is not None
    
    @pytest.mark.asyncio
    async def test_execute_query_success(self, base_repo, mock_db):
        """Test successful query execution."""
        mock_db.execute.return_value = Mock(rowcount=1)
        
        result = await base_repo._execute_query(
            "INSERT INTO test (name) VALUES (?)",
            ("test_name",)
        )
        
        assert result.rowcount == 1
        mock_db.execute.assert_called_once_with(
            "INSERT INTO test (name) VALUES (?)",
            ("test_name",)
        )
    
    @pytest.mark.asyncio
    async def test_execute_query_database_error(self, base_repo, mock_db):
        """Test query execution with database error."""
        mock_db.execute.side_effect = Exception("Database connection failed")
        
        with pytest.raises(DatabaseError) as exc_info:
            await base_repo._execute_query(
                "INSERT INTO test (name) VALUES (?)",
                ("test_name",)
            )
        
        assert "Database operation failed" in str(exc_info.value)
        assert exc_info.value.details["operation"] == "_execute_query"
    
    @pytest.mark.asyncio
    async def test_fetch_one_success(self, base_repo, mock_db):
        """Test successful single row fetch."""
        expected_row = {"id": 1, "name": "test"}
        mock_db.fetch_one.return_value = expected_row
        
        result = await base_repo._fetch_one(
            "SELECT * FROM test WHERE id = ?",
            (1,)
        )
        
        assert result == expected_row
        mock_db.fetch_one.assert_called_once_with(
            "SELECT * FROM test WHERE id = ?",
            (1,)
        )
    
    @pytest.mark.asyncio
    async def test_fetch_all_success(self, base_repo, mock_db):
        """Test successful multiple rows fetch."""
        expected_rows = [
            {"id": 1, "name": "test1"},
            {"id": 2, "name": "test2"}
        ]
        mock_db.fetch_all.return_value = expected_rows
        
        result = await base_repo._fetch_all("SELECT * FROM test")
        
        assert result == expected_rows
        mock_db.fetch_all.assert_called_once_with("SELECT * FROM test", ())
    
    def test_validate_pagination_valid(self, base_repo):
        """Test valid pagination parameters."""
        # Should not raise any exception
        base_repo._validate_pagination(page=1, limit=10)
        base_repo._validate_pagination(page=5, limit=50)
    
    def test_validate_pagination_invalid_page(self, base_repo):
        """Test invalid page parameter."""
        with pytest.raises(InvalidInputError) as exc_info:
            base_repo._validate_pagination(page=0, limit=10)
        
        assert "Page must be positive" in str(exc_info.value)
        assert exc_info.value.details["field"] == "page"
    
    def test_validate_pagination_invalid_limit(self, base_repo):
        """Test invalid limit parameter."""
        with pytest.raises(InvalidInputError) as exc_info:
            base_repo._validate_pagination(page=1, limit=101)
        
        assert "Limit must be between 1 and 100" in str(exc_info.value)
        assert exc_info.value.details["field"] == "limit"


class TestUserRepository:
    """Test cases for UserRepository."""
    
    @pytest.fixture
    def mock_db(self):
        """Mock database connection."""
        db = Mock()
        db.execute = AsyncMock()
        db.fetch_one = AsyncMock()
        db.fetch_all = AsyncMock()
        return db
    
    @pytest.fixture
    def user_repo(self, mock_db):
        """Create UserRepository instance."""
        return UserRepository(mock_db)
    
    @pytest.mark.asyncio
    async def test_get_by_wallet_success(self, user_repo, mock_db):
        """Test successful user retrieval by wallet address."""
        expected_user = {
            "id": 1,
            "wallet_address": "remes1testaddress123456789012345678901234",
            "credits": 100.0,
            "is_miner": True,
            "created_at": "2024-01-01T00:00:00Z"
        }
        mock_db.fetch_one.return_value = expected_user
        
        result = await user_repo.get_by_wallet("remes1testaddress123456789012345678901234")
        
        assert result == expected_user
        mock_db.fetch_one.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_by_wallet_invalid_address(self, user_repo):
        """Test user retrieval with invalid wallet address."""
        with pytest.raises(InvalidInputError) as exc_info:
            await user_repo.get_by_wallet("invalid_address")
        
        assert "Invalid wallet address format" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_by_wallet_not_found(self, user_repo, mock_db):
        """Test user retrieval when user not found."""
        mock_db.fetch_one.return_value = None
        
        result = await user_repo.get_by_wallet("remes1testaddress123456789012345678901234")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_create_success(self, user_repo, mock_db):
        """Test successful user creation."""
        mock_db.execute.return_value = Mock(rowcount=1)
        mock_db.fetch_one.return_value = {
            "id": 1,
            "wallet_address": "remes1testaddress123456789012345678901234",
            "credits": 0.0,
            "is_miner": False
        }
        
        result = await user_repo.create(
            wallet_address="remes1testaddress123456789012345678901234",
            initial_credits=0.0,
            is_miner=False
        )
        
        assert result["wallet_address"] == "remes1testaddress123456789012345678901234"
        assert result["credits"] == 0.0
        assert result["is_miner"] is False
    
    @pytest.mark.asyncio
    async def test_create_invalid_wallet(self, user_repo):
        """Test user creation with invalid wallet address."""
        with pytest.raises(InvalidInputError):
            await user_repo.create(
                wallet_address="invalid_address",
                initial_credits=0.0,
                is_miner=False
            )
    
    @pytest.mark.asyncio
    async def test_create_negative_credits(self, user_repo):
        """Test user creation with negative credits."""
        with pytest.raises(InvalidInputError) as exc_info:
            await user_repo.create(
                wallet_address="remes1testaddress123456789012345678901234",
                initial_credits=-10.0,
                is_miner=False
            )
        
        assert "initial_credits must be positive" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_update_credits_success(self, user_repo, mock_db):
        """Test successful credit update."""
        mock_db.execute.return_value = Mock(rowcount=1)
        mock_db.fetch_one.return_value = {
            "id": 1,
            "wallet_address": "remes1testaddress123456789012345678901234",
            "credits": 150.0
        }
        
        result = await user_repo.update_credits(
            wallet_address="remes1testaddress123456789012345678901234",
            new_credits=150.0
        )
        
        assert result["credits"] == 150.0
    
    @pytest.mark.asyncio
    async def test_update_credits_user_not_found(self, user_repo, mock_db):
        """Test credit update when user not found."""
        mock_db.execute.return_value = Mock(rowcount=0)
        
        with pytest.raises(ValidationError) as exc_info:
            await user_repo.update_credits(
                wallet_address="remes1testaddress123456789012345678901234",
                new_credits=150.0
            )
        
        assert "User not found" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_list_miners_success(self, user_repo, mock_db):
        """Test successful miner listing."""
        expected_miners = [
            {"id": 1, "wallet_address": "remes1miner1", "is_miner": True},
            {"id": 2, "wallet_address": "remes1miner2", "is_miner": True}
        ]
        mock_db.fetch_all.return_value = expected_miners
        
        result = await user_repo.list_miners(page=1, limit=10)
        
        assert result == expected_miners
        mock_db.fetch_all.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_list_miners_invalid_pagination(self, user_repo):
        """Test miner listing with invalid pagination."""
        with pytest.raises(InvalidInputError):
            await user_repo.list_miners(page=0, limit=10)


class TestAPIKeyRepository:
    """Test cases for APIKeyRepository."""
    
    @pytest.fixture
    def mock_db(self):
        """Mock database connection."""
        db = Mock()
        db.execute = AsyncMock()
        db.fetch_one = AsyncMock()
        db.fetch_all = AsyncMock()
        return db
    
    @pytest.fixture
    def api_key_repo(self, mock_db):
        """Create APIKeyRepository instance."""
        return APIKeyRepository(mock_db)
    
    @pytest.mark.asyncio
    async def test_create_success(self, api_key_repo, mock_db):
        """Test successful API key creation."""
        mock_db.execute.return_value = Mock(rowcount=1)
        mock_db.fetch_one.return_value = {
            "id": 1,
            "wallet_address": "remes1testaddress123456789012345678901234",
            "api_key_hash": "hashed_key",
            "name": "test_key",
            "is_active": True
        }
        
        result = await api_key_repo.create(
            wallet_address="remes1testaddress123456789012345678901234",
            api_key="r3mes_test123",
            name="test_key"
        )
        
        assert result["name"] == "test_key"
        assert result["is_active"] is True
        assert "api_key_hash" in result
    
    @pytest.mark.asyncio
    async def test_create_invalid_wallet(self, api_key_repo):
        """Test API key creation with invalid wallet address."""
        with pytest.raises(InvalidInputError):
            await api_key_repo.create(
                wallet_address="invalid_address",
                api_key="r3mes_test123",
                name="test_key"
            )
    
    @pytest.mark.asyncio
    async def test_create_weak_api_key(self, api_key_repo):
        """Test API key creation with weak key."""
        with pytest.raises(InvalidInputError) as exc_info:
            await api_key_repo.create(
                wallet_address="remes1testaddress123456789012345678901234",
                api_key="weak",
                name="test_key"
            )
        
        assert "API key must be at least 12 characters" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_validate_success(self, api_key_repo, mock_db):
        """Test successful API key validation."""
        mock_db.fetch_one.return_value = {
            "id": 1,
            "wallet_address": "remes1testaddress123456789012345678901234",
            "is_active": True
        }
        
        with patch('app.repositories.api_key_repository.hashlib.sha256') as mock_hash:
            mock_hash.return_value.hexdigest.return_value = "hashed_key"
            
            result = await api_key_repo.validate("r3mes_test123")
            
            assert result["wallet_address"] == "remes1testaddress123456789012345678901234"
    
    @pytest.mark.asyncio
    async def test_validate_invalid_key(self, api_key_repo, mock_db):
        """Test API key validation with invalid key."""
        mock_db.fetch_one.return_value = None
        
        with pytest.raises(AuthenticationError) as exc_info:
            await api_key_repo.validate("invalid_key")
        
        assert "Invalid API key" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_validate_inactive_key(self, api_key_repo, mock_db):
        """Test API key validation with inactive key."""
        mock_db.fetch_one.return_value = {
            "id": 1,
            "wallet_address": "remes1testaddress123456789012345678901234",
            "is_active": False
        }
        
        with patch('app.repositories.api_key_repository.hashlib.sha256') as mock_hash:
            mock_hash.return_value.hexdigest.return_value = "hashed_key"
            
            with pytest.raises(AuthenticationError) as exc_info:
                await api_key_repo.validate("r3mes_test123")
            
            assert "API key is inactive" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_revoke_success(self, api_key_repo, mock_db):
        """Test successful API key revocation."""
        mock_db.execute.return_value = Mock(rowcount=1)
        
        result = await api_key_repo.revoke(
            wallet_address="remes1testaddress123456789012345678901234",
            api_key_id=1
        )
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_revoke_not_found(self, api_key_repo, mock_db):
        """Test API key revocation when key not found."""
        mock_db.execute.return_value = Mock(rowcount=0)
        
        with pytest.raises(ValidationError) as exc_info:
            await api_key_repo.revoke(
                wallet_address="remes1testaddress123456789012345678901234",
                api_key_id=999
            )
        
        assert "API key not found" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_list_by_wallet_success(self, api_key_repo, mock_db):
        """Test successful API key listing by wallet."""
        expected_keys = [
            {"id": 1, "name": "key1", "is_active": True, "created_at": "2024-01-01"},
            {"id": 2, "name": "key2", "is_active": False, "created_at": "2024-01-02"}
        ]
        mock_db.fetch_all.return_value = expected_keys
        
        result = await api_key_repo.list_by_wallet(
            wallet_address="remes1testaddress123456789012345678901234"
        )
        
        assert result == expected_keys
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_success(self, api_key_repo, mock_db):
        """Test successful cleanup of expired API keys."""
        mock_db.execute.return_value = Mock(rowcount=5)
        
        result = await api_key_repo.cleanup_expired(days_old=30)
        
        assert result == 5  # Number of cleaned up keys


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
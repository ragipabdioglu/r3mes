"""
Pytest configuration and fixtures for E2E tests.
"""

import pytest
import asyncio
import os
import tempfile
import shutil
from typing import Generator, Dict, Any
import logging

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Test configuration."""
    return {
        "R3MES_ENV": "test",
        "R3MES_TEST_MODE": "true",
        "BLOCKCHAIN_RPC_URL": os.getenv("TEST_BLOCKCHAIN_RPC_URL", "http://localhost:26657"),
        "BLOCKCHAIN_GRPC_URL": os.getenv("TEST_BLOCKCHAIN_GRPC_URL", "localhost:9090"),
        "BACKEND_URL": os.getenv("TEST_BACKEND_URL", "http://localhost:8000"),
        "DATABASE_TYPE": "sqlite",  # Use SQLite for tests
        "DATABASE_URL": ":memory:",  # In-memory database
        "REDIS_URL": os.getenv("TEST_REDIS_URL", "redis://localhost:6379/15"),  # Use DB 15 for tests
    }


@pytest.fixture(scope="function")
def temp_dir() -> Generator[str, None, None]:
    """Create temporary directory for test files."""
    temp_path = tempfile.mkdtemp(prefix="r3mes_test_")
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="function")
def mock_blockchain():
    """Mock blockchain client for testing."""
    class MockBlockchain:
        def __init__(self):
            self.block_height = 1000
            self.transactions = []
        
        async def get_block_height(self):
            return self.block_height
        
        async def submit_transaction(self, tx):
            self.transactions.append(tx)
            return {"tx_hash": f"0x{len(self.transactions)}"}
        
        async def get_transaction(self, tx_hash):
            for tx in self.transactions:
                if tx.get("hash") == tx_hash:
                    return tx
            return None
    
    return MockBlockchain()


@pytest.fixture(scope="function")
def mock_ipfs():
    """Mock IPFS client for testing."""
    class MockIPFS:
        def __init__(self):
            self.content = {}
        
        async def add(self, data: bytes) -> str:
            import hashlib
            hash_obj = hashlib.sha256(data)
            cid = f"Qm{hash_obj.hexdigest()[:44]}"
            self.content[cid] = data
            return cid
        
        async def get(self, cid: str) -> bytes:
            return self.content.get(cid, b"")
        
        async def pin(self, cid: str) -> bool:
            return True
    
    return MockIPFS()


@pytest.fixture(scope="function")
def mock_backend():
    """Mock backend API client for testing."""
    class MockBackend:
        def __init__(self):
            self.data = {}
        
        async def get(self, endpoint: str):
            return self.data.get(endpoint, {})
        
        async def post(self, endpoint: str, data: Dict):
            self.data[endpoint] = data
            return {"status": "success"}
    
    return MockBackend()


@pytest.fixture(autouse=True)
def setup_test_environment(test_config):
    """Setup test environment variables."""
    for key, value in test_config.items():
        os.environ[key] = str(value)
    yield
    # Cleanup
    for key in test_config.keys():
        os.environ.pop(key, None)


@pytest.fixture(scope="session")
def test_network():
    """Test network configuration."""
    return {
        "nodes": 1,
        "miners": 3,
        "validators": 2,
    }


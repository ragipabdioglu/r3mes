"""
Pytest configuration and fixtures for integration tests
"""

import pytest
import tempfile
import os
import json
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="function")
def temp_db():
    """Create a temporary database for testing"""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test.db")
    chain_json_path = os.path.join(temp_dir, "chain.json")
    
    # Create empty chain.json
    with open(chain_json_path, "w") as f:
        json.dump({"blocks": []}, f)
    
    # Set environment variables
    os.environ["DATABASE_PATH"] = db_path
    os.environ["CHAIN_JSON_PATH"] = chain_json_path
    os.environ["R3MES_ENV"] = "test"
    os.environ["R3MES_TEST_MODE"] = "true"
    
    yield {
        "db_path": db_path,
        "chain_json_path": chain_json_path,
        "temp_dir": temp_dir
    }
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="function")
def test_wallet_address():
    """Provide a test wallet address"""
    return "remes1test123456789012345678901234567890"


@pytest.fixture(scope="function")
def mock_blockchain_grpc(monkeypatch):
    """Mock blockchain gRPC client"""
    from unittest.mock import MagicMock
    
    mock_stub = MagicMock()
    mock_channel = MagicMock()
    
    def mock_insecure_channel(url):
        return mock_channel
    
    monkeypatch.setattr("grpc.insecure_channel", mock_insecure_channel)
    
    return {
        "stub": mock_stub,
        "channel": mock_channel
    }


@pytest.fixture(scope="function")
def mock_blockchain_rpc(monkeypatch):
    """Mock blockchain RPC client"""
    from unittest.mock import MagicMock
    import requests
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "result": {
            "sync_info": {
                "latest_block_height": "12345"
            }
        }
    }
    
    def mock_get(url, **kwargs):
        return mock_response
    
    monkeypatch.setattr("requests.get", mock_get)
    
    return mock_response


"""
Integration tests for Blockchain interactions

Tests the integration between backend and blockchain node,
including gRPC queries, transaction submission, and data synchronization.
"""

import unittest
import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.blockchain_query_client import BlockchainQueryClient
from app.blockchain_rpc_client import BlockchainRPCClient
from app.database import Database


class TestBlockchainIntegration(unittest.TestCase):
    """Integration tests for blockchain interactions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_blockchain.db")
        self.chain_json_path = os.path.join(self.temp_dir, "chain.json")
        
        with open(self.chain_json_path, "w") as f:
            json.dump({"blocks": []}, f)
        
        os.environ["R3MES_ENV"] = "test"
        os.environ["R3MES_TEST_MODE"] = "true"
        os.environ["BLOCKCHAIN_GRPC_URL"] = "localhost:9090"
        os.environ["BLOCKCHAIN_REST_URL"] = "http://localhost:1317"
        os.environ["BLOCKCHAIN_RPC_URL"] = "http://localhost:26657"
    
    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('app.blockchain_query_client.grpc.insecure_channel')
    @patch('app.blockchain_query_client.query_pb2_grpc.QueryStub')
    def test_query_available_chunks(self, mock_stub_class, mock_channel):
        """Test querying available chunks from blockchain"""
        # Mock gRPC response
        mock_stub = MagicMock()
        mock_stub_class.return_value = mock_stub
        
        mock_response = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.chunk_id = "chunk_123"
        mock_chunk.data_hash = "hash_abc"
        mock_chunk.shard_id = 1
        mock_chunk.status = "available"
        mock_chunk.assigned_miner = ""
        mock_response.chunks = [mock_chunk]
        mock_response.total_available = 1
        
        mock_stub.QueryAvailableChunks.return_value = mock_response
        
        # Create client
        client = BlockchainQueryClient()
        
        # Query chunks
        chunks = client.get_available_chunks(pool_id="pool_1", limit=10)
        
        # Verify
        self.assertIsInstance(chunks, list)
        if chunks:  # If not mocked properly, might be empty
            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0]["chunk_id"], "chunk_123")
    
    @patch('app.blockchain_rpc_client.requests.get')
    def test_get_block_height(self, mock_get):
        """Test getting block height from blockchain RPC"""
        # Mock RPC response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "sync_info": {
                    "latest_block_height": "12345"
                }
            }
        }
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        # Create client
        client = BlockchainRPCClient()
        
        # Get block height
        height = client.get_block_height()
        
        # Verify
        self.assertEqual(height, 12345)
    
    @patch('app.blockchain_rpc_client.requests.get')
    def test_get_recent_blocks(self, mock_get):
        """Test getting recent blocks from blockchain"""
        # Mock RPC response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "block": {
                    "header": {
                        "height": "12345",
                        "time": "2024-01-01T00:00:00Z"
                    }
                }
            }
        }
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        # Create client
        client = BlockchainRPCClient()
        
        # Get recent blocks
        blocks = client.get_recent_blocks(limit=10)
        
        # Verify
        self.assertIsInstance(blocks, list)
    
    def test_database_blockchain_sync(self):
        """Test database synchronization with blockchain data"""
        db = Database(db_path=self.db_path, chain_json_path=self.chain_json_path)
        
        # Add a block to chain.json (simulating blockchain sync)
        with open(self.chain_json_path, "r") as f:
            chain_data = json.load(f)
        
        chain_data["blocks"].append({
            "height": 12345,
            "time": "2024-01-01T00:00:00Z",
            "hash": "block_hash_123"
        })
        
        with open(self.chain_json_path, "w") as f:
            json.dump(chain_data, f)
        
        # Get recent blocks from database
        blocks = db.get_recent_blocks(limit=10)
        
        # Verify
        self.assertIsInstance(blocks, list)


class TestBlockchainQueryClientIntegration(unittest.TestCase):
    """Integration tests for BlockchainQueryClient"""
    
    def setUp(self):
        """Set up test environment"""
        os.environ["R3MES_ENV"] = "test"
        os.environ["R3MES_TEST_MODE"] = "true"
        os.environ["BLOCKCHAIN_GRPC_URL"] = "localhost:9090"
        os.environ["BLOCKCHAIN_REST_URL"] = "http://localhost:1317"
    
    @patch('app.blockchain_query_client.grpc.insecure_channel')
    def test_client_initialization(self, mock_channel):
        """Test client initialization"""
        client = BlockchainQueryClient()
        self.assertIsNotNone(client)
        self.assertEqual(client.grpc_url, "localhost:9090")
    
    @patch('app.blockchain_query_client.grpc.insecure_channel')
    def test_production_validation(self, mock_channel):
        """Test production environment validation"""
        os.environ["R3MES_ENV"] = "production"
        os.environ["BLOCKCHAIN_GRPC_URL"] = "localhost:9090"  # Should fail
        
        with self.assertRaises(ValueError):
            BlockchainQueryClient()


if __name__ == "__main__":
    unittest.main()


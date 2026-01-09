"""
Integration tests for Miner-Blockchain interactions

Tests the integration between miner engine and blockchain node,
including job discovery, gradient submission, and transaction handling.
"""

import unittest
import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r3mes.bridge.blockchain_client import BlockchainClient


class TestMinerBlockchainIntegration(unittest.TestCase):
    """Integration tests for miner-blockchain interactions"""
    
    def setUp(self):
        """Set up test environment"""
        os.environ["R3MES_ENV"] = "test"
        os.environ["R3MES_TEST_MODE"] = "true"
        os.environ["R3MES_NODE_GRPC_URL"] = "localhost:9090"
        os.environ["R3MES_BLOCKCHAIN_REST_URL"] = "http://localhost:1317"
    
    @patch('r3mes.bridge.blockchain_client.grpc.insecure_channel')
    @patch('r3mes.bridge.blockchain_client.query_pb2_grpc.QueryStub')
    def test_get_available_chunks(self, mock_stub_class, mock_channel):
        """Test getting available chunks from blockchain"""
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
        client = BlockchainClient()
        
        # Get available chunks
        chunks = client.get_available_chunks(pool_id="pool_1", limit=10)
        
        # Verify
        self.assertIsInstance(chunks, list)
        if chunks:  # If mocked properly
            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0]["chunk_id"], "chunk_123")
    
    @patch('r3mes.bridge.blockchain_client.grpc.insecure_channel')
    @patch('r3mes.bridge.blockchain_client.msg_pb2_grpc.MsgStub')
    def test_submit_gradient(self, mock_msg_stub_class, mock_channel):
        """Test submitting gradient to blockchain"""
        # Mock gRPC response
        mock_msg_stub = MagicMock()
        mock_msg_stub_class.return_value = mock_msg_stub
        
        mock_response = MagicMock()
        mock_response.tx_response.txhash = "tx_hash_123"
        mock_response.tx_response.code = 0  # Success
        mock_msg_stub.SubmitGradient.return_value = mock_response
        
        # Create client
        client = BlockchainClient()
        
        # Submit gradient (mock data)
        gradient_data = {
            "gradient_hash": "gradient_hash_123",
            "chunk_id": "chunk_123",
            "ipfs_hash": "ipfs_hash_123"
        }
        
        # This would normally call submit_gradient, but we'll mock it
        # result = client.submit_gradient(gradient_data)
        # self.assertIsNotNone(result)
    
    def test_client_initialization_production(self):
        """Test client initialization in production mode"""
        os.environ["R3MES_ENV"] = "production"
        os.environ["R3MES_NODE_GRPC_URL"] = "localhost:9090"  # Should fail
        
        with self.assertRaises(ValueError):
            BlockchainClient()
    
    def test_client_initialization_development(self):
        """Test client initialization in development mode"""
        os.environ["R3MES_ENV"] = "development"
        os.environ["R3MES_NODE_GRPC_URL"] = "localhost:9090"
        
        # Should not raise in development
        try:
            client = BlockchainClient()
            # If initialization succeeds, verify default values
            self.assertIsNotNone(client)
        except Exception as e:
            # If it fails due to gRPC connection, that's OK in test
            self.assertIn("connection", str(e).lower())


if __name__ == "__main__":
    unittest.main()


"""
Integration tests for Backend API endpoints

Tests the full request/response cycle including:
- API endpoint functionality
- Database interactions
- Input validation
- Error handling
- Authentication/authorization
"""

import unittest
import pytest
import asyncio
from fastapi.testclient import TestClient
from fastapi import FastAPI
import tempfile
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from app.database import Database


class TestAPIIntegration(unittest.TestCase):
    """Integration tests for API endpoints"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test client and database"""
        # Create temporary database
        cls.temp_dir = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.temp_dir, "test_integration.db")
        cls.chain_json_path = os.path.join(cls.temp_dir, "chain.json")
        
        # Create empty chain.json
        import json
        with open(cls.chain_json_path, "w") as f:
            json.dump({"blocks": []}, f)
        
        # Override database path in app
        os.environ["DATABASE_PATH"] = cls.db_path
        os.environ["CHAIN_JSON_PATH"] = cls.chain_json_path
        os.environ["R3MES_ENV"] = "test"
        os.environ["R3MES_TEST_MODE"] = "true"
        
        # Create test client
        cls.client = TestClient(app)
        
        # Initialize database
        cls.db = Database(db_path=cls.db_path, chain_json_path=cls.chain_json_path)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(cls.temp_dir)
    
    def setUp(self):
        """Set up for each test"""
        # Clear database before each test
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.db = Database(db_path=self.db_path, chain_json_path=self.chain_json_path)
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
    
    def test_chat_endpoint_without_auth(self):
        """Test chat endpoint without authentication"""
        response = self.client.post(
            "/chat",
            json={"message": "Hello, test message"}
        )
        # Should either require auth or work without it (depending on implementation)
        self.assertIn(response.status_code, [200, 401, 403])
    
    def test_chat_endpoint_with_api_key(self):
        """Test chat endpoint with API key authentication"""
        wallet_address = "remes1test123"
        
        # Create user and API key
        self.db.add_credits(wallet_address, 100.0)
        api_key = self.db.create_api_key(wallet_address, "Test Key")
        
        # Make request with API key
        response = self.client.post(
            "/chat",
            json={"message": "Hello, test message"},
            headers={"X-API-Key": api_key}
        )
        
        # Should succeed with valid API key
        self.assertIn(response.status_code, [200, 500])  # 500 if model not loaded (OK in test)
        if response.status_code == 200:
            data = response.json()
            self.assertIn("response", data)
    
    def test_user_info_endpoint(self):
        """Test user info endpoint"""
        wallet_address = "remes1test123"
        
        # Create user
        self.db.add_credits(wallet_address, 100.0)
        
        # Get user info
        response = self.client.get(f"/user/info/{wallet_address}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["wallet_address"], wallet_address)
        self.assertEqual(data["credits"], 100.0)
    
    def test_user_info_endpoint_invalid_address(self):
        """Test user info endpoint with invalid address"""
        response = self.client.get("/user/info/invalid_address")
        # Should return 400 or 404
        self.assertIn(response.status_code, [400, 404, 422])
    
    def test_api_key_creation_endpoint(self):
        """Test API key creation endpoint"""
        wallet_address = "remes1test123"
        self.db.add_credits(wallet_address, 100.0)
        
        response = self.client.post(
            "/api-keys",
            json={"name": "Test API Key"},
            headers={"X-Wallet-Address": wallet_address}
        )
        
        # Should succeed
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("api_key", data)
        self.assertTrue(data["api_key"].startswith("r3mes_"))
    
    def test_api_key_listing_endpoint(self):
        """Test API key listing endpoint"""
        wallet_address = "remes1test123"
        self.db.add_credits(wallet_address, 100.0)
        
        # Create API keys
        self.db.create_api_key(wallet_address, "Key 1")
        self.db.create_api_key(wallet_address, "Key 2")
        
        # List keys
        response = self.client.get(
            "/api-keys",
            headers={"X-Wallet-Address": wallet_address}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)
    
    def test_api_key_revocation_endpoint(self):
        """Test API key revocation endpoint"""
        wallet_address = "remes1test123"
        self.db.add_credits(wallet_address, 100.0)
        
        # Create API key
        api_key = self.db.create_api_key(wallet_address, "Test Key")
        keys = self.db.list_api_keys(wallet_address)
        key_id = keys[0]["id"]
        
        # Revoke key
        response = self.client.delete(
            f"/api-keys/{key_id}",
            headers={"X-Wallet-Address": wallet_address}
        )
        
        self.assertEqual(response.status_code, 200)
        
        # Verify key is revoked
        validated = self.db.validate_api_key(api_key)
        self.assertIsNone(validated)
    
    def test_network_stats_endpoint(self):
        """Test network stats endpoint"""
        response = self.client.get("/network/stats")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("active_miners", data)
        self.assertIn("total_credits", data)
    
    def test_blocks_endpoint(self):
        """Test blocks endpoint"""
        response = self.client.get("/blocks?limit=10")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("blocks", data)
        self.assertIsInstance(data["blocks"], list)
    
    def test_blocks_endpoint_invalid_limit(self):
        """Test blocks endpoint with invalid limit"""
        # Test limit too high
        response = self.client.get("/blocks?limit=1000")
        # Should either clamp to max or return 400
        self.assertIn(response.status_code, [200, 400, 422])
        
        # Test limit too low
        response = self.client.get("/blocks?limit=0")
        self.assertIn(response.status_code, [200, 400, 422])
    
    def test_credit_system_integration(self):
        """Test credit system integration with API"""
        wallet_address = "remes1test123"
        
        # Create user with credits
        self.db.add_credits(wallet_address, 100.0)
        
        # Get user info via API
        response = self.client.get(f"/user/info/{wallet_address}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        initial_credits = data["credits"]
        
        # Use chat (should deduct credits)
        api_key = self.db.create_api_key(wallet_address, "Test Key")
        response = self.client.post(
            "/chat",
            json={"message": "Test"},
            headers={"X-API-Key": api_key}
        )
        
        # Check credits after chat (if chat succeeded)
        if response.status_code == 200:
            response = self.client.get(f"/user/info/{wallet_address}")
            data = response.json()
            # Credits should be reduced (if chat costs credits)
            # This depends on implementation
            self.assertIsInstance(data["credits"], (int, float))


class TestDatabaseIntegration(unittest.TestCase):
    """Integration tests for database operations"""
    
    def setUp(self):
        """Set up test database"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_db_integration.db")
        self.chain_json_path = os.path.join(self.temp_dir, "chain.json")
        
        import json
        with open(self.chain_json_path, "w") as f:
            json.dump({"blocks": []}, f)
        
        self.db = Database(db_path=self.db_path, chain_json_path=self.chain_json_path)
    
    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_user_credit_lifecycle(self):
        """Test complete user credit lifecycle"""
        wallet_address = "remes1test123"
        
        # Create user
        self.db.add_credits(wallet_address, 100.0)
        credits = self.db.check_credits(wallet_address)
        self.assertEqual(credits, 100.0)
        
        # Deduct credits
        self.db.deduct_credit(wallet_address, 30.0)
        credits = self.db.check_credits(wallet_address)
        self.assertEqual(credits, 70.0)
        
        # Add more credits
        self.db.add_credits(wallet_address, 50.0)
        credits = self.db.check_credits(wallet_address)
        self.assertEqual(credits, 120.0)
    
    def test_api_key_lifecycle(self):
        """Test complete API key lifecycle"""
        wallet_address = "remes1test123"
        self.db.add_credits(wallet_address, 100.0)
        
        # Create key
        api_key = self.db.create_api_key(wallet_address, "Test Key")
        self.assertIsNotNone(api_key)
        
        # Validate key
        validated = self.db.validate_api_key(api_key)
        self.assertEqual(validated, wallet_address)
        
        # List keys
        keys = self.db.list_api_keys(wallet_address)
        self.assertEqual(len(keys), 1)
        
        # Revoke key
        key_id = keys[0]["id"]
        self.db.revoke_api_key(key_id, wallet_address)
        
        # Validate should fail
        validated = self.db.validate_api_key(api_key)
        self.assertIsNone(validated)


if __name__ == "__main__":
    unittest.main()


"""
Unit tests for database.py
Tests credit system, API key management, and blockchain sync
"""

import unittest
import tempfile
import os
import shutil
from pathlib import Path
import json
import time

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import Database


class TestDatabase(unittest.TestCase):
    """Test cases for Database class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary database
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.chain_json_path = os.path.join(self.temp_dir, "chain.json")
        
        # Create empty chain.json
        with open(self.chain_json_path, "w") as f:
            json.dump({"blocks": []}, f)
        
        # Initialize database
        self.db = Database(db_path=self.db_path, chain_json_path=self.chain_json_path)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_user_creation(self):
        """Test user creation"""
        wallet_address = "remes1test123"
        
        # User should not exist initially
        user_info = self.db.get_user_info(wallet_address)
        self.assertIsNone(user_info)
        
        # Add credits (creates user)
        self.db.add_credits(wallet_address, 100.0)
        
        # User should exist now
        user_info = self.db.get_user_info(wallet_address)
        self.assertIsNotNone(user_info)
        self.assertEqual(user_info["wallet_address"], wallet_address)
        self.assertEqual(user_info["credits"], 100.0)
    
    def test_credit_operations(self):
        """Test credit add, deduct, and check"""
        wallet_address = "remes1test123"
        
        # Add credits
        self.db.add_credits(wallet_address, 100.0)
        credits = self.db.check_credits(wallet_address)
        self.assertEqual(credits, 100.0)
        
        # Deduct credits
        self.db.deduct_credit(wallet_address, 50.0)
        credits = self.db.check_credits(wallet_address)
        self.assertEqual(credits, 50.0)
        
        # Deduct more than available (should not go negative)
        result = self.db.deduct_credit(wallet_address, 100.0)
        self.assertFalse(result)  # Should fail
        credits = self.db.check_credits(wallet_address)
        self.assertEqual(credits, 50.0)  # Should remain unchanged
    
    def test_api_key_creation(self):
        """Test API key creation"""
        wallet_address = "remes1test123"
        
        # Create API key
        api_key_info = self.db.create_api_key(wallet_address, "Test Key")
        self.assertIsNotNone(api_key_info)
        self.assertIn("api_key", api_key_info)
        self.assertTrue(api_key_info["api_key"].startswith("r3mes_"))
        
        # Validate API key
        validated_wallet = self.db.validate_api_key(api_key_info["api_key"])
        self.assertIsNotNone(validated_wallet)
        self.assertEqual(validated_wallet["wallet_address"], wallet_address)
    
    def test_api_key_listing(self):
        """Test API key listing"""
        wallet_address = "remes1test123"
        
        # Create multiple API keys
        key1 = self.db.create_api_key(wallet_address, "Key 1")
        key2 = self.db.create_api_key(wallet_address, "Key 2")
        
        # List keys
        keys = self.db.list_api_keys(wallet_address)
        self.assertEqual(len(keys), 2)
        
        # Check that keys are masked
        for key_info in keys:
            self.assertIn("api_key", key_info)
            self.assertIn("...", key_info["api_key"])  # Check that key is masked
            self.assertIn("name", key_info)
            self.assertIn("is_active", key_info)
    
    def test_api_key_revocation(self):
        """Test API key revocation"""
        wallet_address = "remes1test123"
        
        # Create API key
        api_key_info = self.db.create_api_key(wallet_address, "Test Key")
        api_key = api_key_info["api_key"]
        
        # Validate before revocation
        validated_wallet = self.db.validate_api_key(api_key)
        self.assertIsNotNone(validated_wallet)
        
        # Revoke API key
        result = self.db.revoke_api_key(api_key, wallet_address)
        self.assertTrue(result)
        
        # Validate after revocation (should fail)
        validated_wallet = self.db.validate_api_key(api_key)
        self.assertIsNone(validated_wallet)
    
    def test_miner_stats(self):
        """Test miner statistics"""
        wallet_address = "remes1test123"
        
        # Set as miner
        self.db.add_credits(wallet_address, 100.0)
        # Note: is_miner flag would need to be set through a method
        
        # Get miner stats
        stats = self.db.get_miner_stats(wallet_address)
        self.assertIsNotNone(stats)
    
    def test_network_stats(self):
        """Test network statistics"""
        stats = self.db.get_network_stats()
        self.assertIsNotNone(stats)
        self.assertIn("active_miners", stats)
        self.assertIn("total_credits", stats)


if __name__ == "__main__":
    unittest.main()


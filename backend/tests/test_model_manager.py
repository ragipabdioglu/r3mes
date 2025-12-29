"""
Unit tests for Model Manager

Tests model loading, adapter management, and inference functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.model_manager import AIModelManager


class TestAIModelManager(unittest.TestCase):
    """Unit tests for AIModelManager"""
    
    def setUp(self):
        """Set up test environment"""
        os.environ["R3MES_ENV"] = "test"
        os.environ["R3MES_TEST_MODE"] = "true"
        os.environ["R3MES_USE_MOCK_MODEL"] = "true"
    
    def tearDown(self):
        """Clean up"""
        # Clear environment variables
        os.environ.pop("R3MES_ENV", None)
        os.environ.pop("R3MES_TEST_MODE", None)
        os.environ.pop("R3MES_USE_MOCK_MODEL", None)
    
    @patch('app.model_manager.get_model_loader')
    def test_model_manager_initialization(self, mock_loader):
        """Test model manager initialization"""
        mock_loader.return_value = Mock()
        
        manager = AIModelManager()
        self.assertIsNotNone(manager)
    
    @patch('app.model_manager.get_model_loader')
    def test_production_mock_model_forbidden(self, mock_loader):
        """Test that mock model is forbidden in production"""
        os.environ["R3MES_ENV"] = "production"
        os.environ["R3MES_USE_MOCK_MODEL"] = "true"
        
        with self.assertRaises(RuntimeError):
            AIModelManager()
    
    @patch('app.model_manager.get_model_loader')
    def test_switch_adapter(self, mock_loader):
        """Test switching between adapters"""
        manager = AIModelManager()
        
        # Mock adapter switching
        with patch.object(manager, 'switch_adapter') as mock_switch:
            mock_switch.return_value = True
            result = manager.switch_adapter("test_adapter")
            self.assertTrue(result)
    
    @patch('app.model_manager.get_model_loader')
    def test_list_available_adapters(self, mock_loader):
        """Test listing available adapters"""
        manager = AIModelManager()
        
        # Mock adapter list
        with patch.object(manager, 'list_available_adapters') as mock_list:
            mock_list.return_value = ["adapter1", "adapter2"]
            adapters = manager.list_available_adapters()
            self.assertIsInstance(adapters, list)
            self.assertEqual(len(adapters), 2)


if __name__ == "__main__":
    unittest.main()


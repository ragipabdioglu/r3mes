"""
Unit tests for Semantic Router

Tests semantic routing, embedding-based adapter selection, and fallback mechanisms.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.semantic_router import SemanticRouter


class TestSemanticRouter(unittest.TestCase):
    """Unit tests for SemanticRouter"""
    
    def setUp(self):
        """Set up test environment"""
        os.environ["R3MES_ENV"] = "test"
        os.environ["R3MES_TEST_MODE"] = "true"
    
    def tearDown(self):
        """Clean up"""
        os.environ.pop("R3MES_ENV", None)
        os.environ.pop("R3MES_TEST_MODE", None)
    
    def test_router_initialization(self):
        """Test router initialization"""
        router = SemanticRouter()
        self.assertIsNotNone(router)
        self.assertTrue(router.use_semantic)
        self.assertEqual(router.similarity_threshold, 0.7)
    
    def test_router_initialization_keyword_only(self):
        """Test router initialization with semantic disabled"""
        router = SemanticRouter(use_semantic=False)
        self.assertFalse(router.use_semantic)
        self.assertIsNotNone(router.keyword_router)
    
    @patch('app.semantic_router.SentenceTransformer')
    def test_route_selection_with_semantic(self, mock_sentence_transformer):
        """Test route selection using semantic similarity"""
        # Mock embedding model
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_sentence_transformer.return_value = mock_model
        
        router = SemanticRouter(use_semantic=True, similarity_threshold=0.5)
        
        # Mock cosine similarity calculation
        with patch('app.semantic_router.np.dot') as mock_dot:
            mock_dot.return_value = 0.8  # High similarity
            
            # Test route selection
            adapter = router.route("How do I write Python code?")
            
            # Should return an adapter (either semantic or keyword fallback)
            self.assertIsNotNone(adapter)
            self.assertIsInstance(adapter, str)
    
    def test_route_selection_keyword_fallback(self):
        """Test route selection with keyword fallback"""
        router = SemanticRouter(use_semantic=False)
        
        # Test with code-related message
        adapter = router.route("How to write a Python function?")
        self.assertIsNotNone(adapter)
        self.assertIsInstance(adapter, str)
    
    def test_add_route_example(self):
        """Test adding example to route"""
        router = SemanticRouter()
        
        # Add example to existing route
        router.add_route_example("coder_adapter", "New coding example")
        
        # Verify example was added
        self.assertIn("New coding example", router.route_definitions["coder_adapter"])
    
    def test_route_definitions_exist(self):
        """Test that route definitions are initialized"""
        router = SemanticRouter()
        
        # Check that route definitions exist
        self.assertIn("coder_adapter", router.route_definitions)
        self.assertIn("law_adapter", router.route_definitions)
        self.assertIn("default_adapter", router.route_definitions)
        
        # Check that each route has examples
        for route, examples in router.route_definitions.items():
            self.assertIsInstance(examples, list)
            self.assertGreater(len(examples), 0)


if __name__ == "__main__":
    unittest.main()


"""Tests for qdrant_init module."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.qdrant_init import get_qdrant_connection
from qdrant_client import QdrantClient


class TestQdrantInit(unittest.TestCase):
    """Test Qdrant initialization utilities."""

    def setUp(self):
        """Load environment variables before each test."""
        load_dotenv()

    def test_get_qdrant_connection_returns_client(self):
        """Test that get_qdrant_connection returns a QdrantClient instance."""
        with patch.dict(os.environ, {
            'QDRANT_URL': 'http://test-url:6333',
            'QDRANT_API_KEY': 'test-key-123'
        }):
            with patch('qdrant_lib.src.qdrant_init.load_dotenv'):
                client = get_qdrant_connection()
                self.assertIsInstance(client, QdrantClient)

    def test_get_qdrant_connection_uses_env_vars(self):
        """Test that get_qdrant_connection uses QDRANT_URL and QDRANT_API_KEY."""
        with patch.dict(os.environ, {
            'QDRANT_URL': 'http://test-url:6333',
            'QDRANT_API_KEY': 'test-key-123'
        }):
            with patch('qdrant_lib.src.qdrant_init.load_dotenv'):
                client = get_qdrant_connection()
                self.assertIsNotNone(client)
                self.assertIsInstance(client, QdrantClient)

    def test_missing_qdrant_url_exits(self):
        """Test that missing QDRANT_URL causes SystemExit."""
        with patch.dict(os.environ, {
            'QDRANT_API_KEY': 'test-key-123'
        }, clear=True):
            with patch('qdrant_lib.src.qdrant_init.load_dotenv'):
                with patch('builtins.print') as mock_print:
                    with self.assertRaises(SystemExit) as context:
                        get_qdrant_connection()
                    self.assertEqual(context.exception.code, 1)
                    mock_print.assert_called_with('Error: QDRANT_URL not defined in environment variables')

    def test_missing_qdrant_api_key_exits(self):
        """Test that missing QDRANT_API_KEY causes SystemExit."""
        with patch.dict(os.environ, {
            'QDRANT_URL': 'http://test-url:6333'
        }, clear=True):
            with patch('src.qdrant_init.load_dotenv'):
                with self.assertRaises(SystemExit) as context:
                    get_qdrant_connection()
                self.assertEqual(context.exception.code, 1)

    def test_both_env_vars_required(self):
        """Test that both QDRANT_URL and QDRANT_API_KEY are required."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('qdrant_lib.src.qdrant_init.load_dotenv'):
                with patch('builtins.print'):
                    with self.assertRaises(SystemExit):
                        get_qdrant_connection()


if __name__ == '__main__':
    unittest.main()

"""Tests for qdrant_init module."""

import os
import unittest
from unittest.mock import patch
from dotenv import load_dotenv

from qdrant_lib.qdrant_init import get_qdrant_connection
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
            with patch('qdrant_lib.qdrant_init.load_dotenv'):
                client = get_qdrant_connection()
                self.assertIsInstance(client, QdrantClient)

    def test_get_qdrant_connection_uses_env_vars(self):
        """Test that get_qdrant_connection uses QDRANT_URL and QDRANT_API_KEY."""
        with patch.dict(os.environ, {
            'QDRANT_URL': 'http://test-url:6333',
            'QDRANT_API_KEY': 'test-key-123'
        }):
            with patch('qdrant_lib.qdrant_init.load_dotenv'):
                client = get_qdrant_connection()
                self.assertIsNotNone(client)
                self.assertIsInstance(client, QdrantClient)

    def test_missing_qdrant_url_raises(self):
        """Test that missing QDRANT_URL raises EnvironmentError."""
        with patch.dict(os.environ, {
            'QDRANT_API_KEY': 'test-key-123'
        }, clear=True):
            with patch('qdrant_lib.qdrant_init.load_dotenv'):
                with self.assertRaises(EnvironmentError) as context:
                    get_qdrant_connection()
                self.assertEqual(str(context.exception), 'QDRANT_URL not defined in environment variables')

    def test_missing_qdrant_api_key_raises(self):
        """Test that missing QDRANT_API_KEY raises EnvironmentError."""
        with patch.dict(os.environ, {
            'QDRANT_URL': 'http://test-url:6333'
        }, clear=True):
            with patch('qdrant_lib.qdrant_init.load_dotenv'):
                with self.assertRaises(EnvironmentError) as context:
                    get_qdrant_connection()
                self.assertEqual(str(context.exception), 'QDRANT_API_KEY not defined in environment variables')

    def test_both_env_vars_required(self):
        """Test that both QDRANT_URL and QDRANT_API_KEY are required."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('qdrant_lib.qdrant_init.load_dotenv'):
                with self.assertRaises(EnvironmentError):
                    get_qdrant_connection()

    def test_get_qdrant_connection_memory_instance(self):
        """Test that get_qdrant_connection creates an in-memory instance with location=':memory:'."""
        client = get_qdrant_connection(location=":memory:")
        self.assertIsInstance(client, QdrantClient)

    def test_memory_instance_does_not_require_env_vars(self):
        """Test that in-memory instance can be created without environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('qdrant_lib.qdrant_init.load_dotenv'):
                client = get_qdrant_connection(location=":memory:")
                self.assertIsInstance(client, QdrantClient)


if __name__ == '__main__':
    unittest.main()

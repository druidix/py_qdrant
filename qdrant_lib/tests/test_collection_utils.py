"""Tests for collection_utils module."""

import unittest

from qdrant_lib.qdrant_init import get_qdrant_connection
from qdrant_lib.collection_utils import get_or_create_collection
from qdrant_client import models

from qdrant_lib.tests.testutils import TEST_COLLECTION_PREFIX, cleanup_test_collections, pre_test_cleanup

TEST_COLLECTION_NEW = f"{TEST_COLLECTION_PREFIX}get_or_create_new"
TEST_COLLECTION_EXISTING = f"{TEST_COLLECTION_PREFIX}get_or_create_existing"
TEST_COLLECTION_CUSTOM_CONFIG = f"{TEST_COLLECTION_PREFIX}get_or_create_custom_config"
ALL_TEST_COLLECTIONS = [TEST_COLLECTION_NEW, TEST_COLLECTION_EXISTING, TEST_COLLECTION_CUSTOM_CONFIG]


class TestCollectionUtils(unittest.TestCase):
    """Test Qdrant collection utilities."""

    @classmethod
    def setUpClass(cls):
        """Connect to Qdrant and remove any leftover test collections from a prior run."""
        cls.client = get_qdrant_connection()
        pre_test_cleanup(cls.client, ALL_TEST_COLLECTIONS)

    def test_creates_collection_when_missing(self):
        """get_or_create_collection creates the collection if it doesn't exist."""
        collections_before = {c.name for c in self.client.get_collections().collections}
        self.assertNotIn(TEST_COLLECTION_NEW, collections_before)

        result = get_or_create_collection(
            client=self.client,
            collection_name=TEST_COLLECTION_NEW,
        )

        self.assertIsInstance(result, models.CollectionInfo)
        collections_after = {c.name for c in self.client.get_collections().collections}
        self.assertIn(TEST_COLLECTION_NEW, collections_after)

        cleanup_test_collections(self.client, [TEST_COLLECTION_NEW])

    def test_returns_existing_collection_without_recreating(self):
        """get_or_create_collection returns a reference without recreating an existing collection."""
        self.client.create_collection(
            collection_name=TEST_COLLECTION_EXISTING,
            vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
        )

        result = get_or_create_collection(
            client=self.client,
            collection_name=TEST_COLLECTION_EXISTING,
        )

        self.assertIsInstance(result, models.CollectionInfo)
        collections = [c.name for c in self.client.get_collections().collections]
        self.assertEqual(collections.count(TEST_COLLECTION_EXISTING), 1)

        cleanup_test_collections(self.client, [TEST_COLLECTION_EXISTING])

    def test_respects_custom_vectors_config(self):
        """get_or_create_collection uses a custom vectors_config when creating a new collection."""
        custom_config = models.VectorParams(size=8, distance=models.Distance.DOT)

        get_or_create_collection(
            client=self.client,
            collection_name=TEST_COLLECTION_CUSTOM_CONFIG,
            vectors_config=custom_config,
        )

        info = self.client.get_collection(TEST_COLLECTION_CUSTOM_CONFIG)
        self.assertEqual(info.config.params.vectors.size, 8)
        self.assertEqual(info.config.params.vectors.distance, models.Distance.DOT)

        cleanup_test_collections(self.client, [TEST_COLLECTION_CUSTOM_CONFIG])


if __name__ == '__main__':
    unittest.main()

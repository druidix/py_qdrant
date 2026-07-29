"""Helpers for creating and tearing down Qdrant collections used in tests.

All collections managed by these helpers must be named with
``TEST_COLLECTION_PREFIX`` as a guardrail against accidentally deleting
real, non-test collections.
"""

import logging
from collections.abc import Iterable

from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

TEST_COLLECTION_PREFIX = "test_"


def _assert_test_collection_names(names: Iterable[str]) -> None:
    non_test_names = [name for name in names if not name.startswith(TEST_COLLECTION_PREFIX)]
    if non_test_names:
        raise ValueError(
            f"Refusing to operate on collections without the '{TEST_COLLECTION_PREFIX}' "
            f"prefix: {non_test_names}"
        )


def _delete_collections(client: QdrantClient, names: Iterable[str]) -> list[str]:
    """Delete the named collections, returning names that failed to delete."""
    names = list(names)
    _assert_test_collection_names(names)

    existing = {coll.name for coll in client.get_collections().collections}
    failures = []

    for name in names:
        if name not in existing:
            continue
        try:
            client.delete_collection(name)
        except Exception:
            logger.warning("Failed to delete test collection %r", name, exc_info=True)
            failures.append(name)

    return failures


def pre_test_cleanup(client: QdrantClient, names: Iterable[str]) -> None:
    """
    Delete any leftover test collections before a test run begins.

    Raises:
        RuntimeError: If any of the named collections could not be deleted.
    """
    failures = _delete_collections(client, names)
    if failures:
        raise RuntimeError(
            f"Fatal: pre-test cleanup failed to delete existing test collections: {failures}"
        )


def cleanup_test_collections(client: QdrantClient, names: Iterable[str]) -> None:
    """
    Delete test collections after a successful test run.

    Raises:
        RuntimeError: If any of the named collections could not be deleted.
    """
    failures = _delete_collections(client, names)
    if failures:
        raise RuntimeError(
            f"Fatal: post-test cleanup failed to delete test collections: {failures}"
        )

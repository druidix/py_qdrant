"""Qdrant collection utilities."""

from qdrant_client import QdrantClient, models


def get_or_create_collection(
    *,
    client: QdrantClient,
    collection_name: str,
    vectors_config: models.VectorParams | None = None,
) -> models.CollectionInfo:
    """
    Return a reference to an existing collection, creating it first if needed.

    Args:
        client: An authenticated QdrantClient connection.
        collection_name: Name of the collection to fetch or create.
        vectors_config: Vector parameters to use if the collection must be
            created. Defaults to a freshly constructed size=4, COSINE config.

    Returns:
        CollectionInfo: A reference to the (possibly newly created) collection.
    """
    if vectors_config is None:
        vectors_config = models.VectorParams(size=4, distance=models.Distance.COSINE)

    collections_list = client.get_collections()
    collection_exists = any(
        coll.name == collection_name for coll in collections_list.collections
    )

    if not collection_exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
        )

    return client.get_collection(collection_name)

"""Qdrant collection utilities."""

from qdrant_client import QdrantClient, models


def get_or_create_collection(
    *,
    client: QdrantClient,
    collection_name: str,
    vectors_config: models.VectorParams | dict[str, models.VectorParams] | None = None,
    sparse_vectors_config: dict[str, models.SparseVectorParams] | None = None,
    **kwargs,
) -> models.CollectionInfo:
    """
    Return a reference to an existing collection, creating it first if needed.

    Args:
        client: An authenticated QdrantClient connection.
        collection_name: Name of the collection to fetch or create.
        vectors_config: Vector parameters to use if the collection must be
            created. Can be a single VectorParams or a dict of named vectors.
            Defaults to a freshly constructed size=4, COSINE config.
        sparse_vectors_config: Sparse vector parameters to use if the collection
            must be created. Maps sparse vector names to SparseVectorParams.
        **kwargs: Additional keyword arguments to pass to
            client.create_collection, such as hnsw_config, optimizers_config,
            quantization_config, etc.

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
        create_kwargs = {
            "collection_name": collection_name,
            "vectors_config": vectors_config,
            **kwargs,
        }
        if sparse_vectors_config is not None:
            create_kwargs["sparse_vectors_config"] = sparse_vectors_config

        client.create_collection(**create_kwargs)

    return client.get_collection(collection_name)

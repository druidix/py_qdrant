#!/usr/bin/env python3

from qdrant_lib import (
    authenticate_huggingface,
    get_or_create_collection,
    get_qdrant_connection,
)
from qdrant_client import models
from sentence_transformers import SentenceTransformer
import time

client = get_qdrant_connection()

collection_name = "day3_hybrid_search"

# Create hybrid collection
collection = get_or_create_collection(
    collection_name=collection_name,
    vectors_config={
        "dense": models.VectorParams(size=384, distance=models.Distance.COSINE)
    },
    sparse_vectors_config={
        "sparse": models.SparseVectorParams(
            index=models.SparseIndexParams(on_disk=False)
        )
    }
)


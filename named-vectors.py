#!/usr/bin/env python3

# NOTE:  The Qdrant course had us build this script up to the point
# of the upsert, but didn't actually do anything else with it.

from qdrant_client import models
from qdrant_lib import get_qdrant_connection, get_or_create_collection
import json

client = get_qdrant_connection()
collection_name = "kaushik_named_vectors_1"

named_vector_coll = get_or_create_collection(
    client=client, 
    collection_name=collection_name,
    vectors_config={
        "image": models.VectorParams(size=4, distance=models.Distance.DOT),
        "text": models.VectorParams(size=5, distance=models.Distance.COSINE),
    },
    sparse_vectors_config={
        "text-sparse": models.SparseVectorParams()
        },
)

# Qdrant accepts mixed dense/sparse vectors in a dict.
# Pylance can't fully express that union so using the
# ignore directive to silence the warning
client.upsert(
    collection_name=collection_name,
    points=[
        models.PointStruct(
            id=1,
            vector={
                "image": [0.9, 0.1, 0.1, 0.2],
                "text": [0.4, 0.7, 0.1, 0.8, 0.1],
                "text-sparse": {
                    "indices": [1, 3, 5, 7],
                    "values": [0.1, 0.2, 0.3, 0.4],
                }, # type: ignore
            },
        ),
    ],
)


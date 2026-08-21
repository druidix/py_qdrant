#!/usr/bin/env python3

from qdrant_client import models

from qdrant_lib import (
    get_or_create_collection,
    get_qdrant_connection,
)

client = get_qdrant_connection()

# We create two collections below for comparison purposes
collection_name_standard = "sparse_vectors_collection"

# Create the collection with sparse vectors
get_or_create_collection(
    collection_name=collection_name_standard,
    sparse_vectors_config={ #vector named "sparse_vector"
        "sparse_vector": models.SparseVectorParams(),
    },
)

collection_name_custom_index = "sparse_vectors_collection_custom_index"

get_or_create_collection(
    collection_name=collection_name_custom_index,
    sparse_vectors_config={
        "sparse_vector": models.SparseVectorParams(
            index=models.SparseIndexParams( #inverted index parameters
                full_scan_threshold=0, #full scan search, not using inverted index
                on_disk=False, #where inverted index is stored
                datatype=models.VectorStorageDatatype("float32") #precision of values stored in inverted index

            )
        ),
    },
)

# Insert vectors into the collection
client.upsert(
    collection_name=collection_name_standard,
    points=[
        models.PointStruct(
            id=1,
            payload={},
            vector={ #vector named "sparse_vector"
                "sparse_vector": models.SparseVector(
                    indices=[1, 2, 3], #uint32, from 0 to 4_294_967_295
                    values=[0.2, -0.2, 0.2] #stored as floats
                )
            },
        ),
        models.PointStruct(
            id=2,
            payload={},
            vector={ #vector named "sparse_vector"
                "sparse_vector": models.SparseVector(
                    indices=[1, 5], #uint32, from 0 to 4_294_967_295
                    values=[0.1, 0.1] #stored as floats
                )
            },
        ),
    ],
)
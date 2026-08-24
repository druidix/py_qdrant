#!/usr/bin/env python3

from qdrant_client import models
from qdrant_lib import (
    authenticate_huggingface,
    get_or_create_collection,
    get_qdrant_connection,
)

#Qdrant/bm25 is downloaded from the Hugging Face Hub the first time FastEmbed
#runs, so the token must be in the environment before the upsert below.
authenticate_huggingface()

client = get_qdrant_connection()
collection_name = "bm25_vectors_collection"

get_or_create_collection(
    client= client,
    collection_name=collection_name,
    sparse_vectors_config={
        "bm25_sparse_vector": models.SparseVectorParams(
            modifier=models.Modifier.IDF #Inverse Document Frequency
        ),
    },
)

grocery_items_descriptions = [
    "Grated hard cheese",
    "White crusty bread roll",
    "Mac and cheese"
]

#Estimating the average length of documents in the corpus
avg_document_length = sum(len(description.split()) for description in grocery_items_descriptions) / len(grocery_items_descriptions)

print(f"Average document length: {avg_document_length}")

client.upsert(
    collection_name="bm25_vectors_collection",
    points=[
        models.PointStruct(
            id=i,
            payload={"text": description}, #meta data, descriptions text in human-readable format
            vector={
                "bm25_sparse_vector": models.Document( #to run FastEmbed under the hood
                    text=description,
                    model="Qdrant/bm25",
                    options={"avg_len": avg_document_length} #To pass BM25 parameters, here we're using default k & b for the BM25 formula
                )
           },
        ) for i, description in enumerate(grocery_items_descriptions)
    ],
)

print(
        client.query_points(
        collection_name="bm25_vectors_collection",
        using="bm25_sparse_vector",
        limit=3,
        query=models.Document(  #to run FastEmbed under the hood
            text="cheese",
            model="Qdrant/bm25"
        ),
        with_vectors=True,
    )
)
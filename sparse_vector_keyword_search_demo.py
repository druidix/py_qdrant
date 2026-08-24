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
collection_name_bm_25 = "bm25_vector_collection"

get_or_create_collection(
    client= client,
    collection_name=collection_name_bm_25,
    sparse_vectors_config={
        "bm25_sparse_vector": models.SparseVectorParams(
            modifier=models.Modifier.IDF #Inverse Document Frequency
        ),
    },
)

grocery_item_descriptions = [
    "Grated hard cheese",
    "White crusty bread roll",
    "Mac and cheese"
]

#Estimating the average length of documents in the corpus
avg_document_length = sum(len(description.split()) for description in grocery_item_descriptions) / len(grocery_item_descriptions)

print(f"Average document length: {avg_document_length}")

client.upsert(
    collection_name=collection_name_bm_25,
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
        ) for i, description in enumerate(grocery_item_descriptions)
    ],
)

print(f"\nBM25 RETRIEVAL\n",
        client.query_points(
        collection_name=collection_name_bm_25,
        using="bm25_sparse_vector",
        limit=3,
        query=models.Document(  #to run FastEmbed under the hood
            text="cheese",
            model="Qdrant/bm25"
        ),
        with_vectors=True,
    )
)

collection_name_splade="splade_vector_collection"

get_or_create_collection(
    client=client,
    collection_name=collection_name_splade,
    sparse_vectors_config={
        "splade_sparse_vector": models.SparseVectorParams(),
    },
)

client.upsert(
    collection_name=collection_name_splade,
    points=[
        models.PointStruct(
            id=i,
            payload={"text": description}, #meta data, descriptions text in human-readable format
            vector={
                "splade_sparse_vector": models.Document( #to run FastEmbed under the hood
                    text=description,
                    model="prithivida/Splade_PP_en_v1"
                )
           },
        ) for i, description in enumerate(grocery_item_descriptions)
    ],
)

print(f"\nSEMANTIC NEURAL RETRIEVAL\n",
    client.query_points(
        collection_name="splade_vector_collection",
        using="splade_sparse_vector",
        limit=3,
        query=models.Document(
            text="A not soft cheese",
            model="prithivida/Splade_PP_en_v1"
        ),
        with_vectors=True,
    )
)
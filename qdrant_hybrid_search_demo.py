#!/usr/bin/env python3

import uuid
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

collection_name_hybrid = "hybrid_search_demo"

# Create our collection with both sparse (bm25) and dense vectors
get_or_create_collection(
    client=client,
    collection_name=collection_name_hybrid,
    vectors_config={
        "dense": models.VectorParams(
            distance=models.Distance.COSINE,
            size=384,
        ),
    },
    sparse_vectors_config={
        "sparse": models.SparseVectorParams(
            modifier=models.Modifier.IDF
        )
    }
)
documents = [
    "Aged Gouda develops a crystalline texture and nutty flavor profile after 18 months of maturation.",
    "Mature Gouda cheese becomes grainy and develops a rich, buttery taste with extended aging.",
    "Brie cheese features a soft, creamy interior surrounded by an edible white rind.",
    "This French cheese has a flowing, buttery center encased in a bloomy white crust.",
    "Fresh mozzarella pairs beautifully with ripe tomatoes and basil leaves.",
    "Classic Margherita pizza topped with tomato sauce, mozzarella, and fresh basil.",
    "Parmesan requires at least 12 months of cave aging to develop its signature sharp taste.",
    "Parmigiano-Reggiano's distinctive piquant flavor comes from extended maturation in controlled environments.",
    "Grilled cheese sandwiches are the ultimate American comfort food for cold winter days.",
    "Croque Monsieur combines ham and Gruyère in France's answer to the toasted cheese sandwich.",
]

client.upsert(
    collection_name=collection_name_hybrid,
    points=[
        models.PointStruct(
            id=uuid.uuid4().hex,
            vector={
                "dense": models.Document(
                    text=doc,
                    model="sentence-transformers/all-MiniLM-L6-v2",
                ),
                "sparse": models.Document(
                    text=doc,
                    model="Qdrant/bm25",
                ),
            },
            payload={"text": doc},
        )
        for i, doc in enumerate(documents)
    ]
)

def dense_search(query: str) -> list[models.ScoredPoint]:
    response = client.query_points(
        collection_name=collection_name_hybrid,
        query=models.Document(
            text=query,
            model="sentence-transformers/all-MiniLM-L6-v2",
        ),
        using="dense",
        limit=3,
    )
    return response.points

def sparse_search(query: str) -> list[models.ScoredPoint]:
    response = client.query_points(
        collection_name=collection_name_hybrid,
        query=models.Document(
            text=query,
            model="Qdrant/bm25",
        ),
        using="sparse",
        limit=3,
    )
    return response.points

queries = [
    "nutty aged cheese",
    "soft French cheese",
    "pizza ingredients",
    "a good lunch",
]

for query in queries:
    print("Query:", query)

    dense_results = dense_search(query)
    print("Dense Results:")
    for result in dense_results:
        print("\t-", result.payload["text"], result.score)

    sparse_results = sparse_search(query)
    print("Sparse Results:")
    for result in sparse_results:
        print("\t-", result.payload["text"], result.score)
    print()
    
    def rrf_search(query: str) -> list[models.ScoredPoint]:
        response = client.query_points(
            collection_name=collection_name_hybrid,
            prefetch=[
                models.Prefetch(
                    query=models.Document(
                        text=query,
                        model="Qdrant/bm25",
                    ),
                    using="sparse",
                    limit=3,
                ),
                models.Prefetch(
                    query=models.Document(
                        text=query,
                        model="sentence-transformers/all-MiniLM-L6-v2",
                    ),
                    using="dense",
                    limit=3,
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=3,
        )
        return response.points

for query in queries:
    print("Query:", query)

    rrf_results = rrf_search(query)
    print("RRF Results:")
    for result in rrf_results:
        print("\t-", result.payload["text"], result.score)
    print()
    
def dbsf_search(query: str) -> list[models.ScoredPoint]:
    response = client.query_points(
        collection_name=collection_name_hybrid,
        prefetch=[
            models.Prefetch(
                query=models.Document(
                    text=query,
                    model="Qdrant/bm25",
                ),
                using="sparse",
                limit=3,
            ),
            models.Prefetch(
                query=models.Document(
                    text=query,
                    model="sentence-transformers/all-MiniLM-L6-v2",
                ),
                using="dense",
                limit=3,
            )
        ],
        query=models.FusionQuery(fusion=models.Fusion.DBSF),
        limit=3,
    )
    return response.points

for query in queries:
    print("Query:", query)

    dbsf_results = dbsf_search(query)
    print("DBSF Results:")
    for result in dbsf_results:
        print("\t-", result.payload["text"], result.score)
    print()

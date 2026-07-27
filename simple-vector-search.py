#!/usr/bin/env python3

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
import json
import jsonpickle
import os
import sys

load_dotenv()

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

if not qdrant_url:
    print("Error: QDRANT_URL not defined in environment variables")
    sys.exit(1)
if not qdrant_api_key:
    print("Error: QDRANT_API_KEY not defined in environment variables")
    sys.exit(1)

client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

collection_name = 'kaushik_test_collection_1'

collections_list = client.get_collections()
collection_exists = any(coll.name == collection_name for coll in collections_list.collections)

if not collection_exists:
    print(f"DEBUG: Collection '{collection_name}' not found. Creating collection and adding points.")

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
    )

    client.create_payload_index(
        collection_name=collection_name,
        field_name="category",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

    points=[
        models.PointStruct(
            id=1,
            vector=[0.9, 0.1, 0.1, 0.8], # High affordability, high innovation
            payload={"name": "Budget Smartphone", "category": "electronics", "price": 299},
        ),
        models.PointStruct(
            id=2,
            vector=[0.2, 0.9, 0.8, 0.5], # High quality, high popularity
            payload={"name": "Bestselling Novel", "category": "books", "price": 19},
        ),
        models.PointStruct(
            id=3,
            vector=[0.8, 0.3, 0.2, 0.9], # High affordability, high innovation (similar to ID 1)
            payload={"name": "Smart Home Hub", "category": "electronics", "price": 89},
        ),

        # Low affordability, high quality, moderate popularity, high innovation
        models.PointStruct(
            id=4,
            vector=[0.2, 0.9, 0.3, 0.9],
            payload={"name": "Wireless earbuds", "category": "electronics", "price": 189},
        ),

        # High affordability, high quality, high popularity, low innovation
        models.PointStruct(
            id=5,
            vector=[0.9, 0.9, 0.7, 0.1],
            payload={"name": "Wired earbuds", "category": "electronics", "price": 19},
        ),

        # High affordability, moderate quality, high popularity, low innovation
        models.PointStruct(
            id=6,
            vector=[0.9, 0.4, 0.8, 0.1],
            payload={"name": "Umbrella", "category": "housewares", "price": 24},
        ),

        # High affordability, low quality, moderate popularity, low innovation
        models.PointStruct(
            id=7,
            vector=[0.9, 0.3, 0.4, 0.1],
            payload={"name": "Scissors", "category": "housewares", "price": 7},
        ),

        # High affordability, moderate quality, high popularity, high innovation
        models.PointStruct(
            id=8,
            vector=[0.9, 0.6, 0.7, 0.7],
            payload={"name": "E-book reader", "category": "electronics", "price": 67},
        ),
    ]

    client.upsert(collection_name=collection_name, points=points)
else:
    print(f"DEBUG: Collection '{collection_name}' already exists. Skipping creation and upsert.")

# Define a query vector for "affordable and innovative"
query_vector = [0.85, 0.2, 0.1, 0.9]

# 1. Basic similarity search
basic_results = client.query_points(collection_name, query=query_vector)
# print(f"\n\nBasic search results:", json.dumps(basic_results[0], indent=2))

print(f"\n\nBasic Results:\n")

results_data = [
    {
        "id": point.id,
        "score": point.score,
        "payload": point.payload
    }
    for point in basic_results.points
]

print(json.dumps(results_data, indent=2))

# 2. Filtered search (only find electronics)
filtered_results = client.query_points(
    collection_name,
    query=query_vector,
    query_filter=models.Filter(
        must=[models.FieldCondition(key="category", match=models.MatchValue(value="electronics"))]
    ),
)

print(f"\n\nFiltered Results:\n")

results_data = [
    {
        "id": point.id,
        "score": point.score,
        "payload": point.payload
    }
    for point in filtered_results.points
]

print(json.dumps(results_data, indent=2))
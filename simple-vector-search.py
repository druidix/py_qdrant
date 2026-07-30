#!/usr/bin/env python3

import json

from qdrant_client import models
from qdrant_lib import get_qdrant_connection, get_or_create_collection

client = get_qdrant_connection()

collection_name = 'kaushik_test_collection_1'

get_or_create_collection(
    client=client,
    collection_name=collection_name,
    vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
)

client.create_payload_index(
    collection_name=collection_name,
    field_name="category",
    field_schema=models.PayloadSchemaType.KEYWORD,
)

points = [
    models.PointStruct(
        id=1,
        vector=[0.9, 0.1, 0.1, 0.8],
        payload={"name": "Budget Smartphone", "category": "electronics", "price": 299},
    ),
    models.PointStruct(
        id=2,
        vector=[0.2, 0.9, 0.8, 0.5],
        payload={"name": "Bestselling Novel", "category": "books", "price": 19},
    ),
    models.PointStruct(
        id=3,
        vector=[0.8, 0.3, 0.2, 0.9],
        payload={"name": "Smart Home Hub", "category": "electronics", "price": 89},
    ),
    models.PointStruct(
        id=4,
        vector=[0.2, 0.9, 0.3, 0.9],
        payload={"name": "Wireless earbuds", "category": "electronics", "price": 189},
    ),
    models.PointStruct(
        id=5,
        vector=[0.9, 0.9, 0.7, 0.1],
        payload={"name": "Wired earbuds", "category": "electronics", "price": 19},
    ),
    models.PointStruct(
        id=6,
        vector=[0.9, 0.4, 0.8, 0.1],
        payload={"name": "Umbrella", "category": "housewares", "price": 24},
    ),
    models.PointStruct(
        id=7,
        vector=[0.9, 0.3, 0.4, 0.1],
        payload={"name": "Scissors", "category": "housewares", "price": 7},
    ),
    models.PointStruct(
        id=8,
        vector=[0.9, 0.6, 0.7, 0.7],
        payload={"name": "E-book reader", "category": "electronics", "price": 67},
    ),
]

client.upsert(collection_name=collection_name, points=points)

query_vector = [0.85, 0.2, 0.1, 0.9]

basic_results = client.query_points(collection_name, query=query_vector)

print("\n\nBasic Results:\n")

results_data = [
    {
        "id": point.id,
        "score": point.score,
        "payload": point.payload
    }
    for point in basic_results.points
]

print(json.dumps(results_data, indent=2))

filtered_results = client.query_points(
    collection_name,
    query=query_vector,
    query_filter=models.Filter(
        must=[models.FieldCondition(key="category", match=models.MatchValue(value="electronics"))]
    ),
)

print("\n\nFiltered Results:\n")

results_data = [
    {
        "id": point.id,
        "score": point.score,
        "payload": point.payload
    }
    for point in filtered_results.points
]

print(json.dumps(results_data, indent=2))

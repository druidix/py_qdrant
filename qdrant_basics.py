#!/usr/bin/env python3

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
import os

load_dotenv()

api_key = os.getenv("QDRANT_API_KEY")
if not api_key:
    print("Error: QDRANT_API_KEY not defined in environment variables")
    exit(1)

client = QdrantClient(url="https://2ec3c57d-71b6-449e-9d63-6c9404bfade3.us-west-1-0.aws.cloud.qdrant.io",
                      api_key=api_key)

# Retrieve and display the list of collections
collections = client.get_collections()
print(f"\nExisting collections:\n", collections)

my_test_collection = "kaushik_test"

collection_exists = any(coll.name == my_test_collection for coll in client.get_collections().collections)

if not collection_exists:
    client.create_collection(
        collection_name=my_test_collection,
        vectors_config=models.VectorParams(
            size=4,  # Dimensionality of the vectors
            distance=models.Distance.COSINE  # Distance metric for similarity search
        )
    )
    print(f"\n\nCreated collection: {my_test_collection}")
else:
    print(f"\n\nCollection {my_test_collection} already exists\n\n")

# Define the vectors to be inserted
points = [
    models.PointStruct(
        id=1,
        vector=[0.1, 0.2, 0.3, 0.4],  # 4D vector
        payload={"category": "example"}  # Metadata (optional)
    ),
    models.PointStruct(
        id=2,
        vector=[0.2, 0.3, 0.4, 0.5],
        payload={"category": "demo"}
    )
]

# Insert vectors into the collection
client.upsert(
    collection_name=my_test_collection,
    points=points
)

print(f'', client.get_collection(my_test_collection), "\n\n")

query_vector = [0.08, 0.14, 0.33, 0.28]

search_results = client.query_points(
    collection_name=my_test_collection,
    query=query_vector,
    limit=1  # Return the top 1 most similar vector
)

print(f"\n\nSearch results:", search_results, "\n\n")

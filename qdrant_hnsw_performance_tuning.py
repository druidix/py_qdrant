#!/usr/bin/env python3
"""
Uploads the dbpedia 100K-point dataset to Qdrant for HNSW performance tuning.

By default, the script is idempotent: if the collection already exists and
already holds all 100K points, the upload is skipped.

Options:
    --clean     Delete the collection before running, so it is rebuilt and
                repopulated from scratch. Use this to reset the collection
                after changing its config (e.g. hnsw_config) or to discard a
                partial/corrupted upload.
"""

import argparse
import time

import numpy as np
from datasets import load_dataset
from qdrant_client import models
from tqdm import tqdm

from qdrant_lib import (
    get_openai_connection,
    get_or_create_collection,
    get_qdrant_connection,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--clean",
    action="store_true",
    help="Delete the collection before running, starting over from scratch.",
)
args = parser.parse_args()

client = get_qdrant_connection()

def upload_batch_without_indexes(start_idx, end_idx):
    points = []
    for i in range(start_idx, min(end_idx, total_points)):
        example = ds['train'][i]

        # Get the embedding
        embedding = example['text-embedding-3-large-1536-embedding']

        # Create payload
        payload = {
            'text': example['text'],
            'title': example['title'],
            '_id': example['_id'],
            'length': len(example['text']),
            'has_numbers': any(char.isdigit() for char in example['text'])
        }

        points.append(models.PointStruct(
            id=i,
            vector=embedding,
            payload=payload
        ))

    if points:
        client.upload_points(collection_name=collection_name, points=points)
        return len(points)
    return 0

def get_query_embedding(text):
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            dimensions=1536
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting OpenAI embedding: {e}")
        print("Using random vector as fallback...")
        return np.random.normal(0, 1, 1536).tolist()

try:
    ds = load_dataset("Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-100K")
    collection_name = "dbpedia_100K"

    if args.clean:
        deleted = client.delete_collection(collection_name=collection_name)
        if deleted:
            print(f"Deleted existing collection {collection_name}\n")

    collection = get_or_create_collection(
        client=client,
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=1536,
            distance=models.Distance.COSINE
        ),
        hnsw_config=models.HnswConfigDiff(
            m=0,
            ef_construct=100,
            full_scan_threshold=10000
        ),
        strict_mode_config=models.StrictModeConfig(
            enabled=False,
            unindexed_filtering_retrieve=True  # Allow filtering without indexes
        )
    )

    print(f"Collection {collection_name} initialized\n")
    print("Dataset info:\n")
    print(ds)

    print("\nFirst example (proper access):")
    first_example = ds['train'][0]
    print(first_example)

    print("\nDataset features:")
    print(ds['train'].features)

    print("\nAvailable columns:")
    print(ds['train'].column_names)
    
    batch_size = 10000
    total_points = len(ds['train'])

    existing_points = client.count(collection_name=collection_name, exact=True).count

    if existing_points >= total_points:
        print(f"\nCollection {collection_name} already has {existing_points} points; skipping upload\n")
    else:
        print(f"Uploading {total_points} points in batches of {batch_size}")

        # Upload all batches
        total_uploaded = 0
        for i in tqdm(range(0, total_points, batch_size), desc="Uploading points"):
            uploaded = upload_batch_without_indexes(i, i + batch_size)
            total_uploaded += uploaded

        print(f"\nUpload completed! Total points uploaded: {total_uploaded}")

    openai_client = get_openai_connection()
    new_query = "artificial intelligence"

    # Get the embedding
    query_embedding = get_query_embedding(new_query)

    print(f"Embedding dimensions: {len(query_embedding)}")
    print(f"First 5 values: {query_embedding[:5]}")

    # Create a text-based filter, reused by both filtered-search benchmarks below
    text_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="text",
                match=models.MatchText(text="data")
            )
        ]
    )

    ### UNINDEXED FILTERED SEARCH ###
    # Runs while hnsw_config.m is still 0 and no payload index exists yet,
    # so this is a true "no indexes at all" baseline.
    print("Testing filtering without payload indexes")

    # Run multiple times for more reliable measurement
    unindexed_times = []
    for i in range(3):
        start_time = time.time()
        response = client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=10,
            search_params=models.SearchParams(hnsw_ef=100),
            query_filter=text_filter
        )
        unindexed_times.append((time.time() - start_time) * 1000)

    unindexed_filter_time = sum(unindexed_times) / len(unindexed_times)

    print(f"Filtered search (WITHOUT index): {unindexed_filter_time:.2f}ms")
    print(f"Individual times: {[f'{t:.2f}ms' for t in unindexed_times]}")
    print(f"Found {len(response.points)} matching results")
    if response.points:
        print(f"Top result: '{response.points[0].payload['text']}'\nScore: {response.points[0].score:.4f}")
    else:
        print("No results found - try a different filter term")

    ### BUILD PAYLOAD INDEX + HNSW GRAPH ###
    # Payload indexes must exist BEFORE the HNSW graph is built for the graph
    # to be filter-aware (see warning in the course docs). So create the
    # payload index first, while hnsw_config.m is still 0, and only then
    # trigger the HNSW build, so later searches benefit from the
    # filter-aware HNSW graph.
    client.create_payload_index(
        collection_name=collection_name,
        field_name="text",
        wait=True,
        field_schema=models.TextIndexParams(
            type="text",
            tokenizer="word",
            phrase_matching=False
        )
    )
    print("Payload index created for 'text' field")

    collection_info = client.get_collection(collection_name)
    current_m = collection_info.config.hnsw_config.m

    if current_m == 16:
        print("\nHNSW m already set to 16; skipping reindex\n")
    else:
        client.update_collection(
            collection_name=collection_name,
            hnsw_config=models.HnswConfigDiff(
                m=16,  # Updated from 0 to 16
            )
        )
        print("\nHNSW indexing enabled with m=16\n")

    ### BASELINE UNFILTERED SEARCH ###
    # Runs after the HNSW graph (m=16) is built, so this measures HNSW
    # search rather than the flat/brute-force scan used while m=0.
    print("\nRunning baseline performance test...\n")

    # Warm up the RAM index/vectors cache with a test query
    print("Warming up caches...\n")
    client.query_points(collection_name=collection_name, query=query_embedding, limit=1)

    # Measure vector search performance
    search_times = []
    for _ in range(3):  # Multiple runs for a stable average
        start_time = time.time()
        response = client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=10
        )
        search_time = (time.time() - start_time) * 1000
        search_times.append(search_time)

    baseline_time = sum(search_times) / len(search_times)

    print(f"Average search time: {baseline_time:.2f}ms")
    print(f"Search times: {[f'{t:.2f}ms' for t in search_times]}")
    print(f"Found {len(response.points)} results")
    print(f"Top result: '{response.points[0].payload['title']}' (score: {response.points[0].score:.4f})")

    ### INDEXED FILTERED SEARCH ###
    indexed_times = []
    for i in range(3):
        start_time = time.time()
        response = client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=10,
            search_params=models.SearchParams(hnsw_ef=100),
            query_filter=text_filter
        )
        indexed_times.append((time.time() - start_time) * 1000)

    indexed_filter_time = sum(indexed_times) / len(indexed_times)

    print(f"Filtered search (WITH index): {indexed_filter_time:.2f}ms")
    print(f"Individual times: {[f'{t:.2f}ms' for t in indexed_times]}")
    print(f"Overhead vs baseline: {indexed_filter_time - baseline_time:.2f}ms")
    print(f"Found {len(response.points)} matching results")
    if response.points:
        print(f"Top result: '{response.points[0].payload['text']}'\nScore: {response.points[0].score:.4f}")
    else:
        print("No results found - try a different filter term")

finally:
    client.close()
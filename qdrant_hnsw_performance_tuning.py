#!/usr/bin/env python3

from datasets import load_dataset
from qdrant_client import models
from tqdm import tqdm
import openai
import time
import os

from qdrant_lib import get_qdrant_connection, get_or_create_collection

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

try:
    ds = load_dataset("Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-100K")
    collection_name = "dbpedia_100K"

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

finally:
    client.close()
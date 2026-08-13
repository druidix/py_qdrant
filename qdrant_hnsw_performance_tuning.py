#!/usr/bin/env python3

from datasets import load_dataset
from qdrant_client import models
from tqdm import tqdm
import openai
import time
import os

from qdrant_lib import get_qdrant_connection, get_or_create_collection

client = get_qdrant_connection()

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
finally:
    client.close()
#!/usr/bin/env python3
"""
Day 3 pitstop project: hybrid search benchmark over a book dataset.

Loads data/hybrid_search_test_data.txt (500 books, each with a title and
description) into a Qdrant collection with both dense (semantic) and sparse
(BM25) vectors, then compares dense-only, sparse-only, and RRF-fused hybrid
search across a set of test queries, reporting average and P95 latency for
each method.
"""

import json
import statistics
import time
from typing import Callable

from qdrant_client import QdrantClient, models

from qdrant_lib import (
    authenticate_huggingface,
    get_or_create_collection,
    get_qdrant_connection,
)

DATA_PATH = "data/hybrid_search_test_data.txt"
COLLECTION_NAME = "day3_hybrid_search"
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_SIZE = 384
SPARSE_MODEL = "Qdrant/bm25"
SEARCH_LIMIT = 5
LATENCY_RUNS = 10

# Qdrant/bm25 is downloaded from the Hugging Face Hub the first time FastEmbed
# runs, so the token must be in the environment before the upsert below.
authenticate_huggingface()


def load_books(path: str) -> list[dict]:
    """Read the newline-delimited JSON book dataset into a list of dicts."""
    books = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                books.append(json.loads(line))
    return books


def upsert_books(client: QdrantClient, books: list[dict]) -> None:
    """Embed each book's title+description as dense and sparse vectors and upsert."""
    points = []
    for book in books:
        text = f"{book['title']}. {book['description']}"
        points.append(
            models.PointStruct(
                id=book["id"],
                vector={
                    "dense": models.Document(text=text, model=DENSE_MODEL),
                    "sparse": models.Document(text=text, model=SPARSE_MODEL),
                },
                payload=book,
            )
        )
    client.upload_points(collection_name=COLLECTION_NAME, points=points, wait=True)


def dense_search(client: QdrantClient, query: str) -> list[models.ScoredPoint]:
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=models.Document(text=query, model=DENSE_MODEL),
        using="dense",
        limit=SEARCH_LIMIT,
    )
    return response.points


def sparse_search(client: QdrantClient, query: str) -> list[models.ScoredPoint]:
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=models.Document(text=query, model=SPARSE_MODEL),
        using="sparse",
        limit=SEARCH_LIMIT,
    )
    return response.points


def rrf_search(client: QdrantClient, query: str) -> list[models.ScoredPoint]:
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=models.Document(text=query, model=SPARSE_MODEL),
                using="sparse",
                limit=SEARCH_LIMIT,
            ),
            models.Prefetch(
                query=models.Document(text=query, model=DENSE_MODEL),
                using="dense",
                limit=SEARCH_LIMIT,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=SEARCH_LIMIT,
    )
    return response.points


def print_results(label: str, results: list[models.ScoredPoint]) -> None:
    print(f"{label}:")
    for result in results:
        print(f"\t- {result.payload['title']} ({result.score:.4f})")


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (pct in [0, 100]) over values."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


def measure_latency(
    search_fn: Callable[[QdrantClient, str], list[models.ScoredPoint]],
    client: QdrantClient,
    query: str,
    runs: int = LATENCY_RUNS,
) -> list[float]:
    """Return per-run latencies in milliseconds for search_fn(client, query)."""
    times = []
    for _ in range(runs):
        start = time.time()
        search_fn(client, query)
        times.append((time.time() - start) * 1000)
    return times


client = get_qdrant_connection()

try:
    get_or_create_collection(
        client=client,
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(size=DENSE_SIZE, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF),
        },
    )

    books = load_books(DATA_PATH)
    print(f"Loaded {len(books)} books from {DATA_PATH}")

    existing_points = client.count(collection_name=COLLECTION_NAME, exact=True).count
    if existing_points >= len(books):
        print(f"Collection already has {existing_points} points; skipping upload\n")
    else:
        print(f"Upserting {len(books)} books into '{COLLECTION_NAME}'...")
        upsert_books(client, books)
        print("Upload complete\n")

    queries = [
        "a cozy mystery in a small village",
        "space exploration and survival",
        "a dark fantasy with old magic",
        "second chances and small-town romance",
    ]

    methods = [
        ("Dense", dense_search),
        ("Sparse", sparse_search),
        ("RRF Hybrid", rrf_search),
    ]

    # Comparative results per query
    for query in queries:
        print(f"Query: {query!r}")
        for label, search_fn in methods:
            print_results(label, search_fn(client, query))
        print()

    # Latency benchmarking (average + P95) per method, aggregated across queries
    print("=" * 60)
    print("LATENCY BENCHMARK")
    print("=" * 60)

    # Warm up caches so the first measured run of each method isn't penalized
    # by cold-start cost (model load, connection setup, etc.).
    for _, search_fn in methods:
        search_fn(client, queries[0])

    for label, search_fn in methods:
        all_times = []
        for query in queries:
            all_times.extend(measure_latency(search_fn, client, query))

        avg_time = statistics.mean(all_times)
        p95_time = percentile(all_times, 95)
        print(f"{label:12s} avg: {avg_time:7.2f}ms   p95: {p95_time:7.2f}ms")

finally:
    client.close()

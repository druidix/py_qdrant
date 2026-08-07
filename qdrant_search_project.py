#!/usr/bin/env python3

import importlib.util
import os
import sys
import uuid

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import models
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core import Document
from transformers import AutoTokenizer

# Homegrown utils library
from qdrant_lib import get_qdrant_connection, get_or_create_collection

# Used for text chunking below.
MAX_TOKENS = 256

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

client = get_qdrant_connection()
coll_name = 'kaushik_restaurants'

encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Restaurant data spec:
# 100 entries
# fields:  name, cuisine, menu_items, rating (between 2 and 5, 1 decimal place)
# description (owner-perspective)

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "restaurant_data.py")

def load_restaurant_data(path: str) -> list[dict]:
    """Load the `restaurants` list from the data module at `path`.

    Exits the process with a fatal error if the file is missing or unreadable.
    """
    if not os.path.isfile(path):
        sys.exit(f"FATAL: restaurant data file not found: {path}")

    if not os.access(path, os.R_OK):
        sys.exit(f"FATAL: restaurant data file is not readable: {path}")

    try:
        spec = importlib.util.spec_from_file_location("restaurant_data", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except OSError as e:
        sys.exit(f"FATAL: could not read restaurant data file: {path} ({e})")
    except Exception as e:
        sys.exit(f"FATAL: could not load restaurant data file: {path} ({e})")

    try:
        return module.restaurants
    except AttributeError:
        sys.exit(f"FATAL: '{path}' does not define a 'restaurants' list")


restaurants = load_restaurant_data(DATA_FILE)

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

restaurant_collection = get_or_create_collection(
    client=client,
    collection_name=coll_name,
    vectors_config={
        'fixed': models.VectorParams(size=encoder.get_embedding_dimension(), distance=models.Distance.COSINE),
        'sentence': models.VectorParams(size=encoder.get_embedding_dimension(), distance=models.Distance.COSINE),
        'semantic': models.VectorParams(size=encoder.get_embedding_dimension(), distance=models.Distance.COSINE),

    },
)

client.create_payload_index(
    collection_name=coll_name,
    field_name="cuisine",
    field_schema=models.PayloadSchemaType.KEYWORD,
)

client.create_payload_index(
    collection_name=coll_name,
    field_name="rating",
    field_schema=models.PayloadSchemaType.FLOAT,
)



def fixed_size_chunks(text, size=MAX_TOKENS):
    "Splits text into fixed-size token chunks."
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return [
        tokenizer.decode(tokens[i:i+size], skip_special_tokens=True)
        for i in range(0, len(tokens), size)
    ]

def sentence_splitter(text):
    splitter = SentenceSplitter(chunk_size=MAX_TOKENS, chunk_overlap=40)
    return splitter.split_text(text)

def semantic_splitter(text):
    document = Document(text=text)

    semantic_splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    nodes = semantic_splitter.get_nodes_from_documents([document])  # Pass list of Document objects
    return [n.text for n in nodes]

# Namespace for deterministic point IDs so re-running this script upserts
# existing points instead of accumulating orphaned duplicates.
POINT_ID_NAMESPACE = uuid.UUID("d6e29a3a-8f2e-4f0d-9b0a-9e1a9c9f5b3c")


def make_point_id(restaurant_id: int, chunking: str, chunk_index: int) -> str:
    """Deterministic point ID derived from (restaurant id, chunking method, chunk index)."""
    key = f"{restaurant_id}|{chunking}|{chunk_index}"
    return str(uuid.uuid5(POINT_ID_NAMESPACE, key))

# NOTE:  This truthy check is coarse; it does not account for a previous run that may have crashed
# part-way through.
if restaurant_collection.points_count:
    print(
        f"Collection '{coll_name}' already has "
        f"{restaurant_collection.points_count} points; skipping chunking and upload."
    )
else:
    # Count tokens for each description
    for eatery in restaurants:
        tokens = tokenizer.encode(eatery["description"], add_special_tokens=False)
        print(f"{eatery['name']}: {len(tokens)} tokens")

        # show if it exceeds
        if len(tokens) > MAX_TOKENS:
            print(f"  -  Exceeds", MAX_TOKENS, "token limit by", len(tokens) - MAX_TOKENS, "tokens")
        print()

    points = []

    for eatery in restaurants:
        # Fixed-size
        for chunk_index, chunk in enumerate(fixed_size_chunks(eatery["description"])):
            points.append(models.PointStruct(
                id=make_point_id(eatery["id"], "fixed", chunk_index),
                vector={"fixed": encoder.encode(chunk).tolist()},
                payload={**eatery, "chunk": chunk, "chunking": "fixed"}
            ))

        # Sentence
        for chunk_index, chunk in enumerate(sentence_splitter(eatery["description"])):
            points.append(models.PointStruct(
                id=make_point_id(eatery["id"], "sentence", chunk_index),
                vector={"sentence": encoder.encode(chunk).tolist()},
                payload={**eatery, "chunk": chunk, "chunking": "sentence"}
            ))

        # Semantic
        for chunk_index, chunk in enumerate(semantic_splitter(eatery["description"])):
            points.append(models.PointStruct(
                id=make_point_id(eatery["id"], "semantic", chunk_index),
                vector={"semantic": encoder.encode(chunk).tolist()},
                payload={**eatery, "chunk": chunk, "chunking": "semantic"}
            ))

    client.upload_points(collection_name=coll_name, points=points)
    print(f"Uploaded {len(points)} vectors.")

# Now that we have data populated, let's do some searches.
results = client.query_points(
    collection_name=coll_name,
    query=encoder.encode("").tolist(),
    using="fixed",  # or "sentence" or "semantic"
    limit=3,
)

# Helper function for inspection
def search_and_inspect(query, vector_name, k=3, query_filter=None):
    results = client.query_points(
        collection_name=coll_name,
        query=encoder.encode(query).tolist(),
        using=vector_name,
        limit=k,
        query_filter=query_filter,
        with_payload=True,
    )

    print(f"\nTop {k} results using '{vector_name}' chunks for query: '{query}'\n")
    for i, point in enumerate(results.points, 1):
        payload = point.payload
        print(
            f"{i}. {payload['name']} ({payload['cuisine']})\n"
            f"   Score: {point.score:.4f}\n"
            f"   Rating: {payload['rating']:.1f}\n"
            f"   Chunking: {payload['chunking']}\n"
            f"   Chunk: {payload['chunk']}\n"
        )

search_and_inspect(
    query='authentic Chinese comfort food',
    vector_name='sentence',
    k=5,
    query_filter=models.Filter(
        must=[
            models.FieldCondition(key="cuisine", match=models.MatchValue(value="Chinese")),
            models.FieldCondition(key="rating",  range=models.Range(gte=3.5)),

        ]
    ),
)
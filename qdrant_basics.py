#!/usr/bin/env python3

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
import os

load_dotenv()

api_key = os.getenv("QDRANT_API_KEY")
client = QdrantClient(url="https://2ec3c57d-71b6-449e-9d63-6c9404bfade3.us-west-1-0.aws.cloud.qdrant.io", 
                      api_key=api_key)

# Retrieve and display the list of collections
collections = client.get_collections()
print("Existing collections:", collections)

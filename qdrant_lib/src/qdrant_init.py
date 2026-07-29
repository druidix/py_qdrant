"""Qdrant client initialization utilities."""

from dotenv import load_dotenv
from qdrant_client import QdrantClient
import os
import sys


def get_qdrant_connection() -> QdrantClient:
    """
    Initialize and return a Qdrant client using environment variables.

    Required environment variables:
        QDRANT_URL: The URL of the Qdrant instance
        QDRANT_API_KEY: The API key for authentication

    Raises:
        SystemExit: If required environment variables are not defined

    Returns:
        QdrantClient: An authenticated connection to Qdrant
    """
    load_dotenv()

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url:
        print("Error: QDRANT_URL not defined in environment variables")
        sys.exit(1)
    if not qdrant_api_key:
        print("Error: QDRANT_API_KEY not defined in environment variables")
        sys.exit(1)

    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

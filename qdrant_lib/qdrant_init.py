"""Qdrant client initialization utilities."""

import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient


def get_qdrant_connection(location: str | None = None, **kwargs) -> QdrantClient:
    """
    Initialize and return a Qdrant client.

    If location is ":memory:", creates an in-memory Qdrant instance.
    Otherwise, uses QDRANT_URL and QDRANT_API_KEY from environment variables
    to connect to a cloud instance.

    Args:
        location: Either ":memory:" for in-memory instance, or None to use
            environment variables for cloud connection.
        **kwargs: Additional keyword arguments to pass to QdrantClient,
            such as https, timeout, port, grpc_port, etc.

    Raises:
        EnvironmentError: If location is None and QDRANT_URL or QDRANT_API_KEY
            is not defined in environment variables.

    Returns:
        QdrantClient: A connection to a Qdrant instance (in-memory or cloud).
    """
    if location == ":memory:":
        return QdrantClient(location=":memory:", **kwargs)

    load_dotenv()

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url:
        raise EnvironmentError("QDRANT_URL not defined in environment variables")
    if not qdrant_api_key:
        raise EnvironmentError("QDRANT_API_KEY not defined in environment variables")

    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key, **kwargs)

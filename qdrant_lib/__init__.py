"""Qdrant utilities library for vector operations."""

__version__ = "0.1.0"

from .qdrant_init import get_qdrant_connection
from .collection_utils import get_or_create_collection

__all__ = ["get_qdrant_connection", "get_or_create_collection"]

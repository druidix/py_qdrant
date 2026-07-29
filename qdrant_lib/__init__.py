"""Qdrant utilities library for vector operations."""

__version__ = "0.1.0"

from .src.qdrant_init import get_qdrant_connection

__all__ = ["get_qdrant_connection"]

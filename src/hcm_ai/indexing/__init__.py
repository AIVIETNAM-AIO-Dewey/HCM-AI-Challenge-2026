"""Local vector and lexical stores with optional third-party accelerators."""

from .bm25_store import BM25Store, TextDocument
from .in_memory import InMemoryVectorStore
from .persistence import (
    IndexBuildResult,
    build_text_index,
    build_visual_index,
    load_text_store,
    load_vector_store,
    save_text_store,
    save_vector_store,
    text_index_fingerprint,
    visual_index_fingerprint,
)

__all__ = [
    "BM25Store",
    "IndexBuildResult",
    "InMemoryVectorStore",
    "TextDocument",
    "build_text_index",
    "build_visual_index",
    "load_text_store",
    "load_vector_store",
    "save_text_store",
    "save_vector_store",
    "text_index_fingerprint",
    "visual_index_fingerprint",
]

import faiss
import numpy as np
import json
import os

INDEX_PATH = "data/faiss.index"
META_PATH = "data/metadata.json"
DIMENSION = 384  # matches all-MiniLM-L6-v2


def _init_index() -> faiss.IndexFlatIP:
    """Inner product on normalized vectors = cosine similarity."""
    return faiss.IndexFlatIP(DIMENSION)


def save(index: faiss.Index, metadata: list[dict]):
    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w") as f:
        json.dump(metadata, f)


def load() -> tuple[faiss.Index, list[dict]]:
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r") as f:
        metadata = json.load(f)
    return index, metadata


def exists() -> bool:
    return os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)

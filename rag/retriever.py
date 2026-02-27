from rag.embedder import embed
from rag import store
import numpy as np


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Return top_k most relevant chunks for a query."""
    if not store.exists():
        print("Index not found. Run: python server.py index <file_or_url>")
        return []
    index, metadata = store.load()

    query_vec = embed([query]).astype(np.float32)
    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append(
            {
                "score": float(score),
                "source": metadata[idx]["source"],
                "text": metadata[idx]["text"],
            }
        )

    return results

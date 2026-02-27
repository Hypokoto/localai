from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # fast, good quality, 384-dim
_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print("Loading embedding model on: CPU")
        _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model


def embed(texts: list[str]) -> np.ndarray:
    return get_model().encode(texts, convert_to_numpy=True, normalize_embeddings=True)

import faiss
import numpy as np
from utils.file_loader import load_file
from rag.chunker import chunk_text
from rag.embedder import embed
from rag import store


def index_source(source: str):
    """Index a file path or URL into the FAISS store."""
    print(f"Loading: {source}")
    text = load_file(source)

    print(f"Chunking...")
    chunks = chunk_text(text)
    print(f"  → {len(chunks)} chunks")

    print(f"Embedding...")
    vectors = embed(chunks)

    # Load or create index
    if store.exists():
        index, metadata = store.load()
    else:
        index = store._init_index()
        metadata = []

    index.add(vectors.astype(np.float32))

    for chunk in chunks:
        metadata.append({"source": source, "text": chunk})

    store.save(index, metadata)
    print(f"Done. Total chunks in index: {len(metadata)}")

import argparse
import ollama
from rag.indexer import index_source
from rag.retriever import retrieve

MODEL = "mistral"  # change to whatever model you have pulled


def build_prompt(query: str, context_chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(c["text"] for c in context_chunks)
    return f"""You are a helpful assistant. Answer the question using only the context below.
If the answer isn't in the context, say "I don't have enough information."

Context:
{context}

Question: {query}
Answer:"""


def chat(query: str):
    print("\nRetrieving context...")
    chunks = retrieve(query, top_k=5)

    if not chunks:
        print("No relevant context found. Is your index empty?")
        return

    print(f"Found {len(chunks)} relevant chunks. Querying {MODEL}...\n")
    prompt = build_prompt(query, chunks)

    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    print("Answer:\n")
    print(response["message"]["content"])


def main():
    parser = argparse.ArgumentParser(description="Local RAG CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index a file or URL")
    index_parser.add_argument("source", help="File path or URL to index")

    # Query command
    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("question", help="Your question")

    args = parser.parse_args()

    if args.command == "index":
        index_source(args.source)
    elif args.command == "query":
        chat(args.question)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

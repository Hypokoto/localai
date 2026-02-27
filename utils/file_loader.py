import fitz  # pymupdf
import os
import re
import requests
from bs4 import BeautifulSoup


def load_file(source: str) -> str:
    """Load text from a file path or URL."""
    if source.startswith("http://") or source.startswith("https://"):
        return _load_url(source)

    ext = os.path.splitext(source)[1].lower()

    if ext == ".pdf":
        return _load_pdf(source)
    elif ext in (".md", ".txt"):
        return _load_text(source)
    elif ext in (".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs"):
        return _load_text(source)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _load_pdf(path: str) -> str:
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)


def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _load_url(url: str) -> str:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    # Strip nav/footer noise
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)

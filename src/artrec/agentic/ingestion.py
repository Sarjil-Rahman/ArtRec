from __future__ import annotations

import json
from pathlib import Path

from artrec.agentic.models import Document, DocumentChunk


DEFAULT_DOCUMENT_DIR = Path("data/documents/sample")


def load_sample_documents(document_dir: Path | str = DEFAULT_DOCUMENT_DIR) -> list[Document]:
    """Load bundled synthetic documents from JSON files."""
    base = Path(document_dir)
    documents: list[Document] = []
    for path in sorted(base.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["source_path"] = str(path)
        documents.append(Document(**payload))
    if not documents:
        raise FileNotFoundError(f"No sample documents found under {base}")
    return documents


def chunk_documents(
    documents: list[Document], chunk_size: int = 700, overlap: int = 120
) -> list[DocumentChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks: list[DocumentChunk] = []
    for doc in documents:
        text = doc.text.strip()
        start = 0
        chunk_index = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            if end < len(text):
                split_at = text.rfind(" ", start, end)
                if split_at > start + int(chunk_size * 0.6):
                    end = split_at
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        document_id=doc.document_id,
                        title=doc.title,
                        domain=doc.domain,
                        source_path=doc.source_path,
                        chunk_id=f"{doc.document_id}_chunk_{chunk_index}",
                        text=chunk_text,
                        start_char=start,
                        end_char=end,
                    )
                )
                chunk_index += 1
            if end >= len(text):
                break
            start = max(0, end - overlap)
    return chunks

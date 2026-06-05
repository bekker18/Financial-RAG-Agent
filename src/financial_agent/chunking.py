import hashlib
import re
from typing import Any

from financial_agent.schema import DocumentChunk


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\xa0", " ")

    return text.strip()


def chunk_text(
    text: str,
    metadata: dict[str, Any],
    chunk_size: int = 1800,
    overlap: int = 250,
) -> list[DocumentChunk]:
    """
    Simple character-based chunking.

    chunk_size: 1800 means every chunk is around 1800 characters.
    overlap=250 means neighboring chunks share 250 characters.
    This reduces the chance that important information is split badly.
    """

    text = clean_text(text)

    if not text:
        return []

    chunks: list[DocumentChunk] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            raw_id = (
                f"{metadata.get('ticker', '')}-"
                f"{metadata.get('form', '')}-"
                f"{metadata.get('filing_date', '')}-"
                f"{metadata.get('accession_number', '')}-"
                f"{chunk_index}"
            )
            chunk_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()

            chunk_metadata = dict(metadata)
            chunk_metadata["chunk_index"] = chunk_index

            chunks.append(
                DocumentChunk(id=chunk_id, text=chunk, metadata=chunk_metadata)
            )

        chunk_index += 1
        start += chunk_size - overlap

    return chunks

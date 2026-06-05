from dataclasses import dataclass
from typing import Any

from chromadb.api.types import Metadata


@dataclass
class FilingMetadata:
    ticker: str
    cik: str
    company_name: str
    form: str
    filing_date: str
    accession_number: str
    primary_document: str
    url: str


@dataclass
class DocumentChunk:
    id: str
    text: str
    metadata: dict[str, Any]


@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: Metadata
    distance: float | None = None

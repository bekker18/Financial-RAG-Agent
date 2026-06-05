from typing import Any, cast

import chromadb
from chromadb.api.types import Embeddings, Metadatas, QueryResult, Where

from financial_agent.config import get_settings
from financial_agent.embeddings import GeminiEmbedder
from financial_agent.schema import DocumentChunk, RetrievedChunk


class FinancialVectorStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedder = GeminiEmbedder()

        # Stores the vector database on disk.
        self.client = chromadb.PersistentClient(
            path=self.settings.chroma_path,
        )

        self.collection = self.client.get_or_create_collection(
            name=self.settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.settings.chroma_collection)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name=self.settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[DocumentChunk], batch_size: int = 32) -> None:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]

            ids = [chunk.id for chunk in batch]
            documents = [chunk.text for chunk in batch]
            metadatas: Metadatas = [
                self._clean_metadata(chunk.metadata) for chunk in batch
            ]
            embeddings = cast(
                Embeddings,
                [self.embedder.embed_document(chunk.text) for chunk in batch],
            )

            # Saves chunks + embeddings + metadata.
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )

    def search(
        self,
        query: str,
        top_k: int = 5,
        ticker: str | None = None,
    ) -> list[RetrievedChunk]:
        query_embedding = self.embedder.embed_query(query)

        where: Where | None = None
        if ticker:
            where = {"ticker": ticker.upper()}

        # Retrieves the closest chunks to the user question.
        results: QueryResult = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: list[RetrievedChunk] = []

        ids = (results["ids"] or [[]])[0]
        documents = (results["documents"] or [[]])[0]
        metadatas = (results["metadatas"] or [[]])[0]
        distances = (results["distances"] or [[]])[0]

        for chunk_id, text, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            retrieved.append(
                RetrievedChunk(
                    id=chunk_id,
                    text=text,
                    metadata=metadata,
                    distance=distance,
                )
            )

        return retrieved

    @staticmethod
    def _clean_metadata(
        metadata: dict[str, Any],
    ) -> dict[str, str | int | float | bool]:
        """
        Chroma metadata values should be simple scalar types.
        """

        clean: dict[str, str | int | float | bool] = {}

        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                clean[key] = value
            elif value is None:
                clean[key] = ""
            else:
                clean[key] = str(value)

        return clean

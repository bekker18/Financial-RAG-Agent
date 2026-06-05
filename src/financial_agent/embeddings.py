from google import genai
from google.genai import types

from financial_agent.config import get_settings


class GeminiEmbedder:
    def __init__(self) -> None:
        self.settings = get_settings()

        if not self.settings.gemini_api_key:
            raise ValueError("Missing GEMINI_API_KEY in environment.")

        self.client = genai.Client(api_key=self.settings.gemini_api_key)

    def embed_document(self, text: str) -> list[float]:
        """
        Embedding for stored document chunks.
        """

        prompt = f"Retrieval document: {text}"

        return self._embed(prompt)

    def embed_query(self, query: str) -> list[float]:
        """
        Embedding for user queries.
        """

        prompt = f"Retrieval query: {query}"

        return self._embed(prompt)

    def _embed(self, text: str) -> list[float]:
        result = self.client.models.embed_content(
            model=self.settings.gemini_embedding_model,
            contents=text,
            config=types.EmbedContentConfig(
                output_dimensionality=self.settings.embedding_dim,
            ),
        )

        if not result.embeddings:
            raise ValueError("Embedding API response missing embeddings.")

        embedding_values = result.embeddings[0].values
        if not isinstance(embedding_values, list):
            raise ValueError("Embedding API response missing values.")

        return list(embedding_values)

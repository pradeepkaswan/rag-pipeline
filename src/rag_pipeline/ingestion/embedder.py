"""Embedder for generating text embeddings using OpenAI."""

from __future__ import annotations

from openai import OpenAI, RateLimitError, APIError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from rag_pipeline.config.settings import settings
from rag_pipeline.utils.logger import get_logger

MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

logger = get_logger(__name__)


class Embedder:
    """Embedder for generating text embeddings using OpenAI's text-embedding-3-small model."""

    def __init__(self):
        """Initialize OpenAI client with API key from settings."""
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = MODEL
        self.dimension = EMBEDDING_DIMENSION

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
    )
    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text string.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        response = self.client.embeddings.create(
            input=text,
            model=self.model,
            dimensions=self.dimension,
        )
        return response.data[0].embedding

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
    )
    def embed_batch(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed.
            batch_size: Maximum batch size for API call (OpenAI supports up to ~2048).

        Returns:
            List of embedding vectors.
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.model,
                dimensions=self.dimension,
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            logger.info(f"Embedded batch {i // batch_size + 1}: {len(batch)} texts")

        return embeddings

"""
NVIDIA NIM embedding client for BGE-M3 with API key rotation.
Handles batched API calls with retry logic, progress tracking, and multi-key load distribution.
"""

import logging
import time
from itertools import cycle
from typing import Optional

from openai import OpenAI

from backend.config import settings

logger = logging.getLogger(__name__)


class NvidiaEmbedder:
    """
    Generate embeddings using NVIDIA NIM API (BGE-M3).
    Supports multiple API keys for distributing rate limits.
    Uses the OpenAI-compatible endpoint at integrate.api.nvidia.com.
    """

    # 40 RPM per key on NVIDIA Build free tier
    RATE_LIMIT_RPM_PER_KEY = 40

    def __init__(self):
        keys = settings.all_api_keys
        if not keys:
            raise ValueError("NVIDIA_API_KEY not set. Check your .env file.")

        # Build a pool of OpenAI clients — one per key
        self.clients = [
            OpenAI(api_key=k, base_url=settings.nvidia_base_url)
            for k in keys
        ]
        self._client_cycle = cycle(range(len(self.clients)))
        self.model = settings.nvidia_embed_model
        self.batch_size = settings.embed_batch_size
        self.dimension: Optional[int] = None

        # Rate limiting: total RPM = 40 × num_keys
        total_rpm = self.RATE_LIMIT_RPM_PER_KEY * len(self.clients)
        self._min_interval = 60.0 / total_rpm  # seconds between requests
        self._last_request_time = 0.0

        logger.info(
            f"Embedder initialized with {len(self.clients)} API key(s) — "
            f"model={self.model}, batch_size={self.batch_size}, "
            f"effective rate={total_rpm} RPM"
        )

    def _next_client(self) -> OpenAI:
        """Round-robin across API key clients."""
        idx = next(self._client_cycle)
        return self.clients[idx]

    def embed_texts(self, texts: list[str],
                    input_type: str = "passage",
                    progress_callback=None) -> list[list[float]]:
        """
        Embed a list of texts using batched NVIDIA API calls.

        Args:
            texts: List of text strings to embed.
            input_type: "passage" for documents, "query" for queries.
            progress_callback: Optional fn(done, total) for progress updates.

        Returns:
            List of embedding vectors.
        """
        all_embeddings = []
        total = len(texts)
        logger.info(f"Embedding {total:,} texts in batches of {self.batch_size}")

        for i in range(0, total, self.batch_size):
            batch = texts[i:i + self.batch_size]

            # Truncate very long texts (API limit)
            batch = [t[:8192] if len(t) > 8192 else t for t in batch]

            # Rate throttle: wait if we're calling too fast
            elapsed = time.time() - self._last_request_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)

            embeddings = self._embed_batch_with_retry(batch, input_type)
            self._last_request_time = time.time()
            all_embeddings.extend(embeddings)

            if progress_callback:
                progress_callback(len(all_embeddings), total)

            if (i + self.batch_size) % (self.batch_size * 10) == 0:
                logger.info(f"Embedded {len(all_embeddings):,} / {total:,}")

        # Capture dimension from first embedding
        if all_embeddings and self.dimension is None:
            self.dimension = len(all_embeddings[0])
            logger.info(f"Embedding dimension: {self.dimension}")

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        result = self._embed_batch_with_retry([query[:8192]], input_type="query")
        return result[0]

    def _embed_batch_with_retry(self, texts: list[str],
                                 input_type: str = "passage",
                                 max_retries: int = 5) -> list[list[float]]:
        """Embed a batch with exponential backoff retry and key rotation."""
        for attempt in range(max_retries):
            client = self._next_client()
            try:
                response = client.embeddings.create(
                    input=texts,
                    model=self.model,
                    encoding_format="float",
                    extra_body={"input_type": input_type, "truncate": "END"},
                )
                return [item.embedding for item in response.data]

            except Exception as e:
                error_str = str(e)
                if "rate" in error_str.lower() or "429" in error_str:
                    wait = 2 ** attempt * 2
                    logger.warning(
                        f"Rate limited on key #{next(self._client_cycle) % len(self.clients)}, "
                        f"rotating + waiting {wait}s (attempt {attempt + 1})"
                    )
                    time.sleep(wait)
                elif "timeout" in error_str.lower() or "connection" in error_str.lower():
                    wait = 2 ** attempt
                    logger.warning(f"Connection error, retrying in {wait}s: {e}")
                    time.sleep(wait)
                elif attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"Embed error (attempt {attempt + 1}): {e}")
                    time.sleep(wait)
                else:
                    logger.error(f"Embedding failed after {max_retries} attempts: {e}")
                    raise

        raise RuntimeError("Embedding failed after all retries")

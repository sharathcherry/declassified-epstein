"""
NVIDIA NIM reranker client using llama-3.2-nv-rerankqa-1b-v2.
Reranks retrieved chunks for precision before LLM generation.
"""

import logging
import time

import requests

from backend.config import settings

logger = logging.getLogger(__name__)


class NvidiaReranker:
    """
    Rerank retrieved documents using NVIDIA NIM reranking API.
    Uses the /v1/ranking endpoint.
    """

    def __init__(self):
        if not settings.has_nvidia_key:
            raise ValueError("NVIDIA_API_KEY not set. Check your .env file.")

        self.api_key = settings.nvidia_api_key
        self.model = settings.nvidia_rerank_model
        self.base_url = settings.nvidia_base_url
        self.top_k = settings.rerank_top_k

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Rerank documents using NVIDIA NIM reranker.

        Args:
            query: The search query.
            documents: List of chunk dicts with 'text' field.
            top_k: Number of top results to return.

        Returns:
            Reranked documents with rerank_score added.
        """
        if not documents:
            return []

        top_k = top_k or self.top_k

        # Extract texts for the API
        texts = [doc.get("text", "")[:4096] for doc in documents]

        try:
            reranked = self._call_rerank_api(query, texts)
        except Exception as e:
            logger.warning(f"Reranking failed, returning original order: {e}")
            return documents[:top_k]

        # Map scores back to documents
        results = []
        for item in reranked[:top_k]:
            idx = item["index"]
            if idx < len(documents):
                doc = {**documents[idx]}
                doc["rerank_score"] = item["logit"]
                doc["original_rank"] = idx + 1
                results.append(doc)

        logger.debug(f"Reranked {len(documents)} → top {len(results)}")
        return results

    def _call_rerank_api(
        self, query: str, texts: list[str], max_retries: int = 3
    ) -> list[dict]:
        """Call the NVIDIA ranking API with retry logic."""
        url = f"{self.base_url}/ranking"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "query": {"text": query},
            "passages": [{"text": t} for t in texts],
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url, json=payload, headers=headers, timeout=30
                )
                response.raise_for_status()
                data = response.json()
                rankings = data.get("rankings", [])
                return sorted(rankings, key=lambda x: x.get("logit", 0), reverse=True)

            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:
                    wait = 2 ** attempt * 2
                    logger.warning(f"Reranker rate limited, waiting {wait}s")
                    time.sleep(wait)
                else:
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"Reranker error (attempt {attempt + 1}): {e}")
                    time.sleep(wait)
                else:
                    raise

        return []

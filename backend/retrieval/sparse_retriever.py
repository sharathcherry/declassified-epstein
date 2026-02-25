"""
BM25 sparse retriever for keyword-based document search.
Complements dense retrieval for hybrid search.
"""

import logging
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from backend.config import DATA_DIR, settings

logger = logging.getLogger(__name__)

BM25_FILE = DATA_DIR / "bm25_index.pkl"


class SparseRetriever:
    """BM25-based sparse retrieval over chunk corpus."""

    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.metadata: list[dict] = []
        self.tokenized_corpus: list[list[str]] = []

    def build(self, chunks: list) -> None:
        """Build BM25 index from chunks."""
        self.metadata = [c.to_dict() if hasattr(c, 'to_dict') else c for c in chunks]

        # Tokenize
        self.tokenized_corpus = [
            self._tokenize(m.get("text", "")) for m in self.metadata
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info(f"Built BM25 index over {len(self.tokenized_corpus):,} chunks")

    def search(self, query: str, top_k: int = 50) -> list[dict]:
        """Search using BM25 scoring."""
        if self.bm25 is None:
            return []

        import numpy as np

        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Fast top-k using argpartition (O(n) instead of O(n log n) argsort)
        if len(scores) > top_k:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            # Sort only the top-k by score descending
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        else:
            top_indices = np.argsort(scores)[::-1]

        results = []
        has_metadata = len(self.metadata) >= len(scores)

        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                break
            if has_metadata and idx < len(self.metadata):
                result = {**self.metadata[idx], "score": score}
            else:
                result = {"id": f"bm25_{idx}", "score": score}
            results.append(result)

        return results

    def save(self) -> None:
        """Persist BM25 index to disk."""
        data = {
            "tokenized_corpus": self.tokenized_corpus,
            "metadata": self.metadata,
        }
        with open(BM25_FILE, "wb") as f:
            pickle.dump(data, f)
        logger.info("Saved BM25 index to disk")

    def load(self) -> bool:
        """Load BM25 index from disk. Handles both formats:
        - Local format: dict with 'tokenized_corpus' and 'metadata'
        - Kaggle format: raw BM25Okapi object
        Metadata is NOT loaded here to avoid duplication — main.py shares it.
        """
        if not BM25_FILE.exists():
            return False

        try:
            with open(BM25_FILE, "rb") as f:
                data = pickle.load(f)

            if isinstance(data, dict):
                # Local format
                self.tokenized_corpus = data["tokenized_corpus"]
                self.metadata = data.get("metadata", [])
                self.bm25 = BM25Okapi(self.tokenized_corpus)
            elif isinstance(data, BM25Okapi):
                # Kaggle format: raw BM25 object — already initialized
                self.bm25 = data
                self.tokenized_corpus = []
                # Don't load chunk_metadata.pkl here — main.py will share it
                # from vector_store to avoid loading 210 MB twice
                self.metadata = []
            else:
                logger.error(f"Unknown BM25 format: {type(data)}")
                return False

            logger.info(f"Loaded BM25 index: {self.bm25.corpus_size:,} chunks")
            return True
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return False

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + lowercase tokenization."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return [w for w in text.split() if len(w) > 1]

    @property
    def is_ready(self) -> bool:
        return self.bm25 is not None

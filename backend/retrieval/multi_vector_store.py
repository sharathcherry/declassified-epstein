"""
Multi-vector FAISS store for summary-level retrieval.
Indexes chunk summaries alongside passage embeddings for 3-way hybrid fusion.

Each chunk gets two representations:
1. Passage embedding (original text) — stored in main VectorStore
2. Summary embedding (LLM-generated one-liner) — stored here
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from backend.config import DATA_DIR, settings

logger = logging.getLogger(__name__)

SUMMARY_INDEX_FILE = DATA_DIR / "summary_faiss_index.bin"
SUMMARY_META_FILE = DATA_DIR / "summary_metadata.pkl"
SUMMARIES_FILE = DATA_DIR / "chunk_summaries.pkl"

# Prompt for generating chunk summaries
SUMMARY_PROMPT = """Summarize this document excerpt in ONE concise sentence (max 30 words).
Focus on: WHO, WHAT, WHEN, WHERE.

Text: {text}

One-sentence summary:"""


class MultiVectorStore:
    """
    Secondary FAISS index storing chunk summary embeddings.
    Queries match against high-level intent (summaries) while
    full text is returned from the primary store.
    """

    def __init__(self):
        self.index = None
        self.metadata: list[dict] = []
        self.summaries: dict[str, str] = {}  # chunk_id → summary text
        self.dimension: Optional[int] = None
        self.enabled = settings.enable_multi_vector

    def generate_summaries(
        self, chunks: list, llm_client, progress_callback=None
    ) -> dict[str, str]:
        """
        Generate one-line summaries for all chunks using the LLM.

        Args:
            chunks: List of Chunk objects or dicts.
            llm_client: LLM client for generation.
            progress_callback: Optional fn(done, total).

        Returns:
            {chunk_id: summary_text}
        """
        if not self.enabled:
            return {}

        # Check for cached summaries
        if SUMMARIES_FILE.exists():
            try:
                with open(SUMMARIES_FILE, "rb") as f:
                    self.summaries = pickle.load(f)
                logger.info(f"Loaded {len(self.summaries):,} cached summaries")
                return self.summaries
            except Exception:
                pass

        if not llm_client or not llm_client.available:
            logger.warning("LLM not available for summary generation")
            return {}

        summaries = {}
        total = len(chunks)
        batch_size = 5  # Process 5 at a time to manage API calls

        for i, chunk in enumerate(chunks):
            chunk_dict = chunk.to_dict() if hasattr(chunk, 'to_dict') else chunk
            chunk_id = chunk_dict.get("id", str(i))
            text = chunk_dict.get("text", "")[:1500]  # Truncate for summary

            if not text.strip():
                continue

            try:
                summary = llm_client.generate(
                    system_prompt="You are a document summarizer. Output only the summary.",
                    user_prompt=SUMMARY_PROMPT.format(text=text),
                    max_tokens=60,
                    temperature=0.0,
                )
                summary = summary.strip().strip('"').strip("'")
                if summary and len(summary) > 10:
                    summaries[chunk_id] = summary
            except Exception as e:
                logger.debug(f"Summary generation failed for chunk {chunk_id}: {e}")

            if progress_callback and (i + 1) % 100 == 0:
                progress_callback(i + 1, total)

        self.summaries = summaries

        # Cache summaries
        with open(SUMMARIES_FILE, "wb") as f:
            pickle.dump(summaries, f)

        logger.info(f"Generated {len(summaries):,} summaries for {total:,} chunks")
        return summaries

    def build(self, summary_embeddings: list[list[float]], chunks: list) -> None:
        """Build the summary FAISS index."""
        import faiss

        if not summary_embeddings:
            logger.warning("No summary embeddings to index")
            return

        vectors = np.array(summary_embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)

        self.dimension = vectors.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(vectors)

        self.metadata = [c.to_dict() if hasattr(c, 'to_dict') else c for c in chunks]

        logger.info(
            f"Built summary vector index: {self.index.ntotal:,} vectors, "
            f"dimension={self.dimension}"
        )

    def save(self) -> None:
        """Persist summary index to disk."""
        import faiss

        if self.index is None:
            return

        faiss.write_index(self.index, str(SUMMARY_INDEX_FILE))
        with open(SUMMARY_META_FILE, "wb") as f:
            pickle.dump(self.metadata, f)

        logger.info(f"Saved summary index ({self.index.ntotal:,} vectors)")

    def load(self) -> bool:
        """Load summary index from disk."""
        import faiss

        if not SUMMARY_INDEX_FILE.exists() or not SUMMARY_META_FILE.exists():
            return False

        try:
            self.index = faiss.read_index(str(SUMMARY_INDEX_FILE))
            with open(SUMMARY_META_FILE, "rb") as f:
                self.metadata = pickle.load(f)
            if SUMMARIES_FILE.exists():
                with open(SUMMARIES_FILE, "rb") as f:
                    self.summaries = pickle.load(f)

            self.dimension = self.index.d
            logger.info(f"Loaded summary index: {self.index.ntotal:,} vectors")
            return True
        except Exception as e:
            logger.error(f"Failed to load summary index: {e}")
            return False

    def search(self, query_embedding: list[float], top_k: int = 30) -> list[dict]:
        """Search the summary index."""
        import faiss

        if self.index is None or self.index.ntotal == 0:
            return []

        query = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            result = {**self.metadata[idx], "summary_score": float(score)}
            # Attach summary text if available
            chunk_id = result.get("id", "")
            if chunk_id in self.summaries:
                result["summary"] = self.summaries[chunk_id]
            results.append(result)

        return results

    @property
    def is_ready(self) -> bool:
        return self.index is not None and self.index.ntotal > 0

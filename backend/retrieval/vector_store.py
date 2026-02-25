"""
FAISS vector store for dense retrieval.
Handles index creation, persistence, and similarity search.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from backend.config import DATA_DIR, settings

logger = logging.getLogger(__name__)

FAISS_INDEX_FILE = DATA_DIR / "faiss_index.bin"
METADATA_FILE = DATA_DIR / "chunk_metadata.pkl"


class VectorStore:
    """
    FAISS-backed vector store with metadata sidecar.
    Uses IndexFlatIP (inner product) since BGE-M3 produces normalized vectors.
    """

    def __init__(self):
        self.index = None
        self.metadata: list[dict] = []  # chunk metadata aligned with FAISS index
        self.dimension: Optional[int] = None

    def build(self, embeddings: list[list[float]], chunks: list) -> None:
        """Build a new FAISS index from embeddings and chunk metadata."""
        import faiss

        vectors = np.array(embeddings, dtype=np.float32)
        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(vectors)

        self.dimension = vectors.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(vectors)

        # Store chunk metadata
        self.metadata = [c.to_dict() if hasattr(c, 'to_dict') else c for c in chunks]

        logger.info(
            f"Built FAISS index: {self.index.ntotal:,} vectors, "
            f"dimension={self.dimension}"
        )

    def save(self) -> None:
        """Persist index and metadata to disk."""
        import faiss

        if self.index is None:
            return

        faiss.write_index(self.index, str(FAISS_INDEX_FILE))
        with open(METADATA_FILE, "wb") as f:
            pickle.dump(self.metadata, f)

        logger.info(f"Saved FAISS index ({self.index.ntotal:,} vectors) to disk")

    def load(self) -> bool:
        """Load index from disk using memory-mapped IO to save RAM."""
        import faiss

        if not FAISS_INDEX_FILE.exists() or not METADATA_FILE.exists():
            return False

        try:
            # Memory-mapped loading: keeps index on disk, pages in as needed
            # This drops memory from ~1.6 GB to nearly zero at startup
            try:
                self.index = faiss.read_index(str(FAISS_INDEX_FILE), faiss.IO_FLAG_MMAP)
                logger.info("FAISS loaded with memory-mapped IO")
            except Exception:
                # Fallback to regular loading if mmap not supported
                self.index = faiss.read_index(str(FAISS_INDEX_FILE))
                logger.info("FAISS loaded (regular IO)")

            with open(METADATA_FILE, "rb") as f:
                self.metadata = pickle.load(f)

            self.dimension = self.index.d
            logger.info(
                f"Loaded FAISS index: {self.index.ntotal:,} vectors, "
                f"dimension={self.dimension}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return False

    def search(self, query_embedding: list[float], top_k: int = 50) -> list[dict]:
        """
        Search for similar vectors.
        Returns list of {metadata + score} dicts, sorted by relevance.
        """
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
            result = {**self.metadata[idx], "score": float(score)}
            results.append(result)

        return results

    @property
    def is_ready(self) -> bool:
        return self.index is not None and self.index.ntotal > 0

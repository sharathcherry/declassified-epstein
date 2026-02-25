"""
Local embedding cache using numpy for fast loading.
- Saves to embedding_cache.npz (fast, ~5s to load) instead of pickle (30+ min)
- Backwards-compatible: loads old .pkl if .npz doesn't exist yet
"""

import hashlib
import logging
import pickle
from pathlib import Path

import numpy as np

from backend.config import DATA_DIR

logger = logging.getLogger(__name__)

NPZ_FILE   = DATA_DIR / "embedding_cache.npz"
CACHE_FILE = DATA_DIR / "embedding_cache.pkl"   # legacy, kept for compatibility


class EmbeddingCache:
    """
    Hash-based embedding cache backed by numpy arrays.
    Maps text content hash → embedding vector index in a compact matrix.
    Loads in <5 seconds vs 30+ minutes for the old pickle format.
    """

    def __init__(self):
        self._hashes: list[str] = []
        self._matrix: np.ndarray | None = None   # shape (N, dim)
        self._index: dict[str, int] = {}          # hash → row in matrix
        self._pending: dict[str, list[float]] = {} # new embeddings not yet flushed
        self._load()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self):
        """Load cache — prefer fast .npz, fall back to .pkl."""
        if NPZ_FILE.exists():
            self._load_npz()
        elif CACHE_FILE.exists():
            logger.warning(
                "Slow .pkl cache found — loading once to convert. "
                "This takes 10-30 min but only happens once."
            )
            self._load_pkl_and_convert()
        else:
            logger.info("No embedding cache found — starting fresh")

    def _load_npz(self):
        try:
            t0 = __import__("time").time()
            data = np.load(str(NPZ_FILE), allow_pickle=True)
            self._hashes = list(data["hashes"])
            self._matrix = data["embeddings"]            # float32 (N, dim)
            self._index  = {h: i for i, h in enumerate(self._hashes)}
            elapsed = __import__("time").time() - t0
            logger.info(
                f"Loaded {len(self._index):,} cached embeddings from .npz in {elapsed:.1f}s"
            )
        except Exception as e:
            logger.warning(f"npz load failed ({e}), trying .pkl")
            self._load_pkl_and_convert()

    def _load_pkl_and_convert(self):
        with open(CACHE_FILE, "rb") as f:
            old_cache: dict[str, list] = pickle.load(f)
        hashes = list(old_cache.keys())
        matrix = np.array(list(old_cache.values()), dtype=np.float32)
        del old_cache
        np.savez_compressed(str(NPZ_FILE), hashes=hashes, embeddings=matrix)
        self._hashes = hashes
        self._matrix = matrix
        self._index  = {h: i for i, h in enumerate(hashes)}
        logger.info(f"Converted .pkl → .npz ({len(self._index):,} embeddings saved)")

    # ── Saving ────────────────────────────────────────────────────────────────

    def save(self):
        """Persist any pending new embeddings to .npz."""
        if not self._pending:
            return
        try:
            new_h = list(self._pending.keys())
            new_e = np.array(list(self._pending.values()), dtype=np.float32)

            if self._matrix is not None:
                matrix = np.vstack([self._matrix, new_e])
                hashes = self._hashes + new_h
            else:
                matrix = new_e
                hashes = new_h

            np.savez_compressed(str(NPZ_FILE), hashes=hashes, embeddings=matrix)
            self._hashes = hashes
            self._matrix = matrix
            self._index  = {h: i for i, h in enumerate(hashes)}
            self._pending.clear()
            logger.info(f"Saved {len(self._index):,} embeddings to .npz")
        except Exception as e:
            logger.error(f"Cache save failed: {e}")

    # ── Public API ────────────────────────────────────────────────────────────

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, text: str):
        """Return cached embedding as list[float], or None on miss."""
        h = self._hash(text)
        if h in self._pending:
            return self._pending[h]
        if h in self._index and self._matrix is not None:
            return self._matrix[self._index[h]].tolist()
        return None

    def put(self, text: str, embedding: list[float]):
        """Store a new embedding (buffered until save() is called)."""
        self._pending[self._hash(text)] = embedding

    def get_batch(self, texts: list[str]) -> tuple[list, list[int]]:
        """
        Batch lookup. Returns:
          results     — list of embeddings (None for misses)
          miss_indices — indices that need embedding
        """
        results, misses = [], []
        for i, text in enumerate(texts):
            emb = self.get(text)
            results.append(emb)
            if emb is None:
                misses.append(i)
        return results, misses

    def put_batch(self, texts: list[str], embeddings: list[list[float]]):
        for text, emb in zip(texts, embeddings):
            self.put(text, emb)

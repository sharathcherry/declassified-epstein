"""
Near-duplicate and corrupted document detection.
Uses MinHash-like approach for fast duplicate identification.
"""

import hashlib
import logging
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

# Minimum document length to be considered valid
MIN_DOC_LENGTH = 50

# Similarity threshold for near-duplicates (Jaccard on shingles)
DEDUP_THRESHOLD = 0.85


class Deduplicator:
    """Detect and remove near-duplicate and corrupted documents."""

    @classmethod
    def deduplicate(cls, documents: dict[str, dict]) -> dict[str, dict]:
        """
        Remove corrupted and near-duplicate documents.
        Returns filtered copy with dedup stats.
        """
        total = len(documents)
        logger.info(f"Starting deduplication on {total:,} documents...")

        # Step 1: Remove corrupted (too short, empty, garbage)
        valid = {}
        corrupted = 0
        for fn, doc in documents.items():
            text = doc.get("text", "")
            if cls._is_corrupted(text):
                corrupted += 1
            else:
                valid[fn] = doc

        logger.info(f"Removed {corrupted:,} corrupted documents")

        # Step 2: Remove exact duplicates (by text hash)
        seen_hashes: dict[str, str] = {}  # hash → first filename
        unique = {}
        exact_dupes = 0

        for fn, doc in valid.items():
            text_hash = hashlib.md5(doc["text"].encode()).hexdigest()
            if text_hash not in seen_hashes:
                seen_hashes[text_hash] = fn
                unique[fn] = doc
            else:
                exact_dupes += 1

        logger.info(f"Removed {exact_dupes:,} exact duplicates")

        # Step 3: Near-duplicate detection via shingle comparison
        # For large corpora, we group by length bucket for efficiency
        near_dupes = cls._find_near_duplicates(unique)
        for fn in near_dupes:
            unique.pop(fn, None)

        logger.info(f"Removed {len(near_dupes):,} near-duplicates")
        logger.info(
            f"Deduplication complete: {len(unique):,} unique docs "
            f"(removed {total - len(unique):,} total)"
        )

        return unique

    @staticmethod
    def _is_corrupted(text: str) -> bool:
        """Check if a document is corrupted or garbage."""
        if not text or len(text.strip()) < MIN_DOC_LENGTH:
            return True

        stripped = text.strip()

        # Mostly non-alphanumeric
        alpha_ratio = sum(c.isalnum() or c.isspace() for c in stripped) / max(len(stripped), 1)
        if alpha_ratio < 0.3:
            return True

        # Single repeated character
        if len(set(stripped.replace(' ', ''))) < 3:
            return True

        return False

    @classmethod
    def _find_near_duplicates(cls, documents: dict[str, dict]) -> set[str]:
        """Find near-duplicate documents using text shingles."""
        # Group by approximate length bucket for efficiency
        buckets: dict[int, list[tuple[str, str]]] = defaultdict(list)
        for fn, doc in documents.items():
            text = doc.get("text", "")
            bucket = len(text) // 200  # 200-char buckets
            buckets[bucket].append((fn, text))

        duplicates = set()

        for bucket_id, items in buckets.items():
            if len(items) < 2:
                continue

            # Compare within bucket (and adjacent buckets are handled by overlap)
            shingles_cache = {}
            for i, (fn1, text1) in enumerate(items):
                if fn1 in duplicates:
                    continue

                if fn1 not in shingles_cache:
                    shingles_cache[fn1] = cls._get_shingles(text1)
                s1 = shingles_cache[fn1]

                for j in range(i + 1, min(i + 20, len(items))):  # Compare up to 20 neighbors
                    fn2, text2 = items[j]
                    if fn2 in duplicates:
                        continue

                    if fn2 not in shingles_cache:
                        shingles_cache[fn2] = cls._get_shingles(text2)
                    s2 = shingles_cache[fn2]

                    if cls._jaccard(s1, s2) >= DEDUP_THRESHOLD:
                        duplicates.add(fn2)

        return duplicates

    @staticmethod
    def _get_shingles(text: str, k: int = 5) -> set[str]:
        """Generate character k-shingles from text."""
        text = re.sub(r'\s+', ' ', text.lower().strip())
        if len(text) < k:
            return {text}
        return {text[i:i + k] for i in range(len(text) - k + 1)}

    @staticmethod
    def _jaccard(s1: set, s2: set) -> float:
        """Jaccard similarity between two sets."""
        if not s1 or not s2:
            return 0.0
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        return intersection / union if union > 0 else 0.0

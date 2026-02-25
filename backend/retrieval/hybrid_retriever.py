"""
Hybrid retrieval: fuse dense (FAISS), sparse (BM25), summary, and entity signals
using reciprocal rank fusion (RRF) with deduplication.
"""

import logging
from typing import Optional

from backend.config import settings

logger = logging.getLogger(__name__)


class HybridRetriever:
    """4-way hybrid retriever with RRF fusion and deduplication."""

    def __init__(self, vector_store, sparse_retriever, multi_vector=None,
                 entity_retriever=None, k=60):
        self.vector_store = vector_store
        self.sparse = sparse_retriever
        self.multi_vector = multi_vector
        self.entity_retriever = entity_retriever
        self.k = k  # RRF constant
        logger.info("Hybrid retriever initialized (dense + sparse + summary + entity)")

    def search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 50,
        doc_type_filter: Optional[str] = None,
        keyword_boost: Optional[str] = None,
        hyde_embedding: Optional[list[float]] = None,
        entity_names: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Multi-stage hybrid search with 4-way RRF fusion + deduplication.
        """
        # ── Stage 1: Parallel retrieval ─────────────────────

        # Dense retrieval (use HyDE embedding if available)
        dense_embed = hyde_embedding if hyde_embedding is not None else query_embedding
        dense_results = self.vector_store.search(dense_embed, top_k=settings.dense_top_k)

        # Sparse retrieval
        sparse_results = self.sparse.search(query, top_k=settings.sparse_top_k)
        sparse_results = [
            r for r in sparse_results
            if r.get("text", "") and len(r.get("text", "")) > 50
        ]

        # Multi-vector summary retrieval (3rd signal)
        summary_results = []
        if self.multi_vector and self.multi_vector.is_ready:
            summary_results = self.multi_vector.search(
                query_embedding, top_k=settings.summary_top_k
            )

        # Entity retrieval (4th signal)
        entity_results = []
        if self.entity_retriever and entity_names:
            entity_results = self.entity_retriever.search(
                entity_names, top_k=settings.entity_top_k
            )

        # ── Stage 2: 4-way RRF fusion ──────────────────────

        rrf_scores: dict[str, float] = {}
        rrf_docs: dict[str, dict] = {}

        # Dense rankings (weight: 1.0)
        for rank, result in enumerate(dense_results):
            cid = result.get("id", str(rank))
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (self.k + rank + 1)
            rrf_docs[cid] = result

        # Sparse rankings (weight: 1.0)
        for rank, result in enumerate(sparse_results):
            cid = result.get("id", f"sparse_{rank}")
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (self.k + rank + 1)
            if cid not in rrf_docs:
                rrf_docs[cid] = result

        # Summary rankings (weight: 0.8)
        summary_weight = 0.8
        for rank, result in enumerate(summary_results):
            cid = result.get("id", f"summary_{rank}")
            rrf_scores[cid] = rrf_scores.get(cid, 0) + summary_weight / (self.k + rank + 1)
            if cid not in rrf_docs:
                rrf_docs[cid] = result

        # Entity rankings (weight: 1.2 — entity signal gets slight boost)
        entity_weight = 1.2
        for rank, result in enumerate(entity_results):
            cid = result.get("id", f"entity_{rank}")
            rrf_scores[cid] = rrf_scores.get(cid, 0) + entity_weight / (self.k + rank + 1)
            if cid not in rrf_docs:
                rrf_docs[cid] = result
            # Preserve entity_score for composite scoring
            if cid in rrf_docs and "entity_score" in result:
                rrf_docs[cid]["entity_score"] = result["entity_score"]
                rrf_docs[cid]["entity_overlap"] = result.get("entity_overlap", 0)
                rrf_docs[cid]["matched_entities"] = result.get("matched_entities", [])

        # ── Stage 3: Keyword boosting ──────────────────────

        if keyword_boost:
            kw_lower = keyword_boost.lower()
            for cid, doc in rrf_docs.items():
                text = doc.get("text", "").lower()
                if kw_lower in text:
                    rrf_scores[cid] *= 1.3

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # ── Stage 4: Deduplicate by text similarity ────────

        results = []
        seen_texts = []

        for cid in sorted_ids[:top_k * 3]:
            doc = rrf_docs[cid]
            text = doc.get("text", "")

            if not text or len(text) < 30:
                continue

            mid = len(text) // 4
            text_sig = text[mid:mid+300].strip().lower()
            start_sig = text[50:250].strip().lower() if len(text) > 250 else text.strip().lower()

            is_dup = False
            for seen_mid, seen_start in seen_texts:
                if (text_sig == seen_mid or
                    _text_overlap(text_sig, seen_mid) > 0.6 or
                    _text_overlap(start_sig, seen_start) > 0.7):
                    is_dup = True
                    break

            if is_dup:
                continue

            seen_texts.append((text_sig, start_sig))

            doc["rrf_score"] = rrf_scores[cid]

            # Track which retrieval sources found this result
            doc["retrieval_sources"] = []
            if any(r.get("id") == cid for r in dense_results):
                doc["retrieval_sources"].append("dense")
            if any(r.get("id") == cid for r in sparse_results):
                doc["retrieval_sources"].append("sparse")
            if any(r.get("id") == cid for r in summary_results):
                doc["retrieval_sources"].append("summary")
            if any(r.get("id") == cid for r in entity_results):
                doc["retrieval_sources"].append("entity")

            # Apply type filter
            if doc_type_filter:
                if doc.get("doc_type", "").upper() != doc_type_filter.upper():
                    continue

            results.append(doc)
            if len(results) >= top_k:
                break

        logger.debug(
            f"Hybrid search: {len(dense_results)} dense + {len(sparse_results)} sparse "
            f"+ {len(summary_results)} summary + {len(entity_results)} entity "
            f"→ {len(results)} fused results"
        )

        return results


def _text_overlap(a: str, b: str) -> float:
    """Quick character-level overlap ratio between two strings."""
    if not a or not b:
        return 0.0
    a_grams = set(a[i:i+3] for i in range(len(a) - 2))
    b_grams = set(b[i:i+3] for i in range(len(b) - 2))
    if not a_grams or not b_grams:
        return 0.0
    intersection = a_grams & b_grams
    return len(intersection) / max(len(a_grams), len(b_grams))

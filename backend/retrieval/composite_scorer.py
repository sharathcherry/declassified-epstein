"""
Composite Scorer: Unified scoring across semantic, keyword, entity, and graph signals.
Applies boosting modifiers for temporal relevance, entity type match,
multi-source confirmation, and community relevance.
"""

import logging
import math
from typing import Optional

from backend.config import settings

logger = logging.getLogger(__name__)


class CompositeScorer:
    """
    Unified scoring function:
    FinalScore(d, q) = w_s · Semantic(d, q) + w_k · Keyword(d, q) + w_e · Entity(d, q) + w_g · Graph(d, q)

    With multiplicative boosting modifiers.
    """

    def __init__(self, entity_store=None, graph_builder=None):
        self.entity_store = entity_store
        self.graph_builder = graph_builder

        # Load weights from config
        self.w_semantic = settings.composite_weights_semantic
        self.w_keyword = settings.composite_weights_keyword
        self.w_entity = settings.composite_weights_entity
        self.w_graph = settings.composite_weights_graph

    def score_results(
        self,
        results: list[dict],
        query_entities: list[str] = None,
        query_dates: list[str] = None,
        graph_paths: list = None,
    ) -> list[dict]:
        """
        Apply composite scoring to a set of retrieval results.

        Each result dict should have:
            - rrf_score or score (from hybrid retrieval)
            - entity_score (from entity retriever, if available)
            - text, id, doc_filename, etc.

        Returns results sorted by composite score.
        """
        query_entities = query_entities or []
        query_dates = query_dates or []

        # Build graph score mapping
        graph_scores = self._compute_graph_scores(graph_paths) if graph_paths else {}

        scored = []
        for result in results:
            composite = self._score_single(
                result, query_entities, query_dates, graph_scores
            )
            result["composite_score"] = composite
            scored.append(result)

        scored.sort(key=lambda x: x["composite_score"], reverse=True)
        return scored

    def _score_single(
        self,
        result: dict,
        query_entities: list[str],
        query_dates: list[str],
        graph_scores: dict,
    ) -> float:
        """Compute composite score for a single result."""

        # ── Component scores ──

        # 1. Semantic score (from FAISS cosine similarity or RRF)
        s_semantic = self._normalize_score(
            result.get("rrf_score", result.get("score", 0)),
            max_val=0.05  # RRF scores are typically small
        )

        # 2. Keyword score (from BM25, embedded in RRF)
        # If retrieval_sources includes 'sparse', give keyword credit
        sources = result.get("retrieval_sources", [])
        s_keyword = 0.5 if "sparse" in sources else 0.0
        if "dense" in sources and "sparse" in sources:
            s_keyword = 0.8  # Both signals agree → higher keyword relevance

        # 3. Entity score
        s_entity = self._compute_entity_score(result, query_entities)

        # 4. Graph score
        chunk_id = result.get("id", "")
        s_graph = graph_scores.get(chunk_id, 0.0)

        # ── Weighted combination ──
        base_score = (
            self.w_semantic * s_semantic +
            self.w_keyword * s_keyword +
            self.w_entity * s_entity +
            self.w_graph * s_graph
        )

        # ── Boosting modifiers (multiplicative) ──
        boost = 1.0

        # Temporal relevance: +10% if chunk dates match query dates
        if query_dates:
            boost *= self._temporal_boost(result, query_dates)

        # Multi-source confirmation: +20% if entity appears in multiple docs
        boost *= self._multi_source_boost(result, query_entities)

        # Community relevance: +10% if entity is in relevant community
        boost *= self._community_boost(result, query_entities)

        return base_score * boost

    def _compute_entity_score(self, result: dict, query_entities: list[str]) -> float:
        """
        Entity overlap score:
        EntityScore(q, d) = |E_q ∩ E_d| / |E_q| · log(1 + mention_count(d))
        """
        if not query_entities or not self.entity_store or not self.entity_store.is_ready:
            return result.get("entity_score", 0.0)

        # Get entities in this chunk
        chunk_id = result.get("id", "")
        chunk_entities = set()

        if chunk_id:
            chunk_ents = self.entity_store.get_entities_in_chunk(chunk_id)
            chunk_entities = set(e.lower() for e in chunk_ents)

        # Also check text content
        text = result.get("text", "")
        if text and not chunk_entities:
            text_lower = text.lower()
            for qe in query_entities:
                if qe.lower() in text_lower:
                    chunk_entities.add(qe.lower())

        # Compute overlap
        query_set = set(e.lower() for e in query_entities)
        overlap = query_set & chunk_entities

        if not query_set:
            return 0.0

        overlap_ratio = len(overlap) / len(query_set)

        # Log-scaled mention count boost
        mention_count = sum(
            self.entity_store.lookup(e).get("count", 0)
            for e in overlap
            if self.entity_store.lookup(e)
        )
        count_boost = math.log(1 + mention_count) / 10.0  # Normalize

        return min(overlap_ratio * (1 + count_boost), 1.0)

    def _temporal_boost(self, result: dict, query_dates: list[str]) -> float:
        """Boost if chunk dates overlap with query dates."""
        chunk_text = result.get("text", "")
        import re
        chunk_years = set(re.findall(r'\b(19[5-9]\d|20[0-2]\d)\b', chunk_text))

        if not chunk_years:
            return 1.0  # No temporal context

        query_years = set(query_dates)
        if query_years & chunk_years:
            return 1.10  # +10% boost
        return 0.95  # Slight penalty for date mismatch

    def _multi_source_boost(self, result: dict, query_entities: list[str]) -> float:
        """Boost if matched entities appear in multiple documents."""
        if not query_entities or not self.entity_store or not self.entity_store.is_ready:
            return 1.0

        max_docs = 1
        for ent_name in query_entities:
            info = self.entity_store.lookup(ent_name)
            if info:
                n_docs = len(info.get("source_docs", info.get("source_documents", [])))
                max_docs = max(max_docs, n_docs)

        if max_docs > 3:
            return 1.20  # +20% for multi-source confirmation
        elif max_docs > 1:
            return 1.10  # +10%
        return 1.0

    def _community_boost(self, result: dict, query_entities: list[str]) -> float:
        """Boost if entities share a graph community."""
        if not self.graph_builder or not self.graph_builder.is_ready:
            return 1.0

        if not query_entities:
            return 1.0

        # Check if any query entity and chunk entity share a community
        chunk_text = result.get("text", "").lower()
        for ent_name in query_entities:
            if ent_name.lower() in chunk_text:
                comm = self.graph_builder.node_to_community.get(ent_name)
                if comm is not None:
                    return 1.10  # +10% community relevance boost

        return 1.0

    @staticmethod
    def _normalize_score(score: float, max_val: float = 1.0) -> float:
        """Normalize a score to 0-1 range."""
        if max_val <= 0:
            return 0.0
        return min(max(score / max_val, 0.0), 1.0)

    @staticmethod
    def _compute_graph_scores(graph_paths: list) -> dict[str, float]:
        """Map chunk IDs to graph path scores."""
        scores: dict[str, float] = {}
        for path in graph_paths:
            for cid in path.evidence_chunks:
                if cid not in scores or path.score > scores[cid]:
                    scores[cid] = path.score
        return scores

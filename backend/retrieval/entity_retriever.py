"""
Entity Retriever: Entity-first retrieval path.
Given entity names, finds all chunks mentioning them and scores by entity relevance.
Supports entity expansion via co-occurrence graph.
"""

import logging
import math
from collections import Counter
from typing import Optional

from backend.config import settings

logger = logging.getLogger(__name__)


class EntityRetriever:
    """
    Entity-first retrieval: entity names → chunk IDs → scored results.
    Complements dense/sparse retrieval as the 4th signal.
    """

    def __init__(self, entity_store, chunk_lookup=None):
        """
        Args:
            entity_store: EntityStore instance
            chunk_lookup: Dict mapping chunk_id → chunk dict (text, metadata)
        """
        self.entity_store = entity_store
        self.chunk_lookup = chunk_lookup or {}

    def set_chunk_lookup(self, chunks: list[dict]):
        """Build chunk_id → chunk mapping from a list of chunk dicts."""
        self.chunk_lookup = {}
        for chunk in chunks:
            cid = chunk.get("id", "")
            if cid:
                self.chunk_lookup[cid] = chunk

    def search(
        self,
        entity_names: list[str],
        top_k: int = 20,
        expand: bool = True,
        max_expansion: int = 5,
    ) -> list[dict]:
        """
        Retrieve chunks by entity names.

        Args:
            entity_names: Entity names to search for
            top_k: Number of results to return
            expand: Whether to include co-occurring entities
            max_expansion: Max co-occurring entities to add
        """
        if not self.entity_store or not self.entity_store.is_ready:
            return []

        if not entity_names:
            return []

        # Resolve entities
        resolved = []
        for name in entity_names:
            info = self.entity_store.lookup(name)
            if info:
                resolved.append(info)

        if not resolved:
            return []

        # Get primary chunk IDs
        all_chunk_ids = set()
        primary_chunks = set()
        entity_weights: dict[str, float] = {}

        for info in resolved:
            chunk_ids = info.get("chunk_ids", [])
            primary_chunks.update(chunk_ids)
            all_chunk_ids.update(chunk_ids)
            entity_weights[info["name"]] = 1.0  # Primary entities have weight 1.0

        # Expand via co-occurrence
        if expand:
            for info in resolved:
                co_occurring = info.get("co_occurring", [])
                if isinstance(co_occurring, list):
                    for co_ent in co_occurring[:max_expansion]:
                        co_name = co_ent.get("name", co_ent) if isinstance(co_ent, dict) else co_ent
                        co_info = self.entity_store.lookup(co_name)
                        if co_info:
                            expansion_chunks = set(co_info.get("chunk_ids", []))
                            all_chunk_ids.update(expansion_chunks)
                            entity_weights[co_name] = 0.5  # Expanded entities have lower weight

        # Score chunks
        scored_chunks = []
        query_entity_names = set(e.lower() for e in entity_names)

        for cid in all_chunk_ids:
            chunk = self.chunk_lookup.get(cid)
            if not chunk:
                continue

            # Score: how many query entities appear in this chunk
            chunk_entities = self.entity_store.get_entities_in_chunk(cid)
            chunk_entity_lower = set(e.lower() for e in chunk_entities)

            # Entity overlap with query
            overlap = query_entity_names & chunk_entity_lower
            overlap_score = len(overlap) / len(query_entity_names) if query_entity_names else 0

            # Boost if chunk is from primary entity set (not expansion)
            is_primary = cid in primary_chunks
            primary_boost = 1.5 if is_primary else 1.0

            # Entity count boost (log-scaled)
            entity_count = len(chunk_entities)
            count_boost = math.log(1 + entity_count) / 3.0

            # Final score
            score = (overlap_score * 0.6 + count_boost * 0.4) * primary_boost

            result = {**chunk, "entity_score": score, "entity_overlap": overlap_score}
            result["matched_entities"] = list(overlap)
            scored_chunks.append(result)

        # Sort by entity score
        scored_chunks.sort(key=lambda x: x["entity_score"], reverse=True)

        logger.debug(
            f"Entity retrieval: {len(entity_names)} entities → "
            f"{len(all_chunk_ids)} chunks → {min(top_k, len(scored_chunks))} results"
        )

        return scored_chunks[:top_k]

    def get_entity_chunk_matrix(self, entity_names: list[str]) -> dict[str, list[str]]:
        """
        Get entity → chunk_ids mapping for a set of entities.
        Useful for O(1) lookups.
        """
        matrix = {}
        for name in entity_names:
            matrix[name] = self.entity_store.get_chunk_ids(name)
        return matrix

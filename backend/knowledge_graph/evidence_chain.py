"""
Evidence Chain Generator: Build human-readable reasoning chains from graph paths.
Collects supporting evidence from chunks and computes chain confidence.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from backend.config import settings

logger = logging.getLogger(__name__)


@dataclass
class EvidenceChain:
    """A single evidence chain connecting entities."""
    entities: list[str] = field(default_factory=list)
    relationships: list[str] = field(default_factory=list)
    supporting_chunks: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    path_description: str = ""
    temporal_range: tuple = ("", "")
    chain_id: str = ""

    def to_dict(self) -> dict:
        return {
            "entities": self.entities,
            "relationships": self.relationships,
            "supporting_chunks": self.supporting_chunks,
            "confidence": round(self.confidence, 3),
            "path_description": self.path_description,
            "temporal_range": list(self.temporal_range),
            "chain_id": self.chain_id,
        }


class EvidenceChainGenerator:
    """
    Generate evidence chains from graph traversal paths.
    Each chain includes entities, relationships, supporting documents,
    confidence scores, and temporal context.
    """

    def __init__(self, entity_store=None, chunk_lookup: dict = None):
        self.entity_store = entity_store
        self.chunk_lookup = chunk_lookup or {}

    def set_chunk_lookup(self, chunks: list[dict]):
        """Build chunk_id → chunk dict mapping."""
        self.chunk_lookup = {}
        for chunk in chunks:
            cid = chunk.get("id", "")
            if cid:
                self.chunk_lookup[cid] = chunk

    def generate_chains(
        self,
        traversal_paths: list,  # list[TraversalPath]
        max_chains: int = None,
    ) -> list[EvidenceChain]:
        """
        Generate evidence chains from graph traversal paths.

        Args:
            traversal_paths: Paths from GraphTraversal.multi_hop_search()
            max_chains: Max number of chains to generate (default from config)
        """
        max_chains = max_chains or settings.evidence_chain_max

        chains = []
        for i, path in enumerate(traversal_paths[:max_chains * 2]):  # Over-fetch for filtering
            chain = self._path_to_chain(path, chain_id=f"chain_{i}")
            if chain and chain.confidence > 0:
                chains.append(chain)

        # Sort by confidence
        chains.sort(key=lambda c: c.confidence, reverse=True)

        # Deduplicate chains with similar entity sets
        deduped = self._deduplicate_chains(chains)

        return deduped[:max_chains]

    def _path_to_chain(self, path, chain_id: str = "") -> Optional[EvidenceChain]:
        """Convert a TraversalPath to an EvidenceChain with supporting evidence."""
        entities = path.entities
        relationships = path.edge_types
        evidence_chunk_ids = path.evidence_chunks

        # Collect supporting chunk data
        supporting_chunks = []
        all_dates = set()

        for cid in evidence_chunk_ids[:10]:  # Limit evidence chunks
            chunk = self.chunk_lookup.get(cid)
            if chunk:
                text = chunk.get("text", "")
                # Extract a relevant snippet (first 300 chars)
                snippet = text[:300].strip()
                if len(text) > 300:
                    snippet += "..."

                supporting_chunks.append({
                    "chunk_id": cid,
                    "snippet": snippet,
                    "doc_filename": chunk.get("doc_filename", chunk.get("filename", "Unknown")),
                })

        # Get temporal range from entity store
        if self.entity_store and self.entity_store.is_ready:
            for ent_name in entities:
                dates = self.entity_store.get_dates_for_entity(ent_name)
                all_dates.update(dates)

        temporal_range = ("", "")
        if all_dates:
            sorted_dates = sorted(all_dates)
            temporal_range = (sorted_dates[0], sorted_dates[-1])

        # Build description
        description = self._build_description(entities, relationships)

        # Compute confidence
        confidence = path.score

        return EvidenceChain(
            entities=entities,
            relationships=relationships,
            supporting_chunks=supporting_chunks,
            confidence=confidence,
            path_description=description,
            temporal_range=temporal_range,
            chain_id=chain_id,
        )

    def _build_description(self, entities: list[str], relationships: list[str]) -> str:
        """Build a human-readable description of the evidence chain."""
        if not entities:
            return ""

        if len(entities) == 1:
            return f"Entity: {entities[0]}"

        parts = []
        for i in range(len(entities) - 1):
            rel = relationships[i] if i < len(relationships) else "connected_to"
            # Human-readable relationship name
            rel_display = rel.replace("_", " ")
            parts.append(f"{entities[i]} →({rel_display})→ {entities[i + 1]}")

        return " | ".join(parts)

    def _deduplicate_chains(self, chains: list[EvidenceChain]) -> list[EvidenceChain]:
        """Remove chains with identical entity sets."""
        seen_entity_sets = []
        unique_chains = []

        for chain in chains:
            entity_key = tuple(sorted(chain.entities))
            if entity_key not in seen_entity_sets:
                seen_entity_sets.append(entity_key)
                unique_chains.append(chain)

        return unique_chains

    def generate_from_entities(
        self,
        query_entities: list[str],
        graph_traversal,
        max_chains: int = None,
    ) -> list[EvidenceChain]:
        """
        Convenience method: run graph traversal + chain generation in one call.
        """
        max_chains = max_chains or settings.evidence_chain_max

        paths = graph_traversal.multi_hop_search(
            start_entities=query_entities,
            max_hops=settings.graph_max_hops,
        )

        return self.generate_chains(paths, max_chains=max_chains)

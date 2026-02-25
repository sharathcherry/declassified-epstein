"""
Graph Traversal: Multi-hop reasoning engine over the knowledge graph.
Discovers indirect entity connections through configurable BFS traversal
with path scoring and evidence collection.
"""

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

from backend.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TraversalPath:
    """A path discovered through graph traversal."""
    entities: list[str] = field(default_factory=list)
    edge_types: list[str] = field(default_factory=list)
    edge_weights: list[float] = field(default_factory=list)
    evidence_chunks: list[str] = field(default_factory=list)
    hops: int = 0
    score: float = 0.0
    description: str = ""


class GraphTraversal:
    """
    Multi-hop reasoning engine over a NetworkX knowledge graph.
    Supports BFS traversal with configurable depth, edge type filtering,
    and path scoring with hop decay.
    """

    def __init__(self, graph: nx.Graph, node_to_community: dict = None):
        self.graph = graph
        self.node_to_community = node_to_community or {}

    def multi_hop_search(
        self,
        start_entities: list[str],
        max_hops: int = None,
        min_edge_weight: float = 0.1,
        edge_type_filter: Optional[list[str]] = None,
        target_entities: Optional[list[str]] = None,
    ) -> list[TraversalPath]:
        """
        BFS from start entities, discovering connected entities.
        Returns scored paths with evidence chains.

        Args:
            start_entities: Starting entity names
            max_hops: Maximum traversal depth (default from config)
            min_edge_weight: Minimum edge weight to traverse
            edge_type_filter: Only follow edges of these types
            target_entities: If set, find paths TO these entities
        """
        max_hops = max_hops or settings.graph_max_hops
        hop_decay = settings.graph_hop_decay

        # Resolve start entities to graph nodes
        start_nodes = self._resolve_nodes(start_entities)
        if not start_nodes:
            return []

        target_nodes = None
        if target_entities:
            target_nodes = set(self._resolve_nodes(target_entities))

        all_paths: list[TraversalPath] = []

        for start in start_nodes:
            paths = self._bfs_paths(
                start, max_hops, min_edge_weight,
                edge_type_filter, target_nodes, hop_decay
            )
            all_paths.extend(paths)

        # Sort by score descending
        all_paths.sort(key=lambda p: p.score, reverse=True)

        return all_paths

    def find_paths_between(
        self,
        entity1: str,
        entity2: str,
        max_hops: int = None,
    ) -> list[TraversalPath]:
        """Find all paths between two entities up to max_hops."""
        max_hops = max_hops or settings.graph_max_hops
        hop_decay = settings.graph_hop_decay

        n1 = self._resolve_node(entity1)
        n2 = self._resolve_node(entity2)

        if not n1 or not n2:
            return []

        paths = []
        try:
            # Find all simple paths up to max_hops
            for path_nodes in nx.all_simple_paths(self.graph, n1, n2, cutoff=max_hops):
                tp = self._build_traversal_path(path_nodes, hop_decay)
                paths.append(tp)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

        paths.sort(key=lambda p: p.score, reverse=True)
        return paths[:20]  # Limit to top 20 paths

    def get_neighborhood(
        self,
        entity: str,
        depth: int = 1,
        min_weight: float = 1.0,
    ) -> dict:
        """Get entity neighborhood up to given depth."""
        node = self._resolve_node(entity)
        if not node:
            return {"entity": entity, "neighbors": [], "edges": []}

        neighbors = []
        edges = []
        visited = {node}
        queue = deque([(node, 0)])

        while queue:
            current, d = queue.popleft()
            if d >= depth:
                continue

            for neighbor in self.graph.neighbors(current):
                edge_data = self.graph[current][neighbor]
                weight = edge_data.get("weight", 0)

                if weight < min_weight:
                    continue

                edges.append({
                    "source": current,
                    "target": neighbor,
                    "weight": weight,
                    "type": edge_data.get("type", "mentioned_with"),
                })

                if neighbor not in visited:
                    visited.add(neighbor)
                    node_data = self.graph.nodes.get(neighbor, {})
                    neighbors.append({
                        "name": neighbor,
                        "type": node_data.get("label", "UNKNOWN"),
                        "count": node_data.get("count", 0),
                        "depth": d + 1,
                        "community": self.node_to_community.get(neighbor),
                    })
                    queue.append((neighbor, d + 1))

        neighbors.sort(key=lambda x: x["count"], reverse=True)

        return {
            "entity": node,
            "neighbors": neighbors,
            "edges": edges,
            "total_connections": len(neighbors),
        }

    def _bfs_paths(
        self,
        start: str,
        max_hops: int,
        min_weight: float,
        type_filter: Optional[list[str]],
        targets: Optional[set[str]],
        hop_decay: float,
    ) -> list[TraversalPath]:
        """BFS traversal from a start node."""
        paths = []
        visited = {start}
        # Queue: (current_node, path_so_far)
        queue = deque([(start, [start])])

        while queue:
            current, path = queue.popleft()

            if len(path) - 1 >= max_hops:
                continue

            for neighbor in self.graph.neighbors(current):
                edge_data = self.graph[current][neighbor]
                weight = edge_data.get("weight", 0)

                # Weight filter
                if weight < min_weight:
                    continue

                # Type filter
                if type_filter:
                    edge_type = edge_data.get("type", "mentioned_with")
                    if edge_type not in type_filter:
                        continue

                new_path = path + [neighbor]
                tp = self._build_traversal_path(new_path, hop_decay)

                # If targeting specific entities, only keep paths that reach them
                if targets:
                    if neighbor in targets:
                        paths.append(tp)
                else:
                    # Keep all paths of length > 1
                    if len(new_path) > 1:
                        paths.append(tp)

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, new_path))

        return paths

    def _build_traversal_path(self, path_nodes: list[str], hop_decay: float) -> TraversalPath:
        """Build a TraversalPath from a list of node names."""
        edge_types = []
        edge_weights = []
        evidence_chunks = []

        for i in range(len(path_nodes) - 1):
            n1, n2 = path_nodes[i], path_nodes[i + 1]
            edge_data = self.graph.get_edge_data(n1, n2, {})

            edge_types.append(edge_data.get("type", "mentioned_with"))
            edge_weights.append(edge_data.get("weight", 0))

            chunks = edge_data.get("chunks", edge_data.get("evidence_chunks", set()))
            if isinstance(chunks, set):
                chunks = list(chunks)
            evidence_chunks.extend(chunks[:5])

        # Score: product of edge weights × hop decay
        n_hops = len(path_nodes) - 1
        if edge_weights and n_hops > 0:
            # Normalize weights to 0-1 range
            max_w = max(edge_weights) if max(edge_weights) > 0 else 1
            norm_weights = [w / max_w for w in edge_weights]
            weight_product = 1.0
            for w in norm_weights:
                weight_product *= max(w, 0.01)  # Floor at 0.01
            score = weight_product * (hop_decay ** (n_hops - 1))
        else:
            score = 0.0

        # Build description
        parts = []
        for i in range(len(path_nodes) - 1):
            parts.append(f"{path_nodes[i]} →({edge_types[i]})→ {path_nodes[i + 1]}")
        description = " | ".join(parts)

        return TraversalPath(
            entities=path_nodes,
            edge_types=edge_types,
            edge_weights=edge_weights,
            evidence_chunks=list(set(evidence_chunks))[:20],
            hops=n_hops,
            score=score,
            description=description,
        )

    def _resolve_nodes(self, names: list[str]) -> list[str]:
        """Resolve entity names to graph node names (case-insensitive)."""
        resolved = []
        for name in names:
            node = self._resolve_node(name)
            if node:
                resolved.append(node)
        return resolved

    def _resolve_node(self, name: str) -> Optional[str]:
        """Resolve a single entity name to a graph node."""
        if name in self.graph:
            return name
        for n in self.graph.nodes:
            if n.lower() == name.lower():
                return n
        return None

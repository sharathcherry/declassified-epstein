"""
GraphRAG-powered knowledge graph builder.

Upgrades from simple co-occurrence to:
1. Entity co-occurrence graph (weighted edges)
2. Leiden community detection (hierarchical clustering)
3. LLM-generated community summaries
4. Community-aware retrieval (match queries to communities)
"""

import json
import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Optional

import networkx as nx

from backend.config import DATA_DIR, settings

logger = logging.getLogger(__name__)

GRAPH_FILE = DATA_DIR / "knowledge_graph.pkl"
GRAPH_JSON_FILE = DATA_DIR / "knowledge_graph.json"
COMMUNITIES_FILE = DATA_DIR / "graph_communities.pkl"

# Prompt for generating community summaries
COMMUNITY_SUMMARY_PROMPT = """You are analyzing a community of related entities from the Epstein Files document corpus.

Entities in this community:
{entities}

Key relationships (co-occurrences):
{relationships}

Write a 2-3 sentence summary describing:
1. Who/what are the main entities in this community
2. How they are connected
3. What theme or context links them

Summary:"""


class GraphBuilder:
    """
    GraphRAG knowledge graph with Leiden community detection.
    Nodes = entities, Edges = co-occurrence, Communities = Leiden clusters.
    """

    def __init__(self):
        self.graph = nx.Graph()
        self.communities: dict[int, dict] = {}  # community_id → {nodes, summary, ...}
        self.node_to_community: dict[str, int] = {}  # entity → community_id

    def build(self, entities: list[dict], min_count: int = 2) -> None:
        """
        Build the entity co-occurrence graph.

        Args:
            entities: List of entity dicts with text/normalized, label, chunk_id.
            min_count: Minimum occurrences for an entity to be included.
        """
        # Group entities by chunk
        chunk_entities: dict[str, list[dict]] = defaultdict(list)
        entity_counts: dict[str, int] = defaultdict(int)
        entity_chunks: dict[str, set] = defaultdict(set)

        for ent in entities:
            chunk_id = ent.get("chunk_id", "")
            name = ent.get("normalized", ent["text"].strip().title())
            if len(name) < 2:
                continue
            chunk_entities[chunk_id].append({
                "name": name,
                "label": ent["label"],
                "source": ent.get("source", ""),
            })
            entity_counts[name] += 1
            entity_chunks[name].add(chunk_id)

        # Filter by min_count
        valid_entities = {n for n, c in entity_counts.items() if c >= min_count}

        # Add nodes
        for name in valid_entities:
            labels = [e["label"] for es in chunk_entities.values()
                      for e in es if e["name"] == name]
            dominant_label = max(set(labels), key=labels.count) if labels else "UNKNOWN"

            self.graph.add_node(
                name,
                label=dominant_label,
                count=entity_counts[name],
                num_chunks=len(entity_chunks[name]),
            )

        # Add edges from co-occurrence with typed metadata
        for chunk_id, ents in chunk_entities.items():
            chunk_names = list({e["name"] for e in ents if e["name"] in valid_entities})

            for i in range(len(chunk_names)):
                for j in range(i + 1, len(chunk_names)):
                    n1, n2 = chunk_names[i], chunk_names[j]
                    if self.graph.has_edge(n1, n2):
                        self.graph[n1][n2]["weight"] += 1
                        self.graph[n1][n2]["chunks"].add(chunk_id)
                        self.graph[n1][n2]["evidence_chunks"].append(chunk_id)
                    else:
                        self.graph.add_edge(
                            n1, n2,
                            weight=1,
                            chunks={chunk_id},
                            type="mentioned_with",
                            evidence_chunks=[chunk_id],
                        )

        logger.info(
            f"Built graph: {self.graph.number_of_nodes():,} nodes, "
            f"{self.graph.number_of_edges():,} edges"
        )

    def detect_communities(self, resolution: Optional[float] = None) -> dict[int, dict]:
        """
        Run Leiden community detection on the graph.

        Returns:
            {community_id: {nodes: [...], size: int, density: float}}
        """
        if self.graph.number_of_nodes() == 0:
            return {}

        resolution = resolution or settings.graphrag_resolution

        try:
            import leidenalg
            import igraph as ig

            # Convert NetworkX → iGraph
            ig_graph = ig.Graph.from_networkx(self.graph)

            # Get edge weights
            weights = ig_graph.es["weight"] if "weight" in ig_graph.es.attributes() else None

            # Run Leiden
            partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.RBConfigurationVertexPartition,
                weights=weights,
                resolution_parameter=resolution,
                n_iterations=10,
            )

            # Map back to NetworkX nodes
            nx_nodes = list(self.graph.nodes())
            communities = {}
            self.node_to_community = {}

            for comm_id, members in enumerate(partition):
                comm_nodes = [nx_nodes[i] for i in members if i < len(nx_nodes)]
                if len(comm_nodes) < 2:
                    continue

                # Compute community density
                subgraph = self.graph.subgraph(comm_nodes)
                density = nx.density(subgraph) if len(comm_nodes) > 1 else 0

                communities[comm_id] = {
                    "nodes": comm_nodes,
                    "size": len(comm_nodes),
                    "density": round(density, 3),
                    "summary": "",  # Will be generated by LLM
                }

                for node in comm_nodes:
                    self.node_to_community[node] = comm_id

            self.communities = communities
            logger.info(
                f"Leiden detected {len(communities)} communities "
                f"(resolution={resolution})"
            )

            return communities

        except ImportError:
            logger.warning(
                "leidenalg/igraph not installed. "
                "Falling back to NetworkX Louvain communities."
            )
            return self._fallback_louvain()

    def _fallback_louvain(self) -> dict[int, dict]:
        """Fallback community detection using NetworkX's built-in Louvain."""
        try:
            partition = nx.community.louvain_communities(
                self.graph,
                resolution=settings.graphrag_resolution,
            )

            communities = {}
            self.node_to_community = {}

            for comm_id, members in enumerate(partition):
                comm_nodes = list(members)
                if len(comm_nodes) < 2:
                    continue

                subgraph = self.graph.subgraph(comm_nodes)
                density = nx.density(subgraph) if len(comm_nodes) > 1 else 0

                communities[comm_id] = {
                    "nodes": comm_nodes,
                    "size": len(comm_nodes),
                    "density": round(density, 3),
                    "summary": "",
                }

                for node in comm_nodes:
                    self.node_to_community[node] = comm_id

            self.communities = communities
            logger.info(f"Louvain fallback: {len(communities)} communities")
            return communities

        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            return {}

    def generate_community_summaries(self, llm_client, max_communities: int = 50) -> None:
        """
        Generate LLM summaries for each community (GraphRAG-style).
        """
        if not llm_client or not llm_client.available:
            logger.warning("LLM not available for community summaries")
            return

        sorted_comms = sorted(
            self.communities.items(),
            key=lambda x: x[1]["size"],
            reverse=True,
        )

        for comm_id, comm_data in sorted_comms[:max_communities]:
            if comm_data.get("summary"):
                continue

            nodes = comm_data["nodes"][:20]  # Limit for prompt
            entities_text = "\n".join(
                f"- {n} ({self.graph.nodes[n].get('label', '?')}, "
                f"mentioned {self.graph.nodes[n].get('count', 0)} times)"
                for n in nodes
            )

            # Get top relationships
            relationships = []
            subgraph = self.graph.subgraph(nodes)
            sorted_edges = sorted(
                subgraph.edges(data=True),
                key=lambda x: x[2].get("weight", 0),
                reverse=True,
            )
            for u, v, data in sorted_edges[:10]:
                relationships.append(
                    f"- {u} ↔ {v} (co-occur {data.get('weight', 0)} times)"
                )

            rel_text = "\n".join(relationships) if relationships else "No strong relationships."

            try:
                summary = llm_client.generate(
                    system_prompt="You are an investigative analyst.",
                    user_prompt=COMMUNITY_SUMMARY_PROMPT.format(
                        entities=entities_text,
                        relationships=rel_text,
                    ),
                    max_tokens=200,
                    temperature=0.0,
                )
                comm_data["summary"] = summary.strip()
            except Exception as e:
                logger.debug(f"Community summary failed for {comm_id}: {e}")

        summaries_generated = sum(1 for c in self.communities.values() if c.get("summary"))
        logger.info(f"Generated {summaries_generated} community summaries")

    def get_relevant_communities(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Find communities most relevant to a query.
        Uses keyword overlap between query and community nodes/summaries.
        """
        if not self.communities:
            return []

        query_words = set(w.lower() for w in query.split() if len(w) > 2)
        scored = []

        for comm_id, comm_data in self.communities.items():
            score = 0

            # Score by entity name overlap
            for node in comm_data["nodes"]:
                node_words = set(w.lower() for w in node.split() if len(w) > 2)
                overlap = len(query_words & node_words)
                score += overlap * 2

            # Score by summary text overlap
            summary = comm_data.get("summary", "").lower()
            for word in query_words:
                if word in summary:
                    score += 1

            if score > 0:
                scored.append({
                    "community_id": comm_id,
                    "score": score,
                    "nodes": comm_data["nodes"][:15],
                    "size": comm_data["size"],
                    "summary": comm_data.get("summary", ""),
                    "density": comm_data.get("density", 0),
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def get_community_context(self, query: str) -> str:
        """
        Generate community context to inject into LLM prompt.
        Returns a text block describing relevant communities.
        """
        relevant = self.get_relevant_communities(query, top_k=3)
        if not relevant:
            return ""

        context_parts = ["## Relevant Entity Communities\n"]
        for comm in relevant:
            if comm.get("summary"):
                entities_str = ", ".join(comm["nodes"][:10])
                context_parts.append(
                    f"**Community** ({comm['size']} entities): {entities_str}\n"
                    f"Summary: {comm['summary']}\n"
                )

        return "\n".join(context_parts)

    # ── Existing query methods (preserved) ─────────────────────

    def get_entity(self, name: str) -> dict | None:
        """Get entity details, connections, and community."""
        if name not in self.graph:
            for n in self.graph.nodes:
                if n.lower() == name.lower():
                    name = n
                    break
            else:
                return None

        node_data = self.graph.nodes[name]
        neighbors = []
        for neighbor in self.graph.neighbors(name):
            edge_data = self.graph[name][neighbor]
            neighbors.append({
                "name": neighbor,
                "label": self.graph.nodes[neighbor].get("label", "UNKNOWN"),
                "weight": edge_data.get("weight", 1),
            })

        neighbors.sort(key=lambda x: x["weight"], reverse=True)

        community_id = self.node_to_community.get(name)
        community_info = None
        if community_id is not None and community_id in self.communities:
            comm = self.communities[community_id]
            community_info = {
                "id": community_id,
                "size": comm["size"],
                "summary": comm.get("summary", ""),
            }

        return {
            "name": name,
            "label": node_data.get("label", "UNKNOWN"),
            "count": node_data.get("count", 0),
            "num_chunks": node_data.get("num_chunks", 0),
            "connections": neighbors[:50],
            "degree": self.graph.degree(name),
            "community": community_info,
        }

    def search_entities(self, query: str, limit: int = 50) -> list[dict]:
        """Search entities by name."""
        query_lower = query.lower()
        results = []

        for name, data in self.graph.nodes(data=True):
            if query_lower in name.lower():
                results.append({
                    "name": name,
                    "label": data.get("label", "UNKNOWN"),
                    "count": data.get("count", 0),
                    "degree": self.graph.degree(name),
                    "community_id": self.node_to_community.get(name),
                })

        results.sort(key=lambda x: x["count"], reverse=True)
        return results[:limit]

    def get_relationships(self, entity1: str, entity2: str) -> dict | None:
        """Get relationship between two entities."""
        n1 = n2 = None
        for n in self.graph.nodes:
            if n.lower() == entity1.lower():
                n1 = n
            if n.lower() == entity2.lower():
                n2 = n

        if not n1 or not n2:
            return None

        result = {
            "entity1": n1,
            "entity2": n2,
            "community1": self.node_to_community.get(n1),
            "community2": self.node_to_community.get(n2),
            "same_community": (
                self.node_to_community.get(n1) == self.node_to_community.get(n2)
                and self.node_to_community.get(n1) is not None
            ),
        }

        if self.graph.has_edge(n1, n2):
            edge = self.graph[n1][n2]
            result.update({
                "direct_connection": True,
                "weight": edge.get("weight", 0),
                "co_occurrences": edge.get("weight", 0),
            })
        else:
            try:
                path = nx.shortest_path(self.graph, n1, n2)
                result.update({
                    "direct_connection": False,
                    "path": path,
                    "path_length": len(path) - 1,
                })
            except nx.NetworkXNoPath:
                result.update({
                    "direct_connection": False,
                    "path": None,
                    "message": "No connection found.",
                })

        return result

    def get_graph_json(self) -> dict:
        """Export graph as JSON for frontend visualization (with communities)."""
        nodes = []
        for name, data in self.graph.nodes(data=True):
            nodes.append({
                "id": name,
                "label": data.get("label", "UNKNOWN"),
                "count": data.get("count", 0),
                "degree": self.graph.degree(name),
                "community": self.node_to_community.get(name, -1),
            })

        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "weight": data.get("weight", 1),
                "type": data.get("type", "mentioned_with"),
            })

        nodes.sort(key=lambda x: x["degree"], reverse=True)

        communities_summary = []
        for cid, cdata in sorted(self.communities.items(), key=lambda x: x[1]["size"], reverse=True):
            communities_summary.append({
                "id": cid,
                "size": cdata["size"],
                "density": cdata.get("density", 0),
                "summary": cdata.get("summary", ""),
                "top_nodes": cdata["nodes"][:5],
            })

        return {
            "nodes": nodes[:500],
            "edges": edges[:2000],
            "communities": communities_summary[:50],
        }

    # ── Persistence ────────────────────────────────────────────

    def save(self) -> None:
        """Save graph and communities to disk."""
        with open(GRAPH_FILE, "wb") as f:
            pickle.dump(self.graph, f)

        with open(COMMUNITIES_FILE, "wb") as f:
            pickle.dump({
                "communities": self.communities,
                "node_to_community": self.node_to_community,
            }, f)

        self._export_json()
        logger.info("Saved knowledge graph and communities")

    def load(self) -> bool:
        """Load graph and communities from disk."""
        if not GRAPH_FILE.exists():
            return False
        try:
            with open(GRAPH_FILE, "rb") as f:
                self.graph = pickle.load(f)

            if COMMUNITIES_FILE.exists():
                with open(COMMUNITIES_FILE, "rb") as f:
                    comm_data = pickle.load(f)
                    self.communities = comm_data.get("communities", {})
                    self.node_to_community = comm_data.get("node_to_community", {})

            logger.info(
                f"Loaded graph: {self.graph.number_of_nodes():,} nodes, "
                f"{len(self.communities)} communities"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load graph: {e}")
            return False

    def _export_json(self) -> None:
        """Export graph to JSON file."""
        data = self.get_graph_json()
        with open(GRAPH_JSON_FILE, "w") as f:
            json.dump(data, f)

    @property
    def is_ready(self) -> bool:
        return self.graph.number_of_nodes() > 0

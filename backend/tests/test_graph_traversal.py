"""Tests for multi-hop graph traversal and path scoring."""

import os
import sys

import networkx as nx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.fixture
def sample_graph():
    """Build a small test graph with known topology."""
    G = nx.Graph()
    G.add_node("Epstein", label="PERSON", count=300)
    G.add_node("Maxwell", label="PERSON", count=200)
    G.add_node("Andrew", label="PERSON", count=100)
    G.add_node("Palm Beach", label="GPE", count=80)
    G.add_node("Wexner", label="PERSON", count=50)

    G.add_edge("Epstein", "Maxwell", weight=120, type="associated_with",
               chunks={"c1", "c2"}, evidence_chunks=["c1", "c2"])
    G.add_edge("Epstein", "Palm Beach", weight=80, type="located_at",
               chunks={"c1"}, evidence_chunks=["c1"])
    G.add_edge("Maxwell", "Andrew", weight=30, type="mentioned_with",
               chunks={"c3"}, evidence_chunks=["c3"])
    G.add_edge("Epstein", "Wexner", weight=25, type="employed_by",
               chunks={"c4"}, evidence_chunks=["c4"])

    return G


@pytest.fixture
def traversal(sample_graph):
    from backend.knowledge_graph.graph_traversal import GraphTraversal
    return GraphTraversal(sample_graph, node_to_community={"Epstein": 0, "Maxwell": 0, "Andrew": 0})


class TestGraphTraversal:

    def test_multi_hop_search_1hop(self, traversal):
        paths = traversal.multi_hop_search(["Epstein"], max_hops=1)
        assert len(paths) > 0
        # All paths should be 1-hop
        for p in paths:
            assert p.hops == 1

    def test_multi_hop_search_2hop(self, traversal):
        paths = traversal.multi_hop_search(["Epstein"], max_hops=2)
        # Should find Epstein → Maxwell → Andrew (2-hop)
        two_hop = [p for p in paths if p.hops == 2]
        assert len(two_hop) > 0

    def test_path_scoring_decay(self, traversal):
        paths = traversal.multi_hop_search(["Epstein"], max_hops=3)
        one_hop = [p for p in paths if p.hops == 1]
        two_hop = [p for p in paths if p.hops == 2]

        if one_hop and two_hop:
            # 1-hop should score higher than 2-hop (with decay)
            max_1 = max(p.score for p in one_hop)
            max_2 = max(p.score for p in two_hop)
            assert max_1 >= max_2

    def test_find_paths_between(self, traversal):
        paths = traversal.find_paths_between("Epstein", "Andrew", max_hops=3)
        assert len(paths) > 0
        # Best path should be Epstein → Maxwell → Andrew
        best = paths[0]
        assert "Epstein" in best.entities
        assert "Andrew" in best.entities

    def test_find_paths_no_connection(self, traversal):
        # Add disconnected node
        traversal.graph.add_node("Isolated", label="PERSON", count=1)
        paths = traversal.find_paths_between("Epstein", "Isolated", max_hops=3)
        assert len(paths) == 0

    def test_edge_type_filter(self, traversal):
        paths = traversal.multi_hop_search(
            ["Epstein"], max_hops=2, edge_type_filter=["associated_with"]
        )
        for p in paths:
            for et in p.edge_types:
                assert et == "associated_with"

    def test_min_weight_filter(self, traversal):
        paths = traversal.multi_hop_search(["Epstein"], max_hops=1, min_edge_weight=50)
        # Should only include edges with weight >= 50 (Epstein-Maxwell=120, Epstein-PB=80)
        for p in paths:
            for w in p.edge_weights:
                assert w >= 50

    def test_neighborhood(self, traversal):
        result = traversal.get_neighborhood("Epstein", depth=1)
        assert result["entity"] == "Epstein"
        assert len(result["neighbors"]) >= 2
        neighbor_names = {n["name"] for n in result["neighbors"]}
        assert "Maxwell" in neighbor_names

    def test_path_description(self, traversal):
        paths = traversal.find_paths_between("Epstein", "Andrew")
        assert len(paths) > 0
        desc = paths[0].description
        assert "Epstein" in desc
        assert "Andrew" in desc
        assert "→" in desc

    def test_evidence_chunks_collected(self, traversal):
        paths = traversal.multi_hop_search(["Epstein"], max_hops=1)
        for p in paths:
            assert len(p.evidence_chunks) > 0

    def test_case_insensitive_resolve(self, traversal):
        paths = traversal.multi_hop_search(["epstein"], max_hops=1)
        assert len(paths) > 0

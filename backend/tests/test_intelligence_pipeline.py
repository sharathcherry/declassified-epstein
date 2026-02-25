"""Integration test: full intelligence pipeline end-to-end (unit test with mocks)."""

import os
import sys

import networkx as nx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestEvidenceChainGeneration:
    """Test evidence chain pipeline: traversal → chain generation."""

    @pytest.fixture
    def graph(self):
        G = nx.Graph()
        G.add_node("Epstein", label="PERSON", count=300)
        G.add_node("Maxwell", label="PERSON", count=200)
        G.add_node("Andrew", label="PERSON", count=100)
        G.add_edge("Epstein", "Maxwell", weight=120, type="associated_with",
                   chunks={"c1"}, evidence_chunks=["c1", "c2"])
        G.add_edge("Maxwell", "Andrew", weight=30, type="mentioned_with",
                   chunks={"c3"}, evidence_chunks=["c3"])
        return G

    @pytest.fixture
    def chunk_data(self):
        return [
            {"id": "c1", "text": "Epstein and Maxwell shared financial ties. Document source.", "doc_filename": "fin_01.pdf"},
            {"id": "c2", "text": "Maxwell organized travel logistics for Epstein.", "doc_filename": "travel_02.pdf"},
            {"id": "c3", "text": "Andrew was reportedly introduced by Maxwell at a party.", "doc_filename": "social_03.pdf"},
        ]

    def test_chain_from_traversal(self, graph, chunk_data):
        from backend.knowledge_graph.graph_traversal import GraphTraversal
        from backend.knowledge_graph.evidence_chain import EvidenceChainGenerator

        traversal = GraphTraversal(graph)
        chain_gen = EvidenceChainGenerator()
        chain_gen.set_chunk_lookup(chunk_data)

        paths = traversal.multi_hop_search(["Epstein"], max_hops=2)
        assert len(paths) > 0

        chains = chain_gen.generate_chains(paths, max_chains=5)
        assert len(chains) > 0

        # Validate chain structure
        for chain in chains:
            assert len(chain.entities) >= 2
            assert chain.confidence >= 0
            assert chain.path_description != ""

    def test_chain_to_dict(self, graph, chunk_data):
        from backend.knowledge_graph.graph_traversal import GraphTraversal
        from backend.knowledge_graph.evidence_chain import EvidenceChainGenerator

        traversal = GraphTraversal(graph)
        chain_gen = EvidenceChainGenerator()
        chain_gen.set_chunk_lookup(chunk_data)

        paths = traversal.multi_hop_search(["Epstein"], max_hops=1)
        chains = chain_gen.generate_chains(paths)

        for chain in chains:
            d = chain.to_dict()
            assert "entities" in d
            assert "relationships" in d
            assert "confidence" in d
            assert "supporting_chunks" in d

    def test_chain_has_evidence_snippets(self, graph, chunk_data):
        from backend.knowledge_graph.graph_traversal import GraphTraversal
        from backend.knowledge_graph.evidence_chain import EvidenceChainGenerator

        traversal = GraphTraversal(graph)
        chain_gen = EvidenceChainGenerator()
        chain_gen.set_chunk_lookup(chunk_data)

        paths = traversal.multi_hop_search(["Epstein"], max_hops=1)
        chains = chain_gen.generate_chains(paths)

        has_evidence = any(len(c.supporting_chunks) > 0 for c in chains)
        assert has_evidence

    def test_deduplication(self, graph, chunk_data):
        from backend.knowledge_graph.graph_traversal import GraphTraversal
        from backend.knowledge_graph.evidence_chain import EvidenceChainGenerator

        traversal = GraphTraversal(graph)
        chain_gen = EvidenceChainGenerator()
        chain_gen.set_chunk_lookup(chunk_data)

        paths = traversal.multi_hop_search(["Epstein"], max_hops=2)
        chains = chain_gen.generate_chains(paths, max_chains=20)

        # No two chains should have identical entity sets
        entity_sets = [tuple(sorted(c.entities)) for c in chains]
        assert len(entity_sets) == len(set(entity_sets))


class TestEntityRetriever:
    """Test entity-first retrieval."""

    @pytest.fixture
    def entity_store(self):
        from backend.knowledge_graph.entity_store import EntityStore
        store = EntityStore()
        store._ready = True
        store.index = {
            "Jeffrey Epstein": {
                "type": "PERSON", "count": 300,
                "chunk_ids": ["c1", "c2", "c3"],
                "source_docs": ["d1.pdf"],
                "dates_associated": ["2005"],
                "co_occurring": [{"name": "Ghislaine Maxwell", "count": 100}],
            },
            "Ghislaine Maxwell": {
                "type": "PERSON", "count": 200,
                "chunk_ids": ["c2", "c4"],
                "source_docs": ["d1.pdf"],
                "dates_associated": ["2005"],
                "co_occurring": [{"name": "Jeffrey Epstein", "count": 100}],
            },
        }
        store._build_secondary_indices()
        return store

    @pytest.fixture
    def chunks(self):
        return [
            {"id": "c1", "text": "Epstein financial docs."},
            {"id": "c2", "text": "Epstein and Maxwell correspondence."},
            {"id": "c3", "text": "Epstein travel records."},
            {"id": "c4", "text": "Maxwell socialite network."},
        ]

    def test_entity_retrieval(self, entity_store, chunks):
        from backend.retrieval.entity_retriever import EntityRetriever
        retriever = EntityRetriever(entity_store)
        retriever.set_chunk_lookup(chunks)

        results = retriever.search(["Jeffrey Epstein"], top_k=5)
        assert len(results) > 0
        assert results[0]["entity_score"] > 0

    def test_entity_retrieval_empty(self, entity_store, chunks):
        from backend.retrieval.entity_retriever import EntityRetriever
        retriever = EntityRetriever(entity_store)
        retriever.set_chunk_lookup(chunks)

        results = retriever.search(["Nonexistent Person"])
        assert len(results) == 0

    def test_entity_expansion(self, entity_store, chunks):
        from backend.retrieval.entity_retriever import EntityRetriever
        retriever = EntityRetriever(entity_store)
        retriever.set_chunk_lookup(chunks)

        results_no_expand = retriever.search(["Jeffrey Epstein"], expand=False)
        results_expand = retriever.search(["Jeffrey Epstein"], expand=True)

        # Expansion should find at least as many results
        assert len(results_expand) >= len(results_no_expand)

"""Tests for EntityStore CRUD, lookup, filtering, and persistence."""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestEntityStore:
    """Test entity store operations."""

    @pytest.fixture
    def store(self):
        from backend.knowledge_graph.entity_store import EntityStore
        store = EntityStore()
        store._ready = True
        store.index = {
            "Jeffrey Epstein": {
                "type": "PERSON", "count": 342,
                "chunk_ids": ["c1", "c2", "c3"],
                "source_docs": ["doc1.pdf", "doc2.pdf"],
                "dates_associated": ["2005", "2008", "2019"],
                "co_occurring": [{"name": "Ghislaine Maxwell", "count": 120}],
            },
            "Ghislaine Maxwell": {
                "type": "PERSON", "count": 218,
                "chunk_ids": ["c2", "c4"],
                "source_docs": ["doc1.pdf"],
                "dates_associated": ["2005", "2019"],
                "co_occurring": [{"name": "Jeffrey Epstein", "count": 120}],
            },
            "Palm Beach": {
                "type": "GPE", "count": 95,
                "chunk_ids": ["c1", "c5"],
                "source_docs": ["doc3.pdf"],
                "dates_associated": ["2005"],
                "co_occurring": [],
            },
            "Title 18": {
                "type": "LAW", "count": 15,
                "chunk_ids": ["c6"],
                "source_docs": ["doc4.pdf"],
                "dates_associated": [],
                "co_occurring": [],
            },
        }
        store._build_secondary_indices()
        return store

    def test_exact_lookup(self, store):
        result = store.lookup("Jeffrey Epstein")
        assert result is not None
        assert result["name"] == "Jeffrey Epstein"
        assert result["count"] == 342

    def test_case_insensitive_lookup(self, store):
        result = store.lookup("jeffrey epstein")
        assert result is not None
        assert result["name"] == "Jeffrey Epstein"

    def test_lookup_missing(self, store):
        assert store.lookup("Nonexistent") is None

    def test_search_substring(self, store):
        results = store.search("Epstein")
        assert len(results) >= 1
        assert results[0]["name"] == "Jeffrey Epstein"

    def test_fuzzy_search(self, store):
        results = store.search_fuzzy("Jeffrey Palm")
        assert len(results) >= 1

    def test_filter_by_type(self, store):
        persons = store.filter_by_type("PERSON")
        assert len(persons) == 2
        laws = store.filter_by_type("LAW")
        assert len(laws) == 1

    def test_filter_by_date_range(self, store):
        results = store.filter_by_date_range("2005", "2008")
        names = {r["name"] for r in results}
        assert "Jeffrey Epstein" in names
        assert "Palm Beach" in names

    def test_get_chunk_ids(self, store):
        ids = store.get_chunk_ids("Jeffrey Epstein")
        assert set(ids) == {"c1", "c2", "c3"}

    def test_get_chunk_ids_multi(self, store):
        ids = store.get_chunk_ids_multi(["Jeffrey Epstein", "Palm Beach"])
        assert "c1" in ids
        assert "c5" in ids

    def test_get_entities_in_chunk(self, store):
        entities = store.get_entities_in_chunk("c2")
        assert "Jeffrey Epstein" in entities
        assert "Ghislaine Maxwell" in entities

    def test_get_entity_profile(self, store):
        profile = store.get_entity_profile("Jeffrey Epstein")
        assert profile is not None
        assert profile["mentions"] == 342
        assert "2019" in profile["dates_associated"]

    def test_add_entity(self, store):
        store.add_entity("Prince Andrew", "PERSON", chunk_id="c10",
                         source_doc="doc5.pdf", dates=["2001"])
        result = store.lookup("Prince Andrew")
        assert result is not None
        assert result["count"] == 1
        assert "c10" in result["chunk_ids"]

    def test_entity_overlap(self, store):
        score = store.get_entity_overlap(
            ["Jeffrey Epstein", "Ghislaine Maxwell"],
            ["Jeffrey Epstein", "Palm Beach"]
        )
        assert 0 < score <= 1.0

    def test_save_and_load(self, store):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name

        try:
            store.save(path)
            assert os.path.exists(path)

            from backend.knowledge_graph.entity_store import EntityStore
            store2 = EntityStore()
            assert store2.load(path)
            assert store2.is_ready
            assert len(store2.index) == len(store.index)
        finally:
            os.unlink(path)

    def test_stats(self, store):
        stats = store.stats()
        assert stats["ready"] is True
        assert stats["total_entities"] == 4
        assert "PERSON" in stats["type_distribution"]

    def test_extract_entities_from_text(self, store):
        found = store.extract_entities_from_text(
            "Jeffrey Epstein was connected to Ghislaine Maxwell in Palm Beach"
        )
        names = {e["name"] for e in found}
        assert "Jeffrey Epstein" in names

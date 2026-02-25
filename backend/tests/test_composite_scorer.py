"""Tests for composite scoring and query parsing."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestQueryParser:
    """Test query decomposition into structured filters."""

    @pytest.fixture
    def parser(self):
        from backend.retrieval.query_parser import QueryParser
        return QueryParser(entity_store=None)

    def test_date_extraction_single(self, parser):
        parsed = parser.parse("What happened in 2005?")
        assert "2005" in parsed.dates_mentioned

    def test_date_extraction_range(self, parser):
        parsed = parser.parse("Events between 1999 and 2005")
        assert "1999" in parsed.dates_mentioned
        assert "2005" in parsed.dates_mentioned

    def test_legal_act_extraction(self, parser):
        parsed = parser.parse("Was Title 18 Section 2255 violated?")
        assert len(parsed.legal_acts) > 0

    def test_intent_entity_lookup(self, parser):
        parsed = parser.parse("Who is Jeffrey Epstein?")
        assert parsed.intent == "entity_lookup"

    def test_intent_relationship(self, parser):
        parsed = parser.parse("What is the connection between them?")
        assert parsed.intent == "relationship"

    def test_intent_timeline(self, parser):
        parsed = parser.parse("Show me the timeline of events")
        assert parsed.intent == "timeline"

    def test_intent_general_search(self, parser):
        parsed = parser.parse("Tell me about the flight logs")
        assert parsed.intent == "search"

    def test_keywords_extraction(self, parser):
        parsed = parser.parse("flight logs Palm Beach 2005")
        assert "flight" in parsed.search_keywords or "logs" in parsed.search_keywords

    def test_structured_filters_dates(self, parser):
        parsed = parser.parse("Events between 2001 and 2005")
        assert "date_from" in parsed.structured_filters
        assert "date_to" in parsed.structured_filters


class TestCompositeScorer:
    """Test composite scoring logic."""

    @pytest.fixture
    def scorer(self):
        from backend.retrieval.composite_scorer import CompositeScorer
        return CompositeScorer()

    def test_score_empty_results(self, scorer):
        scored = scorer.score_results([])
        assert scored == []

    def test_score_basic_results(self, scorer):
        results = [
            {"id": "c1", "text": "Epstein was at Palm Beach.", "rrf_score": 0.05, "retrieval_sources": ["dense", "sparse"]},
            {"id": "c2", "text": "Maxwell traveled to London.", "rrf_score": 0.03, "retrieval_sources": ["dense"]},
        ]
        scored = scorer.score_results(results)
        assert len(scored) == 2
        # All results should have composite_score
        for r in scored:
            assert "composite_score" in r
            assert r["composite_score"] >= 0

    def test_score_ordering(self, scorer):
        results = [
            {"id": "c1", "text": "Epstein Palm Beach 2005", "rrf_score": 0.05, "retrieval_sources": ["dense", "sparse"]},
            {"id": "c2", "text": "Some random text.", "rrf_score": 0.01, "retrieval_sources": []},
        ]
        scored = scorer.score_results(results, query_entities=["Jeffrey Epstein"],
                                       query_dates=["2005"])
        # First result should score higher (has entity mention and date match)
        assert scored[0]["composite_score"] >= scored[1]["composite_score"]

    def test_temporal_boost_applied(self, scorer):
        result = {"id": "c1", "text": "In 2005 Epstein traveled.", "rrf_score": 0.04, "retrieval_sources": []}
        score_with_date = scorer._score_single(result, ["Epstein"], ["2005"], {})

        result2 = {"id": "c2", "text": "Generic text here.", "rrf_score": 0.04, "retrieval_sources": []}
        score_no_date = scorer._score_single(result2, ["Epstein"], ["2005"], {})

        # Temporal boost should make score_with_date slightly higher
        assert score_with_date >= score_no_date

    def test_weights_sum_to_one(self, scorer):
        total = scorer.w_semantic + scorer.w_keyword + scorer.w_entity + scorer.w_graph
        assert abs(total - 1.0) < 0.01

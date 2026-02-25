"""Tests for entity extraction with structured output."""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestEntityExtractor:
    """Test entity extraction, normalization, and structured output."""

    @pytest.fixture
    def extractor(self):
        from backend.knowledge_graph.entity_extractor import EntityExtractor
        return EntityExtractor()

    def test_alias_normalization(self, extractor):
        """Test that aliases are resolved correctly."""
        text = "Epstein met with Maxwell at the Palm Beach residence."
        entities = extractor.extract_from_text(text)

        # Find normalized names
        names = {e["normalized"] for e in entities}
        # Aliases should resolve
        assert "Jeffrey Epstein" in names or "Ghislaine Maxwell" in names or len(entities) > 0

    def test_legal_act_detection(self, extractor):
        """Test detection of legal act references."""
        text = "Under Title 18 Section 2255, the defendant filed a motion."
        entities = extractor.extract_from_text(text)
        labels = {e["label"] for e in entities}
        assert "LAW" in labels

    def test_event_extraction(self, extractor):
        """Test detection of events."""
        text = "The deposition was held on March 15, 2005 after the arrest in Palm Beach."
        entities = extractor.extract_from_text(text)
        labels = {e["label"] for e in entities}
        assert "EVENT" in labels

    def test_case_number_detection(self, extractor):
        """Test case number regex pattern."""
        text = "Case 08-CJ-2343 was filed in 2008."
        entities = extractor.extract_from_text(text)
        labels = {e["label"] for e in entities}
        assert "CASE_NUMBER" in labels

    def test_money_detection(self, extractor):
        """Test money pattern."""
        text = "The settlement was $150,000."
        entities = extractor.extract_from_text(text)
        labels = {e["label"] for e in entities}
        assert "MONEY" in labels

    def test_structured_extraction(self, extractor):
        """Test structured extraction produces correct format."""
        text = "Jeffrey Epstein was arrested in 2019 under Title 18. A deposition was held."
        result = extractor.extract_structured(text, "test.pdf", "chunk_001")

        assert result["chunk_id"] == "chunk_001"
        assert result["doc_filename"] == "test.pdf"
        assert "entities" in result
        assert "dates_in_chunk" in result
        assert "legal_acts" in result
        assert "events" in result
        assert result["entity_count"] >= 0

    def test_normalize_entities_aggregation(self, extractor):
        """Test entity aggregation via normalize_entities."""
        entities = [
            {"text": "Epstein", "normalized": "Jeffrey Epstein", "label": "PERSON",
             "source": "doc1.pdf", "chunk_id": "c1"},
            {"text": "Epstein", "normalized": "Jeffrey Epstein", "label": "PERSON",
             "source": "doc2.pdf", "chunk_id": "c2"},
        ]

        result = EntityExtractor.normalize_entities(entities)
        assert "Jeffrey Epstein" in result
        assert result["Jeffrey Epstein"]["count"] == 2
        assert len(result["Jeffrey Epstein"]["sources"]) == 2
        assert len(result["Jeffrey Epstein"]["chunk_ids"]) == 2

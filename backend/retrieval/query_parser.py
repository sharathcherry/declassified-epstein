"""
Query Parser: Extract entities, dates, legal acts, and intent from user queries.
Decomposes natural language into structured filters for entity-centric retrieval.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from backend.config import settings

logger = logging.getLogger(__name__)


@dataclass
class EntityRef:
    """Reference to an entity found in a query."""
    name: str
    type: str = "UNKNOWN"
    count: int = 0
    resolved: bool = False  # True if found in entity store


@dataclass
class ParsedQuery:
    """Structured query decomposition."""
    raw_query: str
    entities_mentioned: list[EntityRef] = field(default_factory=list)
    dates_mentioned: list[str] = field(default_factory=list)
    legal_acts: list[str] = field(default_factory=list)
    intent: str = "search"  # entity_lookup, relationship, timeline, search
    structured_filters: dict = field(default_factory=dict)
    search_keywords: list[str] = field(default_factory=list)


class QueryParser:
    """
    Parse user queries to extract entities, dates, intent, and filters.
    Uses entity store for resolution + regex patterns for structure.
    """

    YEAR_PAT = re.compile(r'\b(19[5-9]\d|20[0-2]\d)\b')
    DATE_RANGE_PAT = re.compile(
        r'\b(19[5-9]\d|20[0-2]\d)\s*[-–to]+\s*(19[5-9]\d|20[0-2]\d)\b', re.I
    )
    LEGAL_PAT = re.compile(
        r'\b(?:Title\s+\d+|Section\s+\d+[a-z]?|§\s*\d+|U\.?S\.?C\.?\s*§?\s*\d+)\b', re.I
    )

    # Intent classification keywords
    ENTITY_LOOKUP_WORDS = {"who is", "tell me about", "what do we know about", "profile", "details on"}
    RELATIONSHIP_WORDS = {"connection", "relationship", "linked", "connected", "between", "path"}
    TIMELINE_WORDS = {"timeline", "chronolog", "when did", "history of", "over time"}

    def __init__(self, entity_store=None):
        self.entity_store = entity_store

    def parse(self, query: str) -> ParsedQuery:
        """Parse a raw query into structured components."""
        parsed = ParsedQuery(raw_query=query)

        # Extract dates
        parsed.dates_mentioned = self._extract_dates(query)

        # Extract legal acts
        parsed.legal_acts = self._extract_legal_acts(query)

        # Extract entities
        parsed.entities_mentioned = self._extract_entities(query)

        # Classify intent
        parsed.intent = self._classify_intent(query, parsed)

        # Build structured filters
        parsed.structured_filters = self._build_filters(parsed)

        # Extract remaining keywords (non-entity, non-date words)
        parsed.search_keywords = self._extract_keywords(query, parsed)

        return parsed

    def _extract_dates(self, query: str) -> list[str]:
        """Extract year and date references from query."""
        dates = []

        # Check for date ranges
        for m in self.DATE_RANGE_PAT.finditer(query):
            dates.extend([m.group(1), m.group(2)])

        # Single years
        for m in self.YEAR_PAT.finditer(query):
            if m.group(1) not in dates:
                dates.append(m.group(1))

        return sorted(set(dates))

    def _extract_legal_acts(self, query: str) -> list[str]:
        """Extract legal act references from query."""
        return [m.group().strip() for m in self.LEGAL_PAT.finditer(query)]

    def _extract_entities(self, query: str) -> list[EntityRef]:
        """Extract entities from query using entity store lookup."""
        entities = []

        if self.entity_store and self.entity_store.is_ready:
            # Use entity store for fast dictionary-based detection
            found = self.entity_store.extract_entities_from_text(query)
            for ent in found:
                entities.append(EntityRef(
                    name=ent["name"],
                    type=ent.get("type", "UNKNOWN"),
                    count=ent.get("count", 0),
                    resolved=True,
                ))

        # Deduplicate — keep highest count version
        seen = {}
        for e in entities:
            key = e.name.lower()
            if key not in seen or e.count > seen[key].count:
                seen[key] = e

        return list(seen.values())

    def _classify_intent(self, query: str, parsed: ParsedQuery) -> str:
        """Classify query intent based on keywords and structure."""
        query_lower = query.lower()

        # Entity lookup intent
        if any(kw in query_lower for kw in self.ENTITY_LOOKUP_WORDS):
            return "entity_lookup"

        # Relationship intent
        if any(kw in query_lower for kw in self.RELATIONSHIP_WORDS):
            return "relationship"

        # Timeline intent
        if any(kw in query_lower for kw in self.TIMELINE_WORDS):
            return "timeline"

        # If query is basically just an entity name, treat as entity lookup
        if len(parsed.entities_mentioned) == 1 and len(query.split()) <= 4:
            return "entity_lookup"

        # If two entities mentioned, likely a relationship query
        if len(parsed.entities_mentioned) >= 2:
            return "relationship"

        return "search"

    def _build_filters(self, parsed: ParsedQuery) -> dict:
        """Build structured filters from parsed query."""
        filters = {}

        if parsed.dates_mentioned:
            if len(parsed.dates_mentioned) >= 2:
                filters["date_from"] = min(parsed.dates_mentioned)
                filters["date_to"] = max(parsed.dates_mentioned)
            else:
                filters["date_from"] = parsed.dates_mentioned[0]
                filters["date_to"] = parsed.dates_mentioned[0]

        if parsed.entities_mentioned:
            filters["entities"] = [e.name for e in parsed.entities_mentioned]

        if parsed.legal_acts:
            filters["legal_acts"] = parsed.legal_acts

        return filters

    def _extract_keywords(self, query: str, parsed: ParsedQuery) -> list[str]:
        """Extract non-entity, non-date keywords for BM25 search."""
        words = re.sub(r'[^\w\s]', ' ', query.lower()).split()

        # Remove entity names
        entity_words = set()
        for e in parsed.entities_mentioned:
            entity_words.update(e.name.lower().split())

        # Remove date strings
        date_words = set(parsed.dates_mentioned)

        # Remove stopwords
        stopwords = {"the", "a", "an", "is", "was", "were", "are", "in", "on",
                      "at", "to", "for", "of", "with", "and", "or", "but",
                      "who", "what", "when", "where", "how", "did", "do",
                      "about", "between", "from"}

        keywords = [
            w for w in words
            if w not in entity_words
            and w not in date_words
            and w not in stopwords
            and len(w) > 1
        ]

        return keywords

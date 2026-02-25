"""
Entity extractor using spaCy transformer NER (en_core_web_trf) + custom rules.
Extracts people, organizations, locations, dates, flights, money, legal acts,
events, and produces structured entity records with temporal anchoring,
co-occurrence hints, and relationship detection.
"""

import logging
import re
from collections import Counter, defaultdict
from typing import Optional

from backend.config import settings

logger = logging.getLogger(__name__)

# Try to load spaCy
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not installed — entity extraction disabled")


class EntityExtractor:
    """Extract entities using transformer-based NER and custom regex rules."""

    # Custom patterns for legal/investigative documents
    FLIGHT_PATTERN = re.compile(r'\b[A-Z]{2,3}\s?\d{3,4}\b')
    MONEY_PATTERN = re.compile(r'\$[\d,]+(?:\.\d{2})?')
    PHONE_PATTERN = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
    DATE_PATTERN = re.compile(
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*'
        r'\s+\d{1,2},?\s+\d{4}\b',
        re.IGNORECASE,
    )
    CASE_NUMBER = re.compile(r'\b\d{2}-[A-Z]{2,3}-\d{4,}\b')
    YEAR_PATTERN = re.compile(r'\b(19[5-9]\d|20[0-2]\d)\b')

    # Legal act / statute detection
    LEGAL_ACT_PATTERNS = [
        re.compile(r'\bTitle\s+\d+\b', re.I),
        re.compile(r'\bSection\s+\d+[a-z]?\b', re.I),
        re.compile(r'§\s*\d+'),
        re.compile(r'\b\d+\s+U\.?S\.?C\.?\s*§?\s*\d+', re.I),
        re.compile(r'\b(?:Non-Prosecution|Plea)\s+Agreement\b', re.I),
    ]

    # Event detection
    EVENT_PATTERNS = [
        re.compile(
            r'\b(?:deposition|hearing|trial|arraignment|sentencing|arrest|'
            r'indictment|plea\s+(?:deal|agreement))\b', re.I
        ),
        re.compile(r'\b(?:grand\s+jury|search\s+warrant|subpoena)\b', re.I),
        re.compile(r'\b(?:meeting|telephone\s+call|flight|visit|trip)\b', re.I),
    ]

    # Relationship hint patterns
    RELATIONSHIP_PATTERNS = [
        (re.compile(r'(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:met|met with)\s+(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', re.I), "met_with"),
        (re.compile(r'(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:employed|hired|worked for)\s+(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', re.I), "employed_by"),
        (re.compile(r'(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:traveled|flew|visited)\s+(?:to\s+)?(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', re.I), "traveled_to"),
    ]

    # Coreference / alias map for known entities
    ALIAS_MAP = {
        "j. epstein": "Jeffrey Epstein",
        "j epstein": "Jeffrey Epstein",
        "jeffrey e. epstein": "Jeffrey Epstein",
        "epstein": "Jeffrey Epstein",
        "g. maxwell": "Ghislaine Maxwell",
        "g maxwell": "Ghislaine Maxwell",
        "ghislaine": "Ghislaine Maxwell",
        "maxwell": "Ghislaine Maxwell",
        "prince andrew": "Prince Andrew",
        "duke of york": "Prince Andrew",
        "alan dershowitz": "Alan Dershowitz",
        "dershowitz": "Alan Dershowitz",
        "les wexner": "Leslie Wexner",
        "wexner": "Leslie Wexner",
    }

    def __init__(self):
        self.nlp = None
        if SPACY_AVAILABLE:
            model_name = settings.spacy_model
            try:
                self.nlp = spacy.load(model_name)
                self.nlp.max_length = 100000
                logger.info(f"Loaded spaCy model: {model_name}")
            except OSError:
                logger.warning(
                    f"spaCy model '{model_name}' not found. "
                    f"Run: python -m spacy download {model_name}"
                )
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.info("Fell back to en_core_web_sm")
                except OSError:
                    logger.error("No spaCy model available")

    def extract_from_text(self, text: str, doc_filename: str = "") -> list[dict]:
        """
        Extract entities from a text string.
        Returns list of entities: {text, label, start, end, source, normalized}
        """
        entities = []

        # spaCy NER
        if self.nlp:
            truncated = text[:80000] if len(text) > 80000 else text
            doc = self.nlp(truncated)

            for ent in doc.ents:
                if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "DATE", "MONEY",
                                  "NORP", "FAC", "EVENT", "LAW"}:
                    raw_text = ent.text.strip()
                    if len(raw_text) < 2 or raw_text.isdigit():
                        continue

                    normalized = self._normalize_entity(raw_text, ent.label_)
                    entities.append({
                        "text": raw_text,
                        "normalized": normalized,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "source": doc_filename,
                    })

        # Custom pattern extraction
        for match in self.FLIGHT_PATTERN.finditer(text):
            entities.append({
                "text": match.group(),
                "normalized": match.group().replace(" ", ""),
                "label": "FLIGHT",
                "start": match.start(),
                "end": match.end(),
                "source": doc_filename,
            })

        for match in self.MONEY_PATTERN.finditer(text):
            entities.append({
                "text": match.group(),
                "normalized": match.group(),
                "label": "MONEY",
                "start": match.start(),
                "end": match.end(),
                "source": doc_filename,
            })

        for match in self.CASE_NUMBER.finditer(text):
            entities.append({
                "text": match.group(),
                "normalized": match.group(),
                "label": "CASE_NUMBER",
                "start": match.start(),
                "end": match.end(),
                "source": doc_filename,
            })

        # Legal act detection
        for pat in self.LEGAL_ACT_PATTERNS:
            for match in pat.finditer(text):
                entities.append({
                    "text": match.group(),
                    "normalized": match.group().strip(),
                    "label": "LAW",
                    "start": match.start(),
                    "end": match.end(),
                    "source": doc_filename,
                })

        # Event extraction
        for pat in self.EVENT_PATTERNS:
            for match in pat.finditer(text):
                entities.append({
                    "text": match.group(),
                    "normalized": match.group().strip().lower(),
                    "label": "EVENT",
                    "start": match.start(),
                    "end": match.end(),
                    "source": doc_filename,
                })

        return entities

    def extract_structured(self, text: str, doc_filename: str = "", chunk_id: str = "") -> dict:
        """
        Extract structured entity record from a text chunk.
        Returns enriched metadata with entities, legal acts, events, dates,
        and relationship hints — ready for entity store ingestion.
        """
        entities = self.extract_from_text(text, doc_filename)

        # Group by type and deduplicate
        by_type: dict[str, list[str]] = {}
        for e in entities:
            by_type.setdefault(e["label"], []).append(e["normalized"])
        by_type = {k: list(set(v)) for k, v in by_type.items()}

        # Extract year references
        dates = list(set(self.YEAR_PATTERN.findall(text)))

        # Extract legal acts
        legal_acts = by_type.get("LAW", [])

        # Extract events
        events = by_type.get("EVENT", [])

        # Extract relationship hints
        relationship_hints = self._extract_relationships(text)

        return {
            "chunk_id": chunk_id,
            "doc_filename": doc_filename,
            "entities": by_type,
            "dates_in_chunk": sorted(dates),
            "legal_acts": legal_acts,
            "events": events,
            "relationship_hints": relationship_hints,
            "entity_count": sum(len(v) for v in by_type.values()),
        }

    def _extract_relationships(self, text: str) -> list[dict]:
        """Extract relationship hints from syntactic patterns."""
        hints = []
        for pattern, rel_type in self.RELATIONSHIP_PATTERNS:
            for match in pattern.finditer(text):
                groups = match.groups()
                if len(groups) >= 2:
                    hints.append({
                        "entity1": groups[0].strip(),
                        "entity2": groups[1].strip(),
                        "type": rel_type,
                        "evidence": text[max(0, match.start() - 20):match.end() + 20],
                    })
        return hints

    def extract_from_chunks(
        self, chunks: list[dict], progress_callback=None
    ) -> list[dict]:
        """Extract entities from all chunks."""
        all_entities = []
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "")
            filename = chunk.get("doc_filename", chunk.get("filename", ""))

            entities = self.extract_from_text(text, filename)
            for ent in entities:
                ent["chunk_id"] = chunk.get("id", "")
            all_entities.extend(entities)

            if progress_callback and (i + 1) % 500 == 0:
                progress_callback(i + 1, total)

        logger.info(
            f"Extracted {len(all_entities):,} entities from {total:,} chunks"
        )
        return all_entities

    def extract_structured_from_chunks(
        self, chunks: list[dict], progress_callback=None
    ) -> list[dict]:
        """Extract structured entity metadata from all chunks."""
        results = []
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "")
            filename = chunk.get("doc_filename", chunk.get("filename", ""))
            chunk_id = chunk.get("id", "")

            record = self.extract_structured(text, filename, chunk_id)
            results.append(record)

            if progress_callback and (i + 1) % 500 == 0:
                progress_callback(i + 1, total)

        logger.info(
            f"Extracted structured metadata from {total:,} chunks"
        )
        return results

    def _normalize_entity(self, text: str, label: str) -> str:
        """Normalize entity text with coreference resolution."""
        normalized = re.sub(r'\s+', ' ', text).strip()

        lower = normalized.lower()
        if lower in self.ALIAS_MAP:
            return self.ALIAS_MAP[lower]

        if label == "PERSON":
            normalized = normalized.title()
            for prefix in ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sir"]:
                if normalized.startswith(prefix + " "):
                    normalized = normalized[len(prefix) + 1:]

        return normalized

    @staticmethod
    def normalize_entities(entities: list[dict]) -> dict[str, dict]:
        """
        Normalize entity names and aggregate.
        Returns {normalized_name: {label, count, sources, chunk_ids}}.
        """
        name_map: dict[str, dict] = {}

        for ent in entities:
            name = ent.get("normalized", ent["text"].strip().title())
            if len(name) < 2:
                continue

            if name not in name_map:
                name_map[name] = {
                    "label": ent["label"],
                    "count": 0,
                    "sources": set(),
                    "chunk_ids": set(),
                }

            name_map[name]["count"] += 1
            name_map[name]["sources"].add(ent.get("source", ""))
            if ent.get("chunk_id"):
                name_map[name]["chunk_ids"].add(ent["chunk_id"])

        for v in name_map.values():
            v["sources"] = list(v["sources"])
            v["chunk_ids"] = list(v["chunk_ids"])

        return name_map

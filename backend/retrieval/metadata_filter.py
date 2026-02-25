"""
Metadata filter for pre/post-retrieval filtering.

Supports filtering by:
- doc_type: TEXT or IMAGE
- date_range: date strings found in documents
- entities: filter to chunks mentioning specific entities
- source_files: filter to specific source filenames
- min_length: minimum chunk text length
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class MetadataFilter:
    """
    Filter retrieved chunks by metadata attributes.
    Applies filters after retrieval but before reranking.
    """

    @staticmethod
    def apply(
        chunks: list[dict],
        doc_type: Optional[str] = None,
        entities: Optional[list[str]] = None,
        source_files: Optional[list[str]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        min_length: Optional[int] = None,
        exclude_keywords: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Apply metadata filters to retrieved chunks.

        Args:
            chunks: Retrieved chunk dicts.
            doc_type: Filter by document type ("TEXT" or "IMAGE").
            entities: Keep only chunks mentioning any of these entities.
            source_files: Keep only chunks from specific source files.
            date_from: Keep only chunks with dates >= this (YYYY format).
            date_to: Keep only chunks with dates <= this (YYYY format).
            min_length: Minimum text length.
            exclude_keywords: Exclude chunks containing these keywords.

        Returns:
            Filtered list of chunks.
        """
        if not chunks:
            return []

        original_count = len(chunks)
        filtered = chunks

        # Filter by document type
        if doc_type:
            doc_type_upper = doc_type.upper()
            filtered = [
                c for c in filtered
                if c.get("doc_type", "").upper() == doc_type_upper
            ]

        # Filter by entity mentions
        if entities:
            entities_lower = [e.lower() for e in entities]
            filtered = [
                c for c in filtered
                if any(
                    ent in c.get("text", "").lower()
                    for ent in entities_lower
                )
            ]

        # Filter by source files
        if source_files:
            source_set = {s.lower() for s in source_files}
            filtered = [
                c for c in filtered
                if c.get("doc_filename", c.get("filename", "")).lower() in source_set
            ]

        # Filter by date range
        if date_from or date_to:
            filtered = MetadataFilter._filter_by_date(
                filtered, date_from, date_to
            )

        # Filter by minimum length
        if min_length:
            filtered = [
                c for c in filtered
                if len(c.get("text", "")) >= min_length
            ]

        # Exclude keywords
        if exclude_keywords:
            kw_lower = [kw.lower() for kw in exclude_keywords]
            filtered = [
                c for c in filtered
                if not any(kw in c.get("text", "").lower() for kw in kw_lower)
            ]

        logger.debug(
            f"Metadata filter: {original_count} → {len(filtered)} chunks "
            f"(type={doc_type}, entities={entities}, dates={date_from}-{date_to})"
        )

        return filtered

    @staticmethod
    def _filter_by_date(
        chunks: list[dict],
        date_from: Optional[str],
        date_to: Optional[str],
    ) -> list[dict]:
        """
        Filter chunks by year mentions in the text.
        Extracts 4-digit years and checks against range.
        """
        year_pattern = re.compile(r'\b(19[5-9]\d|20[0-2]\d)\b')

        try:
            from_year = int(date_from) if date_from else 0
            to_year = int(date_to) if date_to else 9999
        except ValueError:
            return chunks

        filtered = []
        for chunk in chunks:
            text = chunk.get("text", "")
            years = [int(y) for y in year_pattern.findall(text)]

            if not years:
                # If no dates found, include the chunk (don't over-filter)
                filtered.append(chunk)
                continue

            # Check if any year in the chunk falls in range
            if any(from_year <= y <= to_year for y in years):
                filtered.append(chunk)

        return filtered

    @staticmethod
    def parse_filter_from_query(query: str) -> dict:
        """
        Extract filter hints from a natural language query.

        Examples:
            "documents about Epstein from 2005" → {"date_from": "2005", "date_to": "2005"}
            "text documents mentioning Maxwell" → {"doc_type": "TEXT", "entities": ["Maxwell"]}
        """
        filters = {}

        # Date extraction
        year_match = re.search(
            r'\b(?:from|in|during|year)\s*(\d{4})\b', query, re.IGNORECASE
        )
        if year_match:
            filters["date_from"] = year_match.group(1)
            filters["date_to"] = year_match.group(1)

        year_range = re.search(
            r'\b(\d{4})\s*(?:to|-)\s*(\d{4})\b', query
        )
        if year_range:
            filters["date_from"] = year_range.group(1)
            filters["date_to"] = year_range.group(2)

        # Doc type extraction
        if re.search(r'\btext\s+doc', query, re.IGNORECASE):
            filters["doc_type"] = "TEXT"
        elif re.search(r'\b(?:image|ocr|photo|scan)', query, re.IGNORECASE):
            filters["doc_type"] = "IMAGE"

        return filters

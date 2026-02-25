"""
Citation engine: extracts, validates, and maps source references
from LLM-generated answers back to the original chunks.
"""

import re
import logging

logger = logging.getLogger(__name__)


class CitationEngine:
    """Parse LLM output for [Source N] citations and map to source chunks."""

    @staticmethod
    def extract_citations(answer: str, source_chunks: list[dict]) -> dict:
        """
        Process a generated answer and its source chunks.

        Returns:
            {
                "answer": str (the answer text),
                "citations": [{index, filename, text, score}],
                "confidence": "high" | "medium" | "low" | "none",
                "num_sources": int,
                "is_grounded": bool,
                "warning": str | None,
            }
        """
        # Extract [Source N] references
        cited_indices = set()
        for match in re.finditer(r'\[Source\s*(\d+)\]', answer, re.IGNORECASE):
            idx = int(match.group(1)) - 1  # Convert to 0-indexed
            if 0 <= idx < len(source_chunks):
                cited_indices.add(idx)

        # Build citation list
        citations = []
        for idx in sorted(cited_indices):
            chunk = source_chunks[idx]
            citations.append({
                "index": idx + 1,
                "filename": chunk.get("doc_filename", chunk.get("filename", "Unknown")),
                "text": chunk.get("text", "")[:500],
                "score": chunk.get("rerank_score", chunk.get("rrf_score", chunk.get("score"))),
                "doc_type": chunk.get("doc_type", "UNKNOWN"),
            })

        # Determine grounding confidence
        confidence, is_grounded, warning = CitationEngine._assess_grounding(
            answer, citations, source_chunks
        )

        return {
            "answer": answer,
            "citations": citations,
            "confidence": confidence,
            "num_sources": len(citations),
            "is_grounded": is_grounded,
            "warning": warning,
        }

    @staticmethod
    def _assess_grounding(
        answer: str, citations: list[dict], source_chunks: list[dict]
    ) -> tuple[str, bool, str | None]:
        """
        Assess how well the answer is grounded in sources.
        Returns (confidence, is_grounded, warning).
        """
        answer_lower = answer.lower()
        warning = None

        # "I don't know" is a faithful response
        not_found_phrases = [
            "could not find", "not found in", "no information",
            "does not appear", "not available", "i don't know",
            "cannot determine", "no relevant information",
            "not in the provided", "not mentioned in",
        ]
        for phrase in not_found_phrases:
            if phrase in answer_lower:
                return "high", True, None

        # Check citation presence
        has_citations = bool(citations)

        if not has_citations:
            return "low", False, (
                "⚠️ No source citations found. This answer may not be grounded in the documents."
            )

        # Calculate word overlap between answer and cited sources
        answer_words = set(w.lower() for w in re.findall(r'\b\w{3,}\b', answer))
        source_words = set()
        for c in citations:
            source_words.update(
                w.lower() for w in re.findall(r'\b\w{3,}\b', c.get("text", ""))
            )

        if answer_words:
            overlap = len(answer_words & source_words) / len(answer_words)
        else:
            overlap = 0

        if overlap > 0.4 and len(citations) >= 2:
            return "high", True, None
        elif overlap > 0.25 or has_citations:
            return "medium", True, None
        else:
            warning = "⚠️ Low overlap between answer and cited sources. Verify manually."
            return "low", False, warning

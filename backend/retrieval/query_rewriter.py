"""
LLM-powered query rewriter for improved retrieval.

Strategies:
1. Query Expansion — rephrase for clarity + add relevant terms
2. HyDE (Hypothetical Document Embeddings) — generate a hypothetical answer, embed that
3. Query Decomposition — break multi-part queries into sub-queries
"""

import logging
import re
from typing import Optional

from backend.config import settings

logger = logging.getLogger(__name__)

# ── Prompt templates ────────────────────────────────────────────

EXPANSION_PROMPT = """Rewrite the following search query to improve document retrieval from a legal/investigative document corpus (the Epstein Files). 
Make it more explicit, add relevant related terms, and resolve any ambiguity.

Original query: {query}

Rules:
- Output ONLY the rewritten query, nothing else.
- Keep the original intent.
- Add synonyms and related legal/investigative terms.
- Expand abbreviations.
- Do NOT add information not implied by the original query.

Rewritten query:"""

HYDE_PROMPT = """Given this question about the Epstein Files legal document corpus, write a short paragraph (~100 words) that would be found in a document that answers this question. 
Write as if you are quoting from an actual legal document or deposition transcript.

Question: {query}

Hypothetical document excerpt:"""

DECOMPOSITION_PROMPT = """Break this complex question into 2-3 simpler, atomic sub-questions. Each sub-question should be independently searchable.

Question: {query}

Rules:
- Output only the sub-questions, one per line.
- Each must be a complete, self-contained question.
- Maximum 3 sub-questions.

Sub-questions:"""


class QueryRewriter:
    """
    Multi-strategy query rewriting for improved retrieval.
    Uses the existing LLM client for rewrites.
    """

    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.enabled = settings.enable_query_rewrite

    def rewrite(self, query: str) -> dict:
        """
        Apply all rewriting strategies to a query.

        Returns:
            {
                "original": str,
                "expanded": str,         # Expanded/clarified query
                "hyde_passage": str,      # Hypothetical document for embedding
                "sub_queries": list[str], # Decomposed sub-queries
                "all_queries": list[str], # All unique queries for multi-retrieval
            }
        """
        result = {
            "original": query,
            "expanded": query,
            "hyde_passage": "",
            "sub_queries": [],
            "all_queries": [query],
        }

        if not self.enabled or not self.llm or not self.llm.available:
            return result

        # 1. Query expansion
        try:
            expanded = self.llm.generate(
                system_prompt="You are a search query optimizer.",
                user_prompt=EXPANSION_PROMPT.format(query=query),
                max_tokens=200,
                temperature=0.0,
            )
            expanded = expanded.strip().strip('"').strip("'")
            if expanded and len(expanded) > 5:
                result["expanded"] = expanded
                result["all_queries"].append(expanded)
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")

        # 2. HyDE — hypothetical document generation
        try:
            hyde = self.llm.generate(
                system_prompt="You are a legal document generator.",
                user_prompt=HYDE_PROMPT.format(query=query),
                max_tokens=300,
                temperature=0.3,
            )
            hyde = hyde.strip()
            if hyde and len(hyde) > 20:
                result["hyde_passage"] = hyde
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}")

        # 3. Query decomposition (only for complex queries)
        if self._is_complex(query):
            try:
                decomposed = self.llm.generate(
                    system_prompt="You are a question decomposer.",
                    user_prompt=DECOMPOSITION_PROMPT.format(query=query),
                    max_tokens=300,
                    temperature=0.0,
                )
                sub_qs = [
                    q.strip().lstrip("0123456789.-) ").strip()
                    for q in decomposed.strip().split("\n")
                    if q.strip() and len(q.strip()) > 10
                ]
                if sub_qs:
                    result["sub_queries"] = sub_qs[:3]
                    result["all_queries"].extend(sub_qs[:3])
            except Exception as e:
                logger.warning(f"Query decomposition failed: {e}")

        # Deduplicate
        seen = set()
        unique = []
        for q in result["all_queries"]:
            q_lower = q.lower().strip()
            if q_lower not in seen:
                seen.add(q_lower)
                unique.append(q)
        result["all_queries"] = unique

        logger.info(
            f"Query rewritten: {len(result['all_queries'])} variants, "
            f"HyDE={'yes' if result['hyde_passage'] else 'no'}, "
            f"subs={len(result['sub_queries'])}"
        )

        return result

    @staticmethod
    def _is_complex(query: str) -> bool:
        """Heuristic: is this query complex enough to decompose?"""
        # Complex if: long, has conjunctions, multiple question marks
        if len(query.split()) > 12:
            return True
        if " and " in query.lower() or " or " in query.lower():
            return True
        if query.count("?") > 1:
            return True
        return False

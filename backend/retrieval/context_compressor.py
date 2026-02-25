"""
Context compressor for post-retrieval optimization.

Reduces retrieved context before LLM generation by:
1. Extractive compression — keep only query-relevant sentences
2. Redundancy elimination — remove overlapping content across chunks
3. Max-marginal relevance — diversify the final context set
4. Token budget — fit within LLM context window
"""

import logging
import re
from collections import OrderedDict

from backend.config import settings

logger = logging.getLogger(__name__)


class ContextCompressor:
    """
    Compress retrieved chunks to maximize information density
    within the LLM context window.
    """

    DEFAULT_TOKEN_BUDGET = 6000  # ~24K chars, safe for most LLMs
    CHARS_PER_TOKEN = 4

    def __init__(self):
        self.enabled = settings.enable_context_compression
        self.max_chars = self.DEFAULT_TOKEN_BUDGET * self.CHARS_PER_TOKEN

    def compress(
        self,
        query: str,
        chunks: list[dict],
        token_budget: int | None = None,
    ) -> list[dict]:
        """
        Compress chunks for LLM context.

        Pipeline:
        1. Score sentences by query relevance
        2. Remove near-duplicate sentences across chunks
        3. Apply MMR for diversity
        4. Trim to token budget

        Args:
            query: The user query.
            chunks: Retrieved and reranked chunks.
            token_budget: Max tokens for the compressed context.

        Returns:
            Compressed list of chunks with trimmed text.
        """
        if not self.enabled or not chunks:
            return chunks

        max_chars = (token_budget or self.DEFAULT_TOKEN_BUDGET) * self.CHARS_PER_TOKEN

        # Step 1: Extract and score sentences
        query_words = set(w.lower() for w in re.findall(r'\b\w{3,}\b', query))
        compressed_chunks = []

        for chunk in chunks:
            text = chunk.get("text", "")
            if not text.strip():
                continue

            # Split into sentences
            sentences = self._split_sentences(text)

            # Score each sentence by query relevance
            scored = []
            for sent in sentences:
                sent_words = set(w.lower() for w in re.findall(r'\b\w{3,}\b', sent))
                if not sent_words:
                    continue
                # Jaccard-like relevance score
                overlap = len(query_words & sent_words)
                score = overlap / max(len(query_words), 1)
                # Boost sentences with proper nouns (likely entity mentions)
                proper_noun_count = sum(1 for w in sent.split() if w[0:1].isupper())
                score += proper_noun_count * 0.05
                # Boost sentences with dates
                if re.search(r'\d{4}', sent):
                    score += 0.1
                scored.append((sent, score))

            # Keep top sentences by relevance (at least 3, up to all)
            scored.sort(key=lambda x: x[1], reverse=True)
            keep_count = max(3, int(len(scored) * 0.6))
            kept_sentences = scored[:keep_count]

            # Re-order by original position for coherence
            original_order = {sent: i for i, sent in enumerate(sentences)}
            kept_sentences.sort(
                key=lambda x: original_order.get(x[0], 999)
            )

            compressed_text = " ".join(s for s, _ in kept_sentences)
            compressed_chunks.append({
                **chunk,
                "text": compressed_text,
                "original_length": len(text),
                "compressed_length": len(compressed_text),
                "compression_ratio": round(len(compressed_text) / max(len(text), 1), 2),
            })

        # Step 2: Remove near-duplicate sentences across chunks
        compressed_chunks = self._deduplicate_across_chunks(compressed_chunks)

        # Step 3: Apply MMR diversity selection
        compressed_chunks = self._mmr_select(compressed_chunks, query_words)

        # Step 4: Trim to token budget
        compressed_chunks = self._trim_to_budget(compressed_chunks, max_chars)

        total_original = sum(c.get("original_length", 0) for c in compressed_chunks)
        total_compressed = sum(len(c.get("text", "")) for c in compressed_chunks)
        ratio = round(total_compressed / max(total_original, 1), 2)

        logger.debug(
            f"Context compressed: {len(chunks)} → {len(compressed_chunks)} chunks, "
            f"chars {total_original:,} → {total_compressed:,} ({ratio}x)"
        )

        return compressed_chunks

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences."""
        # Handle common abbreviations to avoid false splits
        text = re.sub(r'(Mr|Mrs|Ms|Dr|Prof|Jr|Sr|vs|etc)\.\s', r'\1DOTSPACE', text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Restore abbreviation dots
        sentences = [s.replace('DOTSPACE', '. ') for s in sentences]
        return [s.strip() for s in sentences if len(s.strip()) > 15]

    @staticmethod
    def _deduplicate_across_chunks(chunks: list[dict]) -> list[dict]:
        """Remove sentences that appear in multiple chunks."""
        seen_sentences: set[str] = set()

        for chunk in chunks:
            text = chunk.get("text", "")
            sentences = re.split(r'(?<=[.!?])\s+', text)
            unique_sentences = []

            for sent in sentences:
                # Normalize for comparison
                normalized = re.sub(r'\s+', ' ', sent.lower().strip())
                if len(normalized) < 20:
                    unique_sentences.append(sent)
                    continue
                if normalized not in seen_sentences:
                    seen_sentences.add(normalized)
                    unique_sentences.append(sent)

            chunk["text"] = " ".join(unique_sentences)

        return [c for c in chunks if len(c.get("text", "").strip()) > 20]

    @staticmethod
    def _mmr_select(
        chunks: list[dict], query_words: set[str], lambda_param: float = 0.7
    ) -> list[dict]:
        """
        Max-Marginal Relevance: balance relevance and diversity.
        """
        if len(chunks) <= 3:
            return chunks

        # Score each chunk
        scored = []
        for chunk in chunks:
            text = chunk.get("text", "")
            chunk_words = set(w.lower() for w in re.findall(r'\b\w{3,}\b', text))
            relevance = len(query_words & chunk_words) / max(len(query_words), 1)
            scored.append((chunk, relevance, chunk_words))

        # Greedy MMR selection
        selected = []
        remaining = list(range(len(scored)))

        while remaining and len(selected) < len(chunks):
            best_idx = None
            best_mmr = -1

            for idx in remaining:
                chunk, rel, words = scored[idx]

                # Max similarity to already-selected
                max_sim = 0
                for sel_idx in selected:
                    _, _, sel_words = scored[sel_idx]
                    intersection = len(words & sel_words)
                    union = len(words | sel_words) or 1
                    sim = intersection / union
                    max_sim = max(max_sim, sim)

                mmr_score = lambda_param * rel - (1 - lambda_param) * max_sim
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
            else:
                break

        return [scored[i][0] for i in selected]

    @staticmethod
    def _trim_to_budget(chunks: list[dict], max_chars: int) -> list[dict]:
        """Trim chunks to fit within character budget."""
        result = []
        total = 0

        for chunk in chunks:
            text = chunk.get("text", "")
            if total + len(text) > max_chars:
                # Trim this chunk to fit
                remaining = max_chars - total
                if remaining > 100:
                    chunk = {**chunk, "text": text[:remaining] + "..."}
                    result.append(chunk)
                break
            result.append(chunk)
            total += len(text)

        return result

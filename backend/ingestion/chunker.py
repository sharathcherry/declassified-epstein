"""
Adaptive document chunker with overlapping windows.
Adjusts chunk size based on document length and preserves paragraph boundaries.
"""

import hashlib
import logging
import re
import uuid
from dataclasses import dataclass, field

from backend.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single chunk of text with metadata."""
    id: str
    text: str
    doc_filename: str
    doc_type: str
    chunk_index: int
    total_chunks: int
    char_start: int
    char_end: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "doc_filename": self.doc_filename,
            "doc_type": self.doc_type,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "char_start": self.char_start,
            "char_end": self.char_end,
            **self.metadata,
        }


class AdaptiveChunker:
    """
    Adaptive chunking strategy for legal documents.
    - Short docs (< threshold): smaller chunks (256 tokens ≈ 1024 chars)
    - Long docs (>= threshold): larger chunks (512 tokens ≈ 2048 chars)
    - Always overlaps to preserve context across boundaries.
    """

    # Approx chars per token for English text
    CHARS_PER_TOKEN = 4

    def __init__(self):
        self.chunk_size_short = settings.chunk_size_short * self.CHARS_PER_TOKEN
        self.chunk_size_long = settings.chunk_size_long * self.CHARS_PER_TOKEN
        self.overlap = settings.chunk_overlap * self.CHARS_PER_TOKEN
        self.threshold = settings.long_doc_threshold

    def chunk_document(self, filename: str, doc: dict) -> list[Chunk]:
        """Chunk a single document into overlapping pieces."""
        text = doc.get("text", "")
        doc_type = doc.get("type", "UNKNOWN")

        if not text.strip():
            return []

        # Choose chunk size based on document length
        if len(text) < self.threshold:
            chunk_size = self.chunk_size_short
        else:
            chunk_size = self.chunk_size_long

        # Try paragraph-aware splitting first
        chunks = self._paragraph_aware_split(text, chunk_size)

        # Build Chunk objects
        result = []
        for i, (chunk_text, start, end) in enumerate(chunks):
            chunk = Chunk(
                id=str(uuid.uuid4().hex[:16]),
                text=chunk_text,
                doc_filename=filename,
                doc_type=doc_type,
                chunk_index=i,
                total_chunks=len(chunks),
                char_start=start,
                char_end=end,
            )
            result.append(chunk)

        return result

    def chunk_corpus(self, documents: dict[str, dict]) -> list[Chunk]:
        """Chunk all documents in the corpus. Returns flat list of chunks."""
        all_chunks = []
        total = len(documents)

        for i, (filename, doc) in enumerate(documents.items()):
            chunks = self.chunk_document(filename, doc)
            all_chunks.extend(chunks)

            if (i + 1) % 5000 == 0:
                logger.info(
                    f"Chunked {i + 1:,} / {total:,} docs "
                    f"({len(all_chunks):,} chunks so far)"
                )

        logger.info(
            f"Chunking complete: {len(all_chunks):,} chunks from "
            f"{total:,} documents (avg {len(all_chunks) / max(total, 1):.1f} chunks/doc)"
        )
        return all_chunks

    def _paragraph_aware_split(self, text: str, chunk_size: int) -> list[tuple[str, int, int]]:
        """
        Split text into chunks, preferring paragraph boundaries.
        Returns list of (chunk_text, char_start, char_end).
        """
        # If text fits in one chunk, return it whole
        if len(text) <= chunk_size:
            return [(text, 0, len(text))]

        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_text = ""
        current_start = 0
        running_pos = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                running_pos += 2  # account for the newlines
                continue

            # If adding this paragraph exceeds chunk size
            if current_text and len(current_text) + len(para) + 2 > chunk_size:
                # Save current chunk
                chunks.append((current_text.strip(), current_start, current_start + len(current_text)))

                # Overlap: keep last portion of current text
                if self.overlap > 0 and len(current_text) > self.overlap:
                    overlap_text = current_text[-self.overlap:]
                    current_start = current_start + len(current_text) - self.overlap
                    current_text = overlap_text + "\n\n" + para
                else:
                    current_start = running_pos
                    current_text = para
            else:
                if current_text:
                    current_text += "\n\n" + para
                else:
                    current_start = running_pos
                    current_text = para

            running_pos += len(para) + 2

        # Last chunk
        if current_text.strip():
            chunks.append((current_text.strip(), current_start, current_start + len(current_text)))

        # Handle case where a single paragraph is larger than chunk_size
        final_chunks = []
        for chunk_text, start, end in chunks:
            if len(chunk_text) > chunk_size * 1.5:
                # Force-split large paragraphs
                sub_chunks = self._force_split(chunk_text, chunk_size, start)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append((chunk_text, start, end))

        return final_chunks

    def _force_split(self, text: str, chunk_size: int, base_offset: int) -> list[tuple[str, int, int]]:
        """Force-split text that won't fit in a single chunk."""
        chunks = []
        pos = 0
        while pos < len(text):
            end = min(pos + chunk_size, len(text))

            # Try to break at sentence boundary
            if end < len(text):
                sentence_break = text.rfind('. ', pos + chunk_size // 2, end)
                if sentence_break > pos:
                    end = sentence_break + 1

            chunk_text = text[pos:end].strip()
            if chunk_text:
                chunks.append((chunk_text, base_offset + pos, base_offset + end))

            pos = max(pos + 1, end - self.overlap)

        return chunks

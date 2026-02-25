"""
OCR text cleaner for noisy legal document corpus.
Removes ONLY formatting artifacts — preserves ALL content.
"""

import re
import logging

logger = logging.getLogger(__name__)


class TextCleaner:
    """Clean and normalize OCR-derived text from legal documents."""

    # ONLY formatting-level cleanup — no content removal
    NOISE_PATTERNS = [
        (re.compile(r'\x0c'), ''),                          # Form feed chars
        (re.compile(r'[\x00-\x08\x0b\x0e-\x1f]'), ''),    # Control chars
        (re.compile(r'(\w)\1{4,}'), r'\1\1\1'),             # Repeated chars (aaaaaa → aaa)
        (re.compile(r'[_]{5,}'), ''),                       # Long underscores
        (re.compile(r'[-]{5,}'), '---'),                    # Long dashes
        (re.compile(r'[=]{5,}'), ''),                       # Long equals
        (re.compile(r'[*]{5,}'), ''),                       # Long asterisks
        (re.compile(r'\.{5,}'), '...'),                     # Long dots
    ]

    @classmethod
    def clean(cls, text: str) -> str:
        """Light cleaning — formatting only, no content removal."""
        if not text or len(text.strip()) < 10:
            return text

        # 1. Remove OCR noise patterns
        for pattern, replacement in cls.NOISE_PATTERNS:
            text = pattern.sub(replacement, text)

        # 2. Normalize whitespace
        text = cls._normalize_whitespace(text)

        # 3. Fix broken line wrapping (common in OCR)
        text = cls._fix_line_wrapping(text)

        # 4. Final cleanup
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
        text = text.strip()

        return text

    @classmethod
    def clean_batch(cls, documents: dict[str, dict]) -> dict[str, dict]:
        """Clean all documents in the corpus. Returns cleaned copy."""
        cleaned = {}
        total = len(documents)

        for i, (filename, doc) in enumerate(documents.items()):
            original_text = doc.get("text", "")
            cleaned_text = cls.clean(original_text)

            if cleaned_text:  # Skip empty after cleaning
                cleaned[filename] = {
                    **doc,
                    "text": cleaned_text,
                    "original_length": len(original_text),
                    "cleaned_length": len(cleaned_text),
                }

            if (i + 1) % 5000 == 0:
                logger.info(f"Cleaned {i + 1:,} / {total:,} documents")

        removed = total - len(cleaned)
        logger.info(
            f"Cleaning complete: {len(cleaned):,} docs kept, "
            f"{removed:,} removed (empty after cleaning)"
        )
        return cleaned

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Normalize all whitespace to single spaces / proper newlines."""
        text = text.replace('\t', '    ')
        lines = text.split('\n')
        lines = [re.sub(r'  +', ' ', line.strip()) for line in lines]
        return '\n'.join(lines)

    @staticmethod
    def _fix_line_wrapping(text: str) -> str:
        """Rejoin lines broken mid-word or mid-sentence by OCR."""
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
        text = re.sub(r'([a-z,;])\n([a-z])', r'\1 \2', text)
        return text

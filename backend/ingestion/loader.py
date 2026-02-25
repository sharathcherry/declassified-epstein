"""
HuggingFace dataset loader for the Epstein Files 20K corpus.
Downloads, parses, and caches the raw document corpus.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

from backend.config import DATA_DIR

logger = logging.getLogger(__name__)

HF_URL = (
    "https://huggingface.co/datasets/teyler/epstein-files-20k/resolve/main/"
    "EPS_FILES_20K_NOV2025.txt"
)
RAW_FILE = DATA_DIR / "raw_corpus.txt"
PARSED_FILE = DATA_DIR / "documents.json"


class DatasetLoader:
    """Download and parse the Epstein Files corpus from HuggingFace."""

    def __init__(self):
        self.documents: dict[str, dict] = {}  # {filename: {text, type, ...}}
        self.status = {
            "phase": "idle",
            "progress": 0,
            "total": 0,
            "message": "",
        }

    # ── Public API ──────────────────────────────────────────

    def load(self) -> dict[str, dict]:
        """Full load pipeline: download → parse → return docs."""
        if PARSED_FILE.exists():
            logger.info("Loading cached parsed documents...")
            self.status.update(phase="loading_cache", message="Loading cached data...")
            with open(PARSED_FILE, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
            logger.info(f"Loaded {len(self.documents):,} documents from cache")
            self.status.update(phase="ready", message=f"{len(self.documents):,} documents loaded")
            return self.documents

        self._download()
        self._parse()
        return self.documents

    def get_stats(self) -> dict:
        """Return corpus statistics."""
        if not self.documents:
            return {}

        text_docs = sum(1 for d in self.documents.values() if d.get("type") == "TEXT")
        image_docs = sum(1 for d in self.documents.values() if d.get("type") == "IMAGE")
        total_chars = sum(len(d.get("text", "")) for d in self.documents.values())
        avg_len = total_chars // max(len(self.documents), 1)

        return {
            "total_documents": len(self.documents),
            "text_documents": text_docs,
            "image_ocr_documents": image_docs,
            "total_characters": total_chars,
            "total_size_mb": f"{total_chars / 1024 / 1024:.1f}",
            "average_doc_length": avg_len,
        }

    def get_document(self, filename: str) -> Optional[dict]:
        """Retrieve a single document by filename."""
        doc = self.documents.get(filename)
        if doc:
            return {"filename": filename, **doc}
        return None

    def list_documents(self, page: int = 1, per_page: int = 20,
                       doc_type: Optional[str] = None) -> dict:
        """Paginated document listing with optional type filter."""
        items = list(self.documents.items())
        if doc_type:
            items = [(fn, d) for fn, d in items if d.get("type") == doc_type.upper()]

        total = len(items)
        start = (page - 1) * per_page
        end = start + per_page
        page_items = items[start:end]

        return {
            "documents": [
                {
                    "filename": fn,
                    "type": d.get("type", "UNKNOWN"),
                    "preview": d.get("text", "")[:200],
                    "length": len(d.get("text", "")),
                }
                for fn, d in page_items
            ],
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": max(1, (total + per_page - 1) // per_page),
        }

    def search(self, query: str, page: int = 1, per_page: int = 20) -> dict:
        """Full-text search across all documents."""
        query_lower = query.lower()
        results = []

        for fn, doc in self.documents.items():
            text = doc.get("text", "")
            text_lower = text.lower()
            if query_lower in text_lower:
                idx = text_lower.index(query_lower)
                start = max(0, idx - 100)
                end = min(len(text), idx + len(query) + 200)
                snippet = text[start:end]
                occurrences = text_lower.count(query_lower)
                results.append({
                    "filename": fn,
                    "type": doc.get("type", "UNKNOWN"),
                    "snippet": snippet,
                    "occurrences": occurrences,
                    "length": len(text),
                })

        results.sort(key=lambda r: r["occurrences"], reverse=True)
        total = len(results)
        start_idx = (page - 1) * per_page
        page_results = results[start_idx:start_idx + per_page]

        return {
            "results": page_results,
            "total": total,
            "page": page,
            "total_pages": max(1, (total + per_page - 1) // per_page),
        }

    # ── Internal ────────────────────────────────────────────

    def _download(self):
        """Download the raw corpus from HuggingFace."""
        if RAW_FILE.exists():
            logger.info("Raw file already exists, skipping download")
            return

        self.status.update(phase="downloading", progress=0, total=0, message="Starting download...")
        logger.info(f"Downloading from {HF_URL}")

        try:
            response = requests.get(HF_URL, stream=True, timeout=600)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            self.status["total"] = total_size

            with open(RAW_FILE, "wb") as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        self.status["progress"] = downloaded
                        self.status["message"] = (
                            f"Downloading... {downloaded / 1024 / 1024:.1f} MB"
                            f" / {total_size / 1024 / 1024:.1f} MB"
                        )

            logger.info(f"Download complete: {downloaded / 1024 / 1024:.1f} MB")
        except Exception as e:
            self.status.update(phase="error", message=f"Download failed: {e}")
            raise

    def _parse(self):
        """Parse the raw text file into structured documents."""
        self.status.update(phase="parsing", progress=0, total=0, message="Counting lines...")

        # Count lines first for progress
        with open(RAW_FILE, "r", encoding="utf-8", errors="replace") as f:
            total_lines = sum(1 for _ in f)

        self.status["total"] = total_lines
        logger.info(f"Parsing {total_lines:,} lines...")

        doc_pattern = re.compile(r'^(IMAGES?-\d+-|TEXT-\d+-)')
        current_filename = None
        current_text = []
        documents = {}

        with open(RAW_FILE, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i % 50000 == 0:
                    self.status["progress"] = i
                    self.status["message"] = (
                        f"Parsing line {i:,} / {total_lines:,} "
                        f"({len(documents):,} docs found)"
                    )

                line = line.rstrip("\n\r")

                # Check if this line starts a new document
                match = doc_pattern.match(line)
                if match:
                    # Save previous document
                    if current_filename and current_text:
                        text = "\n".join(current_text).strip()
                        if text:
                            doc_type = "IMAGE" if "IMAGE" in current_filename else "TEXT"
                            documents[current_filename] = {
                                "text": text,
                                "type": doc_type,
                            }

                    # Start new document
                    # Extract filename from the line
                    parts = line.split(",", 1)
                    current_filename = parts[0].strip().strip('"')
                    current_text = [parts[1].strip().strip('"')] if len(parts) > 1 else []
                elif current_filename:
                    current_text.append(line)

        # Don't forget the last document
        if current_filename and current_text:
            text = "\n".join(current_text).strip()
            if text:
                doc_type = "IMAGE" if "IMAGE" in current_filename else "TEXT"
                documents[current_filename] = {"text": text, "type": doc_type}

        self.documents = documents
        logger.info(f"Parsed {len(documents):,} documents")

        # Cache
        self.status.update(phase="caching", message="Saving parsed documents...")
        with open(PARSED_FILE, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False)

        self.status.update(
            phase="ready",
            progress=total_lines,
            message=f"{len(documents):,} documents parsed and cached",
        )

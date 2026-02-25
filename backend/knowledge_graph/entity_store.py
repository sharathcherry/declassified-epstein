"""
Entity Store: In-memory index for entity → chunk lookups.
Loads entity_index.json from Kaggle pipeline output.
Supports exact match, fuzzy search, type filtering, date range queries,
co-occurrence queries, and incremental updates.
"""

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Optional

from backend.config import DATA_DIR

logger = logging.getLogger(__name__)


class EntityStore:
    """
    Fast in-memory entity index.
    Maps entity names → {type, count, chunk_ids, source_docs, dates, co_occurring}.
    """

    def __init__(self):
        self.index: dict[str, dict] = {}
        self._name_lower_map: dict[str, str] = {}  # lowercase → canonical name
        self._type_index: dict[str, set[str]] = {}  # entity_type → set of names
        self._chunk_to_entities: dict[str, set[str]] = {}  # chunk_id → set of entity names
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready and len(self.index) > 0

    def load(self, path: Optional[str] = None) -> bool:
        """Load entity index from JSON file.
        Supports two formats:
          1. entity_index.json — dict keyed by entity name (preferred)
          2. entity_metadata.json — list of chunk records from NER notebook
        """
        # Try entity_index.json first, then entity_metadata.json
        paths_to_try = []
        if path is not None:
            paths_to_try.append(path)
        else:
            paths_to_try.append(str(DATA_DIR / "entity_index.json"))
            paths_to_try.append(str(DATA_DIR / "entity_metadata.json"))

        for try_path in paths_to_try:
            try:
                with open(try_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)

                if isinstance(raw, dict):
                    # Format 1: already keyed by entity name
                    self.index = raw
                elif isinstance(raw, list):
                    # Format 2: list of chunk records from NER notebook
                    # Convert: [{chunk_id, entities: {TYPE: [names]}, dates_in_chunk}, ...]
                    self.index = self._convert_metadata_list(raw)
                else:
                    logger.warning(f"Unknown entity data format in {try_path}")
                    continue

                self._build_secondary_indices()
                self._ready = True
                logger.info(f"EntityStore loaded: {len(self.index):,} entities from {try_path}")
                return True

            except FileNotFoundError:
                continue
            except Exception as e:
                logger.error(f"Failed to load entity index from {try_path}: {e}")
                continue

        logger.warning("No entity index file found")
        return False

    @staticmethod
    def _convert_metadata_list(records: list[dict]) -> dict[str, dict]:
        """Convert NER notebook output (list of chunk records) to entity name index.
        Memory-optimized: skips co-occurrence for rare entities.
        """
        entity_map: dict[str, dict] = {}

        for rec in records:
            chunk_id = rec.get("chunk_id", "")
            dates = rec.get("dates_in_chunk", [])
            entities = rec.get("entities", {})

            for ent_type, names in entities.items():
                if not isinstance(names, list):
                    continue
                for name in names:
                    name = name.strip()
                    if len(name) < 2:
                        continue

                    if name not in entity_map:
                        entity_map[name] = {
                            "type": ent_type,
                            "count": 0,
                            "chunk_ids": [],
                            "source_docs": [],
                            "dates_associated": [],
                            "co_occurring": [],
                        }

                    entity_map[name]["count"] += 1
                    if chunk_id and chunk_id not in entity_map[name]["chunk_ids"]:
                        entity_map[name]["chunk_ids"].append(chunk_id)
                    for d in dates:
                        if d and d not in entity_map[name]["dates_associated"]:
                            entity_map[name]["dates_associated"].append(d)

        # Build co-occurrence ONLY for entities with count >= 3 (memory optimization)
        from collections import Counter
        significant = {n for n, info in entity_map.items() if info["count"] >= 3}

        chunk_to_entities: dict[str, list[str]] = {}
        for name in significant:
            for cid in entity_map[name]["chunk_ids"]:
                chunk_to_entities.setdefault(cid, []).append(name)

        for name in significant:
            co_counter: Counter = Counter()
            for cid in entity_map[name]["chunk_ids"]:
                for other in chunk_to_entities.get(cid, []):
                    if other != name:
                        co_counter[other] += 1
            entity_map[name]["co_occurring"] = [
                {"name": n, "count": c} for n, c in co_counter.most_common(20)
            ]

        del chunk_to_entities
        logger.info(f"Converted {len(records):,} chunk records → {len(entity_map):,} entities ({len(significant):,} with co-occurrence)")
        return entity_map

    def save(self, path: Optional[str] = None) -> bool:
        """Save entity index to JSON file."""
        if path is None:
            path = str(DATA_DIR / "entity_index.json")

        try:
            # Convert sets to lists for JSON serialization
            serializable = {}
            for name, info in self.index.items():
                entry = {}
                for k, v in info.items():
                    if isinstance(v, set):
                        entry[k] = sorted(list(v))
                    elif isinstance(v, Counter):
                        entry[k] = [
                            {"name": n, "count": c}
                            for n, c in v.most_common(20)
                        ]
                    else:
                        entry[k] = v
                serializable[name] = entry

            with open(path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=1)

            logger.info(f"EntityStore saved: {len(self.index):,} entities to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save entity index: {e}")
            return False

    def _build_secondary_indices(self):
        """Build secondary lookup indices for fast queries."""
        self._name_lower_map = {name.lower(): name for name in self.index}

        self._type_index = {}
        self._chunk_to_entities = {}

        for name, info in self.index.items():
            entity_type = info.get("type", "UNKNOWN")
            self._type_index.setdefault(entity_type, set()).add(name)

            for cid in info.get("chunk_ids", []):
                self._chunk_to_entities.setdefault(cid, set()).add(name)

    def add_entity(self, name: str, entity_type: str, chunk_id: str = "",
                   source_doc: str = "", dates: list[str] = None,
                   co_occurring: list[str] = None):
        """Add or update an entity in the index."""
        if name not in self.index:
            self.index[name] = {
                "type": entity_type,
                "count": 0,
                "chunk_ids": [],
                "source_docs": [],
                "dates_associated": [],
                "co_occurring": [],
            }

        entry = self.index[name]
        entry["count"] += 1

        if chunk_id and chunk_id not in entry["chunk_ids"]:
            entry["chunk_ids"].append(chunk_id)
        if source_doc and source_doc not in entry["source_docs"]:
            entry["source_docs"].append(source_doc)
        if dates:
            for d in dates:
                if d not in entry["dates_associated"]:
                    entry["dates_associated"].append(d)

        # Update secondary indices
        self._name_lower_map[name.lower()] = name
        self._type_index.setdefault(entity_type, set()).add(name)
        if chunk_id:
            self._chunk_to_entities.setdefault(chunk_id, set()).add(name)

    def lookup(self, name: str) -> Optional[dict]:
        """Look up an entity by name. Tries exact → case-insensitive."""
        if not self._ready:
            return None

        if name in self.index:
            return {"name": name, **self.index[name]}

        canonical = self._name_lower_map.get(name.lower())
        if canonical:
            return {"name": canonical, **self.index[canonical]}

        return None

    def search(self, query: str, limit: int = 20) -> list[dict]:
        """Search for entities matching a query string (substring match)."""
        if not self._ready:
            return []

        query_lower = query.lower()
        matches = []

        for name, info in self.index.items():
            if query_lower in name.lower():
                matches.append({"name": name, **info})

        matches.sort(key=lambda x: x.get("count", 0), reverse=True)
        return matches[:limit]

    def search_fuzzy(self, query: str, limit: int = 10) -> list[dict]:
        """Fuzzy search: finds entities where query words appear in the name."""
        if not self._ready:
            return []

        query_words = set(query.lower().split())
        if not query_words:
            return []

        scored = []
        for name, info in self.index.items():
            name_lower = name.lower()
            name_words = set(name_lower.split())

            word_matches = len(query_words & name_words)
            substr_match = 1 if any(w in name_lower for w in query_words) else 0

            if word_matches > 0 or substr_match > 0:
                score = word_matches * 2 + substr_match + info.get("count", 0) / 10000
                scored.append({"name": name, "match_score": score, **info})

        scored.sort(key=lambda x: x["match_score"], reverse=True)
        return scored[:limit]

    def filter_by_type(self, entity_type: str, limit: int = 100) -> list[dict]:
        """Get all entities of a specific type (PERSON, ORG, LAW, etc.)."""
        if not self._ready:
            return []

        entity_type_upper = entity_type.upper()
        names = self._type_index.get(entity_type_upper, set())

        results = [
            {"name": name, **self.index[name]}
            for name in names
            if name in self.index
        ]
        results.sort(key=lambda x: x.get("count", 0), reverse=True)
        return results[:limit]

    def filter_by_date_range(self, date_from: str, date_to: str, limit: int = 100) -> list[dict]:
        """Get entities associated with dates in a range."""
        if not self._ready:
            return []

        results = []
        for name, info in self.index.items():
            dates = info.get("dates_associated", [])
            if any(date_from <= d <= date_to for d in dates):
                results.append({"name": name, **info})

        results.sort(key=lambda x: x.get("count", 0), reverse=True)
        return results[:limit]

    def get_chunk_ids(self, entity_name: str) -> list[str]:
        """Get all chunk IDs where an entity appears."""
        info = self.lookup(entity_name)
        if info:
            return info.get("chunk_ids", [])
        return []

    def get_chunk_ids_multi(self, entity_names: list[str]) -> set[str]:
        """Get chunk IDs for multiple entities (union)."""
        chunk_ids = set()
        for name in entity_names:
            chunk_ids.update(self.get_chunk_ids(name))
        return chunk_ids

    def get_entities_in_chunk(self, chunk_id: str) -> list[str]:
        """Get all entity names that appear in a specific chunk."""
        return list(self._chunk_to_entities.get(chunk_id, set()))

    def get_co_occurring(self, entity_name: str, limit: int = 20) -> list[dict]:
        """Get entities that co-occur with the given entity."""
        info = self.lookup(entity_name)
        if not info:
            return []

        co = info.get("co_occurring", [])
        if isinstance(co, list):
            return co[:limit]
        return []

    def get_dates_for_entity(self, entity_name: str) -> list[str]:
        """Get dates associated with an entity."""
        info = self.lookup(entity_name)
        if info:
            return info.get("dates_associated", [])
        return []

    def get_entity_overlap(self, entity_names: list[str], chunk_entities: list[str]) -> float:
        """
        Compute Jaccard-style overlap between query entities and chunk entities.
        Returns 0.0-1.0.
        """
        if not entity_names:
            return 0.0
        query_set = set(e.lower() for e in entity_names)
        chunk_set = set(e.lower() for e in chunk_entities)
        intersection = query_set & chunk_set
        if not query_set:
            return 0.0
        return len(intersection) / len(query_set)

    def extract_entities_from_text(self, text: str) -> list[dict]:
        """
        Fast entity detection in text using the index as a dictionary.
        No NLP model needed — just checks if known entity names appear in text.
        """
        if not self._ready:
            return []

        text_lower = text.lower()
        found = []

        for name, info in self.index.items():
            if info.get("count", 0) < 3:
                continue
            if name.lower() in text_lower:
                found.append({
                    "name": name,
                    "type": info.get("type", "UNKNOWN"),
                    "count": info.get("count", 0),
                })

        found.sort(key=lambda x: x["count"], reverse=True)
        return found

    def get_entity_profile(self, name: str) -> Optional[dict]:
        """
        Get full entity profile with all connections, timeline, and documents.
        Used by the intelligence API.
        """
        info = self.lookup(name)
        if not info:
            return None

        return {
            "name": info["name"],
            "type": info.get("type", "UNKNOWN"),
            "mentions": info.get("count", 0),
            "dates_associated": info.get("dates_associated", []),
            "co_occurring_entities": info.get("co_occurring", []),
            "source_documents": info.get("source_docs", info.get("source_documents", [])),
            "chunk_ids": info.get("chunk_ids", []),
            "num_chunks": len(info.get("chunk_ids", [])),
            "num_documents": len(info.get("source_docs", info.get("source_documents", []))),
        }

    def stats(self) -> dict:
        """Return summary statistics about the entity index."""
        if not self._ready:
            return {"ready": False}

        type_counts = Counter(info.get("type", "UNKNOWN") for info in self.index.values())
        total_chunks = sum(len(info.get("chunk_ids", [])) for info in self.index.values())

        return {
            "ready": True,
            "total_entities": len(self.index),
            "type_distribution": dict(type_counts),
            "total_chunk_refs": total_chunks,
            "top_entities": [
                {"name": name, "count": info["count"], "type": info["type"]}
                for name, info in sorted(
                    self.index.items(),
                    key=lambda x: x[1].get("count", 0),
                    reverse=True
                )[:10]
            ],
        }

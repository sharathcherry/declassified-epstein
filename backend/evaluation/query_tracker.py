"""
Live RAG query metrics tracker.
Records every chat query's latency, approximate quality metrics, cost,
entity extraction accuracy, graph traversal coverage, and failures.
Thread-safe singleton with ring buffer (last 1000 queries).
"""

import json
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from backend.config import DATA_DIR

logger = logging.getLogger(__name__)

METRICS_FILE = DATA_DIR / "metrics.json"

# NVIDIA NIM pricing (approximate, USD per 1K tokens)
COST_EMBED_PER_1K    = 0.003
COST_RERANK_PER_CALL = 0.002
COST_LLM_PER_1K      = 0.0005


@dataclass
class QueryRecord:
    """Single query record with all tracked metrics."""
    timestamp: float = 0.0
    query: str = ""
    # Latency breakdown (ms)
    latency_rewrite_ms: float = 0
    latency_embed_ms: float = 0
    latency_retrieval_ms: float = 0
    latency_rerank_ms: float = 0
    latency_compress_ms: float = 0
    latency_graph_ms: float = 0
    latency_generation_ms: float = 0
    latency_parse_ms: float = 0
    latency_scoring_ms: float = 0
    latency_evidence_ms: float = 0
    latency_total_ms: float = 0
    # Quality metrics
    ndcg_10: float = 0.0
    mrr: float = 0.0
    recall_10: float = 0.0
    num_sources: int = 0
    confidence: str = ""
    is_grounded: bool = False
    # Entity metrics
    entities_extracted: int = 0
    entities_resolved: int = 0
    evidence_chains: int = 0
    graph_paths_found: int = 0
    # Cost
    estimated_cost_usd: float = 0.0
    # Failure
    is_failure: bool = False
    error_message: str = ""
    failure_mode: str = ""  # no_entities, no_paths, low_confidence, error


class QueryTracker:
    """
    Thread-safe live metrics tracker for RAG queries.
    Stores recent queries in a ring buffer and persists aggregates to disk.
    """

    MAX_HISTORY = 1000

    def __init__(self):
        self._lock = threading.Lock()
        self._history: deque[QueryRecord] = deque(maxlen=self.MAX_HISTORY)
        self._total_queries = 0
        self._total_failures = 0
        self._total_cost = 0.0
        self._recent_failures: deque[dict] = deque(maxlen=50)
        self._failure_modes: dict[str, int] = {}

    # ── Recording ─────────────────────────────────────────────────────────────

    def record(
        self,
        query: str,
        timings: dict,
        results: list[dict],
        citation_result: dict,
    ):
        """Record a successful query with all metrics."""
        rec = QueryRecord(
            timestamp=time.time(),
            query=query[:200],
            # Latency
            latency_rewrite_ms=timings.get("rewrite_ms", 0),
            latency_embed_ms=timings.get("embed_ms", 0),
            latency_retrieval_ms=timings.get("retrieval_ms", 0),
            latency_rerank_ms=timings.get("rerank_ms", 0),
            latency_compress_ms=timings.get("compress_ms", 0),
            latency_graph_ms=timings.get("graph_ms", 0),
            latency_generation_ms=timings.get("generation_ms", 0),
            latency_parse_ms=timings.get("parse_ms", 0),
            latency_scoring_ms=timings.get("scoring_ms", 0),
            latency_evidence_ms=timings.get("evidence_ms", 0),
            latency_total_ms=timings.get("total_ms", 0),
            # Quality
            ndcg_10=self._compute_ndcg(results),
            mrr=self._compute_mrr(results),
            recall_10=self._compute_recall(results),
            num_sources=citation_result.get("num_sources", 0),
            confidence=citation_result.get("confidence", ""),
            is_grounded=citation_result.get("is_grounded", False),
            # Entity metrics
            entities_extracted=self._count_entities(results),
            # Cost
            estimated_cost_usd=self._estimate_cost(query, results, timings),
        )

        # Detect failure modes
        failure_mode = self._detect_failure_mode(results, rec)
        rec.failure_mode = failure_mode

        with self._lock:
            self._history.append(rec)
            self._total_queries += 1
            self._total_cost += rec.estimated_cost_usd
            if failure_mode:
                self._failure_modes[failure_mode] = self._failure_modes.get(failure_mode, 0) + 1

        # Persist every 10 queries
        if self._total_queries % 10 == 0:
            self._persist()

    def record_failure(self, query: str, error: str):
        """Record a failed query."""
        rec = QueryRecord(
            timestamp=time.time(),
            query=query[:200],
            is_failure=True,
            error_message=error[:500],
            failure_mode="error",
        )
        with self._lock:
            self._history.append(rec)
            self._total_queries += 1
            self._total_failures += 1
            self._failure_modes["error"] = self._failure_modes.get("error", 0) + 1
            self._recent_failures.append({
                "query": query[:200],
                "error": error[:500],
                "timestamp": time.time(),
            })

    # ── Quality Metrics ───────────────────────────────────────────────────────

    @staticmethod
    def _compute_ndcg(results: list[dict], k: int = 10) -> float:
        """Approximate NDCG@k using reranker scores as graded relevance."""
        top_k = results[:k]
        if not top_k:
            return 0.0

        scores = []
        for r in top_k:
            score = r.get("rerank_score", r.get("composite_score", r.get("score", 0)))
            scores.append(max(0, float(score)))

        if not scores or max(scores) == 0:
            return 0.0

        max_score = max(scores)
        min_score = min(scores)
        rng = max_score - min_score if max_score != min_score else 1.0
        normalized = [(s - min_score) / rng for s in scores]

        dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(normalized))
        ideal = sorted(normalized, reverse=True)
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def _compute_mrr(results: list[dict]) -> float:
        """MRR based on reranker score threshold."""
        for i, r in enumerate(results):
            score = r.get("rerank_score", r.get("composite_score", r.get("score", 0)))
            if float(score) > 0.5:
                return 1.0 / (i + 1)
        return 1.0 if results else 0.0

    @staticmethod
    def _compute_recall(results: list[dict], k: int = 10) -> float:
        """Approximate Recall@k using reranker score presence."""
        top_k = results[:k]
        if not top_k:
            return 0.0
        reranked = sum(1 for r in top_k if "rerank_score" in r or "composite_score" in r)
        return reranked / len(top_k)

    @staticmethod
    def _count_entities(results: list[dict]) -> int:
        """Count unique entities across results."""
        entities = set()
        for r in results:
            for ent in r.get("matched_entities", []):
                entities.add(ent)
        return len(entities)

    @staticmethod
    def _detect_failure_mode(results: list[dict], rec: QueryRecord) -> str:
        """Detect failure modes for observability."""
        if not results:
            return "no_results"
        if rec.ndcg_10 < 0.3:
            return "low_quality"
        return ""

    @staticmethod
    def _estimate_cost(query: str, results: list[dict], timings: dict) -> float:
        """Estimate USD cost of a single query."""
        cost = 0.0
        cost += (50 / 1000) * COST_EMBED_PER_1K
        if timings.get("rerank_ms", 0) > 0:
            cost += COST_RERANK_PER_CALL
        if timings.get("generation_ms", 0) > 0:
            cost += (2500 / 1000) * COST_LLM_PER_1K
        return round(cost, 6)

    # ── Aggregation ───────────────────────────────────────────────────────────

    def get_metrics(self) -> dict:
        """Return aggregated metrics snapshot."""
        with self._lock:
            history = list(self._history)
            total = self._total_queries
            failures = self._total_failures
            total_cost = self._total_cost
            recent_failures = list(self._recent_failures)
            failure_modes = dict(self._failure_modes)

        if not history:
            return {
                "total_queries": 0,
                "avg_latency_ms": 0,
                "p95_latency_ms": 0,
                "avg_ndcg_10": 0,
                "avg_mrr": 0,
                "avg_recall_10": 0,
                "total_cost_usd": 0,
                "avg_cost_per_query_usd": 0,
                "failure_rate": 0,
                "failure_modes": {},
                "failures": [],
                "latency_breakdown": {},
                "entity_metrics": {},
                "recent_queries": [],
            }

        successful = [h for h in history if not h.is_failure]
        latencies = [h.latency_total_ms for h in successful] if successful else [0]

        sorted_lat = sorted(latencies)
        p95_idx = int(len(sorted_lat) * 0.95)

        return {
            "total_queries": total,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
            "p95_latency_ms": round(sorted_lat[min(p95_idx, len(sorted_lat) - 1)], 1),
            "avg_ndcg_10": round(
                sum(h.ndcg_10 for h in successful) / len(successful), 3
            ) if successful else 0,
            "avg_mrr": round(
                sum(h.mrr for h in successful) / len(successful), 3
            ) if successful else 0,
            "avg_recall_10": round(
                sum(h.recall_10 for h in successful) / len(successful), 3
            ) if successful else 0,
            "total_cost_usd": round(total_cost, 4),
            "avg_cost_per_query_usd": round(total_cost / total, 6) if total else 0,
            "failure_rate": round(failures / total, 3) if total else 0,
            "failure_modes": failure_modes,
            "failures": recent_failures[-10:],
            "latency_breakdown": {
                "parse_ms": round(sum(h.latency_parse_ms for h in successful) / len(successful), 1) if successful else 0,
                "rewrite_ms": round(sum(h.latency_rewrite_ms for h in successful) / len(successful), 1) if successful else 0,
                "embed_ms": round(sum(h.latency_embed_ms for h in successful) / len(successful), 1) if successful else 0,
                "retrieval_ms": round(sum(h.latency_retrieval_ms for h in successful) / len(successful), 1) if successful else 0,
                "scoring_ms": round(sum(h.latency_scoring_ms for h in successful) / len(successful), 1) if successful else 0,
                "rerank_ms": round(sum(h.latency_rerank_ms for h in successful) / len(successful), 1) if successful else 0,
                "compress_ms": round(sum(h.latency_compress_ms for h in successful) / len(successful), 1) if successful else 0,
                "graph_ms": round(sum(h.latency_graph_ms for h in successful) / len(successful), 1) if successful else 0,
                "evidence_ms": round(sum(h.latency_evidence_ms for h in successful) / len(successful), 1) if successful else 0,
                "generation_ms": round(sum(h.latency_generation_ms for h in successful) / len(successful), 1) if successful else 0,
            },
            "entity_metrics": {
                "avg_entities_extracted": round(
                    sum(h.entities_extracted for h in successful) / len(successful), 1
                ) if successful else 0,
                "avg_evidence_chains": round(
                    sum(h.evidence_chains for h in successful) / len(successful), 1
                ) if successful else 0,
                "avg_graph_paths": round(
                    sum(h.graph_paths_found for h in successful) / len(successful), 1
                ) if successful else 0,
            },
            "recent_queries": [
                {
                    "query": h.query,
                    "latency_ms": round(h.latency_total_ms, 1),
                    "ndcg_10": round(h.ndcg_10, 3),
                    "mrr": round(h.mrr, 3),
                    "recall_10": round(h.recall_10, 3),
                    "confidence": h.confidence,
                    "cost_usd": round(h.estimated_cost_usd, 6),
                    "entities_extracted": h.entities_extracted,
                    "failure_mode": h.failure_mode,
                    "is_failure": h.is_failure,
                    "timestamp": h.timestamp,
                }
                for h in list(history)[-20:]
            ],
        }

    def _persist(self):
        """Save aggregated metrics to disk."""
        try:
            metrics = self.get_metrics()
            with open(METRICS_FILE, "w") as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist metrics: {e}")


# Singleton instance
tracker = QueryTracker()

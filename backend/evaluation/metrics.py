"""
Evaluation metrics: Precision@k, Recall@k, MRR, nDCG, latency tracking,
entity recall, and path coverage for entity-centric retrieval.
"""

import logging
import math
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Stores result of a single benchmark query."""
    query_id: str
    query: str
    category: str
    expected_keywords: list[str]
    retrieved_texts: list[str] = field(default_factory=list)
    answer: str = ""
    num_sources: int = 0
    confidence: str = ""
    latency_retrieval_ms: float = 0
    latency_rerank_ms: float = 0
    latency_generation_ms: float = 0
    latency_total_ms: float = 0
    precision_at_5: float = 0
    precision_at_10: float = 0
    recall: float = 0
    mrr: float = 0
    keyword_hit: bool = False


class EvaluationMetrics:
    """Compute retrieval and generation quality metrics."""

    @staticmethod
    def precision_at_k(relevant: list[bool], k: int) -> float:
        """Fraction of top-k results that are relevant."""
        top_k = relevant[:k]
        if not top_k:
            return 0.0
        return sum(top_k) / len(top_k)

    @staticmethod
    def recall_at_k(relevant: list[bool], total_relevant: int, k: int) -> float:
        """Fraction of all relevant docs found in top-k."""
        if total_relevant == 0:
            return 0.0
        top_k = relevant[:k]
        return sum(top_k) / total_relevant

    @staticmethod
    def mrr(relevant: list[bool]) -> float:
        """Mean Reciprocal Rank — rank of first relevant result."""
        for i, is_rel in enumerate(relevant):
            if is_rel:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def ndcg_at_k(relevant: list[bool], k: int) -> float:
        """Normalized Discounted Cumulative Gain."""
        top_k = relevant[:k]
        dcg = sum(
            (1 if rel else 0) / math.log2(i + 2)
            for i, rel in enumerate(top_k)
        )

        ideal = sorted(top_k, reverse=True)
        idcg = sum(
            (1 if rel else 0) / math.log2(i + 2)
            for i, rel in enumerate(ideal)
        )

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def check_keyword_hit(answer: str, expected_keywords: list[str]) -> bool:
        """Check if the answer contains expected keywords."""
        answer_lower = answer.lower()
        for kw in expected_keywords:
            if kw.lower() in answer_lower:
                return True
        return False

    @staticmethod
    def evaluate_retrieval(
        retrieved_texts: list[str], expected_keywords: list[str]
    ) -> list[bool]:
        """Determine relevance of each retrieved chunk based on keyword overlap."""
        relevant = []
        for text in retrieved_texts:
            text_lower = text.lower()
            is_relevant = any(kw.lower() in text_lower for kw in expected_keywords)
            relevant.append(is_relevant)
        return relevant

    @classmethod
    def compute_query_metrics(cls, result: QueryResult) -> QueryResult:
        """Compute all metrics for a single query result."""
        relevant = cls.evaluate_retrieval(result.retrieved_texts, result.expected_keywords)
        total_relevant = sum(relevant)

        result.precision_at_5 = cls.precision_at_k(relevant, 5)
        result.precision_at_10 = cls.precision_at_k(relevant, 10)
        result.recall = cls.recall_at_k(relevant, max(total_relevant, 1), 10)
        result.mrr = cls.mrr(relevant)
        result.keyword_hit = cls.check_keyword_hit(result.answer, result.expected_keywords)

        return result

    @staticmethod
    def aggregate_results(results: list[QueryResult]) -> dict:
        """Aggregate metrics across all benchmark queries."""
        if not results:
            return {}

        by_category: dict[str, list[QueryResult]] = {}
        for r in results:
            by_category.setdefault(r.category, []).append(r)

        summary = {
            "total_queries": len(results),
            "avg_precision_at_5": sum(r.precision_at_5 for r in results) / len(results),
            "avg_precision_at_10": sum(r.precision_at_10 for r in results) / len(results),
            "avg_mrr": sum(r.mrr for r in results) / len(results),
            "avg_recall": sum(r.recall for r in results) / len(results),
            "keyword_hit_rate": sum(r.keyword_hit for r in results) / len(results),
            "avg_latency_total_ms": sum(r.latency_total_ms for r in results) / len(results),
            "avg_latency_retrieval_ms": sum(r.latency_retrieval_ms for r in results) / len(results),
            "avg_latency_rerank_ms": sum(r.latency_rerank_ms for r in results) / len(results),
            "avg_latency_generation_ms": sum(r.latency_generation_ms for r in results) / len(results),
            "by_category": {},
        }

        for cat, cat_results in by_category.items():
            summary["by_category"][cat] = {
                "count": len(cat_results),
                "avg_precision_at_5": sum(r.precision_at_5 for r in cat_results) / len(cat_results),
                "avg_mrr": sum(r.mrr for r in cat_results) / len(cat_results),
                "keyword_hit_rate": sum(r.keyword_hit for r in cat_results) / len(cat_results),
            }

        return summary


class RetrievalMetrics:
    """Entity-centric retrieval quality metrics."""

    @staticmethod
    def recall_at_k(relevant_ids: set, retrieved_ids: list, k: int) -> float:
        """What fraction of relevant docs were retrieved in top-k."""
        if not relevant_ids:
            return 0.0
        retrieved_k = set(retrieved_ids[:k])
        return len(relevant_ids & retrieved_k) / len(relevant_ids)

    @staticmethod
    def precision_at_k(relevant_ids: set, retrieved_ids: list, k: int) -> float:
        """What fraction of top-k retrieved docs are relevant."""
        retrieved_k = retrieved_ids[:k]
        if not retrieved_k:
            return 0.0
        return len(relevant_ids & set(retrieved_k)) / len(retrieved_k)

    @staticmethod
    def mrr(relevant_ids: set, retrieved_ids: list) -> float:
        """Rank of first relevant result."""
        for i, rid in enumerate(retrieved_ids):
            if rid in relevant_ids:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def ndcg_at_k(relevance_scores: list[float], k: int) -> float:
        """nDCG with graded relevance scores."""
        top_k = relevance_scores[:k]
        if not top_k:
            return 0.0

        dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(top_k))
        ideal = sorted(top_k, reverse=True)
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def entity_recall(expected_entities: set[str], found_entities: set[str]) -> float:
        """What fraction of expected entities were found."""
        if not expected_entities:
            return 0.0
        expected_lower = set(e.lower() for e in expected_entities)
        found_lower = set(e.lower() for e in found_entities)
        return len(expected_lower & found_lower) / len(expected_lower)

    @staticmethod
    def path_coverage(expected_paths: list[tuple], found_paths: list[tuple]) -> float:
        """What fraction of expected graph paths were discovered."""
        if not expected_paths:
            return 0.0
        expected_set = set(expected_paths)
        found_set = set(found_paths)
        return len(expected_set & found_set) / len(expected_set)

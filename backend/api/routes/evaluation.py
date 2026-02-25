"""
Evaluation API routes.
"""

import time
from fastapi import APIRouter, HTTPException

from backend.evaluation.benchmarks import BENCHMARK_QUERIES
from backend.evaluation.metrics import EvaluationMetrics, QueryResult
from backend.generation.prompts import SYSTEM_PROMPT, build_qa_prompt
from backend.generation.citation_engine import CitationEngine

router = APIRouter(prefix="/api/eval", tags=["evaluation"])


@router.get("/benchmarks")
async def list_benchmarks():
    """List all benchmark queries."""
    return {"benchmarks": BENCHMARK_QUERIES, "total": len(BENCHMARK_QUERIES)}


@router.post("/run")
async def run_evaluation():
    """Run the full evaluation suite against benchmark queries."""
    from backend.main import app_state

    if not app_state.get("retriever_ready"):
        raise HTTPException(503, "System not ready for evaluation.")

    embedder = app_state["embedder"]
    hybrid = app_state["hybrid_retriever"]
    reranker = app_state.get("reranker")
    llm = app_state.get("llm")

    results = []

    for bq in BENCHMARK_QUERIES:
        qr = QueryResult(
            query_id=bq["id"],
            query=bq["query"],
            category=bq["category"],
            expected_keywords=bq["expected_keywords"],
        )

        try:
            # Retrieval
            t0 = time.time()
            query_embedding = embedder.embed_query(bq["query"])
            retrieved = hybrid.search(
                query=bq["query"],
                query_embedding=query_embedding,
                top_k=50,
            )
            qr.latency_retrieval_ms = (time.time() - t0) * 1000

            # Rerank
            t0 = time.time()
            if reranker and retrieved:
                retrieved = reranker.rerank(bq["query"], retrieved, top_k=10)
            else:
                retrieved = retrieved[:10]
            qr.latency_rerank_ms = (time.time() - t0) * 1000

            qr.retrieved_texts = [r.get("text", "") for r in retrieved]
            qr.num_sources = len(retrieved)

            # Generate
            t0 = time.time()
            if llm and llm.available:
                user_prompt = build_qa_prompt(bq["query"], retrieved)
                qr.answer = llm.generate(
                    system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt
                )
            else:
                qr.answer = "LLM unavailable"
            qr.latency_generation_ms = (time.time() - t0) * 1000

            qr.latency_total_ms = (
                qr.latency_retrieval_ms + qr.latency_rerank_ms + qr.latency_generation_ms
            )

            # Compute metrics
            qr = EvaluationMetrics.compute_query_metrics(qr)

            citation_result = CitationEngine.extract_citations(qr.answer, retrieved)
            qr.confidence = citation_result["confidence"]

        except Exception as e:
            qr.answer = f"Error: {str(e)}"

        results.append(qr)

    # Aggregate
    summary = EvaluationMetrics.aggregate_results(results)

    return {
        "summary": summary,
        "results": [
            {
                "query_id": r.query_id,
                "query": r.query,
                "category": r.category,
                "answer_preview": r.answer[:200],
                "precision_at_5": round(r.precision_at_5, 3),
                "precision_at_10": round(r.precision_at_10, 3),
                "mrr": round(r.mrr, 3),
                "keyword_hit": r.keyword_hit,
                "confidence": r.confidence,
                "latency_total_ms": round(r.latency_total_ms, 1),
            }
            for r in results
        ],
    }

"""
Search API routes: multi-stage hybrid search with entity-aware scoring.
"""

import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/api/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    doc_type: Optional[str] = None
    keyword_boost: Optional[str] = None
    entities: Optional[list[str]] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None


class SearchResponse(BaseModel):
    results: list[dict]
    query: str
    rewritten_query: Optional[str] = None
    total: int
    query_entities: list[dict] = []
    evidence_chains: list[dict] = []
    latency: dict


@router.post("", response_model=SearchResponse)
async def search(req: SearchRequest):
    """Multi-stage hybrid search: parse → embed → 4-way retrieve → filter → rerank → score."""
    from backend.main import app_state

    if not app_state.get("retriever_ready"):
        raise HTTPException(503, "System is still initializing. Please wait.")

    timings = {}
    rewritten_query = None
    query_entities = []

    # Step 1: Query parsing (entity extraction)
    t0 = time.time()
    query_parser = app_state.get("query_parser")
    entity_names = req.entities or []
    parsed = None

    if query_parser:
        parsed = query_parser.parse(req.query)
        if not entity_names:
            entity_names = [e.name for e in parsed.entities_mentioned]
        query_entities = [
            {"name": e.name, "type": e.type, "count": e.count}
            for e in parsed.entities_mentioned
        ]
    timings["parse_ms"] = round((time.time() - t0) * 1000, 1)

    # Step 2: Query rewriting
    t0 = time.time()
    query_rewriter = app_state.get("query_rewriter")
    hyde_embedding = None

    if query_rewriter:
        rewrite_result = query_rewriter.rewrite(req.query)
        search_query = rewrite_result.get("expanded", req.query)
        rewritten_query = search_query if search_query != req.query else None

        if rewrite_result.get("hyde_passage"):
            embedder = app_state["embedder"]
            hyde_embedding = embedder.embed_query(rewrite_result["hyde_passage"])
    else:
        search_query = req.query

    timings["rewrite_ms"] = round((time.time() - t0) * 1000, 1)

    # Step 3: Embed query
    t0 = time.time()
    embedder = app_state["embedder"]
    query_embedding = embedder.embed_query(search_query)
    timings["embed_ms"] = round((time.time() - t0) * 1000, 1)

    # Step 4: Hybrid retrieval (4-way RRF)
    t0 = time.time()
    hybrid = app_state["hybrid_retriever"]
    results = hybrid.search(
        query=search_query,
        query_embedding=query_embedding,
        top_k=req.top_k * 5,
        doc_type_filter=req.doc_type,
        keyword_boost=req.keyword_boost,
        hyde_embedding=hyde_embedding,
        entity_names=entity_names if entity_names else None,
    )
    timings["retrieval_ms"] = round((time.time() - t0) * 1000, 1)

    # Step 5: Metadata filtering
    t0 = time.time()
    from backend.retrieval.metadata_filter import MetadataFilter

    auto_filters = MetadataFilter.parse_filter_from_query(req.query)
    results = MetadataFilter.apply(
        results,
        doc_type=req.doc_type or auto_filters.get("doc_type"),
        entities=req.entities,
        date_from=req.date_from or auto_filters.get("date_from"),
        date_to=req.date_to or auto_filters.get("date_to"),
    )
    timings["filter_ms"] = round((time.time() - t0) * 1000, 1)

    # Step 6: Composite scoring
    t0 = time.time()
    scorer = app_state.get("composite_scorer")
    if scorer and entity_names:
        results = scorer.score_results(
            results,
            query_entities=entity_names,
            query_dates=parsed.dates_mentioned if parsed else [],
        )
    timings["scoring_ms"] = round((time.time() - t0) * 1000, 1)

    # Step 7: Rerank
    t0 = time.time()
    reranker = app_state.get("reranker")
    if reranker and results:
        results = reranker.rerank(req.query, results, top_k=req.top_k)
    else:
        results = results[:req.top_k]
    timings["rerank_ms"] = round((time.time() - t0) * 1000, 1)

    # Step 8: Evidence chains (if entities found)
    evidence_chains = []
    if entity_names:
        t0 = time.time()
        chain_gen = app_state.get("evidence_chain_gen")
        traversal = app_state.get("graph_traversal")
        if chain_gen and traversal:
            paths = traversal.multi_hop_search(entity_names, max_hops=2)
            chains = chain_gen.generate_chains(paths, max_chains=3)
            evidence_chains = [c.to_dict() for c in chains]
        timings["evidence_ms"] = round((time.time() - t0) * 1000, 1)

    timings["total_ms"] = round(sum(timings.values()), 1)

    return SearchResponse(
        results=results[:req.top_k],
        query=req.query,
        rewritten_query=rewritten_query,
        total=len(results),
        query_entities=query_entities,
        evidence_chains=evidence_chains,
        latency=timings,
    )

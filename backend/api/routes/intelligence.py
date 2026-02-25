"""
Intelligence API: Entity-centric intelligence endpoints.
Provides entity profiles, multi-hop paths, timelines, and structured search.
"""

import time
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/api/intelligence", tags=["intelligence"])


class IntelligenceSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    include_evidence: bool = True
    include_graph: bool = True
    use_llm: bool = False  # LLM is optional


class EntityRef(BaseModel):
    name: str
    type: str = "UNKNOWN"
    count: int = 0


class IntelligenceResponse(BaseModel):
    # Core results
    results: list[dict]

    # Entity context
    query_entities: list[dict]
    discovered_entities: list[dict] = []

    # Evidence chains
    evidence_chains: list[dict] = []

    # Metadata
    confidence: float = 0.0
    retrieval_mode: str = "hybrid"
    intent: str = "search"
    latency: dict

    # Optional LLM summary
    summary: Optional[str] = None


@router.get("/entity/{name}")
async def get_entity_profile(name: str):
    """Get full entity profile with all connections, timeline, and documents."""
    from backend.main import app_state

    entity_store = app_state.get("entity_store")
    if not entity_store or not entity_store.is_ready:
        raise HTTPException(503, "Entity store not loaded.")

    profile = entity_store.get_entity_profile(name)
    if not profile:
        # Try fuzzy search
        results = entity_store.search_fuzzy(name, limit=1)
        if results:
            profile = entity_store.get_entity_profile(results[0]["name"])

    if not profile:
        raise HTTPException(404, f"Entity '{name}' not found.")

    # Add graph connections if available
    graph = app_state.get("graph")
    if graph and graph.is_ready:
        entity_data = graph.get_entity(name)
        if entity_data:
            profile["connections"] = entity_data.get("connections", [])[:30]
            profile["community"] = entity_data.get("community")
            profile["degree"] = entity_data.get("degree", 0)

    return profile


@router.get("/path")
async def find_path(
    source: str = Query(..., alias="from"),
    target: str = Query(..., alias="to"),
    max_hops: int = 3,
):
    """Find all paths between two entities."""
    from backend.main import app_state

    graph = app_state.get("graph")
    traversal = app_state.get("graph_traversal")

    if not graph or not graph.is_ready or not traversal:
        raise HTTPException(503, "Knowledge graph not available.")

    t0 = time.time()
    paths = traversal.find_paths_between(source, target, max_hops=max_hops)
    elapsed = round((time.time() - t0) * 1000, 1)

    if not paths:
        return {
            "source": source,
            "target": target,
            "paths": [],
            "message": f"No paths found between '{source}' and '{target}' within {max_hops} hops.",
            "latency_ms": elapsed,
        }

    # Convert to dict
    path_dicts = []
    for p in paths[:10]:  # Top 10 paths
        path_dicts.append({
            "entities": p.entities,
            "relationships": p.edge_types,
            "hops": p.hops,
            "score": round(p.score, 4),
            "description": p.description,
            "evidence_chunks": p.evidence_chunks[:5],
        })

    return {
        "source": source,
        "target": target,
        "paths": path_dicts,
        "total_paths": len(paths),
        "latency_ms": elapsed,
    }


@router.get("/timeline/{name}")
async def get_entity_timeline(name: str):
    """Get chronological events involving an entity."""
    from backend.main import app_state

    entity_store = app_state.get("entity_store")
    if not entity_store or not entity_store.is_ready:
        raise HTTPException(503, "Entity store not loaded.")

    profile = entity_store.get_entity_profile(name)
    if not profile:
        raise HTTPException(404, f"Entity '{name}' not found.")

    dates = profile.get("dates_associated", [])
    co_occurring = profile.get("co_occurring_entities", [])

    # Build timeline entries
    timeline = []
    for date in sorted(dates):
        timeline.append({
            "date": date,
            "entity": name,
            "co_entities": [
                c.get("name", c) if isinstance(c, dict) else c
                for c in co_occurring[:5]
            ],
        })

    return {
        "entity": name,
        "timeline": timeline,
        "date_range": {"from": dates[0] if dates else "", "to": dates[-1] if dates else ""},
        "total_dates": len(dates),
    }


@router.get("/cluster/{entity}")
async def get_entity_cluster(entity: str):
    """Get all entities in the same Leiden community."""
    from backend.main import app_state

    graph = app_state.get("graph")
    if not graph or not graph.is_ready:
        raise HTTPException(503, "Knowledge graph not available.")

    # Find community
    comm_id = graph.node_to_community.get(entity)
    if comm_id is None:
        # Try case-insensitive
        for node in graph.graph.nodes:
            if node.lower() == entity.lower():
                comm_id = graph.node_to_community.get(node)
                entity = node
                break

    if comm_id is None:
        raise HTTPException(404, f"Entity '{entity}' not found in any community.")

    community = graph.communities.get(comm_id, {})
    nodes = community.get("nodes", [])

    # Enrich with entity store data
    entity_store = app_state.get("entity_store")
    enriched_nodes = []
    for node_name in nodes:
        node_info = {"name": node_name}
        node_data = graph.graph.nodes.get(node_name, {})
        node_info["type"] = node_data.get("label", "UNKNOWN")
        node_info["count"] = node_data.get("count", 0)

        if entity_store and entity_store.is_ready:
            profile = entity_store.lookup(node_name)
            if profile:
                node_info["dates"] = profile.get("dates_associated", [])

        enriched_nodes.append(node_info)

    enriched_nodes.sort(key=lambda x: x.get("count", 0), reverse=True)

    return {
        "entity": entity,
        "community_id": comm_id,
        "size": community.get("size", 0),
        "density": community.get("density", 0),
        "summary": community.get("summary", ""),
        "members": enriched_nodes,
    }


@router.post("/search", response_model=IntelligenceResponse)
async def intelligence_search(req: IntelligenceSearchRequest):
    """Evidence-first search with entity context and graph reasoning."""
    from backend.main import app_state

    if not app_state.get("retriever_ready"):
        raise HTTPException(503, "System is still initializing.")

    timings = {}

    # Step 1: Parse query
    t0 = time.time()
    query_parser = app_state.get("query_parser")
    parsed = None
    entity_names = []

    if query_parser:
        parsed = query_parser.parse(req.query)
        entity_names = [e.name for e in parsed.entities_mentioned]

    timings["parse_ms"] = round((time.time() - t0) * 1000, 1)

    # Step 2: Embed query
    t0 = time.time()
    embedder = app_state["embedder"]
    query_embedding = embedder.embed_query(req.query)
    timings["embed_ms"] = round((time.time() - t0) * 1000, 1)

    # Step 3: Hybrid retrieval (4-way)
    t0 = time.time()
    hybrid = app_state["hybrid_retriever"]
    results = hybrid.search(
        query=req.query,
        query_embedding=query_embedding,
        top_k=req.top_k * 3,
        entity_names=entity_names if entity_names else None,
    )
    timings["retrieval_ms"] = round((time.time() - t0) * 1000, 1)

    # Step 4: Graph traversal
    evidence_chains = []
    discovered_entities = []
    if req.include_graph and entity_names:
        t0 = time.time()
        traversal = app_state.get("graph_traversal")
        chain_gen = app_state.get("evidence_chain_gen")

        if traversal and chain_gen:
            paths = traversal.multi_hop_search(entity_names)
            evidence_chains = chain_gen.generate_chains(paths)
            evidence_chains = [c.to_dict() for c in evidence_chains]

            # Collect discovered entities from paths
            seen = set(e.lower() for e in entity_names)
            for path in paths:
                for ent in path.entities:
                    if ent.lower() not in seen:
                        seen.add(ent.lower())
                        discovered_entities.append({"name": ent, "source": "graph"})

        timings["graph_ms"] = round((time.time() - t0) * 1000, 1)

    # Step 5: Composite scoring
    t0 = time.time()
    scorer = app_state.get("composite_scorer")
    if scorer:
        graph_paths = []
        traversal = app_state.get("graph_traversal")
        if traversal and entity_names:
            graph_paths = traversal.multi_hop_search(entity_names)

        results = scorer.score_results(
            results,
            query_entities=entity_names,
            query_dates=parsed.dates_mentioned if parsed else [],
            graph_paths=graph_paths,
        )
    timings["scoring_ms"] = round((time.time() - t0) * 1000, 1)

    # Limit results
    results = results[:req.top_k]

    # Step 6: Optional LLM summary
    summary = None
    if req.use_llm:
        t0 = time.time()
        llm = app_state.get("llm")
        if llm and llm.available and results:
            from backend.generation.prompts import SYSTEM_PROMPT, build_qa_prompt
            user_prompt = build_qa_prompt(req.query, results)
            summary = llm.generate(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
        timings["llm_ms"] = round((time.time() - t0) * 1000, 1)

    # Compute confidence
    confidence = 0.0
    if results:
        top_score = results[0].get("composite_score", results[0].get("rrf_score", 0))
        confidence = min(top_score * 20, 1.0)  # Normalize

    timings["total_ms"] = round(sum(timings.values()), 1)

    return IntelligenceResponse(
        results=results,
        query_entities=[
            {"name": e.name, "type": e.type, "count": e.count}
            for e in (parsed.entities_mentioned if parsed else [])
        ],
        discovered_entities=discovered_entities[:20],
        evidence_chains=evidence_chains,
        confidence=round(confidence, 3),
        retrieval_mode="hybrid_entity" if entity_names else "hybrid",
        intent=parsed.intent if parsed else "search",
        latency=timings,
        summary=summary,
    )

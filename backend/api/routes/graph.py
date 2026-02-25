"""
Knowledge graph API routes.
Includes entity search, multi-hop traversal, path queries, and graph data.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

router = APIRouter(prefix="/api/graph", tags=["graph"])


@router.get("/entities")
async def search_entities(query: str = "", limit: int = 50):
    """Search entities in the knowledge graph."""
    from backend.main import app_state

    graph = app_state.get("graph")
    if not graph or not graph.is_ready:
        raise HTTPException(503, "Knowledge graph not built yet.")

    if query:
        return {"entities": graph.search_entities(query, limit=limit)}

    data = graph.get_graph_json()
    return {"entities": data["nodes"][:limit]}


@router.get("/entity/{name}")
async def get_entity(name: str):
    """Get details and connections for a specific entity."""
    from backend.main import app_state

    graph = app_state.get("graph")
    if not graph or not graph.is_ready:
        raise HTTPException(503, "Knowledge graph not built yet.")

    entity = graph.get_entity(name)
    if not entity:
        raise HTTPException(404, f"Entity '{name}' not found.")

    return entity


@router.get("/relationship")
async def get_relationship(entity1: str, entity2: str):
    """Find the relationship between two entities."""
    from backend.main import app_state

    graph = app_state.get("graph")
    if not graph or not graph.is_ready:
        raise HTTPException(503, "Knowledge graph not built yet.")

    result = graph.get_relationships(entity1, entity2)
    if not result:
        raise HTTPException(404, "One or both entities not found.")

    return result


@router.get("/traverse")
async def multi_hop_traverse(
    entity: str,
    max_hops: int = 2,
    min_weight: float = 1.0,
    edge_type: Optional[str] = None,
):
    """Multi-hop traversal from an entity."""
    from backend.main import app_state

    traversal = app_state.get("graph_traversal")
    if not traversal:
        raise HTTPException(503, "Graph traversal not available.")

    type_filter = [edge_type] if edge_type else None
    paths = traversal.multi_hop_search(
        start_entities=[entity],
        max_hops=max_hops,
        min_edge_weight=min_weight,
        edge_type_filter=type_filter,
    )

    return {
        "entity": entity,
        "paths": [
            {
                "entities": p.entities,
                "relationships": p.edge_types,
                "hops": p.hops,
                "score": round(p.score, 4),
                "description": p.description,
            }
            for p in paths[:30]
        ],
        "total_paths": len(paths),
    }


@router.get("/neighborhood")
async def get_neighborhood(entity: str, depth: int = 1, min_weight: float = 1.0):
    """Get entity neighborhood for visualization."""
    from backend.main import app_state

    traversal = app_state.get("graph_traversal")
    if not traversal:
        raise HTTPException(503, "Graph traversal not available.")

    return traversal.get_neighborhood(entity, depth=depth, min_weight=min_weight)


@router.get("/data")
async def get_graph_data():
    """Get full graph JSON for visualization."""
    from backend.main import app_state

    graph = app_state.get("graph")
    if not graph or not graph.is_ready:
        raise HTTPException(503, "Knowledge graph not built yet.")

    return graph.get_graph_json()

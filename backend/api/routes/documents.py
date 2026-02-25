"""
Document browsing API routes.
"""

from fastapi import APIRouter, HTTPException
from typing import Optional

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.get("")
async def list_documents(page: int = 1, per_page: int = 20, doc_type: Optional[str] = None):
    """Paginated document listing."""
    from backend.main import app_state

    loader = app_state.get("loader")
    if not loader or not loader.documents:
        raise HTTPException(503, "Documents not loaded yet.")

    return loader.list_documents(page=page, per_page=per_page, doc_type=doc_type)


@router.get("/stats")
async def get_stats():
    """Dataset statistics."""
    from backend.main import app_state

    loader = app_state.get("loader")
    if not loader:
        return {"error": "Not loaded yet"}

    stats = loader.get_stats()

    # Add pipeline stats
    stats["chunks_indexed"] = app_state.get("total_chunks", 0)
    stats["graph_nodes"] = app_state.get("graph_nodes", 0)
    stats["graph_edges"] = app_state.get("graph_edges", 0)
    stats["rag_ready"] = app_state.get("retriever_ready", False)
    stats["llm_available"] = app_state.get("llm_available", False)

    return stats


@router.get("/search")
async def search_documents(query: str, page: int = 1, per_page: int = 20):
    """Full-text keyword search across documents."""
    from backend.main import app_state

    loader = app_state.get("loader")
    if not loader or not loader.documents:
        raise HTTPException(503, "Documents not loaded yet.")

    return loader.search(query, page=page, per_page=per_page)


@router.get("/{filename:path}")
async def get_document(filename: str):
    """Get a single document by filename."""
    from backend.main import app_state

    loader = app_state.get("loader")
    if not loader:
        raise HTTPException(503, "Documents not loaded yet.")

    doc = loader.get_document(filename)
    if not doc:
        raise HTTPException(404, f"Document '{filename}' not found.")

    return doc

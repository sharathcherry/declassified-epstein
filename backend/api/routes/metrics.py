"""
Metrics API route — exposes live RAG query metrics.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get("")
async def get_metrics():
    """Return aggregated live query metrics."""
    from backend.evaluation.query_tracker import tracker
    return tracker.get_metrics()


@router.post("/reset")
async def reset_metrics():
    """Reset all tracked metrics."""
    from backend.evaluation.query_tracker import tracker
    tracker.__init__()
    return {"status": "reset"}

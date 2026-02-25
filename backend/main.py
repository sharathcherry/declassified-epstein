"""
Main FastAPI entry point.
Memory-optimized initialization pipeline with shared metadata loading.
"""

import gc
import logging
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings, ROOT_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Shared application state ───────────────────────────────────
app_state: dict = {
    "status": "initializing",
    "phase": "idle",
    "progress_message": "",
    "progress_pct": 0,
    "loader": None,
    "embedder": None,
    "vector_store": None,
    "sparse_retriever": None,
    "multi_vector_store": None,
    "hybrid_retriever": None,
    "graph": None,
    "graph_traversal": None,
    "evidence_chain_gen": None,
    "entity_store": None,
    "entity_retriever": None,
    "query_parser": None,
    "composite_scorer": None,
    "reranker": None,
    "query_rewriter": None,
    "context_compressor": None,
    "llm": None,
    "retriever_ready": False,
    "llm_available": False,
    "total_chunks": 0,
    "features": {
        "api_keys": len(settings.all_api_keys),
        "embed_mode": settings.embed_mode,
        "multi_vector": settings.enable_multi_vector,
    }
}


def _update_status(phase: str, message: str, pct: int = 0):
    app_state["phase"] = phase
    app_state["progress_message"] = message
    app_state["progress_pct"] = pct
    logger.info(f"[{phase}] {message}")


def _initialize_pipeline():
    """
    Background initialization — memory-optimized.
    Fast path: load cached FAISS + BM25 indices (~2 GB).
    Slow path: full pipeline (only if no cached indices).
    """
    try:
        from backend.retrieval.vector_store import VectorStore
        from backend.retrieval.sparse_retriever import SparseRetriever

        vector_store = VectorStore()
        sparse_retriever = SparseRetriever()

        # ── FAST PATH: load pre-built indices from disk ──────────
        _update_status("indexing", "Loading cached indices...", 10)
        faiss_ok = vector_store.load()
        bm25_ok = sparse_retriever.load()

        if faiss_ok and bm25_ok:
            _update_status("indexing", "Loaded cached FAISS + BM25 indices", 50)
            app_state["vector_store"] = vector_store
            app_state["sparse_retriever"] = sparse_retriever
            app_state["total_chunks"] = vector_store.index.ntotal

            # Share metadata: sparse retriever can use vector_store's metadata
            # instead of loading chunk_metadata.pkl twice
            if not sparse_retriever.metadata and vector_store.metadata:
                sparse_retriever.metadata = vector_store.metadata
                logger.info("Shared metadata from vector_store → sparse_retriever")

            gc.collect()

            # Embedder — needed for query-time embedding
            _update_status("embedder", "Loading embedder...", 55)
            embedder = _load_embedder()
            app_state["embedder"] = embedder

            # Mark retriever as ready early so status endpoint works
            app_state["retriever_ready"] = True
            _update_status("retriever", "Core retrieval ready!", 60)

            # ── Load remaining components (non-blocking) ──
            _load_optional_components(vector_store, sparse_retriever)

        else:
            # ── SLOW PATH: full pipeline ──
            _update_status("loading", "No cached indices — running full pipeline...", 0)
            _run_full_pipeline(vector_store, sparse_retriever)

        _update_status("ready", "System ready!", 100)
        app_state["status"] = "ready"

    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        _update_status("error", f"Initialization failed: {str(e)}", 0)
        app_state["status"] = "error"


def _load_embedder():
    """Load the appropriate embedder."""
    if settings.embed_mode == "local":
        try:
            from backend.embeddings.local_embedder import LocalEmbedder
            return LocalEmbedder(batch_size=settings.local_embed_batch_size)
        except Exception as e:
            logger.warning(f"Local embedder failed, using API: {e}")

    from backend.embeddings.nvidia_embedder import NvidiaEmbedder
    return NvidiaEmbedder()


def _load_optional_components(vector_store, sparse_retriever):
    """Load optional components (LLM, entity store, graph, etc.) with memory management."""

    # ── LLM Client ──
    _update_status("llm", "Initializing LLM client...", 62)
    llm = None
    try:
        from backend.generation.llm_client import LLMClient
        llm = LLMClient()
        app_state["llm"] = llm
        app_state["llm_available"] = llm.available
    except Exception as e:
        logger.warning(f"LLM initialization failed: {e}")

    # ── Entity Store (lazy, memory-efficient) ──
    _update_status("entity_store", "Loading entity store...", 65)
    entity_store = None
    try:
        from backend.knowledge_graph.entity_store import EntityStore
        entity_store = EntityStore()
        if entity_store.load():
            app_state["entity_store"] = entity_store
            _update_status("entity_store", f"Entity store: {len(entity_store.index):,} entities", 68)
            gc.collect()
        else:
            logger.warning("Entity index not found — entity features disabled")
    except Exception as e:
        logger.warning(f"Entity store failed: {e}")

    # ── Multi-vector store ──
    multi_vector_store = None
    if settings.enable_multi_vector:
        _update_status("multi_vector", "Loading multi-vector index...", 70)
        try:
            from backend.retrieval.multi_vector_store import MultiVectorStore
            multi_vector_store = MultiVectorStore()
            if multi_vector_store.load():
                app_state["multi_vector_store"] = multi_vector_store
                _update_status("multi_vector", "Multi-vector index loaded", 72)
            else:
                logger.info("Multi-vector index not cached — skipping (will be built on next full run)")
        except Exception as e:
            logger.warning(f"Multi-vector store failed: {e}")

    # ── Entity Retriever ──
    entity_retriever = None
    if entity_store and entity_store.is_ready:
        _update_status("entity_retriever", "Setting up entity retriever...", 73)
        try:
            from backend.retrieval.entity_retriever import EntityRetriever
            entity_retriever = EntityRetriever(entity_store)
            if sparse_retriever.metadata:
                entity_retriever.set_chunk_lookup(sparse_retriever.metadata)
            app_state["entity_retriever"] = entity_retriever
        except Exception as e:
            logger.warning(f"Entity retriever failed: {e}")

    # ── Hybrid Retriever (4-way) ──
    _update_status("retriever", "Building 4-way hybrid retriever...", 75)
    from backend.retrieval.hybrid_retriever import HybridRetriever
    hybrid = HybridRetriever(
        vector_store, sparse_retriever, multi_vector_store,
        entity_retriever=entity_retriever
    )
    app_state["hybrid_retriever"] = hybrid

    # ── Reranker ──
    _update_status("reranker", "Initializing reranker...", 78)
    try:
        from backend.retrieval.reranker import NvidiaReranker
        app_state["reranker"] = NvidiaReranker()
    except Exception as e:
        logger.warning(f"Reranker failed: {e}")

    # ── Query Parser ──
    _update_status("query_parser", "Initializing query parser...", 80)
    try:
        from backend.retrieval.query_parser import QueryParser
        app_state["query_parser"] = QueryParser(entity_store=entity_store)
    except Exception as e:
        logger.warning(f"Query parser failed: {e}")

    # ── Query Rewriter ──
    try:
        from backend.retrieval.query_rewriter import QueryRewriter
        app_state["query_rewriter"] = QueryRewriter(llm_client=llm)
    except Exception as e:
        logger.warning(f"Query rewriter failed: {e}")

    # ── Context Compressor ──
    try:
        from backend.retrieval.context_compressor import ContextCompressor
        app_state["context_compressor"] = ContextCompressor()
    except Exception as e:
        logger.warning(f"Context compressor failed: {e}")

    # ── Knowledge Graph ──
    _update_status("graph", "Loading knowledge graph...", 85)
    try:
        from backend.knowledge_graph.graph_builder import GraphBuilder
        graph = GraphBuilder()
        if graph.load():
            app_state["graph"] = graph
            _update_status("graph", f"Graph loaded: {graph.graph.number_of_nodes():,} nodes", 88)

            # Graph Traversal + Evidence Chains
            from backend.knowledge_graph.graph_traversal import GraphTraversal
            from backend.knowledge_graph.evidence_chain import EvidenceChainGenerator

            graph_traversal = GraphTraversal(graph.graph, graph.node_to_community)
            app_state["graph_traversal"] = graph_traversal

            chain_gen = EvidenceChainGenerator(entity_store=entity_store)
            if sparse_retriever.metadata:
                chain_gen.set_chunk_lookup(sparse_retriever.metadata)
            app_state["evidence_chain_gen"] = chain_gen
        else:
            logger.info("Knowledge graph not cached — graph features disabled")
    except Exception as e:
        logger.warning(f"Knowledge graph failed: {e}")

    # ── Composite Scorer ──
    _update_status("scorer", "Setting up composite scorer...", 95)
    try:
        from backend.retrieval.composite_scorer import CompositeScorer
        scorer = CompositeScorer(
            entity_store=entity_store,
            graph_builder=app_state.get("graph"),
        )
        app_state["composite_scorer"] = scorer
    except Exception as e:
        logger.warning(f"Composite scorer failed: {e}")

    gc.collect()


def _run_full_pipeline(vector_store, sparse_retriever):
    """Full pipeline: load → clean → chunk → embed → index. Memory-heavy."""
    from backend.ingestion.loader import DatasetLoader

    _update_status("loading", "Loading documents...", 5)
    loader = DatasetLoader()
    documents = loader.load()
    app_state["loader"] = loader

    _update_status("cleaning", f"Cleaning {len(documents):,} documents...", 10)
    from backend.ingestion.cleaner import TextCleaner
    documents = TextCleaner.clean_batch(documents)

    _update_status("deduplicating", "Deduplicating...", 18)
    from backend.ingestion.deduplicator import Deduplicator
    documents = Deduplicator.deduplicate(documents)
    loader.documents = documents

    _update_status("chunking", f"Chunking {len(documents):,} documents...", 22)
    from backend.ingestion.chunker import AdaptiveChunker
    chunker = AdaptiveChunker()
    doc_items = dict(list(documents.items())[:settings.max_documents]) if settings.max_documents > 0 else documents
    chunks = chunker.chunk_corpus(doc_items)
    app_state["total_chunks"] = len(chunks)

    # Free documents from memory
    del documents, doc_items
    gc.collect()

    _update_status("embedding", "Loading embedder...", 28)
    embedder = _load_embedder()
    app_state["embedder"] = embedder

    from backend.embeddings.cache import EmbeddingCache
    cache = EmbeddingCache()
    texts = [c.text for c in chunks]
    cached_results, miss_indices = cache.get_batch(texts)

    if miss_indices:
        miss_texts = [texts[i] for i in miss_indices]
        _update_status("embedding", f"Embedding {len(miss_texts):,} chunks...", 33)
        new_embeddings = embedder.embed_texts(miss_texts)
        cache.put_batch(miss_texts, new_embeddings)
        cache.save()

        new_iter = iter(new_embeddings)
        all_embeddings = []
        for cached_emb in cached_results:
            if cached_emb is not None:
                all_embeddings.append(cached_emb)
            else:
                all_embeddings.append(next(new_iter))
    else:
        all_embeddings = cached_results
        _update_status("embedding", "All embeddings from cache", 55)

    _update_status("indexing", "Building FAISS...", 58)
    vector_store.build(all_embeddings, chunks)
    vector_store.save()

    _update_status("indexing", "Building BM25...", 62)
    sparse_retriever.build(chunks)
    sparse_retriever.save()

    app_state["vector_store"] = vector_store
    app_state["sparse_retriever"] = sparse_retriever
    app_state["retriever_ready"] = True

    # Free bulk data
    del all_embeddings, texts, cached_results, chunks
    gc.collect()

    _load_optional_components(vector_store, sparse_retriever)


# ── FastAPI App ─────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background initialization on app startup."""
    thread = threading.Thread(target=_initialize_pipeline, daemon=True)
    thread.start()
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Declassified: AI-Powered Epstein Document Analysis",
    description="Open-source intelligence platform for the unsealed Epstein court documents — 4-way hybrid retrieval, entity graph reasoning, and evidence chains.",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register API routes ────────────────────────────────────────
from backend.api.routes import search, documents, chat, graph, evaluation, metrics, intelligence

app.include_router(search.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(graph.router)
app.include_router(evaluation.router)
app.include_router(metrics.router)
app.include_router(intelligence.router)


@app.get("/api/status")
async def get_status():
    return {
        "status": app_state["status"],
        "phase": app_state["phase"],
        "message": app_state["progress_message"],
        "progress": app_state["progress_pct"],
        "total_chunks": app_state["total_chunks"],
        "retriever_ready": app_state["retriever_ready"],
        "features": {
            **app_state["features"],
            "entity_store": bool(app_state.get("entity_store") and app_state["entity_store"].is_ready),
            "graph_traversal": app_state.get("graph_traversal") is not None,
            "evidence_chains": app_state.get("evidence_chain_gen") is not None,
            "composite_scorer": app_state.get("composite_scorer") is not None,
            "query_parser": app_state.get("query_parser") is not None,
        },
    }


@app.get("/")
async def root():
    return {"message": "Epstein Files Intelligence Platform v3.0 - ONLINE"}

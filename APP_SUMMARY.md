# Epstein Files Intelligence Platform — App Summary

> **Version**: 3.0.0  
> **Stack**: FastAPI (Python) · React 19 + Vite · NVIDIA NIM · FAISS · NetworkX  
> **Architecture**: Entity-centric graph intelligence retrieval system

---

## What This App Does

This is a **retrieval-first intelligence platform** built on top of the [Epstein Files 20K corpus](https://huggingface.co/datasets/teyler/epstein-files-20k) from HuggingFace. It goes beyond a traditional chatbot — entities, relationships, and evidence chains are first-class citizens, and the LLM is **optional**.

**Core Capabilities:**
- **Hybrid Search** — 4-way retrieval fusion (Dense + Sparse + Summary + Entity) with Reciprocal Rank Fusion
- **Entity Intelligence** — Entity profiles, co-occurrence graphs, timeline views, and cluster analysis
- **Knowledge Graph** — Multi-hop graph traversal to discover indirect connections between entities
- **Evidence Chains** — Auto-generated reasoning chains with confidence scores and source documents
- **RAG Chat** — Three-mode chat system: Evidence-only, Evidence + LLM Summary, or Legacy chatbot

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Frontend (React 19 + Vite)                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Chat   │  │Dashboard │  │   Metrics    │  │  Sidebar +   │   │
│  │Interface │  │          │  │  Dashboard   │  │  Connection  │   │
│  └────┬─────┘  └────┬─────┘  └──────┬───────┘  └──────────────┘   │
│       │              │               │                              │
└───────┼──────────────┼───────────────┼──────────────────────────────┘
        │              │               │
        ▼              ▼               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend (v3.0.0)                        │
│                                                                     │
│  API Routers:                                                       │
│  ┌────────────┬──────────┬────────────┬──────────────────────────┐  │
│  │ /api/chat  │/api/search│/api/graph │ /api/intelligence        │  │
│  │ /api/docs  │/api/eval  │/api/metrics│                         │  │
│  └─────┬──────┴─────┬────┴─────┬──────┴────────┬────────────────┘  │
│        │            │          │                │                    │
│  ┌─────▼────────────▼──────────▼────────────────▼──────────────┐   │
│  │               Retrieval Pipeline                              │   │
│  │  Query Parser → Hybrid (4-way RRF) → Composite Scorer        │   │
│  │  → Reranker → Evidence Chain Generator                        │   │
│  └──────────────────────┬────────────────────────────────────────┘   │
│                         │                                            │
│  ┌──────────┬───────────┼────────────┬───────────────────────┐      │
│  │  FAISS   │   BM25    │  Entity    │  Knowledge Graph      │      │
│  │  Index   │   Index   │  Store     │  (NetworkX + Leiden)   │      │
│  └──────────┴───────────┴────────────┴───────────────────────┘      │
│                                                                      │
│  External Services:                                                  │
│  ┌───────────────────────────────────────────────────────────┐       │
│  │  NVIDIA NIM: Embeddings (BGE-M3) · Reranking (Llama 3.2) │       │
│  │              LLM Generation (Llama 3.1 70B)               │       │
│  └───────────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
New folder/
├── backend/                    # FastAPI Python backend
│   ├── main.py                 # App entry point, memory-optimized startup
│   ├── config.py               # Pydantic-settings configuration
│   ├── api/routes/             # 7 API route modules
│   │   ├── chat.py             # 3-mode RAG chat (evidence/summary/legacy)
│   │   ├── search.py           # Multi-stage hybrid search
│   │   ├── intelligence.py     # Entity-centric intelligence API
│   │   ├── graph.py            # Knowledge graph traversal & visualization
│   │   ├── documents.py        # Document browsing
│   │   ├── evaluation.py       # Benchmark evaluation
│   │   └── metrics.py          # Live metrics endpoint
│   ├── retrieval/              # Search & scoring engine
│   │   ├── hybrid_retriever.py # 4-way RRF fusion
│   │   ├── vector_store.py     # FAISS dense retrieval
│   │   ├── sparse_retriever.py # BM25 sparse retrieval
│   │   ├── multi_vector_store.py # Summary-level retrieval
│   │   ├── entity_retriever.py # Entity-first retrieval (4th signal)
│   │   ├── query_parser.py     # Query decomposition & intent classification
│   │   ├── composite_scorer.py # Unified 4-component scoring
│   │   ├── reranker.py         # NVIDIA NIM reranker
│   │   ├── query_rewriter.py   # Query expansion + HyDE
│   │   ├── metadata_filter.py  # Date/type/entity filtering
│   │   └── context_compressor.py # Context window optimization
│   ├── knowledge_graph/        # Graph intelligence layer
│   │   ├── entity_extractor.py # spaCy NER + legal act/event detection
│   │   ├── entity_store.py     # In-memory entity index (JSON-backed)
│   │   ├── graph_builder.py    # NetworkX graph + Leiden communities
│   │   ├── graph_traversal.py  # Multi-hop BFS with path scoring
│   │   └── evidence_chain.py   # Human-readable reasoning chains
│   ├── generation/             # LLM answer generation
│   │   ├── llm_client.py       # NVIDIA NIM LLM (Llama 3.1 70B)
│   │   ├── prompts.py          # Grounded answer prompt templates
│   │   └── citation_engine.py  # Source citation & grounding checks
│   ├── embeddings/             # Embedding engines
│   │   ├── nvidia_embedder.py  # NVIDIA NIM API (BGE-M3)
│   │   └── local_embedder.py   # Local GPU (sentence-transformers)
│   ├── ingestion/              # Data pipeline
│   │   ├── loader.py           # HuggingFace corpus download & parsing
│   │   ├── cleaner.py          # OCR noise removal
│   │   ├── chunker.py          # Adaptive token-based chunking
│   │   └── deduplicator.py     # Near-duplicate detection
│   ├── evaluation/             # Quality metrics
│   │   ├── metrics.py          # Precision, Recall, MRR, nDCG, entity recall
│   │   ├── query_tracker.py    # Live query observability & cost tracking
│   │   └── benchmarks.py       # Benchmark test suite
│   └── tests/                  # Unit tests
│       ├── test_entity_extraction.py
│       ├── test_entity_store.py
│       ├── test_composite_scorer.py
│       └── test_graph_traversal.py / test_intelligence_pipeline.py
│
├── frontend/                   # React 19 + Vite frontend
│   └── src/
│       ├── App.jsx             # Tab-based layout (Chat, Dashboard, Metrics)
│       ├── components/
│       │   ├── ChatInterface.jsx      # RAG chat with sources & citations
│       │   ├── Dashboard.jsx          # System status & corpus stats
│       │   ├── MetricsDashboard.jsx   # Live retrieval quality metrics
│       │   ├── Sidebar.jsx            # Navigation sidebar
│       │   └── ConnectionStatus.jsx   # Backend connection indicator
│       └── services/api.js            # API client
│
├── _pipeline_scripts/          # Offline batch processing
│   ├── build_index.py          # Build FAISS + BM25 indices
│   ├── kaggle_embedder.py      # Kaggle GPU embedding
│   ├── kaggle_dual_gpu_embedder.py  # Dual GPU embedding
│   ├── kaggle_phase1_pipeline.ipynb # Full Phase 1 Kaggle notebook
│   ├── colab_ner_pipeline.ipynb     # Google Colab NER extraction
│   └── merge_caches.py         # Merge embedding caches
│
├── data/                       # Runtime data (indices, caches, models)
├── requirements.txt            # Python dependencies
└── .env                        # API keys & configuration
```

---

## Key Technical Details

### Retrieval Pipeline (7-Stage)

| Stage | Component | Description |
|-------|-----------|-------------|
| 1 | **Query Parser** | Extracts entities, dates, legal acts; classifies intent |
| 2 | **Query Rewriter** | Expands query + generates HyDE passage |
| 3 | **Embedding** | NVIDIA BGE-M3 (API) or local sentence-transformers |
| 4 | **Hybrid Retrieval** | 4-way RRF: Dense (FAISS) + Sparse (BM25) + Summary + Entity |
| 5 | **Composite Scoring** | Weighted formula: 35% semantic + 20% keyword + 25% entity + 20% graph |
| 6 | **Reranking** | NVIDIA Llama 3.2 NV-RerankQA |
| 7 | **Evidence Chains** | Graph traversal → path scoring → human-readable chains |

### Scoring Formula

```
FinalScore(d, q) = 0.35·Semantic + 0.20·Keyword + 0.25·Entity + 0.20·Graph
```

**Boosting modifiers** (multiplicative):
- Temporal relevance: +10% if dates match query
- Entity type match: +15% if entity types align
- Multi-source confirmation: +20% if entity spans multiple documents
- Community relevance: +10% if in relevant Leiden community

### Knowledge Graph

- **Nodes**: Entities (PERSON, ORG, LAW, DATE, LOCATION, EVENT)
- **Edges**: Co-occurrence with typed relationships (associated_with, employed_by, located_at, etc.)
- **Communities**: Leiden algorithm for hierarchical clustering
- **Traversal**: BFS with configurable depth (default 3 hops), hop decay factor 0.7

### Chat Modes

| Mode | Method | Output |
|------|--------|--------|
| **Evidence** | No LLM | Structured results + entity context + evidence chains |
| **Summary** | LLM | Evidence + LLM-generated comprehensive answer |
| **Legacy** | LLM | Traditional chatbot (backward compatible) |

---

## External Dependencies

### Cloud Services (NVIDIA NIM)
| Service | Model | Purpose |
|---------|-------|---------|
| Embeddings | `baai/bge-m3` | Dense vector generation |
| Reranking | `nvidia/llama-3.2-nv-rerankqa-1b-v2` | Precision reranking |
| LLM | `meta/llama-3.1-70b-instruct` | Answer generation (optional) |

### Key Python Libraries
| Library | Purpose |
|---------|---------|
| FastAPI | Web framework |
| FAISS (cpu) | Dense vector similarity search |
| rank-bm25 | Sparse keyword retrieval |
| spaCy (en_core_web_trf) | Transformer-based NER |
| NetworkX | Knowledge graph |
| leidenalg | Community detection |
| sentence-transformers | Local GPU embeddings |
| OpenAI SDK | NVIDIA NIM API client |

### Frontend
| Library | Purpose |
|---------|---------|
| React 19 | UI framework |
| Vite 7 | Build tool / dev server |

---

## API Endpoints

### Chat & Search
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | RAG chat (3 modes) |
| POST | `/api/search` | Multi-stage hybrid search |

### Intelligence (Entity-Centric)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/intelligence/entity/{name}` | Full entity profile |
| GET | `/api/intelligence/path?from=X&to=Y` | Multi-hop paths between entities |
| GET | `/api/intelligence/timeline/{name}` | Chronological entity events |
| GET | `/api/intelligence/cluster/{entity}` | Leiden community members |
| POST | `/api/intelligence/search` | Evidence-first search |

### Knowledge Graph
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/graph/entities` | Search entities |
| GET | `/api/graph/entity/{name}` | Entity details & connections |
| GET | `/api/graph/relationship` | Relationship between two entities |
| GET | `/api/graph/traverse` | Multi-hop traversal |
| GET | `/api/graph/neighborhood` | Entity neighborhood (visualization) |
| GET | `/api/graph/data` | Full graph JSON |

### System
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/status` | System health & initialization progress |
| GET | `/api/documents` | Browse corpus documents |
| GET | `/api/metrics` | Live retrieval quality metrics |
| POST | `/api/evaluation/run` | Run benchmark evaluation |

---

## Data Flow

```
HuggingFace Dataset (20K docs)
        │
        ▼
    Download & Parse (loader.py)
        │
        ▼
    Clean OCR Noise (cleaner.py)
        │
        ▼
    Adaptive Chunking (chunker.py, 512/1024 tokens)
        │
        ▼
    Deduplicate (deduplicator.py)
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
    Embed (NVIDIA BGE-M3 or local)      NER Pipeline (spaCy trf)
        │                                      │
        ├──── FAISS Index (dense)              ├── Entity Store (JSON)
        ├──── BM25 Index (sparse)              ├── Knowledge Graph (NetworkX)
        └──── Multi-Vector (summaries)         └── Community Detection (Leiden)
```

---

## Running the App

```bash
# Backend
cd "New folder"
python -m uvicorn backend.main:app --port 8000

# Frontend
cd frontend
npm run dev
```

The backend loads cached indices on startup (FAISS, BM25, entity store, knowledge graph) and initializes all retrieval components in a memory-optimized background thread.

---

## Offline Processing (Kaggle/Colab)

Heavy compute tasks (embedding 20K docs, NER extraction) are offloaded to Kaggle or Google Colab notebooks:

| Script | Platform | Purpose |
|--------|----------|---------|
| `kaggle_phase1_pipeline.ipynb` | Kaggle | Full Phase 1: load → clean → chunk → embed |
| `kaggle_dual_gpu_embedder.py` | Kaggle | Dual-GPU batch embedding |
| `colab_ner_pipeline.ipynb` | Colab | Named Entity Recognition extraction |
| `build_index.py` | Local | Build FAISS + BM25 from cached embeddings |
| `merge_caches.py` | Local | Merge partial embedding caches |

---
title: Declassified - AI-Powered Epstein Document Analysis
emoji: 🔍
colorFrom: gray
colorTo: red
sdk: docker
app_port: 7860
---

# Declassified: AI-Powered Epstein Document Analysis

Open-source intelligence platform for searching, analyzing, and understanding the unsealed Jeffrey Epstein court documents using entity-centric graph reasoning and 4-way hybrid retrieval.

## Features
- 🔍 4-way hybrid retrieval (dense + sparse + summary + entity)
- 🕸️ Knowledge graph with multi-hop traversal
- 📊 Evidence chain generation with confidence scores
- 🤖 Optional LLM summarization (NVIDIA NIM)
- 📈 Entity profiles, timelines, and cluster exploration

## API Endpoints
- `GET /api/status` — System health
- `POST /api/search` — Hybrid search with entity context
- `POST /api/chat` — RAG chat (evidence/summary/legacy modes)
- `GET /api/intelligence/entity/{name}` — Entity profiles
- `GET /api/graph/data` — Knowledge graph visualization data

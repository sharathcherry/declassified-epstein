"""
RAG Chat API route: full multi-stage pipeline with three modes.
1. Evidence mode (default): Return structured evidence without LLM
2. Summary mode: Evidence + LLM-generated summary
3. Legacy mode: Current chatbot behavior (backward compatible via raw_mode=False)
Supports raw_mode to bypass LLM and return raw source documents.
"""

import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from backend.generation.prompts import SYSTEM_PROMPT, build_qa_prompt
from backend.generation.citation_engine import CitationEngine

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str
    top_k: int = 15
    raw_mode: bool = False  # Skip LLM, return raw documents
    mode: str = "legacy"  # "evidence", "summary", "legacy"
    doc_type: Optional[str] = None
    entities: Optional[list[str]] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    confidence: str
    num_sources: int
    is_grounded: bool
    warning: Optional[str] = None
    query_rewritten: Optional[str] = None
    community_context: Optional[str] = None
    follow_up_questions: list[str] = []
    latency: dict
    # New fields for entity-centric output
    query_entities: list[dict] = []
    evidence_chains: list[dict] = []
    discovered_entities: list[dict] = []


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Full multi-stage RAG pipeline with three modes:
    - evidence: Structured evidence without LLM
    - summary: Evidence + LLM summary
    - legacy: Original chatbot behavior (default)
    """
    from backend.main import app_state
    from backend.evaluation.query_tracker import tracker

    if not app_state.get("retriever_ready"):
        raise HTTPException(503, "System is still initializing. Please wait.")

    timings = {}
    rewritten_query = None
    community_ctx = None

    try:
        # -- Step 1: Query Parsing --
        t0 = time.time()
        query_parser = app_state.get("query_parser")
        entity_names = req.entities or []
        parsed = None
        query_entities = []

        if query_parser:
            parsed = query_parser.parse(req.question)
            if not entity_names:
                entity_names = [e.name for e in parsed.entities_mentioned]
            query_entities = [
                {"name": e.name, "type": e.type, "count": e.count}
                for e in parsed.entities_mentioned
            ]
        timings["parse_ms"] = round((time.time() - t0) * 1000, 1)

        # -- Step 2: Query Rewriting --
        t0 = time.time()
        query_rewriter = app_state.get("query_rewriter")
        rewrite_result = None
        hyde_embedding = None

        if query_rewriter and not req.raw_mode:
            rewrite_result = query_rewriter.rewrite(req.question)
            rewritten_query = rewrite_result.get("expanded")
        else:
            rewrite_result = {"original": req.question, "expanded": req.question,
                              "hyde_passage": "", "all_queries": [req.question]}

        timings["rewrite_ms"] = round((time.time() - t0) * 1000, 1)

        # -- Step 3: Embed query --
        t0 = time.time()
        embedder = app_state["embedder"]
        search_query = rewrite_result.get("expanded", req.question)
        query_embedding = embedder.embed_query(search_query)

        if rewrite_result.get("hyde_passage"):
            hyde_embedding = embedder.embed_query(rewrite_result["hyde_passage"])

        timings["embed_ms"] = round((time.time() - t0) * 1000, 1)

        # -- Step 4: Hybrid Retrieval (4-way RRF) --
        t0 = time.time()
        hybrid = app_state["hybrid_retriever"]
        fetch_k = req.top_k * 5 if not req.raw_mode else req.top_k * 3
        results = hybrid.search(
            query=search_query,
            query_embedding=query_embedding,
            top_k=fetch_k,
            doc_type_filter=req.doc_type,
            hyde_embedding=hyde_embedding,
            entity_names=entity_names if entity_names else None,
        )
        timings["retrieval_ms"] = round((time.time() - t0) * 1000, 1)

        # -- Step 5: Metadata Filtering --
        t0 = time.time()
        from backend.retrieval.metadata_filter import MetadataFilter
        auto_filters = MetadataFilter.parse_filter_from_query(req.question)
        results = MetadataFilter.apply(
            results,
            doc_type=req.doc_type or auto_filters.get("doc_type"),
            entities=req.entities,
            date_from=req.date_from or auto_filters.get("date_from"),
            date_to=req.date_to or auto_filters.get("date_to"),
        )
        timings["filter_ms"] = round((time.time() - t0) * 1000, 1)

        # -- Step 6: Composite scoring --
        t0 = time.time()
        scorer = app_state.get("composite_scorer")
        if scorer and entity_names:
            results = scorer.score_results(
                results,
                query_entities=entity_names,
                query_dates=parsed.dates_mentioned if parsed else [],
            )
        timings["scoring_ms"] = round((time.time() - t0) * 1000, 1)

        # -- Step 7: Rerank --
        t0 = time.time()
        reranker = app_state.get("reranker")
        if reranker and results:
            results = reranker.rerank(req.question, results, top_k=req.top_k)
        else:
            results = results[:req.top_k]
        timings["rerank_ms"] = round((time.time() - t0) * 1000, 1)

        # -- Step 8: Context Compression (skip in raw/evidence mode) --
        t0 = time.time()
        if not req.raw_mode and req.mode not in ("evidence",):
            compressor = app_state.get("context_compressor")
            if compressor:
                results = compressor.compress(req.question, results)
        timings["compress_ms"] = round((time.time() - t0) * 1000, 1)

        # -- Step 9: GraphRAG Community Context --
        t0 = time.time()
        graph = app_state.get("graph")
        if graph and graph.is_ready:
            community_ctx = graph.get_community_context(req.question)
        timings["graph_ms"] = round((time.time() - t0) * 1000, 1)

        # -- Step 10: Evidence chains --
        evidence_chains = []
        discovered_entities = []
        if entity_names:
            t0 = time.time()
            chain_gen = app_state.get("evidence_chain_gen")
            traversal = app_state.get("graph_traversal")
            if chain_gen and traversal:
                paths = traversal.multi_hop_search(entity_names, max_hops=2)
                chains = chain_gen.generate_chains(paths, max_chains=3)
                evidence_chains = [c.to_dict() for c in chains]

                # Collect discovered entities
                seen = set(e.lower() for e in entity_names)
                for path in paths[:10]:
                    for ent in path.entities:
                        if ent.lower() not in seen:
                            seen.add(ent.lower())
                            discovered_entities.append({"name": ent, "source": "graph"})
            timings["evidence_ms"] = round((time.time() - t0) * 1000, 1)

        # ================================================================
        # RAW MODE: Skip LLM, return raw source documents
        # ================================================================
        if req.raw_mode:
            timings["generation_ms"] = 0.0

            raw_sections = []
            for i, r in enumerate(results):
                filename = r.get("doc_filename", r.get("filename", "Unknown"))
                if filename == "Unknown":
                    text_content = r.get("text", "")
                    filename = _extract_doc_id(text_content)

                text = _clean_ocr_text(r.get("text", "No text available"))
                score = r.get("composite_score", r.get("rrf_score", r.get("score", 0)))
                sources_list = r.get("retrieval_sources", [])

                raw_sections.append(
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"📄 **[Source {i+1}]** — {filename}\n"
                    f"Score: {score:.4f} | Via: {', '.join(sources_list) if sources_list else 'hybrid'}\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"{text}\n"
                )

            answer = (
                f"🔍 **RAW DOCUMENT RETRIEVAL** — {len(results)} sources for: \"{req.question}\"\n"
                f"No LLM processing. Showing full, unmodified source text.\n\n"
                + "\n".join(raw_sections)
            )

            timings["total_ms"] = round(sum(timings.values()), 1)

            source_list = []
            for r in results:
                fn = r.get("doc_filename", r.get("filename", "Unknown"))
                if fn == "Unknown":
                    fn = _extract_doc_id(r.get("text", ""))
                source_list.append({
                    "filename": fn,
                    "text": r.get("text", "")[:200],
                    "doc_type": r.get("doc_type", ""),
                    "score": r.get("composite_score", r.get("rrf_score", r.get("score", 0))),
                })

            return ChatResponse(
                answer=answer,
                sources=source_list,
                confidence="raw",
                num_sources=len(results),
                is_grounded=True,
                warning=None,
                query_rewritten=rewritten_query,
                community_context=community_ctx if community_ctx else None,
                follow_up_questions=[],
                latency=timings,
                query_entities=query_entities,
                evidence_chains=evidence_chains,
                discovered_entities=discovered_entities[:10],
            )

        # ================================================================
        # EVIDENCE MODE: Structured evidence without LLM
        # ================================================================
        if req.mode == "evidence":
            timings["generation_ms"] = 0.0
            timings["total_ms"] = round(sum(timings.values()), 1)

            # Build structured answer from evidence
            evidence_sections = []
            for i, r in enumerate(results[:5]):
                filename = r.get("doc_filename", r.get("filename", "Unknown"))
                if filename == "Unknown":
                    filename = _extract_doc_id(r.get("text", ""))
                text = r.get("text", "")[:500]
                score = r.get("composite_score", r.get("rrf_score", 0))
                evidence_sections.append(
                    f"**[{i+1}]** {filename} (score: {score:.3f})\n{text}..."
                )

            answer = (
                f"📋 **Evidence for:** \"{req.question}\"\n\n"
                + "\n\n".join(evidence_sections)
            )

            if evidence_chains:
                answer += "\n\n🔗 **Evidence Chains:**\n"
                for chain in evidence_chains[:3]:
                    answer += f"- {chain['path_description']} (confidence: {chain['confidence']:.2f})\n"

            source_list = [{
                "filename": r.get("doc_filename", r.get("filename", "Unknown")),
                "text": r.get("text", "")[:200],
                "doc_type": r.get("doc_type", ""),
                "score": r.get("composite_score", r.get("rrf_score", r.get("score", 0))),
            } for r in results]

            return ChatResponse(
                answer=answer,
                sources=source_list,
                confidence="evidence",
                num_sources=len(results),
                is_grounded=True,
                query_rewritten=rewritten_query,
                community_context=community_ctx,
                follow_up_questions=[],
                latency=timings,
                query_entities=query_entities,
                evidence_chains=evidence_chains,
                discovered_entities=discovered_entities[:10],
            )

        # ================================================================
        # LLM MODE (legacy / summary): Generate answer from sources
        # ================================================================
        t0 = time.time()
        llm = app_state.get("llm")
        if llm and llm.available:
            user_prompt = build_qa_prompt(req.question, results)
            if community_ctx:
                user_prompt = community_ctx + "\n\n" + user_prompt
            answer = llm.generate(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
        else:
            answer = (
                "LLM not available. Here are the most relevant document excerpts:\n\n"
                + "\n\n".join(
                    f"**[Source {i+1}]** ({r.get('doc_filename', 'Unknown')}): "
                    f"{r.get('text', '')[:300]}..."
                    for i, r in enumerate(results[:5])
                )
            )
        timings["generation_ms"] = round((time.time() - t0) * 1000, 1)

        # -- Parse follow-up questions from answer --
        follow_ups = []
        if "FOLLOW_UP_QUESTIONS:" in answer:
            parts = answer.split("FOLLOW_UP_QUESTIONS:", 1)
            answer = parts[0].strip()
            fq_text = parts[1].strip()
            for line in fq_text.split("\n"):
                line = line.strip().lstrip("- ").strip()
                if line and len(line) > 5:
                    follow_ups.append(line)

        # -- Citation extraction + grounding check --
        citation_result = CitationEngine.extract_citations(answer, results)

        timings["total_ms"] = round(sum(timings.values()), 1)

        # -- Record live metrics --
        tracker.record(req.question, timings, results, citation_result)

        return ChatResponse(
            answer=citation_result["answer"],
            sources=citation_result["citations"],
            confidence=citation_result["confidence"],
            num_sources=citation_result["num_sources"],
            is_grounded=citation_result["is_grounded"],
            warning=citation_result["warning"],
            query_rewritten=rewritten_query,
            community_context=community_ctx if community_ctx else None,
            follow_up_questions=follow_ups[:3],
            latency=timings,
            query_entities=query_entities,
            evidence_chains=evidence_chains,
            discovered_entities=discovered_entities[:10],
        )

    except Exception as e:
        tracker.record_failure(req.question, str(e))
        raise


def _extract_doc_id(text: str) -> str:
    """Extract document reference ID from chunk text."""
    import re
    patterns = [
        r'(HOUSE_OVERSIGHT_\d+)',
        r'(GM_\d+)',
        r'(DOJ_\d+)',
        r'(FBI_\d+)',
        r'(EPSTEIN_\d+)',
        r'(MAXWELL_\d+)',
        r'(GIUFFRE_\d+)',
        r'([A-Z]{2,}_[A-Z]*_?\d{3,})',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return "Document"


def _clean_ocr_text(text: str) -> str:
    """Clean OCR noise from raw document text."""
    import re

    if not text:
        return text

    text = re.sub(r'\d{2}:\d{2}:\d{2}\s*', '', text)
    text = re.sub(r'(?m)^\s*\d{1,2}\s*$', '', text)
    text = re.sub(r'(?m)^\s*\d{1,2}\s+(?=[A-Z])', '', text)

    garbage_patterns = [
        r'Oo O DN OO FF WwW NY =\|',
        r'NO RO PO PNP NM NO \| S\| S\| HS SF S\| S\| S\| S\| S\|',
        r'non BP WO NO -\|- ODO OO WDN OO OT BP WO NYO —',
        r'On Oa bh .*?\d+',
        r'O[nN]\s*O[aA]\s*[bB][hH]\s*[wW][hH]\s*=',
        r'[oO][rR]\d{4}\s*\d+',
        r'\d{2}:\d{2}:\d{2}\s+\d{1,2}\b',
    ]
    for pat in garbage_patterns:
        text = re.sub(pat, '', text)

    noise_strings = [
        'ROUGH DRAFT ONLY',
        'ESQUIRE DEPOSITION SOLUTIONS',
        '(954) 331-4400',
        'Page \\d+ to \\d+ of \\d+',
        r'\d+ of \d+ sheets',
    ]
    for ns in noise_strings:
        text = re.sub(ns, '', text, flags=re.IGNORECASE)

    text = re.sub(r'\n\s*HOUSE_OVERSIGHT_\d+\s*$', '', text)
    text = re.sub(r'\n\s*GM_\d+\s*$', '', text)
    text = re.sub(r'"\s*$', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'http:/\S+', '', text)
    text = re.sub(r'(?m)^\s*\d{2,3}\s*$', '', text)
    text = re.sub(r'(?m)^\d{2}:\d{2}:\d{2}\s+\d+\s+', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'  +', ' ', text)

    lines = [line.strip() for line in text.split('\n')]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()

    return '\n'.join(lines)

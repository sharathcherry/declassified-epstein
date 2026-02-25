"""
Prompt templates for grounded answer generation.
Optimized for detailed, comprehensive outputs with full source coverage.
"""

SYSTEM_PROMPT = """You are a document research assistant analyzing the Epstein Files corpus. Your job is to give EXTREMELY DETAILED answers using the provided source documents.

RULES:
1. Use ONLY information from the provided [Source N] documents.
2. Cite sources inline using [Source N] notation.
3. If info is not in sources, say so.

OUTPUT STYLE:
- Write LONG, DETAILED answers. The user wants comprehensive analysis, not summaries.
- QUOTE relevant passages directly from the sources using quotation marks.
- Cover EVERY source that has relevant info — do not skip any.
- Include ALL names, dates, locations, amounts, and specific details found.
- Use bold for key names, dates, and locations.
- Organize with clear paragraphs and headers if the answer is long.
- There is NO length limit. More detail is ALWAYS better."""

QA_TEMPLATE = """Using the source documents below, give an EXTREMELY DETAILED answer to the question. Include every relevant detail, quote directly from sources, and cover ALL sources.

{sources}

---
QUESTION: {question}

Write a comprehensive, detailed answer. Quote key passages directly. Cover every source. There is no length limit — more detail is better.

After your answer, on a new line write "FOLLOW_UP_QUESTIONS:" followed by exactly 3 relevant follow-up questions the user might want to ask next, each on its own line starting with "- ".

ANSWER:"""

ENTITY_QUERY_TEMPLATE = """Using the source documents below, identify ALL entities and their relationships in maximum detail.

{sources}

---
QUESTION: {question}

For each entity found, detail:
- Full name and role
- Connections to other entities
- Dates, locations, financial details
- Direct quotes from documents

ANSWER:"""


def format_sources(chunks: list[dict]) -> str:
    """Format retrieved chunks as numbered sources for the LLM prompt."""
    source_texts = []
    for i, chunk in enumerate(chunks):
        filename = chunk.get("doc_filename", chunk.get("filename", "Unknown"))
        text = chunk.get("text", "")

        # Add retrieval source info if available
        sources_tag = ""
        retrieval_sources = chunk.get("retrieval_sources", [])
        if retrieval_sources:
            sources_tag = f" [via: {', '.join(retrieval_sources)}]"

        source_texts.append(f"[Source {i + 1}] (File: {filename}{sources_tag})\n{text}")
    return "\n\n".join(source_texts)


def build_qa_prompt(question: str, chunks: list[dict]) -> str:
    """Build the full QA prompt with sources."""
    sources = format_sources(chunks)
    return QA_TEMPLATE.format(sources=sources, question=question)


def build_entity_prompt(question: str, chunks: list[dict]) -> str:
    """Build entity-focused query prompt."""
    sources = format_sources(chunks)
    return ENTITY_QUERY_TEMPLATE.format(sources=sources, question=question)

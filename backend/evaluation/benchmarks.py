"""
Benchmark queries for evaluation.
Covers factual lookup, entity relationships, timeline, and financial queries.
"""

BENCHMARK_QUERIES = [
    # ── Factual Lookup ──────────────────────────────────────
    {
        "id": "fact_01",
        "query": "What properties did Jeffrey Epstein own?",
        "category": "factual",
        "expected_keywords": ["property", "island", "ranch", "mansion", "New York", "Palm Beach"],
    },
    {
        "id": "fact_02",
        "query": "Who were the pilots mentioned in flight logs?",
        "category": "factual",
        "expected_keywords": ["pilot", "flight", "log"],
    },
    {
        "id": "fact_03",
        "query": "What legal proceedings are documented?",
        "category": "factual",
        "expected_keywords": ["court", "case", "lawsuit", "deposition", "trial"],
    },
    {
        "id": "fact_04",
        "query": "What organizations are mentioned in the documents?",
        "category": "factual",
        "expected_keywords": ["foundation", "company", "organization", "LLC"],
    },

    # ── Entity Relationships ────────────────────────────────
    {
        "id": "entity_01",
        "query": "Who is Ghislaine Maxwell and what role did she play?",
        "category": "entity",
        "expected_keywords": ["Ghislaine", "Maxwell"],
    },
    {
        "id": "entity_02",
        "query": "What connections between individuals are documented?",
        "category": "entity",
        "expected_keywords": ["connection", "relationship", "associate"],
    },
    {
        "id": "entity_03",
        "query": "Which lawyers represented the parties involved?",
        "category": "entity",
        "expected_keywords": ["attorney", "lawyer", "counsel", "law firm"],
    },

    # ── Timeline / Chronological ────────────────────────────
    {
        "id": "time_01",
        "query": "What events occurred in 2005?",
        "category": "timeline",
        "expected_keywords": ["2005"],
    },
    {
        "id": "time_02",
        "query": "What is the chronology of legal actions taken?",
        "category": "timeline",
        "expected_keywords": ["date", "filed", "court", "order"],
    },

    # ── Financial ───────────────────────────────────────────
    {
        "id": "fin_01",
        "query": "What financial transactions are documented?",
        "category": "financial",
        "expected_keywords": ["payment", "transfer", "account", "money", "fund"],
    },
    {
        "id": "fin_02",
        "query": "What amounts of money are mentioned in the documents?",
        "category": "financial",
        "expected_keywords": ["$", "million", "thousand", "payment"],
    },

    # ── Anti-hallucination (should result in "not found") ──
    {
        "id": "anti_01",
        "query": "What is the GDP of France?",
        "category": "anti_hallucination",
        "expected_keywords": ["not found", "could not find", "not in the", "no information"],
    },
    {
        "id": "anti_02",
        "query": "What is the weather forecast for tomorrow?",
        "category": "anti_hallucination",
        "expected_keywords": ["not found", "could not find", "not in the", "no information"],
    },
    {
        "id": "anti_03",
        "query": "Explain quantum computing",
        "category": "anti_hallucination",
        "expected_keywords": ["not found", "could not find", "not in the", "no information"],
    },
]

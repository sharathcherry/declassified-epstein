"""
Centralized configuration via pydantic-settings.
All secrets come from .env — nothing is hardcoded.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root (one level above backend/)
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── NVIDIA NIM ──────────────────────────────────────────
    nvidia_api_key: str = ""
    nvidia_api_keys_extra: str = ""  # Comma-separated extra keys for rotation
    nvidia_embed_model: str = "baai/bge-m3"
    nvidia_rerank_model: str = "nvidia/llama-3.2-nv-rerankqa-1b-v2"
    nvidia_llm_model: str = "meta/llama-3.1-70b-instruct"
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"

    # ── App ─────────────────────────────────────────────────
    max_documents: int = 0  # 0 = ALL, no limit
    api_port: int = 8000

    # ── Chunking ────────────────────────────────────────────
    chunk_size_short: int = 512    # tokens for short docs (was 256)
    chunk_size_long: int = 1024   # tokens for long docs (was 512)
    chunk_overlap: int = 100      # token overlap
    long_doc_threshold: int = 3000  # chars to count as "long" (was 2000)

    # ── Retrieval ───────────────────────────────────────────
    dense_top_k: int = 50
    sparse_top_k: int = 50
    summary_top_k: int = 30       # Multi-vector summary index
    rerank_top_k: int = 10
    rrf_k: int = 60  # RRF constant

    # ── Embedding batching ──────────────────────────────────
    embed_batch_size: int = 96
    embed_mode: str = "api"           # "api" = NVIDIA cloud, "local" = GPU
    local_embed_batch_size: int = 32  # Optimal for RTX 3050 Ti (4GB, fp16)

    # ── Advanced Features ───────────────────────────────────
    spacy_model: str = "en_core_web_trf"
    enable_query_rewrite: bool = True
    enable_context_compression: bool = True
    enable_multi_vector: bool = True
    graphrag_resolution: float = 1.0  # Leiden community resolution

    # ── Entity-Centric Retrieval ────────────────────────────
    entity_boost_weight: float = 2.0      # BM25 entity token boost
    graph_max_hops: int = 3               # Multi-hop traversal depth
    graph_hop_decay: float = 0.7          # Score decay per hop
    entity_top_k: int = 20                # Entity retrieval count
    evidence_chain_max: int = 5           # Max evidence chains per query
    composite_weights_semantic: float = 0.35
    composite_weights_keyword: float = 0.20
    composite_weights_entity: float = 0.25
    composite_weights_graph: float = 0.20

    @property
    def has_nvidia_key(self) -> bool:
        return bool(self.nvidia_api_key and self.nvidia_api_key != "nvapi-your_key_here")

    @property
    def all_api_keys(self) -> list[str]:
        """All available NVIDIA API keys for rotation."""
        keys = []
        if self.has_nvidia_key:
            keys.append(self.nvidia_api_key)
        if self.nvidia_api_keys_extra:
            for k in self.nvidia_api_keys_extra.split(","):
                k = k.strip()
                if k and k != "nvapi-your_key_here":
                    keys.append(k)
        return keys


settings = Settings()

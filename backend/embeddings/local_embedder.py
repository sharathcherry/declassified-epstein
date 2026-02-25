"""
Local GPU embedding client using sentence-transformers.
Runs BGE-M3 on your GPU — no API calls, no rate limits, no cost.

VRAM-safe: Aggressively clears GPU memory between micro-batches
to prevent spilling into shared RAM on low-VRAM GPUs (4GB).
"""

import gc
import logging
import time
from typing import Optional

from backend.config import settings

logger = logging.getLogger(__name__)


class LocalEmbedder:
    """
    Generate embeddings locally using sentence-transformers + BGE-M3.
    Optimized for low-VRAM GPUs (4GB) with fp16, small micro-batches,
    and aggressive CUDA memory cleanup.
    """

    def __init__(self, batch_size: int = 32):
        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError:
            raise ImportError(
                "Install sentence-transformers: pip install sentence-transformers"
            )

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        vram_gb = 0

        if self.device == "cuda":
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")

            # Micro-batch sizes tuned per VRAM tier
            # Safe at 32 because convert_to_numpy + empty_cache prevent accumulation
            if vram_gb < 5:
                batch_size = 16   # 4GB VRAM — sweet spot for long texts
            elif vram_gb < 8:
                batch_size = 32
            else:
                batch_size = 64
        else:
            logger.warning("No CUDA GPU found — running on CPU (slow)")
            batch_size = 8

        self.batch_size = batch_size

        logger.info("Loading BGE-M3 model...")
        self.model = SentenceTransformer(
            "BAAI/bge-m3",
            device=self.device,
            model_kwargs={"torch_dtype": "float16"} if self.device == "cuda" else {},
        )

        # Force fp16 on low-VRAM GPUs
        if self.device == "cuda" and vram_gb < 6:
            self.model.half()
            logger.info("Using fp16 for low-VRAM GPU")

        self.dimension = self.model.get_sentence_embedding_dimension()
        self._log_vram("Model loaded")
        logger.info(
            f"Local embedder ready: device={self.device}, "
            f"batch_size={self.batch_size}, dim={self.dimension}"
        )

    def _log_vram(self, label: str = ""):
        """Log current VRAM usage."""
        if self.device == "cuda":
            alloc = self._torch.cuda.memory_allocated() / 1e9
            reserved = self._torch.cuda.memory_reserved() / 1e9
            logger.info(f"[VRAM {label}] allocated={alloc:.2f} GB, reserved={reserved:.2f} GB")

    def _flush_gpu(self):
        """Aggressively flush GPU memory between micro-batches."""
        if self.device == "cuda":
            self._torch.cuda.empty_cache()
            gc.collect()

    def embed_texts(
        self,
        texts: list[str],
        input_type: str = "passage",
        progress_callback=None,
    ) -> list[list[float]]:
        """
        Embed texts locally on GPU/CPU with VRAM-safe micro-batching.

        Processes texts in small micro-batches (8 at a time on 4GB GPUs),
        converting GPU tensors to CPU/list immediately and flushing CUDA
        cache every 10 micro-batches to prevent VRAM overflow.
        """
        total = len(texts)
        logger.info(f"Embedding {total:,} texts locally (micro-batch={self.batch_size})")

        # BGE-M3 uses instruction prefix for queries
        if input_type == "query":
            processed = [
                f"Represent this sentence for searching relevant passages: {t}"
                for t in texts
            ]
        else:
            processed = texts

        start_time = time.time()
        all_embeddings = []
        flush_interval = 10  # Flush CUDA cache every N micro-batches

        for i in range(0, total, self.batch_size):
            batch = processed[i:i + self.batch_size]

            # Truncate to model max length
            batch = [t[:8192] for t in batch]

            # Encode and IMMEDIATELY move to CPU as Python lists
            embeddings = self.model.encode(
                batch,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,  # numpy on CPU, not GPU tensors
            )

            # Convert to plain Python lists (free GPU refs)
            all_embeddings.extend(embeddings.tolist())

            # Flush CUDA cache periodically to reclaim fragmented VRAM
            micro_batch_num = i // self.batch_size
            if micro_batch_num > 0 and micro_batch_num % flush_interval == 0:
                self._flush_gpu()

            if progress_callback and (i + self.batch_size) % (self.batch_size * 10) == 0:
                progress_callback(len(all_embeddings), total)

        # Final flush
        self._flush_gpu()

        elapsed = time.time() - start_time
        rate = total / elapsed if elapsed > 0 else 0
        logger.info(
            f"Embedded {total:,} texts in {elapsed:.1f}s "
            f"({rate:.0f} texts/sec)"
        )
        self._log_vram("After batch complete")

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        result = self.embed_texts([query], input_type="query")
        return result[0]

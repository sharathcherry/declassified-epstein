#!/bin/bash
set -e

echo "=== Downloading data files from HF Dataset ==="
python - <<'EOF'
import os
from huggingface_hub import hf_hub_download

os.makedirs("data", exist_ok=True)

REPO_ID = "sharathcherry03/declassified-data"
FILES = [
    "faiss_index.bin",
    "bm25_index.pkl",
    "chunk_metadata.pkl",
    "entity_metadata.json",
]

for fname in FILES:
    dest = f"data/{fname}"
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        continue
    print(f"  Downloading {fname}...")
    hf_hub_download(
        repo_id=REPO_ID,
        filename=fname,
        repo_type="dataset",
        local_dir="data",
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"  Done: {dest}")

print("=== All data files ready ===")
EOF

echo "=== Starting uvicorn ==="
exec python -m uvicorn backend.main:app --host 0.0.0.0 --port 7860

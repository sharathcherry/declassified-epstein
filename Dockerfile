FROM python:3.12-slim

WORKDIR /app

# System deps for spaCy + FAISS
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model (small model for production — trf is too heavy)
RUN python -m spacy download en_core_web_sm

# Copy app code
COPY backend/ backend/

# Copy runtime data files only
COPY data/faiss_index.bin data/faiss_index.bin
COPY data/bm25_index.pkl data/bm25_index.pkl
COPY data/chunk_metadata.pkl data/chunk_metadata.pkl
COPY data/entity_metadata.json data/entity_metadata.json

# HF Spaces expects port 7860
ENV PORT=7860
EXPOSE 7860

# Run uvicorn
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]

FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt huggingface_hub

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy app code only (no data files in image)
COPY backend/ backend/

# Startup script that downloads data from HF Dataset then runs uvicorn
COPY start.sh start.sh
RUN chmod +x start.sh

ENV PORT=7860
EXPOSE 7860

CMD ["./start.sh"]

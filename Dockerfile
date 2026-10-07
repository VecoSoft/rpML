# FastAPI ML service. CPU-only PyTorch (sentence-transformers would otherwise pull the multi-GB
# CUDA build). The embedding model downloads from Hugging Face on first use into HF_HOME, which
# docker-compose.prod.yml mounts as a volume so it survives rebuilds.

FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app ./app
RUN useradd --system --create-home appuser && mkdir -p /models && chown appuser /models
USER appuser
EXPOSE 8081
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8081", "--workers", "1"]

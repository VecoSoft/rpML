# FastAPI ML service, run by gunicorn with uvicorn workers. Multi-stage: dependencies are installed
# into a virtualenv in the build stage, and only that venv and the app are copied into the runtime
# image. CPU-only PyTorch (sentence-transformers would otherwise pull the multi-GB CUDA build).
# The embedding model downloads from Hugging Face on first use into HF_HOME, which
# docker-compose.prod.yml mounts as a volume so it survives rebuilds.

FROM python:3.11-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /venv
ENV PATH=/venv/bin:$PATH
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1
COPY requirements.txt .
RUN pip install -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
ENV PATH=/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models
RUN useradd --system --uid 10001 --create-home appuser && mkdir -p /models && chown appuser /models
COPY --from=build /venv /venv
COPY --chown=appuser app ./app
USER appuser
EXPOSE 8081
HEALTHCHECK --interval=20s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8081/health', timeout=4).status == 200 else 1)"
# One worker: each worker loads its own copy of the embedding model, and the container has 768 MB.
# The long timeout covers the first request, which loads the model.
CMD ["gunicorn", "app.main:app", "--worker-class", "uvicorn.workers.UvicornWorker", "--workers", "1", \
     "--bind", "0.0.0.0:8081", "--timeout", "120", "--graceful-timeout", "30", "--access-logfile", "-"]

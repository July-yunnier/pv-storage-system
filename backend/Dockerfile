# syntax=docker/dockerfile:1
# Build from the repository root: docker build -f backend/Dockerfile .
FROM python:3.12-slim-bookworm AS dependencies

ARG INSTALL_FORECAST=false
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /build
COPY backend/requirements-runtime.txt backend/requirements-docker-forecast.txt ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install -r requirements-runtime.txt \
    && case "$INSTALL_FORECAST" in \
         true) /opt/venv/bin/pip install -r requirements-docker-forecast.txt \
               && /opt/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.5,<3' ;; \
         false) ;; \
         *) echo 'INSTALL_FORECAST must be true or false' >&2; exit 1 ;; \
       esac \
    && /opt/venv/bin/pip check \
    && /opt/venv/bin/pip freeze > /opt/venv/requirements-installed.txt \
    && /opt/venv/bin/python -c "import importlib.util; assert importlib.util.find_spec('pytest') is None"

FROM python:3.12-slim-bookworm AS runtime

ARG INSTALL_FORECAST=false
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STORAGE_DB_PATH=/app/data/storage.db
WORKDIR /app
# LightGBM needs the OpenMP runtime; no compiler or research/test toolchain is copied.
RUN case "$INSTALL_FORECAST" in \
      true) apt-get update \
            && apt-get install -y --no-install-recommends libgomp1 \
            && rm -rf /var/lib/apt/lists/* ;; \
      false) ;; \
      *) exit 1 ;; \
    esac \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home app \
    && install -d -o app -g app /app/data
COPY --from=dependencies /opt/venv /opt/venv
COPY backend/app/ ./app/
COPY backend/agents/ ./agents/
COPY backend/scripts/ ./scripts/
COPY backend/pv_inference/*.py backend/pv_inference/manifest.json ./pv_inference/
# seed scripts resolve repository configs relative to /app's parent (/).
COPY configs/*.yaml /configs/
COPY THIRD_PARTY_NOTICES.md /app/THIRD_PARTY_NOTICES.md

# Existing application code enables WAL and a 5-second lock wait. Persist the
# whole directory (storage.db + -wal + -shm), writable by UID/GID 10001; one worker.
USER app:app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=3).close()"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

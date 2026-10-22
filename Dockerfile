FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd --create-home --uid 10001 rag && mkdir -p /app/data && chown rag:rag /app/data
COPY core core
COPY ingest ingest
COPY retrieve retrieve
COPY serve serve
COPY eval eval
COPY examples examples
COPY tools tools
COPY config.yaml config.yaml
USER rag
EXPOSE 8000
CMD ["uvicorn", "serve.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]

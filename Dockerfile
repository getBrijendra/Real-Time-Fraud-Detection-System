FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FRAUD_HOME=/app \
    PYTHONPATH=/app/src \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY configs ./configs
COPY src ./src
COPY scripts ./scripts

EXPOSE 8000
CMD ["uvicorn", "fraud.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

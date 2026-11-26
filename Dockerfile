# Gateway + pipeline image (CPU). Models are NOT baked in: mount ./models (see docs/RUN_FOR_REAL.md).
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY shared ./shared
COPY gateway ./gateway
COPY services ./services
COPY pipeline ./pipeline
RUN pip install --no-cache-dir ".[nmt,tts]"
ENV BACKEND_MODE=real
CMD ["python", "-m", "pipeline.server", "run", "--room", "demo"]

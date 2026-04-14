# Base Layer: Optimized for ML inference
FROM python:3.9-slim

# Work Dir: Isolated application environment
WORKDIR /app

# Dependency Layer: Pre-installing libraries to cache layers
# Added prometheus-client for real-time telemetry
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code Layer: Synchronization of enterprise logic
COPY src/ ./src/
COPY api/ ./api/

# Persistence: Preparing the mount point for model weights and baseline metrics
RUN mkdir artifacts

# Network: Exposing port 8000 for FastAPI and Prometheus scraping
EXPOSE 8000

# Execution: Launching with 0.0.0.0 to enable cross-container communication
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
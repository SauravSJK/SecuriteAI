# =================================================================
# SecuriteAI: Inference Engine Image
# Base: Optimized Python 3.10 slim for ML inference
# =================================================================
FROM python:3.10-slim

WORKDIR /app

# 1. INFRASTRUCTURE LAYER: System dependencies for heavy libraries
# 'build-essential' is required for compiling 'torch' and 'pandas' dependencies.
# 'curl' is used for API health checks and prometheus scrapes.
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. DEPENDENCY LAYER: Leveraging Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. CODE LAYER: Full synchronization of enterprise logic
COPY . .

# 4. PERSISTENCE LAYER: GRC-compliant directory structure
# These folders must exist to store 'Feedback' and 'Metrics'.
RUN mkdir -p artifacts/weights artifacts/parameters artifacts/feedback visualizations

# 5. NETWORK LAYER: Exposing the Inference and Metrics port
EXPOSE 8000

# 6. EXECUTION LAYER: High-concurrency ASGI server
# Starts with 0.0.0.0 to enable communication with the Nginx Load Balancer.
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
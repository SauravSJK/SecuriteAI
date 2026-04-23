FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for torch and pandas
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure artifact directories exist
RUN mkdir -p artifacts/weights artifacts/parameters artifacts/feedback visualizations

EXPOSE 8000

# Default command starts the API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
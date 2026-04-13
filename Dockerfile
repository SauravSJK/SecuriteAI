# Base Layer: Start with a lightweight Python environment
FROM python:3.9-slim

# Work Dir: Set the primary folder for the app
WORKDIR /app

# Dependency Layer: Install libraries first (this speeds up future builds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code Layer: Copy the logic files into the container
# Copy individual files to keep the container clean
COPY app.py autoencoder.py clean_log.py feat_eng.py ./

# Create the models directory inside the container
RUN mkdir models

# Execution: Start the FastAPI server when the container boots
# Use 0.0.0.0 to allow the container to communicate with the real computer
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]